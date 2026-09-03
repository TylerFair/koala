"""Load and slice replayable spectroscopic sampler-stage input dumps."""

from __future__ import annotations

import importlib
import os
import pickle
import sys
from dataclasses import dataclass, replace
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import numpyro


SCHEMA_VERSION = 1


def _resolve_identity(identity: str):
    """Resolve ``module.qualname`` while tolerating already-loaded test modules."""
    parts = identity.split(".")
    for module_end in range(len(parts), 0, -1):
        module_name = ".".join(parts[:module_end])
        try:
            module = sys.modules.get(module_name) or importlib.import_module(
                module_name
            )
        except (ImportError, ModuleNotFoundError):
            continue
        value = module
        try:
            for part in parts[module_end:]:
                if part == "<locals>":
                    raise AttributeError(identity)
                value = getattr(value, part)
        except AttributeError:
            continue
        return value
    raise ImportError(f"Could not resolve dumped callable identity {identity!r}.")


def _rebuild_model(spec: Mapping[str, Any]):
    callable_value = _resolve_identity(str(spec["identity"]))
    if bool(spec.get("callable_is_model", False)):
        return callable_value
    kwargs = dict(spec.get("kwargs", {}))
    # Dumps created before the quadratic-uniform prior change have no basis
    # field and must replay the historical U(0,1)^2 coefficient prior.  New
    # pipeline dumps serialize the explicit default in their builder kwargs.
    if (kwargs.get("ld_mode") == "uniform"
            and kwargs.get("ld_profile") == "quadratic"
            and "ld_uniform_basis" not in kwargs):
        kwargs["ld_uniform_basis"] = "coefficients"
    return callable_value(**kwargs)


def _slice_value(value, channel_slice: slice, num_channels: int):
    if (
        isinstance(value, (np.ndarray, jax.Array))
        and value.ndim > 0
        and value.shape[0] == num_channels
    ):
        return value[channel_slice]
    return value


def _slice_meta(meta, channel_slice, num_channels):
    result = dict(meta)
    for key in ("wavelength", "wavelength_err"):
        if key in result:
            result[key] = _slice_value(result[key], channel_slice, num_channels)
    start, stop, step = channel_slice.indices(num_channels)
    result["source_channel_slice"] = [start, stop, step]
    result["num_channels"] = len(range(start, stop, step))
    return result


def _as_key(value):
    array = jnp.asarray(value)
    if array.shape == (2,) and array.dtype == jnp.uint32:
        return array
    # New-style typed keys expose their raw data for serialization. The dump
    # currently stores legacy PRNGKey arrays, but this keeps the reader useful
    # if the producer switches APIs later.
    if array.shape == (2,):
        return array.astype(jnp.uint32)
    raise ValueError(f"Unsupported serialized JAX RNG key shape {array.shape}.")


@dataclass(frozen=True)
class StageInputs:
    """An exact, importable NumPyro stage plus its production inputs."""

    model: Any
    t: Any
    yerr: Any
    y: Any
    init_params: Mapping[str, Any]
    nuts_kwargs: Mapping[str, Any]
    mcmc_kwargs: Mapping[str, Any]
    model_kwargs: Mapping[str, Any]
    chunk_size: int
    meta: Mapping[str, Any]
    channel_varying_kwargs: tuple[str, ...]
    sampler_backend: str
    rng_key: Any
    model_builder: Mapping[str, Any]
    initial_potential: float | None = None
    source_path: str | None = None

    @property
    def num_channels(self) -> int:
        return int(np.shape(self.y)[0])

    @property
    def num_cadences(self) -> int:
        return int(np.shape(self.y)[1])

    def select(self, start: int, end: int) -> "StageInputs":
        """Select channels exactly as ``fit_jwst.get_samples_chunked`` does."""
        if not (0 <= int(start) < int(end) <= self.num_channels):
            raise ValueError(
                f"Channel slice must satisfy 0 <= start < end <= "
                f"{self.num_channels}; received {start}:{end}."
            )
        channel_slice = slice(int(start), int(end))
        varying = frozenset(self.channel_varying_kwargs)
        init_params = {
            name: _slice_value(value, channel_slice, self.num_channels)
            for name, value in self.init_params.items()
        }
        model_kwargs = {
            name: (
                _slice_value(value, channel_slice, self.num_channels)
                if name in varying
                else value
            )
            for name, value in self.model_kwargs.items()
        }
        return replace(
            self,
            yerr=self.yerr[channel_slice],
            y=self.y[channel_slice],
            init_params=init_params,
            model_kwargs=model_kwargs,
            chunk_size=min(self.chunk_size, int(end) - int(start)),
            meta=_slice_meta(
                self.meta, channel_slice, self.num_channels
            ),
            initial_potential=None,
        )

    def recompute_initial_potential(self) -> float:
        init_strategy = dict(self.nuts_kwargs).get(
            "init_strategy",
            numpyro.infer.init_to_value(values=self.init_params),
        )
        model_info = numpyro.infer.util.initialize_model(
            _as_key(self.rng_key),
            self.model,
            init_strategy=init_strategy,
            dynamic_args=False,
            model_args=(self.t, self.yerr),
            model_kwargs={"y": self.y, **dict(self.model_kwargs)},
            validate_grad=False,
        )
        value = model_info.potential_fn(model_info.param_info.z)
        return float(np.asarray(jax.device_get(value)))


def select_channel_slice(inputs: StageInputs, start: int, end: int) -> StageInputs:
    """Functional spelling of :meth:`StageInputs.select`."""
    return inputs.select(start, end)


def load_stage_inputs(
    path: str,
    *,
    validate_potential: bool = True,
    potential_atol: float = 1.0e-8,
    builder_overrides=None,
) -> StageInputs:
    """Load a dump, rebuild its model, and verify the stored initial potential."""
    source_path = os.path.abspath(os.fspath(path))
    with open(source_path, "rb") as stream:
        payload = pickle.load(stream)
    schema = int(payload.get("schema_version", -1))
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported sampler-stage dump schema {schema}; expected "
            f"{SCHEMA_VERSION}."
        )
    model_builder = dict(payload["model_builder"])
    if builder_overrides:
        # Validate the potential with the recorded builder first, then rebuild
        # with the overrides; an override that changes a prior legitimately
        # changes the initial potential.
        if validate_potential:
            load_stage_inputs(
                path,
                validate_potential=True,
                potential_atol=potential_atol,
            )
        validate_potential = False
        model_builder = dict(model_builder)
        model_builder["kwargs"] = {
            **dict(model_builder.get("kwargs", {})),
            **dict(builder_overrides),
        }
    result = StageInputs(
        model=_rebuild_model(model_builder),
        t=payload["t"],
        yerr=payload["yerr"],
        y=payload["indiv_y"],
        init_params=dict(payload["init_params"]),
        nuts_kwargs=dict(payload.get("nuts_kwargs", {})),
        mcmc_kwargs=dict(payload.get("mcmc_kwargs", {})),
        model_kwargs=dict(payload.get("model_kwargs", {})),
        chunk_size=int(payload["chunk_size"]),
        meta=dict(payload.get("meta", {})),
        channel_varying_kwargs=tuple(
            payload.get("channel_varying_kwargs", ())
        ),
        sampler_backend=str(payload.get("sampler_backend", "joint_nuts")),
        rng_key=_as_key(payload["rng_key"]),
        model_builder=model_builder,
        initial_potential=float(payload["initial_potential"]),
        source_path=source_path,
    )
    if validate_potential:
        measured = result.recompute_initial_potential()
        expected = float(result.initial_potential)
        if not np.isclose(measured, expected, rtol=0.0, atol=potential_atol):
            raise ValueError(
                "Rebuilt model initial potential does not match the dump: "
                f"measured={measured:.17g}, stored={expected:.17g}, "
                f"absolute difference={abs(measured - expected):.3e}."
            )
    return result
