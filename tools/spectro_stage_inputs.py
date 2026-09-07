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


def _component_series(array, num_channels):
    array = np.asarray(array)
    if array.ndim < 2 or array.shape[1] != num_channels:
        return []
    trailing = array.shape[2:]
    if not trailing:
        return [("", array)]
    return [
        (
            "[" + ",".join(str(index) for index in component) + "]",
            array[(slice(None), slice(None), *component)],
        )
        for component in np.ndindex(trailing)
    ]


def _reference_channels(array, start, end, candidate_channels):
    if array.ndim < 2 or array.shape[1] == candidate_channels:
        return array
    if array.shape[1] >= end:
        return array[:, start:end]
    raise ValueError(
        f"Reference channel axis {array.shape[1]} cannot provide {start}:{end}."
    )


def _safe_scaled_shift(value, reference, sigma):
    difference = float(value - reference)
    if sigma > 0.0:
        return difference / sigma
    return 0.0 if difference == 0.0 else float(np.copysign(np.inf, difference))


def compare_samples(candidate, reference, start, end):
    """Compare matching per-channel posterior components from two stage runs."""
    num_channels = end - start
    rows = []
    for site in sorted(set(candidate) & set(reference)):
        candidate_components = dict(
            _component_series(np.asarray(candidate[site]), num_channels)
        )
        reference_components = dict(
            _component_series(
                _reference_channels(
                    np.asarray(reference[site]), start, end, num_channels
                ),
                num_channels,
            )
        )
        for suffix in sorted(set(candidate_components) & set(reference_components)):
            for channel in range(num_channels):
                cand = np.asarray(candidate_components[suffix][:, channel], dtype=float)
                ref = np.asarray(reference_components[suffix][:, channel], dtype=float)
                cand_q16, cand_median, cand_q84 = np.percentile(cand, [16, 50, 84])
                ref_q16, ref_median, ref_q84 = np.percentile(ref, [16, 50, 84])
                ref_sigma = float(np.std(ref, ddof=1)) if ref.size > 1 else 0.0
                cand_sigma = float(np.std(cand, ddof=1)) if cand.size > 1 else 0.0
                median_z = abs(_safe_scaled_shift(cand_median, ref_median, ref_sigma))
                sigma_ratio = (
                    cand_sigma / ref_sigma
                    if ref_sigma > 0.0
                    else (1.0 if cand_sigma == 0.0 else float("inf"))
                )
                passed = bool(median_z < 0.1 and 0.9 <= sigma_ratio <= 1.1)
                rows.append({
                    "site": f"{site}{suffix}",
                    "channel": start + channel,
                    "abs_median_shift_ref_sigma": median_z,
                    "sigma_ratio": sigma_ratio,
                    "p16_shift_ref_sigma": _safe_scaled_shift(
                        cand_q16, ref_q16, ref_sigma
                    ),
                    "p84_shift_ref_sigma": _safe_scaled_shift(
                        cand_q84, ref_q84, ref_sigma
                    ),
                    "pass": passed,
                })
    if not rows:
        raise ValueError("Candidate and reference have no comparable per-channel sites.")
    return {
        "pass": all(row["pass"] for row in rows),
        "num_rows": len(rows),
        "num_failed": sum(not row["pass"] for row in rows),
        "rows": rows,
    }
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
        """Select channels exactly as ``koala.sampling.get_samples_chunked`` does."""
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
