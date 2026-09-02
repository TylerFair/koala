"""Shared CPU-only utilities for the NUTS geometry diagnostic.

This module deliberately imports JAX only after the caller has selected the
CPU platform.  It contains no production-pipeline hooks and never submits
work to Slurm.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PRIOR_WIDTHS = {
    "rors": math.sqrt(0.5) - math.sqrt(1.0e-5),
    "log_jitter": math.log(1.0) - math.log(1.0e-6),
    "c1": 1.0,
    "c2": 1.0 - 0.001,
    "c": 0.2,
    "v": 0.2,
    "A": 0.2,
    "A_spot": 1.5,
    "log_tau": math.log(1.0e-1) - math.log(1.0e-3),
    "t0_0": None,  # set from the time span for white light
    "_b_0": 4.0,
    "logD_0": math.log(1.0) - math.log(0.0007),
}


@dataclass
class Problem:
    model: Any
    t: Any
    yerr: Any
    y: Any
    init_params: dict[str, Any]
    model_kwargs: dict[str, Any]
    channel_varying: tuple[str, ...]
    metadata: dict[str, Any]


def json_dump(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)


def jsonable(value: Any) -> Any:
    """Convert NumPy/JAX-heavy trees to strict-JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    try:
        return jsonable(np.asarray(value))
    except Exception:
        return str(value)


def block(tree: Any) -> Any:
    import jax

    return jax.block_until_ready(tree)


def make_spectro_problem(
    channels: int = 40,
    cadences: int = 230,
    seed: int = 20260901,
    *,
    negligible_jitter: bool = False,
    reference_channels: int | None = None,
) -> Problem:
    """Create the production-shaped power-2/informed-LD linear problem.

    The default path reuses the repository benchmark's exact realization.
    ``negligible_jitter`` regenerates the observations at the prior floor and
    is used only to isolate the weakly identified jitter direction.
    """
    import jax
    import jax.numpy as jnp
    from numpyro.handlers import seed as seed_handler
    from numpyro.handlers import substitute, trace

    from tools.benchmark_independent_nuts_gpu import make_problem

    (
        model,
        t,
        yerr,
        y,
        init_params,
        model_kwargs,
        channel_varying,
        metadata,
    ) = make_problem(
        channels,
        cadences,
        seed,
        reference_channels=(channels if reference_channels is None else reference_channels),
    )
    metadata = dict(metadata)
    metadata.update(
        {
            "kind": "synthetic_spectroscopic",
            "channels": int(channels),
            "cadences": int(cadences),
            "seed": int(seed),
            "negligible_jitter": bool(negligible_jitter),
        }
    )
    if negligible_jitter:
        phase = jnp.asarray(metadata["synthetic_channel_phase"], dtype=jnp.float64)
        truth = {
            "rors": (0.145 + 0.002 * phase)[:, None],
            "c1": 0.52 + 0.05 * phase,
            "c2": 0.42 - 0.04 * phase,
            "c": 1.0 + 2.0e-4 * phase,
            "v": 1.5e-3 * phase,
            "log_jitter": jnp.full((channels,), jnp.log(1.0e-6)),
        }
        conditioned = substitute(model, data=truth)
        tr = trace(seed_handler(conditioned, jax.random.PRNGKey(seed + 20))).get_trace(
            t, yerr, y=None, **model_kwargs
        )
        loc = tr["obs"]["fn"].loc
        scale = tr["obs"]["fn"].scale
        noise = jax.random.normal(
            jax.random.PRNGKey(seed + 21), loc.shape, dtype=jnp.float64
        )
        y = loc + scale * noise
        init_params = dict(init_params)
        # The midpoint is the production-style valid initialization.  Starting
        # at the prior floor would map to -infinity in unconstrained space.
        init_params["log_jitter"] = jnp.log(
            jnp.asarray(model_kwargs["precomputed_yerr_per_lc"])
        )
    return Problem(
        model=model,
        t=t,
        yerr=yerr,
        y=y,
        init_params=dict(init_params),
        model_kwargs=dict(model_kwargs),
        channel_varying=tuple(channel_varying),
        metadata=metadata,
    )


def make_whitelight_problem(cadences: int = 2000, seed: int = 20260902) -> Problem:
    import jax
    import jax.numpy as jnp
    from numpyro.handlers import seed as seed_handler
    from numpyro.handlers import substitute, trace

    from models.jaxoplanet import create_whitelight_model

    duration = 0.11693087083333333
    period = 4.05528043
    t = jnp.linspace(-0.30, 0.30, cadences, dtype=jnp.float64)
    yerr = jnp.full((cadences,), 8.0e-5, dtype=jnp.float64)
    prior_params = {
        "period": jnp.asarray([period], dtype=jnp.float64),
        "ecc": jnp.asarray([0.0], dtype=jnp.float64),
        "omega": jnp.asarray([0.0], dtype=jnp.float64),
        "u": jnp.asarray([0.52, 0.42], dtype=jnp.float64),
        "u_sigma": jnp.asarray([0.05, 0.05], dtype=jnp.float64),
    }
    model = create_whitelight_model(
        detrend_type="linear",
        n_planets=1,
        ld_profile="power2",
        ld_mode="informed",
        param_method="duration",
        jaxoplanet_kernel="auto",
    )
    truth = {
        "t0_0": jnp.asarray(0.0),
        "rors_0": jnp.asarray(0.145),
        "_b_0": jnp.asarray(0.4498),
        "logD_0": jnp.asarray(math.log(duration)),
        "c1": jnp.asarray(0.52),
        "c2": jnp.asarray(0.42),
        "log_jitter": jnp.asarray(math.log(4.0e-5)),
        "c": jnp.asarray(1.0),
        "v": jnp.asarray(7.0e-4),
    }
    tr = trace(seed_handler(substitute(model, data=truth), jax.random.PRNGKey(seed))).get_trace(
        t, yerr, y=None, prior_params=prior_params
    )
    loc = tr["obs"]["fn"].loc
    scale = tr["obs"]["fn"].scale
    y = loc + scale * jax.random.normal(
        jax.random.PRNGKey(seed + 1), loc.shape, dtype=jnp.float64
    )
    init = dict(truth)
    # Mimic the current three-stage optimizer outcome: close to the optimum,
    # but leave enough displacement for the explicit MAP solve to be measured.
    init.update(
        {
            "t0_0": jnp.asarray(1.0e-4),
            "rors_0": jnp.asarray(0.144),
            "_b_0": jnp.asarray(0.46),
            "logD_0": jnp.asarray(math.log(duration * 1.01)),
            "c": jnp.asarray(1.0),
            "v": jnp.asarray(0.0),
        }
    )
    return Problem(
        model=model,
        t=t,
        yerr=yerr,
        y=y,
        init_params=init,
        model_kwargs={"prior_params": prior_params},
        channel_varying=(),
        metadata={
            "kind": "synthetic_whitelight",
            "cadences": int(cadences),
            "seed": int(seed),
            "truth": jsonable(truth),
            "current_fit_initialization": (
                "MAP-like start matching fit_jwst.py three-stage optimx solve"
            ),
        },
    )


def load_real_spectro_problem(path: str | Path, channels: int = 40) -> Problem:
    """Load the exact first production block from a replayable stage dump."""
    from tools.spectro_stage_inputs import load_stage_inputs

    stage = load_stage_inputs(str(path), validate_potential=False)
    selected = stage.select(0, min(int(channels), stage.num_channels))
    meta = dict(selected.meta)
    return Problem(
        model=selected.model,
        t=selected.t,
        yerr=selected.yerr,
        y=selected.y,
        init_params=dict(selected.init_params),
        model_kwargs=dict(selected.model_kwargs),
        channel_varying=tuple(selected.channel_varying_kwargs),
        metadata={
            "kind": "real_stage_dump",
            "source_path": str(selected.source_path),
            "stage_label": meta.get("stage_label"),
            "stage_kind": meta.get("stage_kind"),
            "instrument": meta.get("instrument"),
            "channels_in_dump": int(stage.num_channels),
            "channels_selected": int(selected.num_channels),
            "cadences": int(selected.num_cadences),
            "active_window_cadences": meta.get("active_window_cadences"),
            "model_builder_identity": selected.model_builder.get("identity"),
            "model_builder_kwargs": {
                key: value
                for key, value in selected.model_builder.get("kwargs", {}).items()
                if key != "transit_window_indices"
            },
            "dump_nuts_kwargs": dict(selected.nuts_kwargs),
            "dump_mcmc_kwargs": dict(selected.mcmc_kwargs),
        },
    )


def slice_lane(problem: Problem, channel: int) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Return one explicit-channel-axis spectroscopic lane."""
    import jax.numpy as jnp

    err = jnp.asarray(problem.yerr[channel : channel + 1])
    obs = jnp.asarray(problem.y[channel : channel + 1])
    kwargs: dict[str, Any] = {}
    varying = frozenset(problem.channel_varying)
    for name, value in problem.model_kwargs.items():
        array = jnp.asarray(value) if value is not None else None
        kwargs[name] = array[channel : channel + 1] if name in varying else array
    init = {}
    for name, value in problem.init_params.items():
        array = jnp.asarray(value)
        if array.ndim > 0 and array.shape[0] == problem.yerr.shape[0]:
            init[name] = array[channel : channel + 1]
        else:
            init[name] = array
    return err, obs, kwargs, init


def latent_labels(z: Mapping[str, Any]) -> list[str]:
    labels: list[str] = []
    for name in sorted(z):
        shape = tuple(np.shape(z[name]))
        size = int(np.prod(shape, dtype=int))
        if size == 1:
            labels.append(name)
        else:
            labels.extend(f"{name}[{i}]" for i in range(size))
    return labels


def site_base(label: str) -> str:
    return label.split("[", 1)[0]


def summarize_steps(steps: np.ndarray) -> dict[str, Any]:
    steps = np.asarray(steps)
    result = {
        "shape": list(steps.shape),
        "mean": float(np.mean(steps)),
        "median": float(np.median(steps)),
        "p05": float(np.quantile(steps, 0.05)),
        "p25": float(np.quantile(steps, 0.25)),
        "p75": float(np.quantile(steps, 0.75)),
        "p95": float(np.quantile(steps, 0.95)),
        "max": int(np.max(steps)),
    }
    if steps.ndim == 2:
        lane_mean = np.mean(steps, axis=1)
        lane_max = np.max(steps, axis=1)
        result.update(
            {
                "lane_mean_steps_per_draw": float(np.mean(lane_mean)),
                "lane_max_steps_per_draw": float(np.mean(lane_max)),
                "lane_max_to_mean_ratio": float(np.mean(lane_max) / np.mean(lane_mean)),
                "per_lane_mean": np.mean(steps, axis=0).tolist(),
            }
        )
    return result


def sample_moments(x: np.ndarray) -> dict[str, float]:
    from scipy.stats import kurtosis, skew

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "sd": float(np.std(x, ddof=1)),
        "skewness": float(skew(x, bias=False)),
        "excess_kurtosis": float(kurtosis(x, fisher=True, bias=False)),
    }


def effective_sample_sizes(draws: Mapping[str, Any], latent_names: set[str] | None = None) -> dict[str, Any]:
    import jax.numpy as jnp
    from numpyro.diagnostics import effective_sample_size

    result: dict[str, Any] = {}
    flat_values: list[float] = []
    for name, value in draws.items():
        if latent_names is not None and name not in latent_names:
            continue
        array = np.asarray(value)
        if array.shape[0] < 4:
            continue
        ess = np.asarray(effective_sample_size(jnp.asarray(array)[None, ...]))
        result[name] = ess.tolist()
        flat_values.extend(np.ravel(ess).tolist())
    flat = np.asarray(flat_values, dtype=np.float64)
    result["summary"] = {
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }
    return result


def posterior_agreement(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    latent_names: set[str],
) -> dict[str, Any]:
    rows = []
    for name in sorted(latent_names & candidate.keys() & reference.keys()):
        cand = np.asarray(candidate[name])
        ref = np.asarray(reference[name])
        med_c = np.squeeze(np.median(cand, axis=0))
        med_r = np.squeeze(np.median(ref, axis=0))
        sd_c = np.squeeze(np.std(cand, axis=0, ddof=1))
        sd_r = np.squeeze(np.std(ref, axis=0, ddof=1))
        scale = np.maximum(sd_r, np.finfo(np.float64).tiny)
        med_z = np.abs(med_c - med_r) / scale
        sd_z = np.abs(sd_c - sd_r) / scale
        for index in np.ndindex(np.shape(med_z)):
            suffix = "" if np.shape(med_z) == () else "[" + ",".join(map(str, index)) + "]"
            rows.append(
                {
                    "site": name + suffix,
                    "median_delta_sigma": float(np.asarray(med_z)[index]),
                    "sd_delta_sigma": float(np.asarray(sd_z)[index]),
                }
            )
    med = np.asarray([r["median_delta_sigma"] for r in rows])
    sig = np.asarray([r["sd_delta_sigma"] for r in rows])
    return {
        "rows": rows,
        "max_median_delta_sigma": float(np.max(med)),
        "max_sd_delta_sigma": float(np.max(sig)),
        "fraction_medians_within_0p1_sigma": float(np.mean(med <= 0.1)),
        "fraction_sds_within_0p1_sigma": float(np.mean(sig <= 0.1)),
        "all_within_0p1_sigma": bool(np.all(med <= 0.1) and np.all(sig <= 0.1)),
    }


def timed_calls(fn: Any, args: tuple[Any, ...], repeats: int) -> dict[str, Any]:
    values = []
    block(fn(*args))
    for _ in range(repeats):
        start = time.perf_counter()
        block(fn(*args))
        values.append(time.perf_counter() - start)
    array = np.asarray(values)
    return {
        "repeats": int(repeats),
        "seconds": values,
        "median_seconds": float(np.median(array)),
        "min_seconds": float(np.min(array)),
        "max_seconds": float(np.max(array)),
    }
