#!/usr/bin/env python3
"""Benchmark and autotune production-shaped spectroscopic sampler choices.

The suite measures, rather than projects, every requested combination of
sampler backend, trend treatment, and resident channel width.  Each candidate
runs in a fresh subprocess so an OOM is recorded without killing the suite and
GPU peak-memory counters are not contaminated by earlier candidates.

The synthetic problem uses the same jaxoplanet power-2, transit-window, jitter,
and linear-trend model used by ``fit_jwst.py``.  This is a performance pilot,
not a replacement for posterior validation on representative real visits.

Example
-------
python tools/benchmark_spectro_samplers.py --platform gpu \
  --resident-widths 8,16,24,32,40,60,80 \
  --warmup 300 --samples 300 --json a100_spectro_benchmark.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
import os
import platform as python_platform
import resource
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.spectro_benchmark_core import (
    SCHEMA_VERSION,
    VALID_BACKENDS,
    VALID_TREND_MODES,
    atomic_write_json,
    build_measured_comparisons,
    build_trend_comparisons,
    candidate_id,
    parse_choice_list,
    parse_positive_int_list,
    select_candidates,
    summarize_timing,
)
from tools.spectro_manifest_tools import (
    build_batch_plan_from_chunks,
    build_width_selection_from_results,
)


COMPILE_EVENTS = {
    "/jax/core/compile/jaxpr_trace_duration": "jaxpr_trace_seconds",
    "/jax/core/compile/jaxpr_to_mlir_module_duration": "mlir_lowering_seconds",
    "/jax/core/compile/backend_compile_duration": "backend_compile_seconds",
}


class _CompilationEvents:
    def __init__(self):
        self._events: list[tuple[str, float]] = []
        self._lock = threading.Lock()

    def listener(self, event: str, duration_secs: float, **_: Any) -> None:
        if event in COMPILE_EVENTS:
            with self._lock:
                self._events.append((event, float(duration_secs)))

    def mark(self) -> int:
        with self._lock:
            return len(self._events)

    def since(self, mark: int) -> dict[str, Any]:
        totals = {name: 0.0 for name in COMPILE_EVENTS.values()}
        counts = {name.replace("_seconds", "_count"): 0 for name in totals}
        with self._lock:
            selected = list(self._events[mark:])
        for event, duration in selected:
            name = COMPILE_EVENTS[event]
            totals[name] += duration
            counts[name.replace("_seconds", "_count")] += 1
        return {
            **totals,
            **counts,
            "compile_seconds": sum(totals.values()),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument(
        "--backends", default=",".join(VALID_BACKENDS),
        help="Comma-separated sampler backends.",
    )
    parser.add_argument(
        "--trend-modes", default=",".join(VALID_TREND_MODES),
        help="Comma-separated trend treatments.",
    )
    parser.add_argument("--channels", type=int, default=40)
    parser.add_argument(
        "--resident-widths",
        default=None,
        help=(
            "Comma-separated representative chunk widths. Each candidate uses "
            "that many real channels; defaults to --channels."
        ),
    )
    parser.add_argument("--cadences", type=int, default=513)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument(
        "--repeat-runs", type=int, default=2,
        help="Process-local repeats after the cold run used for steady-state timing.",
    )
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--hmc-num-steps", type=int, default=16)
    parser.add_argument("--target-accept", type=float, default=0.8)
    parser.add_argument(
        "--trend-prior-scale", type=float, default=0.1,
        help="Gaussian standard deviation for marginalized c and v.",
    )
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--max-divergence-fraction", type=float, default=0.01)
    parser.add_argument(
        "--max-tree-depth-saturation-fraction", type=float, default=0.05,
        help="Reject NUTS candidates whose worst lane saturates this often.",
    )
    parser.add_argument("--min-channel-ess", type=float, default=50.0)
    parser.add_argument(
        "--max-device-memory-gib", type=float, default=None,
        help="Reject candidates whose measured peak device use exceeds this limit.",
    )
    parser.add_argument("--json", dest="json_path", default="spectro_sampler_benchmark.json")
    parser.add_argument(
        "--emit-width-selection",
        default=None,
        metavar="PATH",
        help="Write a fingerprinted WidthSelection manifest consumable by fit_jwst.py.",
    )
    parser.add_argument(
        "--device-memory-limit-gib",
        "--device-total-memory-gib",
        dest="device_total_memory_gib",
        type=float,
        default=None,
        help=(
            "Override JAX's allocator bytes_limit when constructing a width "
            "manifest. Normally inferred from the measured GPU."
        ),
    )
    parser.add_argument("--width-memory-fraction", type=float, default=0.70)
    parser.add_argument("--manifest-backend", choices=VALID_BACKENDS, default="independent_nuts")
    parser.add_argument(
        "--manifest-trend-mode",
        choices=VALID_TREND_MODES,
        default="gaussian_marginalized",
    )
    parser.add_argument(
        "--pilot-chunk",
        action="append",
        default=[],
        metavar="START:STOP=PATH",
        help=(
            "Independent pilot diagnostic and its exact global channel range; "
            "repeat for every chunk. Comma-separated exact indices are also accepted."
        ),
    )
    parser.add_argument(
        "--emit-batch-plan",
        default=None,
        metavar="PATH",
        help="Aggregate --pilot-chunk inputs into a ChannelBatchPlan manifest.",
    )
    parser.add_argument(
        "--batch-plan-width",
        type=int,
        default=None,
        help="Nominal ordinary-channel width; defaults to emitted WidthSelection.",
    )
    parser.add_argument("--quarantine-width", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    # Private subprocess protocol. These options are intentionally not part of
    # the user-facing workflow; a fresh interpreter owns each GPU case.
    parser.add_argument("--_single-backend", choices=VALID_BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument("--_single-trend-mode", choices=VALID_TREND_MODES, help=argparse.SUPPRESS)
    parser.add_argument("--_single-channels", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_reference-channels", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_single-output", help=argparse.SUPPRESS)
    return parser


def _validate_args(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    backends = parse_choice_list(args.backends, valid=VALID_BACKENDS, label="backends")
    trends = parse_choice_list(args.trend_modes, valid=VALID_TREND_MODES, label="trend modes")
    widths = (
        (args.channels,)
        if args.resident_widths is None
        else parse_positive_int_list(args.resident_widths)
    )
    if args.channels < 1 or args.cadences < 5 or args.samples < 4 or args.warmup < 0:
        raise ValueError("channels >= 1, cadences >= 5, samples >= 4, and warmup >= 0 are required.")
    if args.repeat_runs < 0:
        raise ValueError("repeat-runs must be non-negative.")
    if args.max_tree_depth < 1 or args.hmc_num_steps < 1:
        raise ValueError("tree depth and HMC leapfrog count must be positive.")
    if not 0 < args.target_accept < 1:
        raise ValueError("target-accept must lie strictly between zero and one.")
    if not math.isfinite(args.trend_prior_scale) or args.trend_prior_scale <= 0:
        raise ValueError("trend-prior-scale must be finite and positive.")
    if (
        not 0 <= args.max_divergence_fraction <= 1
        or not 0 <= args.max_tree_depth_saturation_fraction <= 1
        or args.min_channel_ess < 0
    ):
        raise ValueError("quality thresholds are outside their valid ranges.")
    if args.max_device_memory_gib is not None and args.max_device_memory_gib <= 0:
        raise ValueError("max-device-memory-gib must be positive.")
    if args.emit_width_selection:
        if args.device_total_memory_gib is not None and args.device_total_memory_gib <= 0:
            raise ValueError(
                "device-memory-limit-gib must be positive when supplied."
            )
        if args.manifest_backend not in backends:
            raise ValueError("manifest-backend must be included in --backends.")
        if args.manifest_trend_mode not in trends:
            raise ValueError("manifest-trend-mode must be included in --trend-modes.")
    if not 0 < args.width_memory_fraction <= 1:
        raise ValueError("width-memory-fraction must lie in (0, 1].")
    if bool(args.pilot_chunk) != bool(args.emit_batch_plan):
        raise ValueError("--pilot-chunk and --emit-batch-plan must be supplied together.")
    if args.emit_batch_plan and args.batch_plan_width is None and not args.emit_width_selection:
        raise ValueError(
            "--emit-batch-plan requires --batch-plan-width or --emit-width-selection."
        )
    if args.batch_plan_width is not None and args.batch_plan_width < 1:
        raise ValueError("batch-plan-width must be positive.")
    if args.quarantine_width < 1:
        raise ValueError("quarantine-width must be positive.")
    return backends, trends, widths


def _child_command(
    args: argparse.Namespace,
    *,
    backend: str,
    trend_mode: str,
    channels: int,
    reference_channels: int,
    output: str,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--platform", args.platform,
        "--cadences", str(args.cadences),
        "--warmup", str(args.warmup),
        "--samples", str(args.samples),
        "--repeat-runs", str(args.repeat_runs),
        "--max-tree-depth", str(args.max_tree_depth),
        "--hmc-num-steps", str(args.hmc_num_steps),
        "--target-accept", repr(args.target_accept),
        "--trend-prior-scale", repr(args.trend_prior_scale),
        "--seed", str(args.seed),
        "--_single-backend", backend,
        "--_single-trend-mode", trend_mode,
        "--_single-channels", str(channels),
        "--_reference-channels", str(reference_channels),
        "--_single-output", output,
    ]
    return command


def _source_fingerprints() -> dict[str, str]:
    paths = (
        "fit_jwst.py",
        "models/jaxoplanet/builder.py",
        "models/jaxoplanet/core.py",
        "models/jaxoplanet/limb_dark_streamed.py",
        "models/channel_batching.py",
        "models/independent_nuts.py",
        "models/independent_hmc.py",
        "models/linear_marginalization.py",
        "models/trend_marginal.py",
        "tools/benchmark_independent_nuts_gpu.py",
        "tools/benchmark_spectro_samplers.py",
    )
    fingerprints = {}
    for relative in paths:
        path = REPO_ROOT / relative
        if path.is_file():
            fingerprints[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return fingerprints


def _device_memory(jax: Any) -> dict[str, Any]:
    device = jax.devices()[0]
    try:
        raw = device.memory_stats()
    except Exception:
        raw = None
    raw = raw or {}
    peak_keys = ("peak_bytes_in_use", "peak_bytes", "peak_pool_bytes")
    current_keys = ("bytes_in_use", "bytes_reserved")
    peak = next((int(raw[key]) for key in peak_keys if key in raw), None)
    current = next((int(raw[key]) for key in current_keys if key in raw), None)
    memory_limit = int(raw["bytes_limit"]) if "bytes_limit" in raw else None
    max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; this repository is deployed on Linux/Slurm.
    return {
        "peak_device_bytes": peak,
        "current_device_bytes_at_report": current,
        "device_memory_limit_bytes": memory_limit,
        "device_memory_stats_available": bool(raw),
        "process_max_rss_bytes": max_rss * 1024,
        "process_max_rss_assumes_linux_kib": True,
        "peak_device_bytes_scope": (
            "process-lifetime high-water mark after all cold/repeat runs; "
            "includes compilation, warmup, sampling, and postprocessing"
        ),
        "repeat_outputs_released_between_runs": True,
        "allocator_retention_note": (
            "previous draw references are released and Python GC runs between "
            "repeats, but JAX's allocator may retain buffers; the process peak "
            "is intentionally conservative"
        ),
    }


def _make_problem(
    channels: int,
    cadences: int,
    seed: int,
    trend_mode: str,
    trend_scale: float,
    *,
    reference_channels: int,
):
    import jax.numpy as jnp
    import numpy as np

    from models.jaxoplanet import build_transit_window_indices, create_vectorized_model
    from tools.benchmark_independent_nuts_gpu import make_problem

    (
        sampled_model,
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
        reference_channels=reference_channels,
    )
    model_kwargs = dict(model_kwargs)
    init_params = dict(init_params)
    channel_varying = tuple(channel_varying)
    if trend_mode == "sampled_uniform":
        return (
            sampled_model, t, yerr, y, init_params, model_kwargs, channel_varying,
            {
                **metadata,
                "trend_prior": {
                    "family": "independent_uniform",
                    "coefficient_names": ["c", "v"],
                    "lower": [0.9, -0.1],
                    "upper": [1.1, 0.1],
                },
            },
        )

    duration = model_kwargs["mu_duration"]
    period = model_kwargs["PERIOD"]
    t0 = model_kwargs["mu_t0"]
    window_indices = build_transit_window_indices(
        np.asarray(t), np.asarray(period), np.asarray(t0), np.asarray(duration)
    )
    model = create_vectorized_model(
        detrend_type="linear",
        ld_mode="informed",
        trend_mode="gaussian_marginalized",
        n_planets=1,
        ld_profile="power2",
        param_method="duration",
        transit_window="auto",
        transit_window_indices=window_indices,
    )
    model_kwargs["trend_prior_mean"] = jnp.broadcast_to(
        jnp.asarray([1.0, 0.0], dtype=jnp.float64), (channels, 2)
    )
    model_kwargs["trend_prior_scale"] = jnp.full(
        (channels, 2), trend_scale, dtype=jnp.float64
    )
    init_params.pop("c", None)
    init_params.pop("v", None)
    channel_varying += ("trend_prior_mean", "trend_prior_scale")
    return (
        model, t, yerr, y, init_params, model_kwargs, channel_varying,
        {
            **metadata,
            "trend_prior": {
                "family": "independent_gaussian",
                "coefficient_names": ["c", "v"],
                "mean": [1.0, 0.0],
                "scale": [trend_scale, trend_scale],
            },
        },
    )


def _build_reusable_sampler(
    *,
    backend: str,
    model: Any,
    init_params: Mapping[str, Any],
    channels: int,
    channel_varying: tuple[str, ...],
    warmup: int,
    samples: int,
    max_tree_depth: int,
    hmc_num_steps: int,
    target_accept: float,
):
    """Construct the lazy shape-specialized runner used across timing runs."""

    if backend == "joint_nuts":
        from numpyro.infer import MCMC, NUTS
        from numpyro.infer.initialization import init_to_value

        kernel = NUTS(
            model,
            init_strategy=init_to_value(values=init_params),
            dense_mass=False,
            regularize_mass_matrix=True,
            target_accept_prob=target_accept,
            max_tree_depth=max_tree_depth,
        )
        return MCMC(
            kernel,
            num_warmup=warmup,
            num_samples=samples,
            progress_bar=False,
            jit_model_args=True,
        )
    if backend == "independent_nuts":
        from models.independent_nuts import build_independent_nuts_runner

        return build_independent_nuts_runner(
            model,
            num_warmup=warmup,
            num_samples=samples,
            lane_width=channels,
            channel_varying_kwargs=channel_varying,
            dense_mass=True,
            regularize_mass_matrix=True,
            target_accept_prob=target_accept,
            max_tree_depth=max_tree_depth,
        )
    if backend == "independent_hmc":
        from models.independent_hmc import build_independent_hmc_runner

        return build_independent_hmc_runner(
            model,
            num_warmup=warmup,
            num_samples=samples,
            num_steps=hmc_num_steps,
            lane_width=channels,
            channel_varying_kwargs=channel_varying,
            dense_mass=True,
            regularize_mass_matrix=True,
            target_accept_prob=target_accept,
        )
    raise ValueError(f"Unknown backend {backend!r}.")


def _run_sampler_once(
    *, backend: str, model: Any, key: Any, t: Any, yerr: Any, y: Any,
    init_params: Mapping[str, Any], model_kwargs: Mapping[str, Any],
    channel_varying: tuple[str, ...], warmup: int, samples: int,
    max_tree_depth: int, hmc_num_steps: int, target_accept: float,
    reusable_sampler: Any,
):
    import jax
    import numpyro

    if backend == "joint_nuts":
        mcmc = reusable_sampler
        mcmc.run(
            key, t, yerr, y=y,
            extra_fields=("num_steps", "diverging", "accept_prob"),
            **model_kwargs,
        )
        draws = mcmc.get_samples()
        extra = mcmc.get_extra_fields()
        raw = {
            "num_steps": extra["num_steps"],
            "diverging": extra["diverging"],
            "accept_prob": extra["accept_prob"],
            "step_size": mcmc.last_state.adapt_state.step_size,
        }
        return draws, raw

    if backend == "independent_nuts":
        from models.independent_nuts import get_samples_independent

        draws, diagnostics = get_samples_independent(
            model, key, t, yerr, y, init_params,
            num_warmup=warmup,
            num_samples=samples,
            lane_width=int(yerr.shape[0]),
            channel_varying_kwargs=channel_varying,
            dense_mass=True,
            regularize_mass_matrix=True,
            target_accept_prob=target_accept,
            max_tree_depth=max_tree_depth,
            return_diagnostics=True,
            _runner=reusable_sampler,
            **model_kwargs,
        )
    elif backend == "independent_hmc":
        from models.independent_hmc import get_samples_independent_hmc

        draws, diagnostics = get_samples_independent_hmc(
            model, key, t, yerr, y, init_params,
            num_warmup=warmup,
            num_samples=samples,
            num_steps=hmc_num_steps,
            lane_width=int(yerr.shape[0]),
            channel_varying_kwargs=channel_varying,
            dense_mass=True,
            regularize_mass_matrix=True,
            target_accept_prob=target_accept,
            return_diagnostics=True,
            _runner=reusable_sampler,
            **model_kwargs,
        )
    else:  # guarded by argparse and parent validation
        raise ValueError(f"Unknown backend {backend!r}.")
    return draws, {
        "num_steps": diagnostics.num_steps,
        "diverging": diagnostics.diverging,
        "accept_prob": diagnostics.accept_prob,
        "step_size": diagnostics.step_size,
    }


def _finite_float(value: Any) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _case_seeds(
    base_seed: int,
    backend: str,
    trend_mode: str,
) -> tuple[int, int]:
    """Return width-independent problem and sampler seeds for one case family."""
    problem_seed = int(base_seed)
    sampler_seed = (
        problem_seed
        + 1_000_003
        + 1009 * VALID_BACKENDS.index(backend)
        + 9176 * VALID_TREND_MODES.index(trend_mode)
    )
    return problem_seed, sampler_seed


def _summarize_diagnostics(
    raw: Mapping[str, Any],
    backend: str,
    hmc_num_steps: int,
    max_tree_depth: int,
) -> dict[str, Any]:
    import jax
    import numpy as np

    steps = np.asarray(jax.device_get(raw["num_steps"]))
    diverging = np.asarray(jax.device_get(raw["diverging"]), dtype=bool)
    accept = np.asarray(jax.device_get(raw["accept_prob"]))
    step_size = np.asarray(jax.device_get(raw["step_size"]))
    result = {
        "diagnostic_event_count": int(diverging.size),
        "divergences": int(diverging.sum()),
        "divergence_fraction": float(diverging.mean()),
        "mean_accept_prob": float(accept.mean()),
        "mean_num_steps": float(steps.mean()),
        "median_num_steps": float(np.median(steps)),
        "max_num_steps": int(steps.max()),
        "adapted_step_size_min": float(step_size.min()),
        "adapted_step_size_median": float(np.median(step_size)),
        "adapted_step_size_max": float(step_size.max()),
        "joint_divergence_semantics": backend == "joint_nuts",
    }
    if steps.ndim == 2:
        result["mean_per_draw_lockstep_steps"] = float(steps.max(axis=1).mean())
        result["mean_num_steps_per_channel"] = steps.mean(axis=0).tolist()
        result["divergences_per_channel"] = diverging.sum(axis=0).tolist()
        divergence_fractions = diverging.mean(axis=0)
        result["divergence_fraction_per_channel"] = (
            divergence_fractions.tolist()
        )
        result["max_divergence_fraction_per_channel"] = float(
            divergence_fractions.max()
        )
    if backend == "independent_hmc":
        result["fixed_num_steps"] = hmc_num_steps
    else:
        maximum_steps = 2 ** int(max_tree_depth) - 1
        saturated = steps >= maximum_steps
        if saturated.ndim == 2:
            fractions = saturated.mean(axis=0)
            result["tree_depth_saturation_fraction_per_channel"] = (
                fractions.tolist()
            )
            result["tree_depth_saturation_fraction_max"] = float(
                fractions.max()
            )
        else:
            result["tree_depth_saturation_fraction_max"] = float(
                saturated.mean()
            )
    return result


def _summarize_posterior(draws: Mapping[str, Any], steady_seconds: float) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from numpyro.diagnostics import effective_sample_size

    checked_sites = []
    nonfinite_sites = []
    nonfinite_leaf_paths = []
    nonfinite_value_count = 0
    posterior_leaf_count = 0
    for site_name, site_value in sorted(draws.items()):
        site_name = str(site_name)
        checked_sites.append(site_name)
        site_is_finite = True
        path_leaves, _ = jax.tree_util.tree_flatten_with_path(site_value)
        for path, leaf in path_leaves:
            posterior_leaf_count += 1
            array = np.asarray(jax.device_get(leaf))
            try:
                finite = np.isfinite(array)
                leaf_is_finite = bool(np.all(finite))
                leaf_nonfinite_count = int(array.size - np.count_nonzero(finite))
            except TypeError:
                # Posterior sites should be numeric. Treat an unsupported leaf
                # as unsafe instead of letting the quality gate pass silently.
                leaf_is_finite = False
                leaf_nonfinite_count = int(array.size)
            if not leaf_is_finite:
                site_is_finite = False
                nonfinite_value_count += leaf_nonfinite_count
                suffix_parts = []
                for entry in path:
                    value = getattr(
                        entry,
                        "key",
                        getattr(entry, "idx", getattr(entry, "name", entry)),
                    )
                    suffix_parts.append(str(value))
                suffix = "" if not suffix_parts else "." + ".".join(suffix_parts)
                nonfinite_leaf_paths.append(site_name + suffix)
        if not site_is_finite:
            nonfinite_sites.append(site_name)

    rors = np.asarray(jax.device_get(draws["rors"]))
    if rors.ndim == 3 and rors.shape[-1] == 1:
        rors = rors[..., 0]
    if rors.ndim != 2:
        raise ValueError(f"Expected rors draws [draw, channel], got {rors.shape}.")
    ess = []
    for channel in range(rors.shape[1]):
        value = effective_sample_size(jnp.asarray(rors[:, channel])[None, :])
        ess.append(_finite_float(value))
    finite_ess = [value for value in ess if value is not None]
    ess_sum = sum(finite_ess) if len(finite_ess) == len(ess) else None
    return {
        "all_finite": not nonfinite_leaf_paths,
        "posterior_sites_checked": checked_sites,
        "posterior_leaf_count": posterior_leaf_count,
        "nonfinite_sites": nonfinite_sites,
        "nonfinite_leaf_paths": nonfinite_leaf_paths,
        "nonfinite_value_count": nonfinite_value_count,
        "rors_mean_per_channel": [
            _finite_float(value) for value in np.mean(rors, axis=0)
        ],
        "rors_sd_per_channel": [
            _finite_float(value) for value in np.std(rors, axis=0, ddof=1)
        ],
        "rors_median_per_channel": [
            _finite_float(value) for value in np.median(rors, axis=0)
        ],
        "rors_ess_per_channel": ess,
        "rors_ess_min": None if not finite_ess else min(finite_ess),
        "rors_ess_median": None if not finite_ess else statistics.median(finite_ess),
        "rors_ess_sum": ess_sum,
        "rors_ess_sum_per_second": (
            None if ess_sum is None or steady_seconds <= 0 else ess_sum / steady_seconds
        ),
    }


def _runtime_metadata(jax: Any, numpyro: Any) -> dict[str, Any]:
    import jaxlib

    device = jax.devices()[0]
    return {
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpyro": numpyro.__version__,
        "backend": jax.default_backend(),
        "device": str(device),
        "device_kind": getattr(device, "device_kind", None),
        "host": python_platform.node(),
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "jax_platforms_env": os.environ.get("JAX_PLATFORMS"),
        "xla_preallocate_env": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def _run_single_case(args: argparse.Namespace) -> dict[str, Any]:
    # These environment settings happen before the first JAX import in this
    # subprocess. They are also recorded in the result for reproducibility.
    os.environ["JAX_ENABLE_X64"] = "1"
    # ``gpu`` is a user-facing JAX backend class, not a valid value for the
    # explicit JAX_PLATFORMS initialization list in current JAX.  Supplying it
    # there expands to CUDA and ROCm and fails on CUDA-only installations when
    # the absent ROCm backend is initialized.  Pin CUDA directly instead.
    os.environ["JAX_PLATFORMS"] = (
        "cuda" if args.platform == "gpu" else args.platform
    )
    import jax
    import numpyro

    jax.config.update("jax_enable_x64", True)
    actual_backend = jax.default_backend()
    acceptable_backends = (
        {"gpu", "cuda"} if args.platform == "gpu" else {args.platform}
    )
    if actual_backend not in acceptable_backends:
        raise RuntimeError(
            f"Requested JAX platform {args.platform!r}, got {actual_backend!r}."
        )
    backend = args._single_backend
    trend_mode = args._single_trend_mode
    channels = int(args._single_channels)
    reference_channels = (
        channels
        if args._reference_channels is None
        else int(args._reference_channels)
    )
    if reference_channels < channels:
        raise ValueError("Private reference width must be at least case width.")
    # Every width is an exact prefix of one max-width problem realization.
    # Sampler streams may differ by backend/prior family, but never by width.
    problem_seed, sampler_seed = _case_seeds(args.seed, backend, trend_mode)
    (
        model, t, yerr, y, init_params, model_kwargs, channel_varying, problem_metadata,
    ) = _make_problem(
        channels,
        args.cadences,
        problem_seed,
        trend_mode,
        args.trend_prior_scale,
        reference_channels=reference_channels,
    )

    collector = _CompilationEvents()
    jax.monitoring.register_event_duration_secs_listener(collector.listener)
    jax.clear_caches()
    keys = jax.random.split(
        jax.random.PRNGKey(sampler_seed + 31), args.repeat_runs + 1
    )
    reusable_sampler = _build_reusable_sampler(
        backend=backend,
        model=model,
        init_params=init_params,
        channels=channels,
        channel_varying=channel_varying,
        warmup=args.warmup,
        samples=args.samples,
        max_tree_depth=args.max_tree_depth,
        hmc_num_steps=args.hmc_num_steps,
        target_accept=args.target_accept,
    )
    timing_runs = []
    draws = diagnostics_raw = None
    for run_index, key in enumerate(keys):
        if run_index:
            # Do not keep the previous posterior resident while allocating the
            # next one. JAX's allocator can retain freed buffers, so the final
            # device high-water mark remains a conservative process-wide peak.
            draws = None
            diagnostics_raw = None
        gc.collect()
        mark = collector.mark()
        start = time.perf_counter()
        next_draws, next_diagnostics = _run_sampler_once(
            backend=backend,
            model=model,
            key=key,
            t=t,
            yerr=yerr,
            y=y,
            init_params=init_params,
            model_kwargs=model_kwargs,
            channel_varying=channel_varying,
            warmup=args.warmup,
            samples=args.samples,
            max_tree_depth=args.max_tree_depth,
            hmc_num_steps=args.hmc_num_steps,
            target_accept=args.target_accept,
            reusable_sampler=reusable_sampler,
        )
        jax.block_until_ready((next_draws, next_diagnostics))
        wall_seconds = time.perf_counter() - start
        compile_metrics = collector.since(mark)
        timing_runs.append(
            {
                "run_index": run_index,
                "wall_seconds": wall_seconds,
                **compile_metrics,
            }
        )
        draws, diagnostics_raw = next_draws, next_diagnostics
        del next_draws, next_diagnostics
    timing = summarize_timing(timing_runs)
    timing["runs"] = timing_runs
    diagnostics = _summarize_diagnostics(
        diagnostics_raw,
        backend,
        args.hmc_num_steps,
        args.max_tree_depth,
    )
    posterior = _summarize_posterior(draws, timing["steady_state_wall_seconds"])
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "case_id": candidate_id(backend, trend_mode, channels),
        "backend": backend,
        "trend_mode": trend_mode,
        "channels": channels,
        "synthetic_reference_channels": reference_channels,
        "cadences": args.cadences,
        "warmup": args.warmup,
        "samples": args.samples,
        "resident_lane_width": channels if backend != "joint_nuts" else None,
        "max_tree_depth": args.max_tree_depth if backend != "independent_hmc" else None,
        "hmc_num_steps": args.hmc_num_steps if backend == "independent_hmc" else None,
        "target_accept": args.target_accept,
        "dataset": {
            "kind": "synthetic_production_shaped",
            "jaxoplanet_limb_darkening": "power2",
            "detrend_type": "linear",
            **problem_metadata,
        },
        "timing": timing,
        "memory": _device_memory(jax),
        "diagnostics": diagnostics,
        "posterior": posterior,
        "runtime": _runtime_metadata(jax, numpyro),
        "source_sha256": _source_fingerprints(),
        "runner_reused_across_timing_runs": True,
        "problem_seed": problem_seed,
        "sampler_seed": sampler_seed,
    }


def _failure_result(
    backend: str, trend_mode: str, channels: int, completed: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "case_id": candidate_id(backend, trend_mode, channels),
        "backend": backend,
        "trend_mode": trend_mode,
        "channels": channels,
        "returncode": completed.returncode,
        "possible_process_oom": completed.returncode in (-9, 9, 137),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-8000:],
    }


def _width_manifest_provenance(
    args: argparse.Namespace,
    results: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind a resident-width choice to the workload that was measured."""
    matching = [
        result
        for result in results
        if result.get("status") == "ok"
        and result.get("backend") == args.manifest_backend
        and result.get("trend_mode") == args.manifest_trend_mode
    ]
    if not matching:
        raise ValueError("No successful matching cases exist for provenance.")

    signatures = []
    for result in matching:
        dataset = result.get("dataset", {})
        runtime = result.get("runtime", {})
        signature = {
            "artifact_kind": "jwst_spectro_width_selection",
            "benchmark_schema_version": SCHEMA_VERSION,
            "benchmark_platform": args.platform,
            "sampler_backend": args.manifest_backend,
            "trend_inference": args.manifest_trend_mode,
            "num_cadences": int(result["cadences"]),
            "num_warmup": int(result["warmup"]),
            "num_samples": int(result["samples"]),
            "sampler_options": {
                "max_tree_depth": result.get("max_tree_depth"),
                "hmc_num_steps": result.get("hmc_num_steps"),
                "target_accept": float(result["target_accept"]),
                "dense_mass": (
                    False if args.manifest_backend == "joint_nuts" else True
                ),
                "regularize_mass_matrix": True,
            },
            "model": {
                "transit_engine": "jaxoplanet",
                "ld_profile": dataset.get("jaxoplanet_limb_darkening"),
                "ld_mode": "informed",
                "detrend_type": dataset.get("detrend_type"),
                "param_method": "duration",
                "n_planets": 1,
                "transit_window": "auto",
                "window_cadences": dataset.get("window_cadences"),
            },
            "synthetic_realization": {
                "kind": dataset.get("kind"),
                "problem_seed": result.get("problem_seed"),
                "reference_channels": dataset.get(
                    "synthetic_reference_channels"
                ),
                "nested_prefix": dataset.get("synthetic_nested_realization"),
            },
            "runtime": {
                "device_kind": runtime.get("device_kind"),
                "jax": runtime.get("jax"),
                "jaxlib": runtime.get("jaxlib"),
                "numpyro": runtime.get("numpyro"),
                "jax_enable_x64": runtime.get("jax_enable_x64"),
            },
            "source_sha256": result.get("source_sha256", {}),
        }
        signatures.append(signature)
    canonical = {
        json.dumps(item, allow_nan=False, separators=(",", ":"), sort_keys=True)
        for item in signatures
    }
    if len(canonical) != 1:
        raise ValueError(
            "Matching width cases do not share one workload/runtime/source "
            "signature; do not combine measurements from a changing environment."
        )
    return signatures[0]


def _run_suite(args: argparse.Namespace) -> int:
    backends, trends, widths = _validate_args(args)
    reference_channels = max(widths)
    cases = [(backend, trend, width) for trend in trends for backend in backends for width in widths]
    results = []
    with tempfile.TemporaryDirectory(prefix="spectro_benchmark_") as directory:
        for index, (backend, trend_mode, channels) in enumerate(cases, start=1):
            case_name = candidate_id(backend, trend_mode, channels)
            output = str(Path(directory) / f"{case_name}.json")
            print(f"[{index}/{len(cases)}] {case_name}", file=sys.stderr, flush=True)
            completed = subprocess.run(
                _child_command(
                    args,
                    backend=backend,
                    trend_mode=trend_mode,
                    channels=channels,
                    reference_channels=reference_channels,
                    output=output,
                ),
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            # Single-case children persist a structured exception report before
            # exiting with status 2.  Preserve that report instead of replacing
            # it with empty captured streams; it is the only useful diagnosis
            # for failures that happen before XLA emits anything.
            if Path(output).is_file():
                with open(output, "r", encoding="utf-8") as stream:
                    result = json.load(stream)
                if completed.returncode != 0:
                    result.setdefault("schema_version", SCHEMA_VERSION)
                    result.setdefault("status", "failed")
                    result.setdefault(
                        "case_id", candidate_id(backend, trend_mode, channels)
                    )
                    result.setdefault("backend", backend)
                    result.setdefault("trend_mode", trend_mode)
                    result.setdefault("channels", channels)
                    result["returncode"] = completed.returncode
                    result["possible_process_oom"] = completed.returncode in (
                        -9, 9, 137
                    )
                    result["stdout_tail"] = completed.stdout[-4000:]
                    result["stderr_tail"] = completed.stderr[-8000:]
            else:
                result = _failure_result(backend, trend_mode, channels, completed)
                print(
                    f"  failed with return code {completed.returncode}; recorded and continuing",
                    file=sys.stderr,
                    flush=True,
                )
            if args.verbose and completed.stdout:
                print(completed.stdout, file=sys.stderr, end="")
            results.append(result)
            if result["status"] != "ok" and args.fail_fast:
                break

    max_memory = (
        None
        if args.max_device_memory_gib is None
        else int(args.max_device_memory_gib * (1024**3))
    )
    successful = sum(result["status"] == "ok" for result in results)
    autotuning = select_candidates(
        results,
        max_divergence_fraction=args.max_divergence_fraction,
        min_channel_ess=args.min_channel_ess,
        max_device_memory_bytes=max_memory,
        max_tree_depth_saturation_fraction=(
            args.max_tree_depth_saturation_fraction
        ),
    )
    manifest_artifacts: dict[str, Any] = {}
    manifest_errors: list[str] = []
    width_selection = None
    if args.emit_width_selection or args.emit_batch_plan:
        # Importing models.channel_batching initializes JAX through models'
        # package initializer. Do this only after isolated benchmark children
        # have exited, and pin it to the requested platform.
        os.environ["JAX_ENABLE_X64"] = "1"
        os.environ["JAX_PLATFORMS"] = (
            "cuda" if args.platform == "gpu" else args.platform
        )
    if args.emit_width_selection:
        try:
            eligible = {
                entry["case_id"]
                for entry in autotuning["by_trend_mode"][args.manifest_trend_mode]["candidates"]
                if entry["eligible"]
            }
            if args.device_total_memory_gib is not None:
                memory_limit_bytes = int(args.device_total_memory_gib * (1024**3))
                memory_limit_source = "command_line_override"
            else:
                measured_limits = {
                    int(result["memory"]["device_memory_limit_bytes"])
                    for result in results
                    if result.get("status") == "ok"
                    and result.get("backend") == args.manifest_backend
                    and result.get("trend_mode") == args.manifest_trend_mode
                    and result.get("memory", {}).get("device_memory_limit_bytes") is not None
                }
                if len(measured_limits) != 1:
                    raise ValueError(
                        "Could not infer one JAX device bytes_limit from the selected "
                        "measurements; supply --device-memory-limit-gib explicitly."
                    )
                memory_limit_bytes = measured_limits.pop()
                memory_limit_source = "jax_device_memory_stats_bytes_limit"
            width_selection, skipped = build_width_selection_from_results(
                results,
                backend=args.manifest_backend,
                trend_mode=args.manifest_trend_mode,
                total_memory_bytes=memory_limit_bytes,
                memory_fraction=args.width_memory_fraction,
                eligible_case_ids=eligible,
                provenance=_width_manifest_provenance(args, results),
            )
            atomic_write_json(args.emit_width_selection, width_selection.to_manifest())
            manifest_artifacts["width_selection"] = {
                "path": str(Path(args.emit_width_selection).resolve()),
                "fingerprint_sha256": width_selection.fingerprint_sha256,
                "selected_width": width_selection.selected.width,
                "backend": args.manifest_backend,
                "trend_mode": args.manifest_trend_mode,
                "memory_limit_source": memory_limit_source,
                "device_memory_limit_bytes": memory_limit_bytes,
                "skipped_measurements": skipped,
            }
        except Exception as exc:
            manifest_errors.append(f"WidthSelection: {type(exc).__name__}: {exc}")
    if args.emit_batch_plan:
        try:
            nominal_width = (
                args.batch_plan_width
                if args.batch_plan_width is not None
                else width_selection.selected.width
            )
            if args.quarantine_width > nominal_width:
                raise ValueError("quarantine-width cannot exceed the nominal width.")
            plan, sources = build_batch_plan_from_chunks(
                args.pilot_chunk,
                nominal_width=nominal_width,
                quarantine_width=args.quarantine_width,
                require_workload_provenance=True,
            )
            atomic_write_json(args.emit_batch_plan, plan.to_manifest())
            manifest_artifacts["channel_batch_plan"] = {
                "path": str(Path(args.emit_batch_plan).resolve()),
                "fingerprint_sha256": plan.fingerprint_sha256,
                "num_channels": len(plan.original_channel_indices),
                "num_batches": len(plan.batches),
                "nominal_width": plan.nominal_width,
                "quarantine_width": plan.quarantine_width,
                "pilot_sources": sources,
            }
        except Exception as exc:
            manifest_errors.append(f"ChannelBatchPlan: {type(exc).__name__}: {exc}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "suite_status": "ok" if successful == len(results) else ("partial" if successful else "failed"),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "benchmark_scope": (
            "synthetic production-shaped pilot; validate selected settings on "
            "representative real visits before changing scientific defaults"
        ),
        "requested_platform": args.platform,
        "configuration": {
            "backends": list(backends),
            "trend_modes": list(trends),
            "resident_widths": list(widths),
            "synthetic_reference_channels": reference_channels,
            "cadences": args.cadences,
            "warmup": args.warmup,
            "samples": args.samples,
            "repeat_runs": args.repeat_runs,
            "max_tree_depth": args.max_tree_depth,
            "hmc_num_steps": args.hmc_num_steps,
            "target_accept": args.target_accept,
            "max_tree_depth_saturation_fraction": (
                args.max_tree_depth_saturation_fraction
            ),
            "trend_prior_scale": args.trend_prior_scale,
            "seed": args.seed,
        },
        "scientific_comparability": {
            "sampler_comparisons": (
                "same likelihood and prior within each trend mode; every width "
                "is an exact prefix of one max-width synthetic realization and "
                "problem/sampler seeds do not depend on width"
            ),
            "trend_comparisons": (
                "sampled Uniform and marginalized Gaussian trend priors are not identical; "
                "timing ratios do not establish posterior equivalence"
            ),
            "mixed_precision_tested": False,
        },
        "results": results,
        "measured_sampler_comparisons": build_measured_comparisons(results),
        "measured_trend_comparisons": build_trend_comparisons(results),
        "autotuning": autotuning,
        "fit_manifest_artifacts": manifest_artifacts,
        "fit_manifest_errors": manifest_errors,
    }
    if manifest_errors and payload["suite_status"] == "ok":
        payload["suite_status"] = "partial"
    atomic_write_json(args.json_path, payload)
    print(json.dumps({
        "suite_status": payload["suite_status"],
        "successful_cases": successful,
        "attempted_cases": len(results),
        "json": str(Path(args.json_path).resolve()),
        "selected": {
            trend: entry["selected_case_id"]
            for trend, entry in payload["autotuning"]["by_trend_mode"].items()
        },
        "fit_manifest_artifacts": manifest_artifacts,
        "fit_manifest_errors": manifest_errors,
    }, indent=2, sort_keys=True))
    if manifest_errors:
        return 2
    return 0 if successful else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args._single_backend is not None:
            if not args._single_trend_mode or not args._single_channels or not args._single_output:
                parser.error("incomplete private single-case invocation")
            result = _run_single_case(args)
            atomic_write_json(args._single_output, result)
            return 0
        return _run_suite(args)
    except (ValueError, RuntimeError) as exc:
        if args._single_output:
            atomic_write_json(
                args._single_output,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            return 2
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
