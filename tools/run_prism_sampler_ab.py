#!/usr/bin/env python3
"""Run a same-seed direct/grid PRISM independent-NUTS comparison."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import numpy as np

from models.independent_nuts import get_samples_independent
from tools.benchmark_prism_cadence_reduction import _prepare
from tools.loop_fit import _CompilationEvents, _arviz_diagnostics


jax.config.update("jax_enable_x64", True)


COMPARISON_SITES = {
    "depths": "depth",
    "rors": "depth",
    "c1": "limb_darkening",
    "c2": "limb_darkening",
    "c": "trend",
    "v": "trend",
    "A": "trend",
    "log_jitter": "noise",
    "total_error": "noise",
}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--width", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--grid-nodes", type=int, default=769)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, payload):
    path = Path(path)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _diagnostic_summary(diagnostics):
    steps = np.asarray(jax.device_get(diagnostics.num_steps))
    acceptance = np.asarray(jax.device_get(diagnostics.accept_prob))
    diverging = np.asarray(jax.device_get(diagnostics.diverging))
    result = {
        "num_divergences": int(np.count_nonzero(diverging)),
        "num_steps_mean": float(np.mean(steps)),
        "num_steps_median": float(np.median(steps)),
        "num_steps_max": int(np.max(steps)),
        "accept_prob_mean": float(np.mean(acceptance)),
        "step_size_median": float(
            np.median(np.asarray(jax.device_get(diagnostics.step_size)))
        ),
    }
    for name in (
        "map_gradient_norm",
        "map_newton_decrement",
        "map_iterations",
        "map_condition_number",
        "map_hessian_min_eigenvalue",
        "map_hessian_relative_error",
    ):
        value = getattr(diagnostics, name, None)
        if value is not None:
            array = np.asarray(jax.device_get(value))
            finite = array[np.isfinite(array)]
            if finite.size:
                result[f"{name}_median"] = float(np.median(finite))
                result[f"{name}_max"] = float(np.max(finite))
    return result


def _run(stage, key, compilation, warmup, draws):
    nuts_kwargs = dict(stage.nuts_kwargs)
    nuts_kwargs.pop("init_strategy", None)
    mcmc_kwargs = {
        **dict(stage.mcmc_kwargs),
        "num_warmup": int(warmup),
        "num_samples": int(draws),
        "progress_bar": False,
    }
    varying = tuple(
        name
        for name in stage.channel_varying_kwargs
        if name in stage.model_kwargs
    )
    memory_before = dict(jax.devices()[0].memory_stats() or {})
    mark = compilation.mark()
    started = time.perf_counter()
    samples, diagnostics = get_samples_independent(
        stage.model,
        key,
        stage.t,
        stage.yerr,
        stage.y,
        stage.init_params,
        nuts_kwargs=nuts_kwargs,
        mcmc_kwargs=mcmc_kwargs,
        lane_width=stage.num_channels,
        channel_varying_kwargs=varying,
        return_diagnostics=True,
        **dict(stage.model_kwargs),
    )
    samples = jax.device_get(samples)
    wall = time.perf_counter() - started
    compile_stats = compilation.since(mark)
    memory_after = dict(jax.devices()[0].memory_stats() or {})
    result = {
        "wall_seconds": wall,
        "compile": compile_stats,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "sampler": _diagnostic_summary(diagnostics),
        "posterior_diagnostics": _arviz_diagnostics(
            samples, stage.num_channels
        ),
    }
    return samples, result


def _series(array, channels):
    array = np.asarray(array)
    if array.ndim < 2 or array.shape[1] != channels:
        return []
    flat = array.reshape(array.shape[0], channels, -1)
    return [(index, flat[:, :, index]) for index in range(flat.shape[-1])]


def _compare(reference, candidate, channels):
    rows = []
    for site, category in COMPARISON_SITES.items():
        if site not in reference or site not in candidate:
            continue
        ref_components = _series(reference[site], channels)
        new_components = _series(candidate[site], channels)
        if len(ref_components) != len(new_components):
            raise ValueError(f"Sample shape differs for {site!r}.")
        for (component, ref_values), (_, new_values) in zip(
            ref_components, new_components
        ):
            for channel in range(channels):
                ref = ref_values[:, channel]
                new = new_values[:, channel]
                ref_sigma = float(np.std(ref, ddof=1))
                new_sigma = float(np.std(new, ddof=1))
                denominator = max(ref_sigma, np.finfo(float).tiny)
                rows.append(
                    {
                        "category": category,
                        "site": site,
                        "component": component,
                        "channel": channel,
                        "reference_median": float(np.median(ref)),
                        "candidate_median": float(np.median(new)),
                        "reference_mean": float(np.mean(ref)),
                        "candidate_mean": float(np.mean(new)),
                        "reference_sigma": ref_sigma,
                        "candidate_sigma": new_sigma,
                        "abs_median_shift_sigma": float(
                            abs(np.median(new) - np.median(ref)) / denominator
                        ),
                        "abs_mean_shift_sigma": float(
                            abs(np.mean(new) - np.mean(ref)) / denominator
                        ),
                        "sigma_ratio": float(new_sigma / denominator),
                    }
                )
    aggregate = {}
    for category in sorted({row["category"] for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        aggregate[category] = {
            "count": len(selected),
            "median_abs_median_shift_sigma": float(np.median([
                row["abs_median_shift_sigma"] for row in selected
            ])),
            "max_abs_median_shift_sigma": float(np.max([
                row["abs_median_shift_sigma"] for row in selected
            ])),
            "median_abs_mean_shift_sigma": float(np.median([
                row["abs_mean_shift_sigma"] for row in selected
            ])),
            "median_sigma_ratio": float(np.median([
                row["sigma_ratio"] for row in selected
            ])),
            "min_sigma_ratio": float(np.min([
                row["sigma_ratio"] for row in selected
            ])),
            "max_sigma_ratio": float(np.max([
                row["sigma_ratio"] for row in selected
            ])),
        }
    return {"aggregate": aggregate, "rows": rows}


def main():
    args = _parser().parse_args()
    if args.width < 1 or args.warmup < 0 or args.samples < 1:
        raise ValueError("Width/samples must be positive and warmup non-negative.")
    args.output.mkdir(parents=True, exist_ok=True)
    baseline, candidate, groups = _prepare(
        args.dump,
        args.start,
        args.width,
        "auto",
        "combined",
        args.grid_nodes,
    )
    compilation = _CompilationEvents()
    try:
        jax.monitoring.register_event_duration_secs_listener(
            compilation.listener
        )
    except ValueError:
        pass
    key = baseline.rng_key
    report = {
        "dump": str(Path(args.dump).resolve()),
        "device": str(jax.devices()[0]),
        "channels": [args.start, args.start + args.width],
        "warmup": args.warmup,
        "samples": args.samples,
        "grid_nodes": args.grid_nodes,
        "out_of_window_groups": groups,
        "nuts_kwargs": {
            key: value for key, value in baseline.nuts_kwargs.items()
            if key != "init_strategy"
        },
    }

    print("Starting direct baseline", flush=True)
    baseline_samples, report["baseline"] = _run(
        baseline, key, compilation, args.warmup, args.samples
    )
    np.savez_compressed(args.output / "baseline_samples.npz", **baseline_samples)
    _write_json(args.output / "partial_baseline.json", report)
    print(
        f"Direct baseline finished in {report['baseline']['wall_seconds']:.3f} s",
        flush=True,
    )

    print("Starting grid/reduced candidate", flush=True)
    candidate_samples, report["candidate"] = _run(
        candidate, key, compilation, args.warmup, args.samples
    )
    np.savez_compressed(args.output / "candidate_samples.npz", **candidate_samples)
    report["comparison"] = _compare(
        baseline_samples, candidate_samples, args.width
    )
    report["wall_speedup"] = (
        report["baseline"]["wall_seconds"]
        / report["candidate"]["wall_seconds"]
    )
    _write_json(args.output / "report.json", report)
    print(
        f"Candidate finished in {report['candidate']['wall_seconds']:.3f} s; "
        f"speed-up {report['wall_speedup']:.3f}x",
        flush=True,
    )
    print(json.dumps(_jsonable(report["comparison"]["aggregate"]), indent=2))


if __name__ == "__main__":
    main()
