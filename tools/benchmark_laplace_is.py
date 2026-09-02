#!/usr/bin/env python3
"""CPU fidelity/timing study for the Laplace-IS spectroscopic backend.

The script runs selected samplers on one identical synthetic problem or one
replayable stage-input dump and writes detailed per-site/per-channel summaries.
It intentionally reports measured wall time only; it does not extrapolate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.diagnostics import effective_sample_size
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.independent_nuts import get_samples_independent
from models.laplace_is import get_samples_laplace_is


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", default=None)
    parser.add_argument("--channels", type=int, default=40)
    parser.add_argument("--cadences", type=int, default=230)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument(
        "--methods", default="laplace_is,independent_nuts,joint_nuts"
    )
    parser.add_argument("--laplace-draws", type=int, default=4096)
    parser.add_argument("--laplace-rounds", type=int, default=2)
    parser.add_argument("--draw-chunk-size", type=int, default=256)
    parser.add_argument("--student-df", type=float, default=5.0)
    parser.add_argument("--scale-inflation", type=float, default=1.2)
    parser.add_argument("--map-maxiter", type=int, default=80)
    parser.add_argument("--output-mode", choices=("imh", "resample"), default="imh")
    parser.add_argument("--disable-fallback", action="store_true")
    parser.add_argument("--json", required=True)
    return parser


def _problem(args):
    if args.dump:
        from tools.spectro_stage_inputs import load_stage_inputs

        inputs = load_stage_inputs(args.dump)
        if args.channels < inputs.num_channels:
            inputs = inputs.select(0, args.channels)
        model_kwargs = dict(inputs.model_kwargs)
        varying = tuple(
            name for name in inputs.channel_varying_kwargs
            if name in model_kwargs
        )
        return {
            "model": inputs.model,
            "t": inputs.t,
            "yerr": inputs.yerr,
            "y": inputs.y,
            "init": dict(inputs.init_params),
            "kwargs": model_kwargs,
            "varying": varying,
            "meta": {
                **dict(inputs.meta),
                "kind": "stage_dump",
                "path": str(Path(args.dump).resolve()),
                "dump_channel_varying_kwargs": list(
                    inputs.channel_varying_kwargs
                ),
                "used_channel_varying_kwargs": list(varying),
            },
        }

    from tools.benchmark_independent_nuts_gpu import make_problem

    model, t, yerr, y, init, kwargs, varying, meta = make_problem(
        args.channels,
        args.cadences,
        args.seed,
        reference_channels=args.channels,
    )
    return {
        "model": model,
        "t": t,
        "yerr": yerr,
        "y": y,
        "init": init,
        "kwargs": kwargs,
        "varying": varying,
        "meta": {**meta, "kind": "synthetic"},
    }


def _ready(tree):
    return jax.block_until_ready(tree)


def _run(method, problem, args, key):
    common = (
        problem["model"], key, problem["t"], problem["yerr"],
        problem["y"], problem["init"]
    )
    started = time.perf_counter()
    diagnostics = None
    if method == "laplace_is":
        draws, diagnostics = get_samples_laplace_is(
            *common,
            num_warmup=args.warmup,
            num_samples=args.samples,
            lane_width=int(np.shape(problem["y"])[0]),
            channel_varying_kwargs=problem["varying"],
            laplace_is_output=args.output_mode,
            laplace_is_num_draws=args.laplace_draws,
            laplace_is_rounds=args.laplace_rounds,
            laplace_is_draw_chunk_size=args.draw_chunk_size,
            laplace_is_student_df=args.student_df,
            laplace_is_scale_inflation=args.scale_inflation,
            laplace_is_map_maxiter=args.map_maxiter,
            laplace_is_fallback=not args.disable_fallback,
            return_diagnostics=True,
            **problem["kwargs"],
        )
    elif method == "independent_nuts":
        draws, diagnostics = get_samples_independent(
            *common,
            num_warmup=args.warmup,
            num_samples=args.samples,
            lane_width=int(np.shape(problem["y"])[0]),
            channel_varying_kwargs=problem["varying"],
            dense_mass=True,
            regularize_mass_matrix=True,
            target_accept_prob=0.8,
            max_tree_depth=10,
            return_diagnostics=True,
            **problem["kwargs"],
        )
    elif method == "joint_nuts":
        kernel = NUTS(
            problem["model"],
            init_strategy=init_to_value(values=problem["init"]),
            dense_mass=False,
            regularize_mass_matrix=True,
            target_accept_prob=0.8,
            max_tree_depth=10,
        )
        mcmc = MCMC(
            kernel,
            num_warmup=args.warmup,
            num_samples=args.samples,
            progress_bar=False,
            jit_model_args=True,
        )
        mcmc.run(
            key,
            problem["t"],
            problem["yerr"],
            y=problem["y"],
            extra_fields=("num_steps", "diverging", "accept_prob"),
            **problem["kwargs"],
        )
        draws = mcmc.get_samples()
        diagnostics = mcmc.get_extra_fields()
    else:
        raise ValueError(method)
    _ready(draws)
    return draws, diagnostics, time.perf_counter() - started


def _site_arrays(draws, num_channels):
    result = {}
    for site, value in draws.items():
        array = np.asarray(jax.device_get(value), dtype=np.float64)
        if array.ndim < 2 or array.shape[1] != num_channels:
            continue
        result[site] = array.reshape((array.shape[0], num_channels, -1))
    return result


def _summarize(draws, seconds, num_channels):
    sites = _site_arrays(draws, num_channels)
    output = {"wall_seconds": seconds, "sites": {}}
    all_ess = []
    for site, array in sorted(sites.items()):
        ess = np.empty(array.shape[1:], dtype=np.float64)
        for channel in range(array.shape[1]):
            for component in range(array.shape[2]):
                ess[channel, component] = float(
                    effective_sample_size(
                        jnp.asarray(array[:, channel, component])[None, :]
                    )
                )
        all_ess.extend(ess.ravel().tolist())
        output["sites"][site] = {
            "median": np.median(array, axis=0).tolist(),
            "sigma": np.std(array, axis=0, ddof=1).tolist(),
            "q16": np.quantile(array, 0.16, axis=0).tolist(),
            "q84": np.quantile(array, 0.84, axis=0).tolist(),
            "ess": ess.tolist(),
        }
    output["ess_min"] = float(np.min(all_ess))
    output["ess_median"] = float(np.median(all_ess))
    output["ess_sum_per_second"] = float(np.sum(all_ess) / seconds)
    return output


def _comparison(candidate, reference):
    result = {}
    for site in sorted(set(candidate["sites"]) & set(reference["sites"])):
        candidate_site = candidate["sites"][site]
        reference_site = reference["sites"][site]
        med = np.asarray(candidate_site["median"])
        ref_med = np.asarray(reference_site["median"])
        sd = np.asarray(candidate_site["sigma"])
        ref_sd = np.asarray(reference_site["sigma"])
        ref_q16 = np.asarray(reference_site["q16"])
        ref_q84 = np.asarray(reference_site["q84"])
        scale = np.maximum(ref_sd, np.finfo(np.float64).tiny)
        result[site] = {
            "abs_median_shift_over_sigma": (np.abs(med - ref_med) / scale).tolist(),
            "sigma_ratio": (sd / scale).tolist(),
            "q16_shift_over_sigma": (
                (np.asarray(candidate_site["q16"]) - ref_q16) / scale
            ).tolist(),
            "q84_shift_over_sigma": (
                (np.asarray(candidate_site["q84"]) - ref_q84) / scale
            ).tolist(),
        }
    return result


def _diagnostics(method, diagnostics):
    if method == "laplace_is":
        fields = {}
        for name in diagnostics.__dataclass_fields__:
            fields[name] = np.asarray(jax.device_get(getattr(diagnostics, name))).tolist()
        return fields
    if method == "independent_nuts":
        return {
            "mean_num_steps_per_channel": np.asarray(
                jax.device_get(diagnostics.num_steps)
            ).mean(0).tolist(),
            "divergences_per_channel": np.asarray(
                jax.device_get(diagnostics.diverging)
            ).sum(0).tolist(),
        }
    return {
        name: np.asarray(jax.device_get(value)).tolist()
        for name, value in diagnostics.items()
    }


def main():
    args = _parser().parse_args()
    jax.config.update("jax_enable_x64", True)
    problem = _problem(args)
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    valid = {"laplace_is", "independent_nuts", "joint_nuts"}
    if not methods or not set(methods) <= valid:
        raise ValueError(f"Methods must be drawn from {sorted(valid)}.")
    results = {}
    draws_by_method = {}
    for index, method in enumerate(methods):
        print(f"starting {method}", flush=True)
        draws, diagnostics, seconds = _run(
            method, problem, args, jax.random.PRNGKey(args.seed + 100 + index)
        )
        draws_by_method[method] = draws
        results[method] = _summarize(
            draws, seconds, int(np.shape(problem["y"])[0])
        )
        results[method]["diagnostics"] = _diagnostics(method, diagnostics)
        print(f"finished {method} in {seconds:.3f} s", flush=True)

    reference_name = "independent_nuts" if "independent_nuts" in results else methods[-1]
    for method in methods:
        if method != reference_name:
            results[method]["comparison_to_" + reference_name] = _comparison(
                results[method], results[reference_name]
            )
            results[method]["wall_speedup_vs_" + reference_name] = (
                results[reference_name]["wall_seconds"] / results[method]["wall_seconds"]
            )
    payload = {
        "configuration": vars(args),
        "problem": {
            **problem["meta"],
            "channels": int(np.shape(problem["y"])[0]),
            "cadences": int(np.shape(problem["y"])[1]),
        },
        "reference": reference_name,
        "results": results,
    }
    output = Path(args.json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps({
        name: {
            "wall_seconds": value["wall_seconds"],
            "ess_min": value["ess_min"],
            "ess_median": value["ess_median"],
        }
        for name, value in results.items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
