#!/usr/bin/env python3
"""Short real-grid NUTS validation for the fixed-quadratic local-JVP route."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from numpyro.infer import MCMC, NUTS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.benchmark_jaxoplanet_gpu import load_time_grid
from tools.benchmark_jaxoplanet_potential_gpu import PRESETS
from tools.benchmark_jaxoplanet_quadratic_fixed_u_potential_gpu import _build


def _flatten_samples(samples):
    return {
        name: np.asarray(value).reshape(np.asarray(value).shape[0], -1)
        for name, value in samples.items()
    }


def _run(kernel, times, preset, batch, warmup, samples, seed):
    potential, state, runtime, window = _build(kernel, times, batch, preset)

    def fixed_potential(z):
        return potential(z, runtime)

    sampler = MCMC(
        NUTS(
            potential_fn=fixed_potential,
            target_accept_prob=0.9,
            max_tree_depth=10,
        ),
        num_warmup=warmup,
        num_samples=samples,
        num_chains=1,
        progress_bar=False,
    )
    start = time.perf_counter()
    sampler.run(
        jax.random.PRNGKey(seed),
        init_params=state,
        extra_fields=("diverging", "num_steps", "accept_prob"),
    )
    elapsed = time.perf_counter() - start
    draws = _flatten_samples(sampler.get_samples(group_by_chain=False))
    extras = {
        name: np.asarray(value)
        for name, value in sampler.get_extra_fields(group_by_chain=False).items()
    }
    summary = {
        name: {
            "mean": np.mean(value, axis=0).tolist(),
            "sd": np.std(value, axis=0, ddof=1).tolist(),
            "all_finite": bool(np.all(np.isfinite(value))),
        }
        for name, value in draws.items()
    }
    return {
        "elapsed_seconds": elapsed,
        "window": int(window.size),
        "divergences": int(np.sum(extras["diverging"])),
        "median_num_steps": float(np.median(extras["num_steps"])),
        "mean_accept_prob": float(np.mean(extras["accept_prob"])),
        "summary": summary,
    }, draws


def _posterior_comparison(stock, candidate):
    result = {}
    for name in sorted(stock):
        stock_mean = np.mean(stock[name], axis=0)
        candidate_mean = np.mean(candidate[name], axis=0)
        pooled_sd = np.maximum(
            0.5
            * (
                np.std(stock[name], axis=0, ddof=1)
                + np.std(candidate[name], axis=0, ddof=1)
            ),
            np.finfo(np.float64).tiny,
        )
        standardized = np.abs(candidate_mean - stock_mean) / pooled_sd
        result[name] = {
            "max_abs_mean_difference": float(
                np.max(np.abs(candidate_mean - stock_mean), initial=0.0)
            ),
            "max_mean_difference_in_pooled_sd": float(
                np.max(standardized, initial=0.0)
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="hatp12")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=9142)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    times, metadata = load_time_grid(
        preset["path"], attribute="time", transit_t0=preset["t0"]
    )
    stock_result, stock_draws = _run(
        "stock",
        times,
        preset,
        args.batch,
        args.warmup,
        args.samples,
        args.seed,
    )
    candidate_result, candidate_draws = _run(
        "quadratic_local_jvp",
        times,
        preset,
        args.batch,
        args.warmup,
        args.samples,
        args.seed,
    )
    payload = {
        "environment": {
            "device": str(jax.devices()[0]),
            "jax_version": jax.__version__,
            "time_grid": metadata,
        },
        "case": {
            "preset": args.preset,
            "batch": args.batch,
            "warmup": args.warmup,
            "samples": args.samples,
            "seed": args.seed,
            "limb_darkening": "fixed_direct_u1_u2_no_kipping",
        },
        "stock": stock_result,
        "quadratic_local_jvp": candidate_result,
        "posterior_comparison": _posterior_comparison(
            stock_draws, candidate_draws
        ),
    }
    destination = Path(args.json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))

    all_finite = all(
        entry["all_finite"]
        for result in (stock_result, candidate_result)
        for entry in result["summary"].values()
    )
    return 0 if all_finite else 2


if __name__ == "__main__":
    raise SystemExit(main())
