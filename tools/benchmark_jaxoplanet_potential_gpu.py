#!/usr/bin/env python3
"""Benchmark the complete production Jaxoplanet NumPyro potential on GPU.

Unlike the light-curve microbenchmark, this measures the transformed priors,
deterministics, detrending, likelihood, and transit together.  The model data
and fixed white-light geometry remain dynamic JIT arguments, matching
``MCMC(..., jit_model_args=True)``; gradients are taken only with respect to
the unconstrained NUTS state.  Real HAT-P-12 and HAT-P-65 timestamp grids and
their production detrending families are available as presets.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
if __name__ == "__main__":
    for index, token in enumerate(sys.argv[1:], start=1):
        if token == "--platform" and index + 1 < len(sys.argv):
            requested = sys.argv[index + 1]
            if requested in {"cpu", "gpu"}:
                jax.config.update("jax_platform_name", requested)
            break
        if token.startswith("--platform="):
            requested = token.split("=", 1)[1]
            if requested in {"cpu", "gpu"}:
                jax.config.update("jax_platform_name", requested)
            break

import jax.numpy as jnp
import numpy as np
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.jaxoplanet import build_transit_window_indices, create_vectorized_model
from tools.benchmark_jaxoplanet_gpu import (
    Timing,
    _is_resource_exhausted,
    device_memory,
    load_time_grid,
    parse_int_list,
)


PRESETS = {
    "hatp12": {
        "path": (
            "/scratch/midway3/tfairnington/"
            "ASYMM_HAT-P-12_SOSS_ORDER1_FIXEDLD_QUADRATIC_SPOT/"
            "HAT-P-12_NIRISS_SOSS_order1_spectroscopy_data_20LR_50HR.pkl"
        ),
        "t0": 60115.45,
        "period": 3.21305762,
        "duration": 0.090375,
        "impact": 0.343,
        "radius": 0.1406,
        "detrend": "spot_spectroscopic",
    },
    "hatp65": {
        "path": (
            "/scratch/midway3/tfairnington/STELLARINFORMED/"
            "HAT-P-65_PRISM_NRS1_V1_STELLARINFORMEDLD_POWER2_EXPLINEAR/"
            "HAT-P-65_NIRSPEC_PRISM_nrs1_spectroscopy_data_20LR_nativeHR.pkl"
        ),
        "t0": 60470.72,
        "period": 2.60544751,
        "duration": 0.17688375,
        "impact": 0.464,
        "radius": 0.1006,
        "detrend": "explinear",
    },
}


def _block(value):
    return jax.block_until_ready(value)


def _timed_value_grad(fn, state, runtime, *, warmup: int, repeats: int):
    compiled = jax.jit(jax.value_and_grad(fn, argnums=0))
    start = time.perf_counter()
    first = _block(compiled(state, runtime))
    compile_first_ms = 1.0e3 * (time.perf_counter() - start)
    for _ in range(warmup):
        _block(compiled(state, runtime))
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        _block(compiled(state, runtime))
        elapsed.append(1.0e3 * (time.perf_counter() - start))
    return (
        Timing(
            compile_first_ms=compile_first_ms,
            median_ms=float(statistics.median(elapsed)),
            p05_ms=float(np.percentile(elapsed, 5)),
            p95_ms=float(np.percentile(elapsed, 95)),
            repeats=repeats,
        ),
        compiled,
        first,
    )


def _flatten_tree(tree) -> np.ndarray:
    leaves = [np.ravel(np.asarray(value)) for value in jax.tree_util.tree_leaves(tree)]
    return np.concatenate(leaves) if leaves else np.empty(0, dtype=np.float64)


def _build_model_and_state(
    kernel: str,
    times: np.ndarray,
    batch_size: int,
    preset: dict,
):
    n_times = times.size
    t = jnp.asarray(times, dtype=jnp.float64)
    yerr = jnp.full((batch_size, n_times), 7.0e-4, dtype=jnp.float64)
    y = jnp.ones((batch_size, n_times), dtype=jnp.float64)
    period = jnp.asarray([preset["period"]], dtype=jnp.float64)
    duration = jnp.asarray([preset["duration"]], dtype=jnp.float64)
    t0 = jnp.asarray([0.0], dtype=jnp.float64)
    impact = jnp.asarray([preset["impact"]], dtype=jnp.float64)
    window = build_transit_window_indices(times, period, t0, duration)

    model = create_vectorized_model(
        detrend_type=preset["detrend"],
        ld_mode="informed",
        trend_mode="free",
        n_planets=1,
        ld_profile="power2",
        param_method="duration",
        transit_window="auto",
        transit_window_indices=window,
        jaxoplanet_kernel=kernel,
    )
    ld_center = jnp.column_stack(
        (
            jnp.linspace(0.52, 0.58, batch_size, dtype=jnp.float64),
            jnp.linspace(0.37, 0.43, batch_size, dtype=jnp.float64),
        )
    )
    kwargs = {
        "y": y,
        "mu_duration": duration,
        "mu_t0": t0,
        "mu_b": impact,
        "mu_depths": jnp.full(
            (batch_size, 1), preset["radius"] ** 2, dtype=jnp.float64
        ),
        "PERIOD": period,
        "mu_u_ld": ld_center,
        "sigma_u_ld": jnp.full_like(ld_center, 0.08),
        "precomputed_yerr_per_lc": jnp.full(
            (batch_size,), 7.0e-4, dtype=jnp.float64
        ),
    }
    init = {
        "rors": jnp.full(
            (batch_size, 1), preset["radius"], dtype=jnp.float64
        ),
        "c1": ld_center[:, 0],
        "c2": ld_center[:, 1],
        "log_jitter": jnp.full(
            (batch_size,), jnp.log(1.0e-4), dtype=jnp.float64
        ),
        "c": jnp.ones((batch_size,), dtype=jnp.float64),
    }
    if preset["detrend"] == "explinear":
        init.update(
            {
                "v": jnp.zeros((batch_size,), dtype=jnp.float64),
                "A": jnp.zeros((batch_size,), dtype=jnp.float64),
                "log_tau": jnp.full(
                    (batch_size,), jnp.log(0.02), dtype=jnp.float64
                ),
            }
        )
    elif preset["detrend"] == "spot_spectroscopic":
        spot = 1.0e-3 * jnp.exp(-0.5 * jnp.square(t / 0.005))
        kwargs["spot_trend"] = spot
        init["A_spot"] = jnp.ones((batch_size,), dtype=jnp.float64)

    info = initialize_model(
        jax.random.PRNGKey(1729),
        model,
        init_strategy=init_to_value(values=init),
        dynamic_args=True,
        model_args=(t, yerr),
        model_kwargs=kwargs,
    )
    runtime = {"t": t, "yerr": yerr, **kwargs}

    def potential(state, dynamic):
        return info.potential_fn(
            dynamic["t"],
            dynamic["yerr"],
            y=dynamic["y"],
            **{
                name: value
                for name, value in dynamic.items()
                if name not in {"t", "yerr", "y"}
            },
        )(state)

    return potential, info.param_info.z, runtime, window


def run_case(times, batch_size, preset, *, candidate="streamed", warmup, repeats):
    if candidate not in {"streamed", "fused"}:
        raise ValueError("candidate must be 'streamed' or 'fused'")
    stock_fn, stock_state, stock_runtime, window = _build_model_and_state(
        "stock", times, batch_size, preset
    )
    candidate_fn, candidate_state, candidate_runtime, candidate_window = (
        _build_model_and_state(candidate, times, batch_size, preset)
    )
    if not np.array_equal(window, candidate_window):
        raise RuntimeError(f"stock and {candidate} transit windows differ")
    state_diff = _flatten_tree(
        jax.tree_util.tree_map(
            lambda left, right: left - right, stock_state, candidate_state
        )
    )
    if np.max(np.abs(state_diff), initial=0.0) != 0.0:
        raise RuntimeError(
            f"stock and {candidate} unconstrained initial states differ"
        )

    stock_timing, stock_compiled, _ = _timed_value_grad(
        stock_fn, stock_state, stock_runtime, warmup=warmup, repeats=repeats
    )
    candidate_timing, candidate_compiled, _ = _timed_value_grad(
        candidate_fn,
        candidate_state,
        candidate_runtime,
        warmup=warmup,
        repeats=repeats,
    )
    stock_value, stock_grad = _block(stock_compiled(stock_state, stock_runtime))
    candidate_value, candidate_grad = _block(
        candidate_compiled(candidate_state, candidate_runtime)
    )
    grad_reference = _flatten_tree(stock_grad)
    grad_difference = _flatten_tree(
        jax.tree_util.tree_map(
            lambda left, right: left - right, stock_grad, candidate_grad
        )
    )
    selected_names = {"rors", "c1", "c2"}
    transit_reference = _flatten_tree(
        {name: stock_grad[name] for name in selected_names}
    )
    transit_difference = _flatten_tree(
        {
            name: stock_grad[name] - candidate_grad[name]
            for name in selected_names
        }
    )
    fidelity = {
        "potential_abs": float(jnp.abs(stock_value - candidate_value)),
        "potential_abs_per_datum": float(
            jnp.abs(stock_value - candidate_value) / (batch_size * times.size)
        ),
        "gradient_max_abs": float(np.max(np.abs(grad_difference))),
        "gradient_relative_l2": float(
            np.linalg.norm(grad_difference)
            / max(np.linalg.norm(grad_reference), np.finfo(np.float64).tiny)
        ),
        "transit_gradient_relative_l2": float(
            np.linalg.norm(transit_difference)
            / max(np.linalg.norm(transit_reference), np.finfo(np.float64).tiny)
        ),
        "all_finite": bool(
            np.isfinite(np.asarray(stock_value)).all()
            and np.isfinite(np.asarray(candidate_value)).all()
            and np.isfinite(grad_reference).all()
            and np.isfinite(grad_difference).all()
        ),
    }
    return {
        "case": {
            "n_times": int(times.size),
            "window_size": int(window.size),
            "batch_size": int(batch_size),
            "detrend": preset["detrend"],
            "candidate": candidate,
        },
        "fidelity": fidelity,
        "timings": {
            "stock_potential_value_grad": asdict(stock_timing),
            f"{candidate}_potential_value_grad": asdict(candidate_timing),
        },
        "speedup": stock_timing.median_ms / candidate_timing.median_ms,
        "device_memory": device_memory(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), required=True)
    parser.add_argument("--batches", type=parse_int_list, default=[1, 8, 40, 60, 80, 100])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--candidate", choices=("streamed", "fused"), default="streamed"
    )
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args(argv)
    if args.platform != "auto":
        jax.config.update("jax_platform_name", args.platform)
    preset = PRESETS[args.preset]
    times, metadata = load_time_grid(
        preset["path"], attribute="time", transit_t0=preset["t0"]
    )
    print(
        f"JAX {jax.__version__}; backend={jax.default_backend()}; "
        f"device={jax.devices()[0]}; n_times={times.size}"
    )
    results = []
    for batch in args.batches:
        try:
            result = run_case(
                times,
                batch,
                preset,
                candidate=args.candidate,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            case = result["case"]
            timing = result["timings"]
            print(
                f"batch={batch} window={case['window_size']}/{case['n_times']} "
                f"stock={timing['stock_potential_value_grad']['median_ms']:.4f} ms "
                f"{args.candidate}="
                f"{timing[f'{args.candidate}_potential_value_grad']['median_ms']:.4f} ms "
                f"speedup={result['speedup']:.3f}x "
                f"grad_rel={result['fidelity']['transit_gradient_relative_l2']:.3e}"
            )
        except Exception as exc:
            if not _is_resource_exhausted(exc):
                raise
            result = {
                "case": {"n_times": int(times.size), "batch_size": int(batch)},
                "status": "oom",
                "error": str(exc),
            }
            print(f"batch={batch}: OOM")
            jax.clear_caches()
        results.append(result)
    payload = {
        "environment": {
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
            "x64": bool(jax.config.read("jax_enable_x64")),
            "time_grid": metadata,
            "preset": args.preset,
            "candidate": args.candidate,
        },
        "results": results,
    }
    if args.json_path:
        destination = Path(args.json_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
        print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
