#!/usr/bin/env python3
"""Complete fixed-quadratic NumPyro potential benchmark for local JVP.

This constructs the actual vectorized NumPyro model with literal fixed
``u1,u2`` and compares stock jaxoplanet, the algebraic direct-quadratic route,
and the production opt-in local-JVP route.  It exercises the same core router
used by ``fit_jwst.py``; no benchmark-only monkeypatch is involved.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.jaxoplanet import build_transit_window_indices, create_vectorized_model
from tools.benchmark_jaxoplanet_gpu import device_memory, load_time_grid
from tools.benchmark_jaxoplanet_potential_gpu import PRESETS


def _block(value):
    return jax.block_until_ready(value)


def _flatten(tree):
    return np.concatenate(
        [np.ravel(np.asarray(x)) for x in jax.tree_util.tree_leaves(tree)]
    )


def _build(kernel, times, batch_size, preset):
    n_times = times.size
    t = jnp.asarray(times)
    yerr = jnp.full((batch_size, n_times), 7.0e-4)
    y = jnp.ones((batch_size, n_times))
    period = jnp.asarray([preset["period"]])
    duration = jnp.asarray([preset["duration"]])
    t0 = jnp.asarray([0.0])
    impact = jnp.asarray([preset["impact"]])
    window = build_transit_window_indices(times, period, t0, duration)

    model = create_vectorized_model(
        detrend_type=preset["detrend"],
        ld_mode="fixed",
        trend_mode="free",
        n_planets=1,
        ld_profile="quadratic",
        param_method="duration",
        transit_window="auto",
        transit_window_indices=window,
        jaxoplanet_kernel=kernel,
    )
    lanes = jnp.linspace(-1, 1, batch_size)
    fixed_u = jnp.column_stack((0.30 + 0.03 * lanes, 0.20 + 0.02 * lanes))
    kwargs = {
        "y": y,
        "mu_duration": duration,
        "mu_t0": t0,
        "mu_b": impact,
        "mu_depths": jnp.full((batch_size, 1), preset["radius"] ** 2),
        "PERIOD": period,
        "mu_u_ld": fixed_u,
        "ld_fixed": fixed_u,
        "precomputed_yerr_per_lc": jnp.full((batch_size,), 7.0e-4),
    }
    init = {
        "rors": jnp.full((batch_size, 1), preset["radius"]),
        "log_jitter": jnp.full((batch_size,), jnp.log(1.0e-4)),
        "c": jnp.ones((batch_size,)),
    }
    if preset["detrend"] == "explinear":
        init.update(
            v=jnp.zeros((batch_size,)),
            A=jnp.zeros((batch_size,)),
            log_tau=jnp.full((batch_size,), jnp.log(0.02)),
        )
    else:
        kwargs["spot_trend"] = 1.0e-3 * jnp.exp(
            -0.5 * jnp.square(t / 0.005)
        )
        init["A_spot"] = jnp.ones((batch_size,))

    info = initialize_model(
        jax.random.PRNGKey(4021),
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


def _timed(fn, state, runtime, warmup, repeats):
    compiled = jax.jit(jax.value_and_grad(fn))
    start = time.perf_counter()
    _block(compiled(state, runtime))
    compile_first = 1e3 * (time.perf_counter() - start)
    for _ in range(warmup):
        _block(compiled(state, runtime))
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        _block(compiled(state, runtime))
        elapsed.append(1e3 * (time.perf_counter() - start))
    lowered = compiled.lower(state, runtime)
    hlo = str(lowered.compiler_ir(dialect="stablehlo"))
    executable = lowered.compile()
    cost = executable.cost_analysis()
    memory = executable.memory_analysis()
    return {
        "compile_first_ms": compile_first,
        "median_ms": statistics.median(elapsed),
        "p05_ms": float(np.percentile(elapsed, 5)),
        "p95_ms": float(np.percentile(elapsed, 95)),
        "stablehlo_reduce_count": hlo.count("stablehlo.reduce"),
        "stablehlo_dot_general_count": hlo.count("stablehlo.dot_general"),
        "bytes_accessed_estimate": float(cost.get("bytes accessed", np.nan)),
        "flops_estimate": float(cost.get("flops", np.nan)),
        "transcendentals_estimate": float(
            cost.get("transcendentals", np.nan)
        ),
        "compiler_memory": {
            "argument_size_in_bytes": int(memory.argument_size_in_bytes),
            "output_size_in_bytes": int(memory.output_size_in_bytes),
            "alias_size_in_bytes": int(memory.alias_size_in_bytes),
            "temp_size_in_bytes": int(memory.temp_size_in_bytes),
            "generated_code_size_in_bytes": int(
                memory.generated_code_size_in_bytes
            ),
        },
    }, compiled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), required=True)
    parser.add_argument("--batch", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    preset = PRESETS[args.preset]
    times, metadata = load_time_grid(
        preset["path"], attribute="time", transit_t0=preset["t0"]
    )
    stock_fn, stock_state, stock_runtime, stock_window = _build(
        "stock", times, args.batch, preset
    )
    direct_fn, direct_state, direct_runtime, direct_window = _build(
        "quadratic_specialized", times, args.batch, preset
    )
    if not np.array_equal(stock_window, direct_window):
        raise RuntimeError("stock and direct transit windows differ")
    stock_timing, stock_compiled = _timed(
        stock_fn, stock_state, stock_runtime, args.warmup, args.repeats
    )
    stock_memory = device_memory()
    direct_timing, direct_compiled = _timed(
        direct_fn,
        direct_state,
        direct_runtime,
        args.warmup,
        args.repeats,
    )
    direct_memory = device_memory()

    candidate_fn, candidate_state, candidate_runtime, candidate_window = (
        _build("quadratic_local_jvp", times, args.batch, preset)
    )
    if not np.array_equal(stock_window, candidate_window):
        raise RuntimeError("stock and candidate transit windows differ")
    candidate_timing, candidate_compiled = _timed(
        candidate_fn,
        candidate_state,
        candidate_runtime,
        args.warmup,
        args.repeats,
    )
    candidate_memory = device_memory()

    stock_value, stock_grad = _block(stock_compiled(stock_state, stock_runtime))
    direct_value, direct_grad = _block(
        direct_compiled(direct_state, direct_runtime)
    )
    candidate_value, candidate_grad = _block(
        candidate_compiled(candidate_state, candidate_runtime)
    )
    gr = _flatten(stock_grad)

    def fidelity(value, gradient):
        gd = _flatten(gradient) - gr
        return {
            "potential_abs": float(abs(value - stock_value)),
            "gradient_max_abs": float(np.max(np.abs(gd), initial=0)),
            "gradient_relative_l2": float(
                np.linalg.norm(gd) / max(np.linalg.norm(gr), 1e-300)
            ),
            "all_finite": bool(
                np.isfinite(value)
                and np.all(np.isfinite(_flatten(gradient)))
            ),
        }

    direct_fidelity = fidelity(direct_value, direct_grad)
    candidate_fidelity = fidelity(candidate_value, candidate_grad)
    state_sites = (
        sorted(stock_state.keys())
        if isinstance(stock_state, dict)
        else ["<non-dict-state>"]
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
            "window": int(stock_window.size),
            "limb_darkening": "fixed_direct_u1_u2_no_kipping",
            "candidate": "symbolic_zero_aware_local_jvp",
            "unconstrained_state_sites": state_sites,
            "unconstrained_state_size": int(_flatten(stock_state).size),
        },
        "stock": {"timing": stock_timing, "device_memory": stock_memory},
        "current_direct": {
            "timing": direct_timing,
            "device_memory": direct_memory,
            "speedup": (
                stock_timing["median_ms"] / direct_timing["median_ms"]
            ),
            "fidelity": direct_fidelity,
        },
        "local_jvp": {
            "timing": candidate_timing,
            "device_memory": candidate_memory,
            "speedup": (
                stock_timing["median_ms"] / candidate_timing["median_ms"]
            ),
            "fidelity": candidate_fidelity,
        },
    }
    destination = Path(args.json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    all_pass = all(
        item["all_finite"] and item["gradient_relative_l2"] < 1e-8
        for item in (direct_fidelity, candidate_fidelity)
    )
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
