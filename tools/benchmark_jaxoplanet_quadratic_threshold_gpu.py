#!/usr/bin/env python3
"""Find the GPU crossover between direct and local-JVP quadratic kernels.

The workload uses 40 independent channels, fixed literal ``u1,u2`` (no
Kipping coordinates), and gradients with respect to radius, impact,
duration, linear trend, and jitter.  Candidate timings are interleaved to
reduce clock/host-load drift.  Active-window length is the swept variable.
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
if __name__ == "__main__":
    for i, token in enumerate(sys.argv[1:], 1):
        if token == "--platform" and i + 1 < len(sys.argv):
            if sys.argv[i + 1] in {"cpu", "gpu"}:
                jax.config.update("jax_platform_name", sys.argv[i + 1])
            break

import jax.numpy as jnp
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.jaxoplanet.limb_dark_quadratic import light_curve as direct
from models.jaxoplanet.limb_dark_quadratic_local_jvp import (
    light_curve as local_jvp,
)


def _block(x):
    return jax.block_until_ready(x)


def _timing(values):
    values = sorted(values)
    return {
        "median_ms": float(statistics.median(values)),
        "p05_ms": float(np.percentile(values, 5)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def _compiler_metrics(lowered):
    text = str(lowered.compiler_ir(dialect="stablehlo"))
    executable = lowered.compile()
    memory = executable.memory_analysis()
    cost = executable.cost_analysis()
    return {
        "reduce_count": text.count("stablehlo.reduce"),
        "dot_general_count": text.count("stablehlo.dot_general"),
        "temp_size_in_bytes": int(memory.temp_size_in_bytes),
        "bytes_accessed_estimate": float(cost.get("bytes accessed", np.nan)),
        "flops_estimate": float(cost.get("flops", np.nan)),
    }


def make_objective(evaluator, n_active, batch):
    dt = jnp.linspace(-0.5 * 0.17688375, 0.5 * 0.17688375, n_active)
    lanes = jnp.linspace(-1, 1, batch)
    fixed_u = jnp.column_stack((0.30 + 0.03 * lanes, 0.20 + 0.02 * lanes))
    theta = jnp.column_stack(
        (
            0.1006 + 0.002 * lanes,
            0.464 + 0.02 * lanes,
            0.17688375 * (1 + 0.01 * lanes),
            jnp.ones(batch),
            jnp.full(batch, 0.002),
            jnp.full(batch, jnp.log(7e-4)),
        )
    )
    runtime = {"dt": dt, "u": fixed_u}

    def forward(value, dynamic):
        def one(channel, u):
            r, b, duration, offset, slope, _ = channel
            speed = 2 * jnp.sqrt(jnp.maximum(0, (1 + r) ** 2 - b**2)) / duration
            separation = jnp.sqrt((speed * dynamic["dt"]) ** 2 + b**2)
            return evaluator(u, separation, r, order=10) + offset + slope * dynamic["dt"]

        return jax.vmap(one)(value, dynamic["u"])

    # The observations do not affect kernel comparison; keeping them simple
    # prevents a separate truth-kernel trace from contaminating timing.
    phase = jnp.linspace(0, 2 * jnp.pi, n_active)
    observations = 1 + 7e-5 * jnp.sin(phase)[None, :]

    def objective(value, dynamic):
        model = forward(value, dynamic)
        sigma = jnp.exp(value[:, 5:6])
        residual = (model - observations) / sigma
        return 0.5 * jnp.sum(residual**2) + n_active * jnp.sum(value[:, 5])

    return jax.value_and_grad(objective), theta, runtime


def run_size(n_active, batch, warmup, repeats):
    direct_fn, theta, runtime = make_objective(direct, n_active, batch)
    local_fn, local_theta, local_runtime = make_objective(
        local_jvp, n_active, batch
    )
    direct_jit = jax.jit(direct_fn)
    local_jit = jax.jit(local_fn)
    start = time.perf_counter()
    direct_first = _block(direct_jit(theta, runtime))
    direct_compile = 1e3 * (time.perf_counter() - start)
    start = time.perf_counter()
    local_first = _block(local_jit(local_theta, local_runtime))
    local_compile = 1e3 * (time.perf_counter() - start)
    for _ in range(warmup):
        _block(direct_jit(theta, runtime))
        _block(local_jit(local_theta, local_runtime))
    elapsed = {"current_direct": [], "local_jvp": []}
    for iteration in range(repeats):
        order = (
            (("current_direct", direct_jit, theta, runtime),
             ("local_jvp", local_jit, local_theta, local_runtime))
            if iteration % 2 == 0
            else (("local_jvp", local_jit, local_theta, local_runtime),
                  ("current_direct", direct_jit, theta, runtime))
        )
        for name, fn, value, dynamic in order:
            start = time.perf_counter()
            _block(fn(value, dynamic))
            elapsed[name].append(1e3 * (time.perf_counter() - start))
    direct_value, direct_grad = direct_first
    local_value, local_grad = local_first
    gradient_difference = np.asarray(local_grad - direct_grad)
    gradient_reference = np.asarray(direct_grad)
    direct_timing = _timing(elapsed["current_direct"])
    local_timing = _timing(elapsed["local_jvp"])
    direct_timing["compile_first_ms"] = direct_compile
    local_timing["compile_first_ms"] = local_compile
    direct_timing["compiler"] = _compiler_metrics(
        direct_jit.lower(theta, runtime)
    )
    local_timing["compiler"] = _compiler_metrics(
        local_jit.lower(local_theta, local_runtime)
    )
    return {
        "n_active": n_active,
        "batch": batch,
        "current_direct": direct_timing,
        "local_jvp": local_timing,
        "local_speedup_over_direct": (
            direct_timing["median_ms"] / local_timing["median_ms"]
        ),
        "fidelity": {
            "objective_abs_per_datum": float(
                abs(local_value - direct_value) / (batch * n_active)
            ),
            "gradient_max_abs": float(np.max(np.abs(gradient_difference))),
            "gradient_relative_l2": float(
                np.linalg.norm(gradient_difference)
                / max(np.linalg.norm(gradient_reference), 1e-300)
            ),
            "all_finite": bool(
                np.isfinite(local_value)
                and np.all(np.isfinite(np.asarray(local_grad)))
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", default="64,89,128,256,512,1024,2048,4096,8192,16515"
    )
    parser.add_argument("--batch", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]
    results = []
    for size in sizes:
        result = run_size(size, args.batch, args.warmup, args.repeats)
        results.append(result)
        print(
            f"n={size:5d} direct={result['current_direct']['median_ms']:.6f} "
            f"local={result['local_jvp']['median_ms']:.6f} "
            f"local/direct={result['local_speedup_over_direct']:.4f}x "
            f"grad_rel={result['fidelity']['gradient_relative_l2']:.3e}"
        )
    payload = {
        "environment": {
            "jax_version": jax.__version__,
            "device": str(jax.devices()[0]),
        },
        "parameterization": "fixed_direct_u1_u2_no_kipping",
        "results": results,
    }
    destination = Path(args.json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True))
    failed = any(
        not x["fidelity"]["all_finite"]
        or x["fidelity"]["gradient_relative_l2"] > 1e-8
        for x in results
    )
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
