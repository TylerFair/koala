#!/usr/bin/env python3
"""RTX benchmark for the whole-lane native Power-2 local-Jacobian VJP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.jaxoplanet.core import build_transit_phase_offsets
from models.jaxoplanet.experimental_power2_grid import (
    FixedPower2TransitGeometry,
    make_exact_power2_evaluator,
)
from models.jaxoplanet.experimental_power2_native import light_curve
from models.jaxoplanet.experimental_power2_native_local_vjp import (
    make_fixed_geometry_light_curve_local_vjp,
)


RORS_BOUNDS = (float(np.sqrt(1.0e-5)), float(np.sqrt(0.5)))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=16)
    parser.add_argument("--widths", type=int, nargs="+", default=[8, 40, 80])
    parser.add_argument("--n-times", type=int, default=513)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=8128)
    parser.add_argument("--period", type=float, default=2.60544751)
    parser.add_argument("--duration", type=float, default=0.17688375)
    parser.add_argument("--t0", type=float, default=60473.32)
    parser.add_argument("--impact-parameter", type=float, default=0.464)
    parser.add_argument("--time-pickle", required=True)
    parser.add_argument("--time-attribute", default="time")
    parser.add_argument("--json-output")
    return parser


def _ready(value):
    for leaf in jax.tree.leaves(value):
        leaf.block_until_ready()


def _load_times(args):
    with Path(args.time_pickle).open("rb") as stream:
        payload = pickle.load(stream)
    all_times = np.asarray(getattr(payload, args.time_attribute), dtype=np.float64)
    if all_times.ndim != 1 or not np.all(np.diff(all_times) > 0.0):
        raise ValueError("Actual times must be a strictly increasing vector")
    indices = np.linspace(0, all_times.size - 1, args.n_times, dtype=np.int64)
    if np.unique(indices).size != args.n_times:
        raise ValueError("Requested cadence subsample is not unique")
    return all_times[indices], int(all_times.size)


def _profile(function, argument, repeats):
    compiled = jax.jit(function).lower(argument).compile()
    _ready(compiled(argument))
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = compiled(argument)
        _ready(value)
        timings.append(time.perf_counter() - start)
    raw_cost = compiled.cost_analysis() or {}
    cost = {
        key: float(raw_cost[key])
        for key in ("flops", "transcendentals", "bytes accessed")
        if key in raw_cost
    }
    memory = {}
    memory_stats = compiled.memory_analysis()
    if memory_stats is not None:
        for name in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "temp_size_in_bytes",
            "alias_size_in_bytes",
            "generated_code_size_in_bytes",
        ):
            if hasattr(memory_stats, name):
                memory[name] = int(getattr(memory_stats, name))
    return {
        "median_ms": float(np.median(timings) * 1.0e3),
        "p05_ms": float(np.percentile(timings, 5.0) * 1.0e3),
        "p95_ms": float(np.percentile(timings, 95.0) * 1.0e3),
        "min_ms": float(np.min(timings) * 1.0e3),
        "cost_analysis": cost,
        "memory_analysis": memory,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not jax.config.x64_enabled:
        raise RuntimeError("Run with JAX_ENABLE_X64=1")
    geometry = FixedPower2TransitGeometry(
        args.period, args.duration, args.t0, args.impact_parameter
    )
    times, source_cadences = _load_times(args)
    offsets, masks = build_transit_phase_offsets(
        jnp.asarray(times),
        jnp.asarray([geometry.period]),
        jnp.asarray([geometry.t0]),
        jnp.asarray([geometry.duration]),
    )
    dt = offsets[0]
    mask = masks[0]
    local_vjp_forward = make_fixed_geometry_light_curve_local_vjp(
        np.asarray(dt),
        np.asarray(mask),
        impact_parameter=geometry.b,
        duration=geometry.duration,
        order=args.order,
    )

    def native_forward(theta):
        radius = theta[:, 0]
        c_value = theta[:, 1]
        alpha = theta[:, 2]
        speed = 2.0 * jnp.sqrt(
            jnp.maximum(0.0, (1.0 + radius) ** 2 - geometry.b**2)
        ) / geometry.duration
        separation = jnp.sqrt(
            (speed[:, None] * dt[None, :]) ** 2 + geometry.b**2
        )
        signal = light_curve(
            c_value[:, None],
            alpha[:, None],
            separation,
            radius[:, None],
            order=args.order,
        )
        return jnp.where(mask[None, :], signal, 0.0)

    stock_one = make_exact_power2_evaluator(times, geometry, kernel="stock")
    stock_forward = lambda theta: jax.vmap(stock_one)(theta)
    rng = np.random.default_rng(args.seed)
    maximum_width = max(args.widths)
    theta_all = rng.uniform(
        np.asarray([RORS_BOUNDS[0], 0.0, 0.001]),
        np.asarray([RORS_BOUNDS[1], 1.0, 1.0]),
        size=(maximum_width, 3),
    )
    report = {
        "experimental": True,
        "production_wired": False,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "architecture": "whole-lane jacfwd local Jacobian custom VJP",
        "order": args.order,
        "source_cadences": source_cadences,
        "n_times": int(times.size),
        "widths": {},
    }
    for width in args.widths:
        theta = jnp.asarray(theta_all[:width], dtype=jnp.float64)
        time_weights = jnp.linspace(0.7, 1.3, times.size, dtype=jnp.float64)
        lane_weights = jnp.linspace(0.8, 1.2, width, dtype=jnp.float64)
        weights = lane_weights[:, None] * time_weights[None, :]

        def objective(forward):
            return lambda values: jnp.vdot(forward(values), weights)

        baseline_vg = jax.value_and_grad(objective(native_forward))
        local_vjp_vg = jax.value_and_grad(objective(local_vjp_forward))
        stock_vg = jax.value_and_grad(objective(stock_forward))
        baseline_value, baseline_gradient = jax.jit(baseline_vg)(theta)
        local_value, local_gradient = jax.jit(local_vjp_vg)(theta)
        _ready((baseline_value, baseline_gradient, local_value, local_gradient))
        gradient_error = np.asarray(local_gradient - baseline_gradient)
        baseline_gradient_np = np.asarray(baseline_gradient)
        baseline_profile = _profile(baseline_vg, theta, args.repeats)
        local_profile = _profile(local_vjp_vg, theta, args.repeats)
        stock_profile = _profile(stock_vg, theta, args.repeats)
        forward_profile = _profile(native_forward, theta, args.repeats)
        report["widths"][str(width)] = {
            "value_abs_difference": float(
                np.abs(np.asarray(local_value - baseline_value))
            ),
            "gradient_max_abs_difference": float(
                np.max(np.abs(gradient_error))
            ),
            "gradient_relative_l2": float(
                np.linalg.norm(gradient_error)
                / max(np.linalg.norm(baseline_gradient_np), 1.0e-300)
            ),
            "native_forward_floor": forward_profile,
            "ordinary_native_value_gradient": baseline_profile,
            "local_vjp_value_gradient": local_profile,
            "stock_value_gradient": stock_profile,
            "local_vjp_vs_ordinary_speedup": float(
                baseline_profile["median_ms"] / local_profile["median_ms"]
            ),
            "local_vjp_vs_stock_speedup": float(
                stock_profile["median_ms"] / local_profile["median_ms"]
            ),
        }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        Path(args.json_output).write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
