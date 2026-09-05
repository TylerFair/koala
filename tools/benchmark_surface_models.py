#!/usr/bin/env python3
"""Repeatable CPU microbenchmark for JAXoplanet surface evaluators.

This measures a compiled model vector and the gradient of its squared-signal
sum. It characterizes kernel cost; it is not an end-to-end NUTS benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from models.jaxoplanet.core import compute_transit_model
from models.jaxoplanet.surface import compute_surface_model


def _timed(callable_, repeats):
    start = time.perf_counter()
    result = callable_()
    jax.block_until_ready(result)
    compile_seconds = time.perf_counter() - start
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = callable_()
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - start)
    return compile_seconds, 1e3 * float(np.median(samples))


def benchmark(*, cadences=321, repeats=10, order=20, spot_degree=8):
    time_grid = jnp.linspace(-0.1, 0.9, cadences)
    base = {
        "period": 1.0,
        "t0": 0.0,
        "a_rs": 8.0,
        "b": 0.2,
        "rors": 0.1,
        "ecc": 0.0,
        "omega": 0.0,
        "u": jnp.array([0.3, 0.2]),
        "eclipse_depth": 8e-4,
        "dayside_flux": 1e-3,
        "nightside_flux": 2.5e-4,
        "hotspot_offset": 0.3,
        "stellar_rotation_period": 3.7,
    }
    spots = ({"latitude": 0.1, "longitude": 0.2, "radius": 0.2, "contrast": 0.3},)

    def stock(values):
        return compute_transit_model(
            {**base, "a_rs": values[0], "rors": values[1]}, time_grid
        )

    def eclipse(values):
        params = {
            **base,
            "a_rs": values[0],
            "rors": values[1],
            "eclipse_depth": values[2],
        }
        return compute_surface_model(params, time_grid, model="eclipse", order=order)

    def phase_curve(values):
        params = {
            **base,
            "a_rs": values[0],
            "rors": values[1],
            "dayside_flux": values[2],
            "nightside_flux": values[3],
            "hotspot_offset": values[4],
        }
        return compute_surface_model(params, time_grid, model="phase_curve", order=order)

    def spotted(values):
        params = {
            **base,
            "a_rs": values[0],
            "rors": values[1],
            "stellar_spot_contrast": jnp.array([values[2]]),
        }
        return compute_surface_model(
            params,
            time_grid,
            model="transit",
            spots=spots,
            spot_degree=spot_degree,
            order=order,
        )

    cases = (
        ("stock_keplerian", stock, jnp.array([8.0, 0.1])),
        ("eclipse", eclipse, jnp.array([8.0, 0.1, 8e-4])),
        ("phase_curve", phase_curve, jnp.array([8.0, 0.1, 1e-3, 2.5e-4, 0.3])),
        ("stellar_spots", spotted, jnp.array([8.0, 0.1, 0.3])),
    )
    results = {}
    for name, function, point in cases:
        compiled_forward = jax.jit(function)
        forward_compile, forward_warm = _timed(
            lambda: compiled_forward(point), repeats
        )
        objective = lambda values: jnp.sum(jnp.square(function(values)))
        compiled_gradient = jax.jit(jax.value_and_grad(objective))
        gradient_compile, gradient_warm = _timed(
            lambda: compiled_gradient(point), repeats
        )
        results[name] = {
            "forward_compile_seconds": forward_compile,
            "forward_warm_median_ms": forward_warm,
            "value_and_grad_compile_seconds": gradient_compile,
            "value_and_grad_warm_median_ms": gradient_warm,
        }
        print(name, json.dumps(results[name], sort_keys=True), flush=True)
    return {
        "device": str(jax.devices()[0]),
        "cadences": cadences,
        "repeats": repeats,
        "starry_order": order,
        "spot_degree": spot_degree,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cadences", type=int, default=321)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--order", type=int, default=20)
    parser.add_argument("--spot-degree", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark(
        cadences=args.cadences,
        repeats=args.repeats,
        order=args.order,
        spot_degree=args.spot_degree,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
