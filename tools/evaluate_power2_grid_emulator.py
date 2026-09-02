#!/usr/bin/env python3
"""Build, validate, and benchmark the experimental power-2 tensor grid.

The command exits with status 2 when validation fails.  It never writes a grid
artifact that missed either the requested flux or gradient tolerance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.jaxoplanet.experimental_power2_grid import (
    FixedPower2TransitGeometry,
    build_power2_transit_grid,
    make_exact_power2_evaluator,
    save_power2_grid,
    validate_power2_grid,
)
from models.jaxoplanet.experimental_power2_hermite import (
    build_power2_radius_cubic_grid,
    build_power2_radius_hermite_grid,
)


def _positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-npy", help="Optional strictly increasing 1-D cadence array.")
    parser.add_argument("--n-times", type=_positive_int, default=256)
    parser.add_argument("--time-min", type=float, default=-0.09)
    parser.add_argument("--time-max", type=float, default=0.09)
    parser.add_argument("--period", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=0.12)
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--impact-parameter", type=float, default=0.35)
    parser.add_argument("--rors-min", type=float, default=0.08)
    parser.add_argument("--rors-max", type=float, default=0.12)
    parser.add_argument("--rors-nodes", type=_positive_int, default=21)
    parser.add_argument("--c-min", type=float, default=0.2)
    parser.add_argument("--c-max", type=float, default=0.8)
    parser.add_argument("--c-nodes", type=_positive_int, default=9)
    parser.add_argument("--alpha-min", type=float, default=0.3)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--alpha-nodes", type=_positive_int, default=9)
    parser.add_argument(
        "--exact-kernel", choices=("stock", "streamed", "fused", "auto"),
        default="stock",
    )
    parser.add_argument("--build-batch-size", type=_positive_int, default=128)
    parser.add_argument("--derivative-batch-size", type=_positive_int, default=64)
    parser.add_argument(
        "--radius-interpolation",
        choices=("linear", "cubic", "hermite"),
        default="linear",
        help="Hermite stores exact dflux/drors at every knot and is experimental.",
    )
    parser.add_argument("--max-grid-gib", type=float, default=2.0)
    parser.add_argument("--validation-samples-per-cell", type=_positive_int, default=1)
    parser.add_argument("--validation-seed", type=int, default=0)
    parser.add_argument("--flux-tolerance-ppm", type=float, default=1.0)
    parser.add_argument(
        "--gradient-abs-tolerance-ppm-per-unit", type=float, default=25.0
    )
    parser.add_argument("--gradient-relative-tolerance", type=float, default=0.02)
    parser.add_argument("--gradient-points", type=_positive_int, default=32)
    parser.add_argument("--validation-batch-size", type=_positive_int, default=128)
    parser.add_argument("--benchmark-batch", type=_positive_int, default=64)
    parser.add_argument("--benchmark-repeats", type=_positive_int, default=20)
    parser.add_argument("--artifact", help="Write this .npz only after validation passes.")
    parser.add_argument("--json-output", help="Optional path for the complete JSON report.")
    return parser


def _time_ready(function, argument, repeats):
    compile_start = time.perf_counter()
    ready = jax.block_until_ready(function(argument))
    compile_seconds = time.perf_counter() - compile_start
    del ready
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        ready = jax.block_until_ready(function(argument))
        samples.append(time.perf_counter() - start)
        del ready
    return {
        "compile_and_first_seconds": compile_seconds,
        "median_seconds": float(np.median(samples)),
        "min_seconds": float(np.min(samples)),
        "max_seconds": float(np.max(samples)),
    }


def benchmark_grid(grid, *, batch_size: int, repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    lows = np.array(
        [grid.rors_knots[0], grid.c_knots[0], grid.alpha_knots[0]],
        dtype=np.float64,
    )
    highs = np.array(
        [grid.rors_knots[-1], grid.c_knots[-1], grid.alpha_knots[-1]],
        dtype=np.float64,
    )
    theta = jnp.asarray(
        rng.uniform(lows, highs, size=(batch_size, 3)), dtype=jnp.float64
    )
    exact = make_exact_power2_evaluator(
        np.asarray(grid.times), grid.geometry, kernel=grid.exact_kernel
    )

    def approximate(one_theta):
        return grid._interpolate_raw(
            one_theta[0], one_theta[1], one_theta[2]
        )[0]

    exact_forward = jax.jit(jax.vmap(exact))
    grid_forward = jax.jit(jax.vmap(approximate))
    # A fixed weighted scalar mimics the reverse-mode work in a likelihood
    # without timing the enormous full Jacobian used only for validation.
    weights = jnp.linspace(0.7, 1.3, grid.times.size, dtype=jnp.float64)
    exact_value_grad = jax.jit(
        jax.vmap(jax.value_and_grad(lambda x: jnp.vdot(exact(x), weights)))
    )
    grid_value_grad = jax.jit(
        jax.vmap(jax.value_and_grad(lambda x: jnp.vdot(approximate(x), weights)))
    )

    exact_forward_timing = _time_ready(exact_forward, theta, repeats)
    grid_forward_timing = _time_ready(grid_forward, theta, repeats)
    exact_gradient_timing = _time_ready(exact_value_grad, theta, repeats)
    grid_gradient_timing = _time_ready(grid_value_grad, theta, repeats)

    # A local likelihood probe is more informative for HMC than the maximum
    # error in one cadence Jacobian.  Give every parameter point its own exact
    # synthetic observation with the same deterministic 100 ppm residual
    # pattern, then differentiate while holding that observation fixed.
    exact_flux = exact_forward(theta)
    # Pseudorandom residuals avoid the exact odd/even cancellation that a sine
    # pattern has against a symmetric transit derivative.
    residual = jnp.asarray(
        rng.normal(size=(batch_size, grid.times.size)), dtype=jnp.float64
    )
    residual = residual / jnp.sqrt(jnp.mean(jnp.square(residual), axis=1))[:, None]
    sigma = jnp.asarray(100.0e-6, dtype=jnp.float64)
    observations = exact_flux + sigma * residual

    def exact_loglike(one_theta, observed):
        standardized = (observed - exact(one_theta)) / sigma
        return -0.5 * jnp.vdot(standardized, standardized)

    def grid_loglike(one_theta, observed):
        standardized = (observed - approximate(one_theta)) / sigma
        return -0.5 * jnp.vdot(standardized, standardized)

    exact_likelihood = jax.jit(
        jax.vmap(jax.value_and_grad(exact_loglike), in_axes=(0, 0))
    )
    grid_likelihood = jax.jit(
        jax.vmap(jax.value_and_grad(grid_loglike), in_axes=(0, 0))
    )
    exact_logp, exact_logp_grad = jax.block_until_ready(
        exact_likelihood(theta, observations)
    )
    grid_logp, grid_logp_grad = jax.block_until_ready(
        grid_likelihood(theta, observations)
    )
    logp_error = np.abs(np.asarray(grid_logp - exact_logp))
    gradient_error = np.asarray(grid_logp_grad - exact_logp_grad)
    exact_gradient = np.asarray(exact_logp_grad)
    lane_relative = np.linalg.norm(gradient_error, axis=1) / np.maximum(
        np.linalg.norm(exact_gradient, axis=1), 1.0e-30
    )
    domain_width = highs - lows
    scaled_error = gradient_error * domain_width[None, :]
    scaled_exact = exact_gradient * domain_width[None, :]
    lane_scaled_relative = np.linalg.norm(scaled_error, axis=1) / np.maximum(
        np.linalg.norm(scaled_exact, axis=1), 1.0e-30
    )

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator > 0.0 else None

    return {
        "batch_size": int(batch_size),
        "repeats": int(repeats),
        "exact_forward": exact_forward_timing,
        "grid_forward": grid_forward_timing,
        "forward_speedup": ratio(
            exact_forward_timing["median_seconds"],
            grid_forward_timing["median_seconds"],
        ),
        "exact_value_and_gradient": exact_gradient_timing,
        "grid_value_and_gradient": grid_gradient_timing,
        "value_and_gradient_speedup": ratio(
            exact_gradient_timing["median_seconds"],
            grid_gradient_timing["median_seconds"],
        ),
        "local_gaussian_likelihood_probe": {
            "sigma_ppm": 100.0,
            "max_abs_log_likelihood_error": float(np.max(logp_error)),
            "p99_abs_log_likelihood_error": float(
                np.percentile(logp_error, 99.0)
            ),
            "gradient_relative_l2": float(
                np.linalg.norm(gradient_error)
                / max(np.linalg.norm(exact_gradient), 1.0e-30)
            ),
            "max_lane_gradient_relative_l2": float(np.max(lane_relative)),
            "domain_scaled_gradient_relative_l2": float(
                np.linalg.norm(scaled_error)
                / max(np.linalg.norm(scaled_exact), 1.0e-30)
            ),
            "max_lane_domain_scaled_gradient_relative_l2": float(
                np.max(lane_scaled_relative)
            ),
        },
    }


def _load_times(args):
    if args.time_npy:
        return np.asarray(np.load(args.time_npy, allow_pickle=False), dtype=np.float64)
    return np.linspace(args.time_min, args.time_max, args.n_times, dtype=np.float64)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not bool(jax.config.x64_enabled):
        raise RuntimeError(
            "Run with JAX_ENABLE_X64=1; the experimental emulator refuses float32."
        )
    if min(args.rors_nodes, args.c_nodes, args.alpha_nodes) < 2:
        raise ValueError("Every grid axis needs at least two nodes.")
    times = _load_times(args)
    geometry = FixedPower2TransitGeometry(
        period=args.period,
        duration=args.duration,
        t0=args.t0,
        b=args.impact_parameter,
    )
    build_start = time.perf_counter()
    build_kwargs = dict(
        exact_kernel=args.exact_kernel,
        build_batch_size=args.build_batch_size,
        max_grid_bytes=int(args.max_grid_gib * 1024**3),
    )
    grid_arguments = (
        times,
        geometry,
        np.linspace(args.rors_min, args.rors_max, args.rors_nodes),
        np.linspace(args.c_min, args.c_max, args.c_nodes),
        np.linspace(args.alpha_min, args.alpha_max, args.alpha_nodes),
    )
    if args.radius_interpolation == "hermite":
        grid = build_power2_radius_hermite_grid(
            *grid_arguments,
            derivative_batch_size=args.derivative_batch_size,
            **build_kwargs,
        )
    elif args.radius_interpolation == "cubic":
        grid = build_power2_radius_cubic_grid(
            *grid_arguments,
            **build_kwargs,
        )
    else:
        grid = build_power2_transit_grid(*grid_arguments, **build_kwargs)
    build_seconds = time.perf_counter() - build_start

    validation_start = time.perf_counter()
    validation = validate_power2_grid(
        grid,
        samples_per_cell=args.validation_samples_per_cell,
        seed=args.validation_seed,
        flux_tolerance_ppm=args.flux_tolerance_ppm,
        gradient_abs_tolerance_ppm_per_unit=(
            args.gradient_abs_tolerance_ppm_per_unit
        ),
        gradient_relative_tolerance=args.gradient_relative_tolerance,
        gradient_points=args.gradient_points,
        validation_batch_size=args.validation_batch_size,
    )
    validation_seconds = time.perf_counter() - validation_start
    timing = benchmark_grid(
        grid,
        batch_size=args.benchmark_batch,
        repeats=args.benchmark_repeats,
        seed=args.validation_seed + 1,
    )
    report = {
        "experimental": True,
        "radius_interpolation": args.radius_interpolation,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "x64": bool(jax.config.x64_enabled),
        "shape": list(grid.flux_grid.shape),
        "grid_bytes": grid.estimated_bytes,
        "grid_gib": grid.estimated_bytes / 1024**3,
        "fingerprint": grid.fingerprint,
        "build_seconds": build_seconds,
        "validation_seconds": validation_seconds,
        "validation": validation.to_json_dict(),
        "timing": timing,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        Path(args.json_output).write_text(rendered + "\n", encoding="utf-8")
    if validation.passed and args.artifact and args.radius_interpolation == "linear":
        save_power2_grid(args.artifact, grid.with_validation(validation))
    elif validation.passed and args.artifact:
        print(
            "Smooth cubic/Hermite artifacts are evaluation-only and cannot be persisted.",
            file=sys.stderr,
        )
        return 2
    elif args.artifact and not validation.passed:
        print(
            "Validation failed; refusing to write the requested grid artifact.",
            file=sys.stderr,
        )
    return 0 if validation.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
