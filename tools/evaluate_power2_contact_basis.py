#!/usr/bin/env python3
"""Dense adversarial evaluation of the contact-aligned Power-2 basis grid."""

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
from jaxoplanet.core.limb_dark import solution_vector

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.jaxoplanet.experimental_power2_contact_basis import (
    POWER2_ALPHA_PRIOR_BOUNDS,
    POWER2_C_PRIOR_BOUNDS,
    RORS_PRIOR_BOUNDS,
    build_power2_contact_basis_grid,
)
from models.jaxoplanet.experimental_power2_grid import (
    FixedPower2TransitGeometry,
    _evaluate_in_fixed_batches,
    make_exact_power2_evaluator,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rors", type=int, default=513)
    parser.add_argument("--n-full", type=int, default=257)
    parser.add_argument("--n-partial", type=int, default=257)
    parser.add_argument("--build-batch", type=int, default=2048)
    parser.add_argument("--validation-batch", type=int, default=2048)
    parser.add_argument("--random-points", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--period", type=float, default=2.60544751)
    parser.add_argument("--duration", type=float, default=0.17688375)
    parser.add_argument("--t0", type=float, default=60473.32)
    parser.add_argument("--impact-parameter", type=float, default=0.464)
    parser.add_argument("--time-pickle")
    parser.add_argument("--time-attribute", default="time")
    parser.add_argument("--benchmark-batch", type=int, default=64)
    parser.add_argument("--benchmark-repeats", type=int, default=20)
    parser.add_argument("--json-output")
    return parser


def _green_cases(grid):
    c_values = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    alpha_values = np.array([0.001, 0.01, 0.05, 0.2, 0.5, 1.0])
    cases = np.asarray(
        [(c_value, alpha) for c_value in c_values for alpha in alpha_values]
    )
    green = jax.jit(
        jax.vmap(lambda pair: grid._power2_green(pair[0], pair[1]))
    )(jnp.asarray(cases))
    return cases, np.asarray(green)


def _validation_pairs(grid, region, random_points, seed):
    r_axis = np.asarray(grid.rors_knots)
    q_axis = np.asarray(
        grid.full_q_knots if region == "full" else grid.partial_q_knots
    )
    r_mid = 0.5 * (r_axis[:-1] + r_axis[1:])
    q_mid = 0.5 * (q_axis[:-1] + q_axis[1:])
    rr, qq = np.meshgrid(r_mid, q_mid, indexing="ij")
    cell_centers = np.column_stack([rr.ravel(), qq.ravel()])

    rng = np.random.default_rng(seed)
    r_index = rng.integers(0, r_axis.size - 1, size=random_points)
    q_index = rng.integers(0, q_axis.size - 1, size=random_points)
    random = np.column_stack(
        [
            rng.uniform(r_axis[r_index], r_axis[r_index + 1]),
            rng.uniform(q_axis[q_index], q_axis[q_index + 1]),
        ]
    )
    # Adversarial offsets explicitly approach both overlap boundaries.  They
    # complement the heavily clustered q-cell centers.
    selected_r = r_mid[
        np.linspace(0, r_mid.size - 1, min(r_mid.size, 256), dtype=int)
    ]
    eps = np.logspace(-12, -2, 18)
    q_contact = np.unique(np.concatenate([eps, 1.0 - eps]))
    cr, cq = np.meshgrid(selected_r, q_contact, indexing="ij")
    contacts = np.column_stack([cr.ravel(), cq.ravel()])
    return np.concatenate([cell_centers, random, contacts], axis=0)


def dense_basis_validation(grid, *, random_points, seed, batch_size):
    solution = solution_vector(12, order=10)
    ld_cases, green = _green_cases(grid)
    result = {}
    global_worst = None
    all_errors = []
    for region_index, region in enumerate(("full", "partial")):
        pairs = _validation_pairs(
            grid, region, random_points, seed + region_index
        )

        def exact(pair):
            rors, q = pair[0], pair[1]
            separation = (
                q * (1.0 - rors)
                if region == "full"
                else 1.0 - rors + 2.0 * rors * q
            )
            return solution(separation, rors)

        axis = grid.full_q_knots if region == "full" else grid.partial_q_knots
        table = (
            grid.full_solution_grid
            if region == "full"
            else grid.partial_solution_grid
        )

        def approximate(pair):
            return grid._interpolate_solution(table, axis, pair[0], pair[1])

        exact_solution = _evaluate_in_fixed_batches(exact, pairs, batch_size)
        approximate_solution = _evaluate_in_fixed_batches(
            approximate, pairs, batch_size
        )
        # Projection over 30 edge/interior LD cases is exact; all error comes
        # from the geometric basis interpolation.
        flux_error_ppm = np.abs(
            (approximate_solution - exact_solution) @ green.T
        ) * 1.0e6
        all_errors.append(flux_error_ppm.ravel())
        flat = int(np.argmax(flux_error_ppm))
        point_index, ld_index = np.unravel_index(flat, flux_error_ppm.shape)
        worst = {
            "region": region,
            "rors": float(pairs[point_index, 0]),
            "q": float(pairs[point_index, 1]),
            "c": float(ld_cases[ld_index, 0]),
            "alpha": float(ld_cases[ld_index, 1]),
            "error_ppm": float(flux_error_ppm[point_index, ld_index]),
        }
        if global_worst is None or worst["error_ppm"] > global_worst["error_ppm"]:
            global_worst = worst
        result[region] = {
            "n_points": int(pairs.shape[0]),
            "max_abs_flux_error_ppm": worst["error_ppm"],
            "worst": worst,
        }
    combined = np.concatenate(all_errors)
    result.update(
        {
            "n_ld_cases": int(green.shape[0]),
            "max_abs_flux_error_ppm": float(np.max(combined)),
            "p99_abs_flux_error_ppm": float(np.percentile(combined, 99.0)),
            "rms_flux_error_ppm": float(np.sqrt(np.mean(np.square(combined)))),
            "worst": global_worst,
        }
    )
    return result


def contact_gradient_validation(grid, *, batch_size=16):
    r_min, r_max = map(float, RORS_PRIOR_BOUNDS)
    grazing_boundary = 1.0 - grid.geometry.b
    regular = np.linspace(r_min, r_max, 48)
    around_grazing = grazing_boundary + np.array(
        [-1e-3, -1e-4, -1e-6, 1e-6, 1e-4, 1e-3]
    )
    radii = np.unique(np.clip(np.concatenate([regular, around_grazing]), r_min, r_max))
    ld_cases = np.array(
        [[0.0, 0.001], [1.0, 0.001], [1.0, 1.0], [0.5, 0.5]]
    )
    theta = np.asarray(
        [[radius, c_value, alpha] for radius in radii for c_value, alpha in ld_cases]
    )
    offsets = np.array([-100, -10, -1, -0.1, 0.1, 1, 10, 100]) / 86400.0
    lane_times = []
    for radius, _, _ in theta:
        outer = 0.5 * grid.geometry.duration
        if grid.geometry.b < 1.0 - radius:
            inner = outer * np.sqrt(
                ((1.0 - radius) ** 2 - grid.geometry.b**2)
                / ((1.0 + radius) ** 2 - grid.geometry.b**2)
            )
        else:
            inner = 0.0
        positive = np.concatenate([inner + offsets, outer + offsets])
        lane_times.append(np.concatenate([-positive[::-1], positive]))
    lane_times = np.asarray(lane_times)
    exact_solution = solution_vector(12, order=10)

    def exact(one_theta, dt):
        radius, c_value, alpha = one_theta
        speed = 2.0 * jnp.sqrt(
            jnp.maximum(0.0, (1.0 + radius) ** 2 - grid.geometry.b**2)
        ) / grid.geometry.duration
        separation = jnp.sqrt((speed * dt) ** 2 + grid.geometry.b**2)
        vectors = jax.vmap(lambda z: exact_solution(z, radius))(separation)
        green = grid._power2_green(c_value, alpha)
        signal = vectors @ green - 1.0
        return jnp.where(jnp.abs(dt) < 0.5 * grid.geometry.duration, signal, 0.0)

    def approximate(one_theta, dt):
        radius, c_value, alpha = one_theta
        speed = 2.0 * jnp.sqrt(
            jnp.maximum(0.0, (1.0 + radius) ** 2 - grid.geometry.b**2)
        ) / grid.geometry.duration
        separation = jnp.sqrt((speed * dt) ** 2 + grid.geometry.b**2)
        signal, _ = grid.signal_from_separation(
            radius, c_value, alpha, separation
        )
        return jnp.where(jnp.abs(dt) < 0.5 * grid.geometry.duration, signal, 0.0)

    exact_flux = _evaluate_two_argument_batches(exact, theta, lane_times, batch_size)
    approx_flux = _evaluate_two_argument_batches(
        approximate, theta, lane_times, batch_size
    )
    exact_grad = _evaluate_two_argument_batches(
        jax.jacrev(exact, argnums=0), theta, lane_times, batch_size
    )
    approx_grad = _evaluate_two_argument_batches(
        jax.jacrev(approximate, argnums=0), theta, lane_times, batch_size
    )
    gradient_error = approx_grad - exact_grad
    relative = np.linalg.norm(gradient_error, axis=(0, 1)) / np.maximum(
        np.linalg.norm(exact_grad, axis=(0, 1)), 1e-30
    )
    return {
        "n_lanes": int(theta.shape[0]),
        "times_per_lane": int(lane_times.shape[1]),
        "max_abs_flux_error_ppm": float(
            np.max(np.abs(approx_flux - exact_flux)) * 1e6
        ),
        "gradient_max_abs_ppm_per_unit": (
            np.max(np.abs(gradient_error), axis=(0, 1)) * 1e6
        ).tolist(),
        "gradient_relative_l2": relative.tolist(),
        "all_finite": bool(
            np.all(np.isfinite(exact_grad)) and np.all(np.isfinite(approx_grad))
        ),
    }


def _evaluate_two_argument_batches(function, first, second, batch_size):
    size = first.shape[0]
    effective = min(int(batch_size), size)
    batched = jax.jit(jax.vmap(function, in_axes=(0, 0)))
    output = []
    for start in range(0, size, effective):
        stop = min(start + effective, size)
        left = first[start:stop]
        right = second[start:stop]
        count = stop - start
        if count < effective:
            left = np.concatenate([left, np.repeat(left[-1:], effective - count, axis=0)])
            right = np.concatenate([right, np.repeat(right[-1:], effective - count, axis=0)])
        value = jax.block_until_ready(
            batched(jnp.asarray(left), jnp.asarray(right))
        )
        output.append(np.asarray(value)[:count])
    return np.concatenate(output)


def _load_times(args):
    if not args.time_pickle:
        return np.linspace(
            args.t0 - 0.7 * args.duration,
            args.t0 + 0.7 * args.duration,
            4096,
        )
    with Path(args.time_pickle).open("rb") as stream:
        payload = pickle.load(stream)
    times = np.asarray(getattr(payload, args.time_attribute), dtype=np.float64)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("Loaded cadence data must be a finite one-dimensional array.")
    return times


def _timing(function, argument, repeats):
    start = time.perf_counter()
    jax.block_until_ready(function(argument))
    compile_time = time.perf_counter() - start
    measured = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(function(argument))
        measured.append(time.perf_counter() - start)
    return {
        "compile_and_first_seconds": compile_time,
        "median_seconds": float(np.median(measured)),
        "min_seconds": float(np.min(measured)),
    }


def actual_cadence_benchmark(grid, times, *, batch_size, repeats, seed):
    rng = np.random.default_rng(seed)
    lower = np.array([RORS_PRIOR_BOUNDS[0], 0.0, 0.001])
    upper = np.array([RORS_PRIOR_BOUNDS[1], 1.0, 1.0])
    theta = jnp.asarray(rng.uniform(lower, upper, size=(batch_size, 3)))
    exact_one = make_exact_power2_evaluator(times, grid.geometry, kernel="stock")
    exact_forward = jax.jit(jax.vmap(exact_one))
    approximate_forward = jax.jit(
        jax.vmap(lambda one_theta: grid.evaluate_with_status(one_theta, times)[0])
    )
    exact_flux = jax.block_until_ready(exact_forward(theta))
    approximate_flux = jax.block_until_ready(approximate_forward(theta))
    weights = jnp.linspace(0.7, 1.3, times.size, dtype=jnp.float64)
    exact_grad = jax.jit(
        jax.vmap(jax.value_and_grad(lambda x: jnp.vdot(exact_one(x), weights)))
    )
    approximate_grad = jax.jit(
        jax.vmap(
            jax.value_and_grad(
                lambda x: jnp.vdot(grid.evaluate_with_status(x, times)[0], weights)
            )
        )
    )
    exact_forward_timing = _timing(exact_forward, theta, repeats)
    approximate_forward_timing = _timing(approximate_forward, theta, repeats)
    exact_grad_timing = _timing(exact_grad, theta, repeats)
    approximate_grad_timing = _timing(approximate_grad, theta, repeats)
    return {
        "n_times": int(times.size),
        "batch_size": int(batch_size),
        "actual_cadence_max_flux_error_ppm": float(
            np.max(np.abs(np.asarray(approximate_flux - exact_flux))) * 1e6
        ),
        "exact_forward": exact_forward_timing,
        "contact_basis_forward": approximate_forward_timing,
        "forward_speedup": float(
            exact_forward_timing["median_seconds"]
            / approximate_forward_timing["median_seconds"]
        ),
        "exact_value_gradient": exact_grad_timing,
        "contact_basis_value_gradient": approximate_grad_timing,
        "value_gradient_speedup": float(
            exact_grad_timing["median_seconds"]
            / approximate_grad_timing["median_seconds"]
        ),
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not bool(jax.config.x64_enabled):
        raise RuntimeError("Run with JAX_ENABLE_X64=1.")
    geometry = FixedPower2TransitGeometry(
        args.period, args.duration, args.t0, args.impact_parameter
    )
    start = time.perf_counter()
    grid = build_power2_contact_basis_grid(
        geometry,
        n_rors=args.n_rors,
        n_full=args.n_full,
        n_partial=args.n_partial,
        build_batch_size=args.build_batch,
    )
    build_seconds = time.perf_counter() - start
    start = time.perf_counter()
    dense = dense_basis_validation(
        grid,
        random_points=args.random_points,
        seed=args.seed,
        batch_size=args.validation_batch,
    )
    dense_seconds = time.perf_counter() - start
    contact_gradients = contact_gradient_validation(grid)
    times = _load_times(args)
    cadence = actual_cadence_benchmark(
        grid,
        times,
        batch_size=args.benchmark_batch,
        repeats=args.benchmark_repeats,
        seed=args.seed + 101,
    )
    report = {
        "experimental": True,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "geometry": {
            "period": geometry.period,
            "duration": geometry.duration,
            "t0": geometry.t0,
            "b": geometry.b,
        },
        "prior_domain": {
            "rors": list(RORS_PRIOR_BOUNDS),
            "c": list(POWER2_C_PRIOR_BOUNDS),
            "alpha": list(POWER2_ALPHA_PRIOR_BOUNDS),
        },
        "shape": {
            "full": list(grid.full_solution_grid.shape),
            "partial": list(grid.partial_solution_grid.shape),
        },
        "grid_bytes": grid.estimated_bytes,
        "grid_mib": grid.estimated_bytes / 1024**2,
        "build_seconds": build_seconds,
        "dense_validation_seconds": dense_seconds,
        "dense_basis_validation": dense,
        "contact_gradient_validation": contact_gradients,
        "actual_cadence_benchmark": cadence,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        Path(args.json_output).write_text(rendered + "\n")
    return 0 if dense["max_abs_flux_error_ppm"] <= 2.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
