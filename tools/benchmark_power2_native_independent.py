#!/usr/bin/env python3
"""Independent accuracy and GPU timing audit for native Power-2 transits.

This tool is deliberately separate from the production fitter.  It compares
the experimental direct Power-2 contour integral with both a much higher
quadrature order and an adaptive radial-area integral, probes derivatives on
both sides of the analytic contacts, and times the complete fixed-geometry
spectroscopic path against the current degree-12 production implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import IntegrationWarning, quad

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.jaxoplanet.core import build_transit_phase_offsets
from models.jaxoplanet.experimental_power2_grid import (
    FixedPower2TransitGeometry,
    make_exact_power2_evaluator,
)
from models.jaxoplanet.experimental_power2_native import light_curve


RORS_BOUNDS = (float(np.sqrt(1.0e-5)), float(np.sqrt(0.5)))
C_BOUNDS = (0.0, 1.0)
ALPHA_BOUNDS = (0.001, 1.0)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=16)
    parser.add_argument("--reference-order", type=int, default=256)
    parser.add_argument("--convergence-order", type=int, default=128)
    parser.add_argument("--validation-batch", type=int, default=4096)
    parser.add_argument("--gradient-batch", type=int, default=1024)
    parser.add_argument("--independent-points", type=int, default=16)
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
    parser.add_argument("--only-contact-gradients", action="store_true")
    parser.add_argument("--json-output")
    return parser


def _ready(value):
    for leaf in jax.tree.leaves(value):
        leaf.block_until_ready()


def _fixed_batch_evaluate(function, values, batch_size):
    values = np.asarray(values, dtype=np.float64)
    effective = min(int(batch_size), values.shape[0])
    if effective <= 0:
        raise ValueError("batch_size and the number of values must be positive")
    compiled = jax.jit(jax.vmap(function))
    result = []
    for start in range(0, values.shape[0], effective):
        stop = min(start + effective, values.shape[0])
        chunk = values[start:stop]
        count = stop - start
        if count < effective:
            chunk = np.concatenate(
                [chunk, np.repeat(chunk[-1:], effective - count, axis=0)]
            )
        evaluated = compiled(jnp.asarray(chunk, dtype=jnp.float64))
        _ready(evaluated)
        result.append(np.asarray(evaluated)[:count])
    return np.concatenate(result, axis=0)


def _native_scalar(order):
    return lambda row: light_curve(
        row[0], row[1], row[2], row[3], order=order
    )


def _dense_validation_rows():
    r_min, r_max = RORS_BOUNDS
    radii = np.unique(
        np.concatenate(
            [
                np.geomspace(r_min, r_max, 17),
                np.linspace(r_min, r_max, 17),
            ]
        )
    )
    alphas = np.unique(
        np.concatenate(
            [
                np.geomspace(ALPHA_BOUNDS[0], ALPHA_BOUNDS[1], 13),
                np.linspace(ALPHA_BOUNDS[0], ALPHA_BOUNDS[1], 7),
            ]
        )
    )
    c_values = np.asarray([0.0, 0.1, 0.5, 0.9, 1.0])
    full_q = np.asarray(
        [0.0, 1e-6, 1e-4, 0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95,
         0.99, 0.9999, 0.999999, 1.0]
    )
    contact_eps = np.logspace(-12, -2, 6)
    partial_q = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, 17),
                np.sin(0.5 * np.pi * np.linspace(0.0, 1.0, 17)) ** 2,
                contact_eps,
                1.0 - contact_eps,
            ]
        )
    )
    rows = []
    regions = []
    q_values = []
    for radius in radii:
        full_z = (1.0 - radius) * full_q
        partial_z = 1.0 - radius + 2.0 * radius * partial_q
        for alpha in alphas:
            for c_value in c_values:
                rows.extend((c_value, alpha, z, radius) for z in full_z)
                regions.extend([0] * full_z.size)
                q_values.extend(full_q)
                rows.extend((c_value, alpha, z, radius) for z in partial_z)
                regions.extend([1] * partial_z.size)
                q_values.extend(partial_q)
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(regions, dtype=np.int8),
        np.asarray(q_values, dtype=np.float64),
    )


def _error_summary(error, rows, regions, q_values):
    absolute = np.abs(error)
    worst_index = int(np.argmax(absolute))
    worst = rows[worst_index]
    return {
        "max_abs_error_ppm": float(absolute[worst_index] * 1.0e6),
        "p99_abs_error_ppm": float(np.percentile(absolute, 99.0) * 1.0e6),
        "rms_error_ppm": float(np.sqrt(np.mean(np.square(error))) * 1.0e6),
        "worst": {
            "c": float(worst[0]),
            "alpha": float(worst[1]),
            "separation": float(worst[2]),
            "rors": float(worst[3]),
            "region": "partial" if regions[worst_index] else "full",
            "q": float(q_values[worst_index]),
        },
    }


def _radial_moment_reference(separation, radius_ratio, alpha):
    """Independent adaptive annulus integral of blocked ``mu**alpha``."""
    separation = abs(float(separation))
    radius_ratio = abs(float(radius_ratio))
    alpha = float(alpha)
    if separation == 0.0:
        upper = min(radius_ratio, 1.0)
        return (
            2.0
            * np.pi
            / (alpha + 2.0)
            * (1.0 - (1.0 - upper**2) ** (1.0 + 0.5 * alpha))
        )

    def integrand(stellar_radius):
        if radius_ratio >= separation + stellar_radius:
            occulted_angle = 2.0 * np.pi
        elif radius_ratio <= abs(separation - stellar_radius):
            occulted_angle = 0.0
        else:
            cosine = (
                stellar_radius**2 + separation**2 - radius_ratio**2
            ) / (2.0 * stellar_radius * separation)
            occulted_angle = 2.0 * np.arccos(np.clip(cosine, -1.0, 1.0))
        radial_intensity = (
            1.0
            if alpha == 0.0
            else max(0.0, 1.0 - stellar_radius**2) ** (0.5 * alpha)
        )
        return stellar_radius * radial_intensity * occulted_angle

    breakpoints = sorted(
        {
            0.0,
            1.0,
            float(np.clip(abs(separation - radius_ratio), 0.0, 1.0)),
            float(np.clip(separation + radius_ratio, 0.0, 1.0)),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        return sum(
            quad(
                integrand,
                low,
                high,
                epsabs=2.0e-13,
                epsrel=2.0e-13,
                limit=300,
            )[0]
            for low, high in zip(breakpoints[:-1], breakpoints[1:])
            if high > low
        )


def _radial_flux_reference(row):
    c_value, alpha, separation, radius = map(float, row)
    area = _radial_moment_reference(separation, radius, 0.0)
    power = _radial_moment_reference(separation, radius, alpha)
    blocked = (1.0 - c_value) * area + c_value * power
    total = np.pi * (1.0 - c_value) + c_value * 2.0 * np.pi / (alpha + 2.0)
    return -blocked / total


def dense_flux_validation(args):
    rows, regions, q_values = _dense_validation_rows()
    candidate = _fixed_batch_evaluate(
        _native_scalar(args.order), rows, args.validation_batch
    )
    convergence = _fixed_batch_evaluate(
        _native_scalar(args.convergence_order), rows, args.validation_batch
    )
    reference = _fixed_batch_evaluate(
        _native_scalar(args.reference_order), rows, args.validation_batch
    )

    candidate_error = candidate - reference
    convergence_error = convergence - reference
    ranking = np.argsort(np.abs(candidate_error))[::-1]
    selected = []
    seen = set()
    for index in ranking:
        # Avoid spending adaptive-integral time on numerically identical rows.
        key = tuple(np.round(rows[index], decimals=14))
        if key in seen:
            continue
        seen.add(key)
        selected.append(int(index))
        if len(selected) >= args.independent_points:
            break
    independent = []
    independent_errors = []
    reference_errors = []
    for index in selected:
        radial = _radial_flux_reference(rows[index])
        candidate_delta = float(candidate[index] - radial)
        reference_delta = float(reference[index] - radial)
        independent_errors.append(candidate_delta)
        reference_errors.append(reference_delta)
        independent.append(
            {
                "row": {
                    "c": float(rows[index, 0]),
                    "alpha": float(rows[index, 1]),
                    "separation": float(rows[index, 2]),
                    "rors": float(rows[index, 3]),
                    "region": "partial" if regions[index] else "full",
                    "q": float(q_values[index]),
                },
                "order_value": float(candidate[index]),
                "high_order_value": float(reference[index]),
                "radial_value": float(radial),
                "order_vs_radial_ppm": candidate_delta * 1.0e6,
                "high_order_vs_radial_ppm": reference_delta * 1.0e6,
            }
        )
    return {
        "n_points": int(rows.shape[0]),
        "order": args.order,
        "reference_order": args.reference_order,
        "convergence_order": args.convergence_order,
        "order_vs_reference": _error_summary(
            candidate_error, rows, regions, q_values
        ),
        "convergence_vs_reference": _error_summary(
            convergence_error, rows, regions, q_values
        ),
        "independent_radial": {
            "n_points": len(independent),
            "order_max_abs_error_ppm": float(
                np.max(np.abs(independent_errors)) * 1.0e6
            ),
            "high_order_max_abs_error_ppm": float(
                np.max(np.abs(reference_errors)) * 1.0e6
            ),
            "points": independent,
        },
        "passes_two_ppm": bool(
            np.max(np.abs(candidate_error)) * 1.0e6 <= 2.0
            and np.max(np.abs(independent_errors)) * 1.0e6 <= 2.0
        ),
    }


def _contact_gradient_rows():
    r_min, r_max = RORS_BOUNDS
    radii = np.unique(
        np.concatenate(
            [np.geomspace(r_min, r_max, 13), np.linspace(r_min, r_max, 13)]
        )
    )
    alphas = np.asarray([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    c_values = np.asarray([0.1, 0.5, 1.0])
    epsilons = np.concatenate(
        [-np.logspace(-1, -12, 12), [0.0], np.logspace(-12, -1, 12)]
    )
    rows = []
    metadata = []
    for radius in radii:
        scale = 2.0 * radius
        for alpha in alphas:
            for c_value in c_values:
                for contact, location in (
                    ("inner", 1.0 - radius), ("outer", 1.0 + radius)
                ):
                    for epsilon in epsilons:
                        rows.append(
                            (c_value, alpha, location + scale * epsilon, radius)
                        )
                        metadata.append((contact, epsilon))
    return np.asarray(rows), metadata


def _gradient_summary(error, candidate, reference, rows, metadata, mask=None):
    if mask is None:
        mask = np.ones(rows.shape[0], dtype=bool)
    indices = np.flatnonzero(mask)
    selected_error = error[indices]
    selected_candidate = candidate[indices]
    selected_reference = reference[indices]
    parameter_names = ("c", "alpha", "separation", "rors")
    absolute = np.abs(selected_error)
    relative_l2 = np.linalg.norm(selected_error, axis=0) / np.maximum(
        np.linalg.norm(selected_reference, axis=0), 1.0e-300
    )
    worst = []
    for parameter_index, name in enumerate(parameter_names):
        local = int(np.argmax(absolute[:, parameter_index]))
        global_index = int(indices[local])
        row = rows[global_index]
        contact, epsilon = metadata[global_index]
        worst.append(
            {
                "parameter": name,
                "abs_error_per_unit": float(absolute[local, parameter_index]),
                "candidate": float(selected_candidate[local, parameter_index]),
                "reference": float(selected_reference[local, parameter_index]),
                "c": float(row[0]),
                "alpha": float(row[1]),
                "separation": float(row[2]),
                "rors": float(row[3]),
                "contact": contact,
                "scaled_offset": float(epsilon),
            }
        )
    return {
        "n_points": int(indices.size),
        "max_abs_error_per_unit": np.max(absolute, axis=0).tolist(),
        "p99_abs_error_per_unit": np.percentile(absolute, 99.0, axis=0).tolist(),
        "relative_l2": relative_l2.tolist(),
        "max_abs_candidate_per_unit": np.max(
            np.abs(selected_candidate), axis=0
        ).tolist(),
        "max_abs_reference_per_unit": np.max(
            np.abs(selected_reference), axis=0
        ).tolist(),
        "worst": worst,
    }


def contact_gradient_validation(args):
    rows, metadata = _contact_gradient_rows()

    def value_gradient(order):
        scalar = _native_scalar(order)

        def packed(row):
            value, gradient = jax.value_and_grad(scalar)(row)
            return jnp.concatenate([value[None], gradient])

        result = _fixed_batch_evaluate(packed, rows, args.gradient_batch)
        return result[:, 0], result[:, 1:]

    candidate_value, candidate_gradient = value_gradient(args.order)
    convergence_value, convergence_gradient = value_gradient(
        args.convergence_order
    )
    reference_value, reference_gradient = value_gradient(args.reference_order)
    gradient_error = candidate_gradient - reference_gradient
    convergence_gradient_error = convergence_gradient - reference_gradient
    exact_mask = np.asarray([epsilon == 0.0 for _, epsilon in metadata])
    near_mask = np.asarray(
        [0.0 < abs(epsilon) <= 1.0e-6 for _, epsilon in metadata]
    )
    convergence_report = {
        "max_abs_flux_error_ppm": float(
            np.max(np.abs(convergence_value - reference_value)) * 1.0e6
        ),
        "all_offsets": _gradient_summary(
            convergence_gradient_error,
            convergence_gradient,
            reference_gradient,
            rows,
            metadata,
        ),
        "within_scaled_1e-6_of_contact": _gradient_summary(
            convergence_gradient_error,
            convergence_gradient,
            reference_gradient,
            rows,
            metadata,
            near_mask,
        ),
        "exact_contacts": _gradient_summary(
            convergence_gradient_error,
            convergence_gradient,
            reference_gradient,
            rows,
            metadata,
            exact_mask,
        ),
    }
    return {
        "n_points": int(rows.shape[0]),
        "all_values_finite": bool(
            np.all(np.isfinite(candidate_value))
            and np.all(np.isfinite(reference_value))
        ),
        "all_gradients_finite": bool(
            np.all(np.isfinite(candidate_gradient))
            and np.all(np.isfinite(convergence_gradient))
            and np.all(np.isfinite(reference_gradient))
        ),
        "max_abs_flux_error_ppm": float(
            np.max(np.abs(candidate_value - reference_value)) * 1.0e6
        ),
        "parameter_order": ["c", "alpha", "separation", "rors"],
        "all_offsets": _gradient_summary(
            gradient_error,
            candidate_gradient,
            reference_gradient,
            rows,
            metadata,
        ),
        "within_scaled_1e-6_of_contact": _gradient_summary(
            gradient_error,
            candidate_gradient,
            reference_gradient,
            rows,
            metadata,
            near_mask,
        ),
        "exact_contacts": _gradient_summary(
            gradient_error,
            candidate_gradient,
            reference_gradient,
            rows,
            metadata,
            exact_mask,
        ),
        "convergence_order_vs_reference": convergence_report,
    }


def _load_actual_times(args):
    with Path(args.time_pickle).open("rb") as stream:
        payload = pickle.load(stream)
    all_times = np.asarray(getattr(payload, args.time_attribute), dtype=np.float64)
    if all_times.ndim != 1 or not np.all(np.isfinite(all_times)):
        raise ValueError("Cadence times must be a finite one-dimensional array")
    if not np.all(np.diff(all_times) > 0.0):
        raise ValueError("Cadence times must be strictly increasing")
    if args.n_times > all_times.size:
        raise ValueError("n_times exceeds the number of actual cadences")
    indices = np.linspace(0, all_times.size - 1, args.n_times, dtype=np.int64)
    if np.unique(indices).size != args.n_times:
        raise ValueError("Subsampling did not produce the requested unique cadence count")
    return all_times[indices], int(all_times.size)


def _profile(function, argument, repeats):
    start = time.perf_counter()
    compiled = jax.jit(function).lower(argument).compile()
    compile_seconds = time.perf_counter() - start
    _ready(compiled(argument))
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = compiled(argument)
        _ready(result)
        timings.append(time.perf_counter() - start)
    return {
        "compile_seconds": float(compile_seconds),
        "median_ms": float(np.median(timings) * 1.0e3),
        "p05_ms": float(np.percentile(timings, 5.0) * 1.0e3),
        "p95_ms": float(np.percentile(timings, 95.0) * 1.0e3),
        "min_ms": float(np.min(timings) * 1.0e3),
    }


def timing_benchmark(args, geometry):
    times, source_size = _load_actual_times(args)
    offsets, masks = build_transit_phase_offsets(
        jnp.asarray(times),
        jnp.asarray([geometry.period]),
        jnp.asarray([geometry.t0]),
        jnp.asarray([geometry.duration]),
    )
    dt = offsets[0]
    mask = masks[0]
    stock_one = make_exact_power2_evaluator(times, geometry, kernel="stock")

    def stock_forward(theta):
        return jax.vmap(stock_one)(theta)

    def native_forward(theta, order=args.order):
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
            order=order,
        )
        return jnp.where(mask[None, :], signal, 0.0)

    rng = np.random.default_rng(args.seed)
    max_width = max(args.widths)
    low = np.asarray([RORS_BOUNDS[0], C_BOUNDS[0], ALPHA_BOUNDS[0]])
    high = np.asarray([RORS_BOUNDS[1], C_BOUNDS[1], ALPHA_BOUNDS[1]])
    theta_all = rng.uniform(low, high, size=(max_width, 3))
    result = {
        "source_cadences": source_size,
        "subsampled_cadences": int(times.size),
        "first_time": float(times[0]),
        "last_time": float(times[-1]),
        "widths": {},
    }
    for width in args.widths:
        theta = jnp.asarray(theta_all[:width], dtype=jnp.float64)
        weights = jnp.linspace(0.7, 1.3, times.size, dtype=jnp.float64)
        lane_weights = jnp.linspace(0.8, 1.2, width, dtype=jnp.float64)

        def objective(forward):
            return lambda values: jnp.sum(
                forward(values) * lane_weights[:, None] * weights[None, :]
            )

        stock_flux = jax.jit(stock_forward)(theta)
        native_flux = jax.jit(native_forward)(theta)
        high_order_flux = jax.jit(
            lambda values: native_forward(values, args.reference_order)
        )(theta)
        _ready((stock_flux, native_flux, high_order_flux))
        stock_forward_profile = _profile(stock_forward, theta, args.repeats)
        native_forward_profile = _profile(native_forward, theta, args.repeats)
        stock_vg_profile = _profile(
            jax.value_and_grad(objective(stock_forward)), theta, args.repeats
        )
        native_vg_profile = _profile(
            jax.value_and_grad(objective(native_forward)), theta, args.repeats
        )
        result["widths"][str(width)] = {
            "stock_forward": stock_forward_profile,
            "native_forward": native_forward_profile,
            "forward_speedup": float(
                stock_forward_profile["median_ms"]
                / native_forward_profile["median_ms"]
            ),
            "stock_value_gradient": stock_vg_profile,
            "native_value_gradient": native_vg_profile,
            "value_gradient_speedup": float(
                stock_vg_profile["median_ms"] / native_vg_profile["median_ms"]
            ),
            "native_order_vs_high_order_max_flux_error_ppm": float(
                np.max(np.abs(np.asarray(native_flux - high_order_flux))) * 1.0e6
            ),
            "native_direct_vs_stock_degree12_max_difference_ppm": float(
                np.max(np.abs(np.asarray(native_flux - stock_flux))) * 1.0e6
            ),
        }
    return result


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not jax.config.x64_enabled:
        raise RuntimeError("Run with JAX_ENABLE_X64=1")
    if not (2 <= args.order < args.convergence_order < args.reference_order):
        raise ValueError("Require 2 <= order < convergence-order < reference-order")
    geometry = FixedPower2TransitGeometry(
        args.period, args.duration, args.t0, args.impact_parameter
    )

    if args.only_contact_gradients:
        start = time.perf_counter()
        gradients = contact_gradient_validation(args)
        report = {
            "experimental": True,
            "production_wired": False,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "orders": {
                "candidate": args.order,
                "convergence": args.convergence_order,
                "reference": args.reference_order,
            },
            "contact_gradient_validation_seconds": time.perf_counter() - start,
            "contact_gradient_validation": gradients,
        }
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered)
        if args.json_output:
            Path(args.json_output).write_text(rendered + "\n")
        return 0

    start = time.perf_counter()
    dense = dense_flux_validation(args)
    dense_seconds = time.perf_counter() - start
    start = time.perf_counter()
    gradients = contact_gradient_validation(args)
    gradient_seconds = time.perf_counter() - start
    timings = timing_benchmark(args, geometry)
    report = {
        "experimental": True,
        "production_wired": False,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "orders": {
            "candidate": args.order,
            "convergence": args.convergence_order,
            "reference": args.reference_order,
        },
        "prior_domain": {
            "rors": list(RORS_BOUNDS),
            "c": list(C_BOUNDS),
            "alpha": list(ALPHA_BOUNDS),
        },
        "geometry": {
            "period": geometry.period,
            "duration": geometry.duration,
            "t0": geometry.t0,
            "b": geometry.b,
        },
        "dense_flux_validation_seconds": dense_seconds,
        "dense_flux_validation": dense,
        "contact_gradient_validation_seconds": gradient_seconds,
        "contact_gradient_validation": gradients,
        "timing_benchmark": timings,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        Path(args.json_output).write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
