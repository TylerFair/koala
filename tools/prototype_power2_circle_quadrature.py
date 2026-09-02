#!/usr/bin/env python3
"""Prototype direct power-2 transit quadrature for a circular planet.

This is an investigation tool, not a production fitting kernel.  It evaluates
the occulted ``mu**alpha`` moment with Green's theorem: stellar-boundary terms
are analytic and only the planet-boundary arc uses a small, fixed
Gauss--Legendre rule.  The implementation is pure JAX and differentiable away
from the physical contact surfaces.

The command-line validation compares fluxes with an independent adaptive
radial integral, including points clustered around first through fourth
contact.  Passing this exploratory check is not a production accuracy
certificate; see the audit report for the required validation envelope.
"""

from __future__ import annotations

import argparse
import json
import math
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.special import roots_legendre


def _zeta_power(mu2, alpha):
    """Stable Green primitive for the radial moment ``mu**alpha``."""
    mu2 = jnp.clip(mu2, 0.0, 1.0)
    delta = 1.0 - mu2
    exponent = 0.5 * (alpha + 2.0)
    safe_delta = jnp.where(delta > 1.0e-7, delta, 1.0)
    direct = -jnp.expm1(exponent * jnp.log(jnp.maximum(mu2, 1.0e-300)))
    direct = direct / ((alpha + 2.0) * safe_delta)
    # (1-(1-delta)**exponent)/((alpha+2)*delta), through delta**3.
    series = (
        0.5
        - 0.25 * (exponent - 1.0) * delta
        + (exponent - 1.0) * (exponent - 2.0) * delta**2 / 12.0
        - (exponent - 1.0) * (exponent - 2.0)
        * (exponent - 3.0) * delta**3 / 48.0
    )
    return jnp.where(delta > 1.0e-7, direct, series)


def _zeta_power_minus_limb(mu2, alpha):
    """Stable ``zeta(mu, alpha) - zeta(0, alpha)`` evaluation."""
    mu2 = jnp.clip(mu2, 0.0, 1.0)
    delta = 1.0 - mu2
    beta = 0.5 * alpha
    safe_delta = jnp.where(delta > 1.0e-7, delta, 1.0)
    direct = mu2 * (
        -jnp.expm1(beta * jnp.log(jnp.maximum(mu2, 1.0e-300)))
    )
    direct = direct / ((alpha + 2.0) * safe_delta)
    series = (
        beta
        - 0.5 * beta * (beta + 1.0) * delta
        + beta * (beta - 1.0) * (beta + 1.0) * delta**2 / 6.0
    ) / (alpha + 2.0)
    return jnp.where(delta > 1.0e-7, direct, series)


@jax.custom_jvp
def _zero_safe_sqrt(value):
    return jnp.sqrt(value)


@_zero_safe_sqrt.defjvp
def _zero_safe_sqrt_jvp(primals, tangents):
    (value,) = primals
    (value_dot,) = tangents
    result = jnp.sqrt(value)
    tiny = value < 10.0 * jnp.finfo(value.dtype).eps
    denominator = jnp.where(tiny, 1.0, value)
    return result, 0.5 * value_dot * result / denominator


def _kite_area(a, b, c):
    a, b = jnp.minimum(a, b), jnp.maximum(a, b)
    b, c = jnp.minimum(b, c), jnp.maximum(b, c)
    a, b = jnp.minimum(a, b), jnp.maximum(a, b)
    square_area = (
        (a + b + c)
        * (c - a + b)
        * (c + a - b)
        * (a + b - c)
    )
    return _zero_safe_sqrt(jnp.maximum(0.0, square_area))


def _contact_geometry(z, p):
    """Stable kite length and planet/star arc angles for p < 1."""
    partial = (z > jnp.abs(1.0 - p)) & (z < 1.0 + p)
    safe_z = jnp.where(partial, z, 1.0)
    kite = jnp.where(partial, _kite_area(p, safe_z, 1.0), 0.0)
    p_factor = (p - 1.0) * (p + 1.0)
    planet_angle = jnp.arctan2(kite, z * z + p_factor)
    star_angle = jnp.arctan2(kite, z * z - p_factor)
    return kite, planet_angle, star_angle


def make_power2_flux(order: int, *, endpoint_map: str = "sine"):
    """Return a scalar ``flux(z, p, c, alpha)`` JAX function."""
    if endpoint_map not in {"linear", "sine"}:
        raise ValueError("endpoint_map must be 'linear' or 'sine'.")
    roots, weights = roots_legendre(int(order))
    roots = jnp.asarray(roots, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)

    def full_planet(values):
        z, p, c, alpha = values
        theta = math.pi * roots
        cosine = jnp.cos(theta)
        mu2 = jnp.clip(1.0 - z * z - p * p + 2.0 * z * p * cosine, 0.0, 1.0)
        eta = p * p - z * p * cosine
        area = math.pi * p * p
        # Subtract the exact limb value of the Green primitive.  Its contour
        # integral is 2*area/(alpha+2), leaving a residual that vanishes at
        # contact endpoints and has much better quadrature/gradient behavior.
        zeta_residual = _zeta_power(mu2, alpha) - 1.0 / (alpha + 2.0)
        s_alpha = (
            2.0 * area / (alpha + 2.0)
            + math.pi * jnp.sum(weights * zeta_residual * eta)
        )
        norm = math.pi * (1.0 - c + 2.0 * c / (alpha + 2.0))
        return 1.0 - ((1.0 - c) * area + c * s_alpha) / norm

    def partial_overlap(values):
        z, p, c, alpha = values
        planet_cos = jnp.clip((z * z + p * p - 1.0) / (2.0 * z * p), -1.0, 1.0)
        star_cos = jnp.clip((z * z + 1.0 - p * p) / (2.0 * z), -1.0, 1.0)
        theta0 = jnp.arccos(planet_cos)
        phi0 = jnp.arccos(star_cos)
        if endpoint_map == "sine":
            mapped = 0.5 * math.pi * roots
            theta = theta0 * jnp.sin(mapped)
            jacobian = 0.5 * math.pi * theta0 * jnp.cos(mapped)
        else:
            theta = theta0 * roots
            jacobian = jnp.full_like(roots, theta0)
        cosine = jnp.cos(theta)
        mu2 = jnp.clip(1.0 - z * z - p * p + 2.0 * z * p * cosine, 0.0, 1.0)
        eta = p * p - z * p * cosine
        # The same Green integral for alpha=0, simplified analytically.
        area = p * p * theta0 - z * p * jnp.sin(theta0) + phi0
        zeta_residual = _zeta_power(mu2, alpha) - 1.0 / (alpha + 2.0)
        s_alpha = (
            2.0 * area / (alpha + 2.0)
            + jnp.sum(weights * jacobian * zeta_residual * eta)
        )
        norm = math.pi * (1.0 - c + 2.0 * c / (alpha + 2.0))
        return 1.0 - ((1.0 - c) * area + c * s_alpha) / norm

    def flux(z, p, c, alpha):
        values = (jnp.abs(z), jnp.abs(p), c, alpha)
        z_abs, p_abs, _, _ = values
        return jax.lax.cond(
            z_abs >= 1.0 + p_abs,
            lambda _: jnp.asarray(1.0, dtype=jnp.float64),
            lambda inner: jax.lax.cond(
                inner[0] <= 1.0 - inner[1],
                full_planet,
                partial_overlap,
                inner,
            ),
            values,
        )

    return jax.jit(flux)


def make_power2_flux_unified(
    order: int, *, full_endpoint_map: str = "linear"
):
    """Return a scalar Power-2 flux using one branch-free contour integral.

    ``vmap`` lowers an elementwise ``lax.cond`` to selects, so the two overlap
    branches in :func:`make_power2_flux` can both execute on an accelerator.
    This prototype instead selects the arc coordinates and Jacobian before a
    single expensive ``mu**alpha`` quadrature.  It retains the linear map for
    full-planet occultations and the endpoint-clustered sine map in ingress.
    """
    if full_endpoint_map not in {"linear", "sine"}:
        raise ValueError("full_endpoint_map must be 'linear' or 'sine'.")
    roots, weights = roots_legendre(int(order))
    roots = jnp.asarray(roots, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)

    def flux(z, p, c, alpha):
        z = jnp.abs(z)
        p = jnp.abs(p)
        full_planet = z <= 1.0 - p
        kite, theta0, phi0 = _contact_geometry(z, p)

        sine_coordinate = 0.5 * math.pi * roots
        partial_theta = theta0 * jnp.sin(sine_coordinate)
        partial_jacobian = (
            0.5 * math.pi * theta0 * jnp.cos(sine_coordinate)
        )
        if full_endpoint_map == "sine":
            full_theta = math.pi * jnp.sin(sine_coordinate)
            full_jacobian = 0.5 * math.pi**2 * jnp.cos(sine_coordinate)
        else:
            full_theta = math.pi * roots
            full_jacobian = jnp.full_like(roots, math.pi)
        theta = jnp.where(full_planet, full_theta, partial_theta)
        jacobian = jnp.where(full_planet, full_jacobian, partial_jacobian)

        cosine = jnp.cos(theta)
        outer_form = (
            (1.0 - z + p) * (1.0 + z - p)
            - 2.0 * z * p * (1.0 - cosine)
        )
        inner_form = (
            (1.0 - z - p) * (1.0 + z + p)
            + 2.0 * z * p * (1.0 + cosine)
        )
        mu2 = jnp.clip(jnp.where(cosine >= 0.0, outer_form, inner_form), 0.0, 1.0)
        eta = p * p - z * p * cosine
        overlap_area = p * p * theta0 + phi0 - 0.5 * kite
        correction_moment = jnp.sum(
            weights
            * jacobian
            * _zeta_power_minus_limb(mu2, alpha)
            * eta
        )
        power_moment = (
            2.0 * overlap_area / (alpha + 2.0) + correction_moment
        )
        norm = math.pi * (1.0 - c + 2.0 * c / (alpha + 2.0))
        return 1.0 - (
            (1.0 - c) * overlap_area + c * power_moment
        ) / norm

    return jax.jit(flux)


def make_power2_flux_full_circle(order: int, *, endpoint_map: str = "linear"):
    """Return a fixed-circle Green-contour Power-2 flux prototype.

    After subtracting the Green primitive's limb value, the residual is zero
    wherever ``mu == 0``.  The partial planet arc can therefore be extended
    over the whole circular boundary with ``mu2`` clipped to zero outside the
    star.  This removes the variable integration limit (and its inner-contact
    derivative cancellation) at the cost of an interior clipped-support edge.
    """
    if endpoint_map not in {"linear", "sine"}:
        raise ValueError("endpoint_map must be 'linear' or 'sine'.")
    roots, weights = roots_legendre(int(order))
    roots = jnp.asarray(roots, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    sine_coordinate = 0.5 * math.pi * roots
    if endpoint_map == "sine":
        theta = math.pi * jnp.sin(sine_coordinate)
        jacobian = 0.5 * math.pi**2 * jnp.cos(sine_coordinate)
    else:
        theta = math.pi * roots
        jacobian = jnp.full_like(roots, math.pi)
    cosine = jnp.cos(theta)
    sin_half_sq = jnp.sin(0.5 * theta) ** 2
    cos_half_sq = jnp.cos(0.5 * theta) ** 2

    def flux(z, p, c, alpha):
        z = jnp.abs(z)
        p = jnp.abs(p)
        kite, theta0, phi0 = _contact_geometry(z, p)
        overlap_area = p * p * theta0 + phi0 - 0.5 * kite

        outer_form = (
            (1.0 - z + p) * (1.0 + z - p)
            - 4.0 * z * p * sin_half_sq
        )
        inner_form = (
            (1.0 - z - p) * (1.0 + z + p)
            + 4.0 * z * p * cos_half_sq
        )
        mu2 = jnp.clip(
            jnp.where(cosine >= 0.0, outer_form, inner_form), 0.0, 1.0
        )
        eta = p * p - z * p * cosine
        correction_moment = jnp.sum(
            weights
            * jacobian
            * _zeta_power_minus_limb(mu2, alpha)
            * eta
        )
        power_moment = (
            2.0 * overlap_area / (alpha + 2.0) + correction_moment
        )
        norm = math.pi * (1.0 - c + 2.0 * c / (alpha + 2.0))
        return 1.0 - (
            (1.0 - c) * overlap_area + c * power_moment
        ) / norm

    return jax.jit(flux)


def _radial_moment_reference(z: float, p: float, power: float) -> float:
    """Independent adaptive annulus integral of the occulted radial moment."""
    z = abs(float(z))
    p = abs(float(p))
    power = float(power)
    if z >= 1.0 + p:
        return 0.0
    if z == 0.0:
        radius = min(p, 1.0)
        return 2.0 * math.pi * (
            1.0 - (1.0 - radius * radius) ** (0.5 * power + 1.0)
        ) / (power + 2.0)

    full_radius = min(max(p - z, 0.0), 1.0)
    full = 2.0 * math.pi * (
        1.0 - (1.0 - full_radius * full_radius) ** (0.5 * power + 1.0)
    ) / (power + 2.0)
    lower = max(abs(z - p), full_radius)
    upper = min(1.0, z + p)
    if upper <= lower:
        return full

    def integrand(radius):
        if radius == 0.0:
            return 0.0
        cosine = (radius * radius + z * z - p * p) / (2.0 * radius * z)
        angle = 2.0 * math.acos(min(1.0, max(-1.0, cosine)))
        return angle * radius * max(0.0, 1.0 - radius * radius) ** (0.5 * power)

    partial_value, _ = quad(
        integrand,
        lower,
        upper,
        epsabs=2.0e-13,
        epsrel=2.0e-13,
        limit=300,
    )
    return full + partial_value


def reference_flux(z: float, p: float, c: float, alpha: float) -> float:
    area = _radial_moment_reference(z, p, 0.0)
    power_moment = _radial_moment_reference(z, p, alpha)
    norm = math.pi * (1.0 - c + 2.0 * c / (alpha + 2.0))
    return 1.0 - ((1.0 - c) * area + c * power_moment) / norm


def _validation_points(count: int, seed: int):
    rng = np.random.default_rng(seed)
    p_min = math.sqrt(1.0e-5)
    p_max = math.sqrt(0.5)
    points = []
    for index in range(count):
        # The spectroscopic NumPyro prior is substantially wider than the
        # configured initial radii (0.0455--0.218): sqrt(1e-5)--sqrt(0.5).
        p = math.exp(rng.uniform(math.log(p_min), math.log(p_max)))
        c = 1.0 if index % 4 == 0 else rng.uniform(0.0, 1.0)
        # Log-uniform alpha exercises the difficult lower prior boundary.
        alpha = (
            (0.001, 0.01, 0.1, 0.3, 1.0)[index % 5]
            if index % 3 == 0
            else math.exp(rng.uniform(math.log(0.001), math.log(1.0)))
        )
        z = rng.uniform(0.0, 1.0 + p)
        points.append((z, p, c, alpha, "uniform"))
    offsets = np.geomspace(1.0e-12, 3.0e-2, max(2, count // 12))
    for sign in (-1.0, 1.0):
        for contact_name, contact_sign in (("outer", 1.0), ("inner", -1.0)):
            for offset in offsets:
                p = math.exp(rng.uniform(math.log(p_min), math.log(p_max)))
                c = 1.0 if len(points) % 3 == 0 else rng.uniform(0.0, 1.0)
                alpha = math.exp(rng.uniform(math.log(0.001), math.log(1.0)))
                contact = 1.0 + contact_sign * p
                z = min(1.0 + p, max(0.0, contact + sign * offset))
                points.append((z, p, c, alpha, contact_name))
    return points


def validate(
    orders, endpoint_maps, count: int, seed: int, *, implementation: str = "branched"
):
    if implementation not in {"branched", "unified"}:
        raise ValueError("implementation must be 'branched' or 'unified'.")
    points = _validation_points(count, seed)
    theta = np.asarray([point[:4] for point in points], dtype=np.float64)
    reference = np.asarray([reference_flux(*row) for row in theta])
    if implementation == "unified":
        gradient_reference_evaluator = make_power2_flux_unified(
            96, full_endpoint_map="sine"
        )
        gradient_reference_order = 96
    else:
        gradient_reference_evaluator = make_power2_flux(64, endpoint_map="sine")
        gradient_reference_order = 64
    gradient_reference = tuple(
        np.asarray(value)
        for value in jax.vmap(
            jax.grad(
                gradient_reference_evaluator,
                argnums=(0, 1, 2, 3),
            )
        )(*map(jnp.asarray, theta.T))
    )
    reports = []
    for endpoint_map in endpoint_maps:
        for order in orders:
            if implementation == "unified":
                evaluator = make_power2_flux_unified(
                    order, full_endpoint_map=endpoint_map
                )
            else:
                evaluator = make_power2_flux(order, endpoint_map=endpoint_map)
            approximate = np.asarray(
                jax.vmap(evaluator)(*map(jnp.asarray, theta.T))
            )
            error_ppm = np.abs(approximate - reference) * 1.0e6
            worst = int(np.argmax(error_ppm))
            gradients = jax.vmap(
                jax.grad(evaluator, argnums=(0, 1, 2, 3))
            )(*map(jnp.asarray, theta.T))
            finite_gradients = all(
                np.all(np.isfinite(np.asarray(value))) for value in gradients
            )
            gradient_arrays = tuple(np.asarray(value) for value in gradients)
            gradient_error = tuple(
                candidate - reference_value
                for candidate, reference_value in zip(
                    gradient_arrays, gradient_reference, strict=True
                )
            )
            gradient_error_flat = np.concatenate(
                [value.ravel() for value in gradient_error]
            )
            gradient_reference_flat = np.concatenate(
                [value.ravel() for value in gradient_reference]
            )
            worst_gradient = []
            for name, value, reference_value in zip(
                ("z", "p", "c", "alpha"),
                gradient_error,
                gradient_reference,
                strict=True,
            ):
                index = int(np.argmax(np.abs(value)))
                worst_gradient.append(
                    {
                        "parameter": name,
                        "error_ppm_per_unit": float(value[index] * 1.0e6),
                        "reference_derivative": float(reference_value[index]),
                        "parameters": theta[index].tolist(),
                        "point_class": points[index][4],
                    }
                )
            reports.append(
                {
                    "order": int(order),
                    "implementation": implementation,
                    "partial_arc_endpoint_map": endpoint_map,
                    "n_points": len(points),
                    "max_abs_error_ppm": float(error_ppm[worst]),
                    "p99_abs_error_ppm": float(np.percentile(error_ppm, 99.0)),
                    "rms_error_ppm": float(np.sqrt(np.mean(error_ppm**2))),
                    "worst_parameters": theta[worst].tolist(),
                    "worst_class": points[worst][4],
                    "all_first_derivatives_finite": bool(finite_gradients),
                    "gradient_reference_order": gradient_reference_order,
                    "max_abs_gradient_error_ppm_per_unit": {
                        name: float(np.max(np.abs(value)) * 1.0e6)
                        for name, value in zip(
                            ("z", "p", "c", "alpha"),
                            gradient_error,
                            strict=True,
                        )
                    },
                    "gradient_relative_l2": float(
                        np.linalg.norm(gradient_error_flat)
                        / max(np.linalg.norm(gradient_reference_flat), 1.0e-300)
                    ),
                    "worst_gradient_by_parameter": worst_gradient,
                }
            )
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", default="8,12,16,20,24")
    parser.add_argument("--endpoint-maps", default="sine")
    parser.add_argument(
        "--implementation", choices=("branched", "unified"), default="branched"
    )
    parser.add_argument("--random-points", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    orders = tuple(int(value) for value in args.orders.split(","))
    endpoint_maps = tuple(value.strip() for value in args.endpoint_maps.split(","))
    if any(order < 2 for order in orders) or args.random_points < 1:
        parser.error("orders must be >=2 and random-points must be positive")
    if any(value not in {"linear", "sine"} for value in endpoint_maps):
        parser.error("endpoint-maps choices are linear and sine")
    print(
        json.dumps(
            validate(
                orders,
                endpoint_maps,
                args.random_points,
                args.seed,
                implementation=args.implementation,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
