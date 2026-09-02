"""Experimental native Power-2 transit kernel.

This module evaluates the spherical-planet Power-2 law directly instead of
first projecting it onto a degree-12 polynomial limb-darkening basis.  It is
deliberately not connected to the production fitting path yet.

For ``I(mu) = 1 - c + c * mu**alpha``, the blocked ``mu**alpha`` flux is
reduced with Green's theorem to one line integral around the occultor limb.
The constant value of the Green primitive at the stellar limb is integrated
analytically.  The remaining integrand vanishes at contact, and the same
endpoint-clustered coordinate is used on both sides of inner contact.  These
two details avoid the value cancellation and derivative mismatch of a naive
contour integral.

The expensive numerical work has one power-law term, rather than the roughly
ten terms evaluated by the degree-12 polynomial route.
"""

from __future__ import annotations

from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_legendre


Array = jax.Array


def _green_zeta_power2_minus_limb(mu_squared: Array, alpha: Array) -> Array:
    """Return the Power-2 Green primitive minus its stellar-limb value.

    With ``rho**2 = 1-mu**2``, the unsubtracted primitive is

    ``(1-mu**(alpha+2)) / ((alpha+2)*rho**2)``.

    Subtracting ``1/(alpha+2)`` makes the numerical integrand vanish at the
    stellar limb.  The expression below uses ``expm1`` and a centre expansion
    to remain stable at both ends of the stellar radius.
    """

    dtype = jnp.result_type(mu_squared, alpha)
    mu_squared = jnp.clip(jnp.asarray(mu_squared, dtype=dtype), 0.0, 1.0)
    alpha = jnp.asarray(alpha, dtype=dtype)
    radius_squared = 1.0 - mu_squared
    beta = 0.5 * alpha
    safe_radius_squared = jnp.where(radius_squared > 1.0e-7, radius_squared, 1.0)
    direct = mu_squared * (
        -jnp.expm1(beta * jnp.log(jnp.maximum(mu_squared, jnp.finfo(dtype).tiny)))
    )
    direct = direct / ((alpha + 2.0) * safe_radius_squared)
    series = (
        beta
        - 0.5 * beta * (beta + 1.0) * radius_squared
        + beta
        * (beta - 1.0)
        * (beta + 1.0)
        * jnp.square(radius_squared)
        / 6.0
    ) / (alpha + 2.0)
    return jnp.where(radius_squared > 1.0e-7, direct, series)


@jax.custom_jvp
def _zero_safe_sqrt(value: Array) -> Array:
    return jnp.sqrt(value)


@_zero_safe_sqrt.defjvp
def _zero_safe_sqrt_jvp(primals, tangents):
    (value,), (value_dot,) = primals, tangents
    result = _zero_safe_sqrt(value)
    tiny = value < 10.0 * jnp.finfo(value.dtype).eps
    denominator = jnp.where(tiny, 1.0, value)
    return result, 0.5 * value_dot * result / denominator


def _kite_area(a: Array, b: Array, c: Array) -> Array:
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


def _contact_geometry(separation: Array, radius_ratio: Array):
    """Return stable planet/star half-angles and twice the kite area."""

    partial = (
        (separation > jnp.abs(1.0 - radius_ratio))
        & (separation < 1.0 + radius_ratio)
    )
    safe_separation = jnp.where(partial, separation, 1.0)
    kite = jnp.where(
        partial, _kite_area(radius_ratio, safe_separation, 1.0), 0.0
    )
    radius_factor = (radius_ratio - 1.0) * (radius_ratio + 1.0)
    planet_angle = jnp.arctan2(
        kite, jnp.square(separation) + radius_factor
    )
    star_angle = jnp.arctan2(
        kite, jnp.square(separation) - radius_factor
    )
    return kite, planet_angle, star_angle


@partial(jax.jit, static_argnames=("order",))
def light_curve(
    c: Array,
    alpha: Array,
    separation: Array,
    radius_ratio: Array,
    *,
    order: int = 16,
) -> Array:
    """Return the normalized Power-2 transit signal (zero out of transit).

    Parameters broadcast in the usual NumPy fashion.  The returned convention
    matches :func:`jaxoplanet.core.limb_dark.light_curve`: values are negative
    during transit and zero otherwise.

    ``order=16`` is the conservative experimental default.  This remains an
    evaluation-only kernel until it passes posterior and NUTS-equivalence
    tests in addition to the pointwise two-ppm validation gate.
    """

    if order < 2:
        raise ValueError("Power-2 contour quadrature order must be at least two.")

    roots_np, weights_np = roots_legendre(order)
    roots = jnp.asarray(roots_np)
    weights = jnp.asarray(weights_np)

    @partial(jnp.vectorize, signature="(),(),(),()->()")
    def evaluate_scalar(c_i, alpha_i, separation_i, radius_ratio_i):
        dtype = jnp.result_type(c_i, alpha_i, separation_i, radius_ratio_i)
        c_i = jnp.asarray(c_i, dtype=dtype)
        alpha_i = jnp.asarray(alpha_i, dtype=dtype)
        separation_i = jnp.abs(jnp.asarray(separation_i, dtype=dtype))
        radius_ratio_i = jnp.abs(jnp.asarray(radius_ratio_i, dtype=dtype))
        roots_i = roots.astype(dtype)
        weights_i = weights.astype(dtype)

        full_planet = separation_i <= 1.0 - radius_ratio_i
        kite, planet_angle, star_angle = _contact_geometry(
            separation_i, radius_ratio_i
        )

        # A sine coordinate clusters nodes at the two stellar-limb endpoints
        # during ingress/egress.  Use the same map for full-planet
        # occultations so the finite-order approximation and its derivatives
        # remain continuous across inner contact.  Only one power-law
        # quadrature is emitted.
        sine_coordinate = 0.5 * math.pi * roots_i
        partial_theta = planet_angle * jnp.sin(sine_coordinate)
        partial_jacobian = (
            0.5 * math.pi * planet_angle * jnp.cos(sine_coordinate)
        )
        full_theta = math.pi * jnp.sin(sine_coordinate)
        full_jacobian = 0.5 * math.pi**2 * jnp.cos(sine_coordinate)
        theta = jnp.where(full_planet, full_theta, partial_theta)
        jacobian = jnp.where(full_planet, full_jacobian, partial_jacobian)

        cosine = jnp.cos(theta)
        sin_half = jnp.sin(0.5 * theta)
        cos_half = jnp.cos(0.5 * theta)
        outer_form = (
            (1.0 - separation_i + radius_ratio_i)
            * (1.0 + separation_i - radius_ratio_i)
            - 4.0
            * separation_i
            * radius_ratio_i
            * jnp.square(sin_half)
        )
        inner_form = (
            (1.0 - separation_i - radius_ratio_i)
            * (1.0 + separation_i + radius_ratio_i)
            + 4.0
            * separation_i
            * radius_ratio_i
            * jnp.square(cos_half)
        )
        mu_squared = jnp.clip(
            jnp.where(cosine >= 0.0, outer_form, inner_form), 0.0, 1.0
        )
        boundary_form = jnp.square(radius_ratio_i) - (
            separation_i * radius_ratio_i * cosine
        )
        overlap_area = (
            jnp.square(radius_ratio_i) * planet_angle
            + star_angle
            - 0.5 * kite
        )
        correction_moment = jnp.sum(
            weights_i
            * jacobian
            * _green_zeta_power2_minus_limb(mu_squared, alpha_i)
            * boundary_form
        )
        blocked_power = (
            2.0 * overlap_area / (alpha_i + 2.0) + correction_moment
        )

        blocked_flux = (1.0 - c_i) * overlap_area + c_i * blocked_power
        total_flux = math.pi * (
            1.0 - c_i + 2.0 * c_i / (alpha_i + 2.0)
        )
        return -blocked_flux / total_flux

    return evaluate_scalar(c, alpha, separation, radius_ratio)


__all__ = ["light_curve"]
