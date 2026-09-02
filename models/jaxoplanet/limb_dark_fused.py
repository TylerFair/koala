"""Fused polynomial limb-darkening kernel for jaxoplanet 0.1.0.

This is an exact algebraic contraction of the Agol et al. solution vector used
by :mod:`jaxoplanet.core.limb_dark`.  It keeps the same Green-basis transform,
contact branches, and order-10 Gauss--Legendre quadrature as the stock code,
but contracts each high-order integral with its Green coefficient while the
power recurrence is live.  Consequently, no ``(n_times, degree)`` solution or
quadrature-power tensor is materialized.

Portions are derived from jaxoplanet 0.1.0, Copyright (c) 2021-2024 Simons
Foundation, Inc., under the MIT License.  The required notice is retained in
``models/jaxoplanet/JAXOPLANET_LICENSE``.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_legendre

from jaxoplanet.light_curves.utils import vectorize
from jaxoplanet.utils import zero_safe_sqrt

from .limb_dark_streamed import greens_basis_transform, kappas, s0s2


@partial(jax.jit, static_argnames=("order",))
def light_curve(u, b, r, *, order: int = 10):
    """Return the stock polynomial-LD signal with a fused ``solution @ g``."""
    u = jnp.asarray(u)
    if u.ndim != 1:
        raise ValueError("u must be a one-dimensional coefficient vector.")
    if u.shape[0] == 0:
        g = jnp.full((1,), 1.0 / jnp.pi, dtype=jnp.result_type(u, b, r))
    else:
        g = greens_basis_transform(u)
        g /= jnp.pi * (g[0] + g[1] / 1.5)
    return _contracted_solution(g, b, r, order=order) - 1


def _contracted_solution(g, b, r, *, order: int):
    """Evaluate ``solution_vector(len(g)-1)(b, r) @ g`` directly."""
    l_max = len(g) - 1

    @partial(jnp.vectorize, signature="(),()->()")
    def impl(separation, radius):
        separation = jnp.abs(separation)
        radius = jnp.abs(radius)
        area, kappa0, kappa1 = kappas(separation, radius)

        no_occ = jnp.greater_equal(separation, 1 + radius)
        full_occ = jnp.less_equal(1 + separation, radius)
        inactive = jnp.logical_or(no_occ, full_occ)
        safe_separation = jnp.where(
            inactive, jnp.ones_like(separation), separation
        )

        b2 = jnp.square(safe_separation)
        r2 = jnp.square(radius)
        s0, s2 = s0s2(
            safe_separation, radius, b2, r2, area, kappa0, kappa1
        )
        s0 = jnp.where(no_occ, jnp.pi, s0)
        s0 = jnp.where(full_occ, jnp.zeros_like(s0), s0)

        contracted = g[0] * s0
        if l_max < 1:
            return contracted

        factor = 4 * safe_separation * radius
        small_factor = jnp.less(factor, 10 * jnp.finfo(factor.dtype).eps)
        safe_factor = jnp.where(small_factor, jnp.ones_like(factor), factor)
        k2 = jnp.maximum(
            jnp.zeros_like(safe_factor),
            (
                1
                - r2
                - b2
                + 2 * safe_separation * radius
            )
            / safe_factor,
        )

        roots, weights = roots_legendre(order)
        roots = jnp.asarray(roots, dtype=factor.dtype)
        weights = jnp.asarray(weights, dtype=factor.dtype)
        rng = 0.5 * kappa0
        angle = rng * roots + rng
        cosine = jnp.cos(angle)
        sin2 = 0.5 * (1 - cosine)

        omz2 = jnp.maximum(
            0, r2 + b2 - 2 * safe_separation * radius * cosine
        )
        z2 = 1 - omz2
        small_z2 = jnp.less(z2, 10 * jnp.finfo(omz2.dtype).eps)
        safe_z2 = jnp.where(small_z2, jnp.ones_like(z2), z2)
        z3 = safe_z2 * zero_safe_sqrt(safe_z2)
        small_omz2 = jnp.less(omz2, 10 * jnp.finfo(omz2.dtype).eps)
        safe_omz2 = jnp.where(small_omz2, jnp.ones_like(z2), omz2)
        p1_integrand = (
            2
            * radius
            * (radius - safe_separation * cosine)
            * (1 - z3)
            / (3 * safe_omz2)
        )
        p1_integrand = jnp.where(
            small_omz2, jnp.zeros_like(p1_integrand), p1_integrand
        )
        p1 = rng * jnp.sum(p1_integrand * weights)
        p1 = jnp.where(inactive, jnp.zeros_like(p1), p1)
        contracted = contracted + g[1] * (
            -p1 - 2 * (kappa1 - jnp.pi) / 3
        )

        if l_max >= 2:
            contracted = contracted + g[2] * jnp.where(
                inactive, jnp.zeros_like(s2), s2
            )

        if l_max >= 3:
            f0 = jnp.maximum(
                jnp.zeros_like(r2),
                jnp.where(
                    small_factor,
                    1 - r2,
                    safe_factor * (k2 - sin2),
                ),
            )
            geometry = 2 * radius * (
                radius - safe_separation + 2 * safe_separation * sin2
            )
            weighted_integrand = jnp.zeros_like(geometry)
            odd_power = f0 ** 1.5
            even_power = f0 ** 2
            for degree in range(3, l_max + 1):
                power = odd_power if degree % 2 else even_power
                weighted_integrand = (
                    weighted_integrand - g[degree] * power * geometry
                )
                if degree % 2:
                    odd_power = odd_power * f0
                else:
                    even_power = even_power * f0
            high_orders = rng * jnp.sum(weighted_integrand * weights)
            contracted = contracted + jnp.where(
                inactive, jnp.zeros_like(high_orders), high_orders
            )

        return contracted

    return impl(b, r)


def limb_dark_light_curve(orbit, *u, order: int = 10):
    """Orbit-level drop-in wrapper using the fused core evaluator."""
    if u:
        ld_u = jnp.concatenate(
            [jnp.atleast_1d(jnp.asarray(item)) for item in u]
        )
    else:
        ld_u = jnp.array([])

    @vectorize
    def light_curve_impl(time):
        if jnp.ndim(time) != 0:
            raise ValueError(
                "limb_dark_light_curve expects a scalar before vectorization."
            )
        r_star = orbit.central_radius
        x, y, z = orbit.relative_position(time)
        separation = jnp.sqrt(x**2 + y**2) / r_star
        radius = orbit.radius / r_star
        result = light_curve(ld_u, separation, radius, order=order)
        return jnp.where(z > 0, result, 0)

    return jax.jit(light_curve_impl)


__all__ = ["light_curve", "limb_dark_light_curve"]
