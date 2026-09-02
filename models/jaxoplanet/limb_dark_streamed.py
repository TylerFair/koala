"""Memory-efficient jaxoplanet 0.1.0 polynomial limb-darkening kernel.

This module preserves the public and numerical conventions of
``jaxoplanet.light_curves.limb_dark_light_curve`` (quadrature order 10 by
default).  The only substantial difference is in the high-order quadrature:
instead of materializing every ``f0 ** (n / 2)`` value, the even and odd
powers are advanced as two recurrence streams and reduced immediately.

Portions are derived from jaxoplanet 0.1.0, Copyright (c) 2021-2024 Simons
Foundation, Inc., under the MIT License.  The required notice is retained in
``models/jaxoplanet/JAXOPLANET_LICENSE``.
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import binom, roots_legendre

from jaxoplanet.light_curves.utils import vectorize
from jaxoplanet.utils import zero_safe_sqrt


@partial(jax.jit, static_argnames=("order",))
def light_curve(u, b, r, *, order: int = 10):
    """Return the polynomial limb-darkened transit signal for ``b`` and ``r``.

    The signature and returned signal (zero out of transit, negative in
    transit after orbit-level masking) match
    :func:`jaxoplanet.core.limb_dark.light_curve` from jaxoplanet 0.1.0.
    """
    u = jnp.asarray(u)
    if u.ndim != 1:
        raise ValueError("u must be a one-dimensional coefficient vector.")
    if u.shape[0] == 0:
        g = jnp.full((1,), 1.0 / jnp.pi, dtype=jnp.result_type(u, b, r))
    else:
        g = greens_basis_transform(u)
        g /= jnp.pi * (g[0] + g[1] / 1.5)
    s = solution_vector(len(g) - 1, order=order)(b, r)
    return s @ g - 1


def solution_vector(l_max: int, order: int = 10) -> Callable:
    """Build the Agol et al. solution vector evaluator."""
    n_max = l_max + 1

    @partial(jnp.vectorize, signature=f"(),()->({n_max})")
    def impl(b, r):
        b = jnp.abs(b)
        r = jnp.abs(r)
        area, kappa0, kappa1 = kappas(b, r)

        no_occ = jnp.greater_equal(b, 1 + r)
        full_occ = jnp.less_equal(1 + b, r)
        cond = jnp.logical_or(no_occ, full_occ)
        b_ = jnp.where(cond, jnp.ones_like(b), b)

        b2 = jnp.square(b_)
        r2 = jnp.square(r)
        s0, s2 = s0s2(b_, r, b2, r2, area, kappa0, kappa1)
        s0 = jnp.where(no_occ, jnp.pi, s0)
        s0 = jnp.where(full_occ, jnp.zeros_like(s0), s0)
        s2 = jnp.where(cond, jnp.zeros_like(s2), s2)

        values = [s0[None]]
        if l_max >= 1:
            p = p_integral(order, l_max, b_, r, b2, r2, kappa0)
            p = jnp.where(cond, jnp.zeros_like(p), p)
            values.append(-p[:1] - 2 * (kappa1 - jnp.pi) / 3)
            if l_max >= 2:
                values.append(s2[None])
            if l_max >= 3:
                values.append(-p[1:])
        return jnp.concatenate(values, axis=0)

    return impl


def greens_basis_transform(u):
    """Transform polynomial coefficients to the Green basis (0.1.0 parity)."""
    dtype = jnp.dtype(u)
    u = jnp.concatenate((-jnp.ones(1, dtype=dtype), u))
    size = len(u)
    indices = np.arange(size)
    arg = binom(indices[None, :], indices[:, None]) @ u
    p = (-1) ** (indices + 1) * arg
    g = [jnp.zeros((), dtype=dtype) for _ in range(size + 2)]
    for n in range(size - 1, 1, -1):
        g[n] = p[n] / (n + 2) + g[n + 2]
    g[1] = p[1] + 3 * g[3]
    g[0] = p[0] + 2 * g[2]
    return jnp.stack(g[:-2])


def kappas(b, r):
    b2 = jnp.square(b)
    factor = (r - 1) * (r + 1)
    cond = jnp.logical_and(jnp.greater(b, jnp.abs(1 - r)), jnp.less(b, 1 + r))
    b_ = jnp.where(cond, b, jnp.ones_like(b))
    area = jnp.where(cond, kite_area(r, b_, jnp.ones_like(r)), jnp.zeros_like(r))
    return area, jnp.arctan2(area, b2 + factor), jnp.arctan2(area, b2 - factor)


def s0s2(b, r, b2, r2, area, kappa0, kappa1):
    bpr = b + r
    onembpr2 = (1 + bpr) * (1 - bpr)
    eta2 = 0.5 * r2 * (r2 + 2 * b2)

    s0_lrg = jnp.pi * (1 - r2)
    s2_lrg = 2 * s0_lrg + 4 * jnp.pi * (eta2 - 0.5)

    alens = kappa1 + r2 * kappa0 - area * 0.5
    s0_sml = jnp.pi - alens
    s2_sml = 2 * s0_sml + 2 * (
        -(jnp.pi - kappa1)
        + 2 * eta2 * kappa0
        - 0.25 * area * (1 + 5 * r2 + b2)
    )

    delta = 4 * b * r
    cond = jnp.greater(onembpr2 + delta, delta)
    return jnp.where(cond, s0_lrg, s0_sml), jnp.where(cond, s2_lrg, s2_sml)


def p_integral(order: int, l_max: int, b, r, b2, r2, kappa0):
    """Evaluate quadrature while keeping only one limb order live at a time."""
    factor = 4 * b * r
    k2_cond = jnp.less(factor, 10 * jnp.finfo(factor.dtype).eps)
    safe_factor = jnp.where(k2_cond, jnp.ones_like(factor), factor)
    k2 = jnp.maximum(
        jnp.zeros_like(safe_factor),
        (1 - r2 - b2 + 2 * b * r) / safe_factor,
    )

    roots, weights = roots_legendre(order)
    roots = jnp.asarray(roots, dtype=factor.dtype)
    weights = jnp.asarray(weights, dtype=factor.dtype)
    rng = 0.5 * kappa0
    angle = rng * roots + rng
    cosine = jnp.cos(angle)
    # sin^2(angle / 2), expressed without a separate sine temporary.
    sin2 = 0.5 * (1 - cosine)

    terms = []
    if l_max >= 1:
        omz2 = jnp.maximum(0, r2 + b2 - 2 * b * r * cosine)
        z2 = 1 - omz2
        small_z2 = jnp.less(z2, 10 * jnp.finfo(omz2.dtype).eps)
        safe_z2 = jnp.where(small_z2, jnp.ones_like(z2), z2)
        z3 = safe_z2 * zero_safe_sqrt(safe_z2)
        small_omz2 = jnp.less(omz2, 10 * jnp.finfo(omz2.dtype).eps)
        safe_omz2 = jnp.where(small_omz2, jnp.ones_like(z2), omz2)
        result = 2 * r * (r - b * cosine) * (1 - z3) / (3 * safe_omz2)
        result = jnp.where(small_omz2, jnp.zeros_like(result), result)
        terms.append(jnp.sum(result * weights))

    if l_max >= 3:
        f0 = jnp.maximum(
            jnp.zeros_like(r2),
            jnp.where(k2_cond, 1 - r2, safe_factor * (k2 - sin2)),
        )
        geometry = 2 * r * (r - b + 2 * b * sin2)
        odd_power = f0 ** 1.5
        even_power = f0 ** 2
        for n in range(3, l_max + 1):
            power = odd_power if n % 2 else even_power
            terms.append(jnp.sum(power * geometry * weights))
            if n % 2:
                odd_power = odd_power * f0
            else:
                even_power = even_power * f0

    return rng * jnp.stack(terms)


def kite_area(a, b, c):
    def sort2(left, right):
        return jnp.minimum(left, right), jnp.maximum(left, right)

    a, b = sort2(a, b)
    b, c = sort2(b, c)
    a, b = sort2(a, b)
    square_area = (
        (a + (b + c))
        * (c - (a - b))
        * (c + (a - b))
        * (a + (b - c))
    )
    return zero_safe_sqrt(jnp.maximum(0, square_area))


def limb_dark_light_curve(orbit, *u, order: int = 10) -> Callable:
    """Orbit-level drop-in replacement for jaxoplanet's light-curve builder."""
    if u:
        ld_u = jnp.concatenate([jnp.atleast_1d(jnp.asarray(item)) for item in u])
    else:
        ld_u = jnp.array([])

    @vectorize
    def light_curve_impl(time):
        if jnp.ndim(time) != 0:
            raise ValueError("limb_dark_light_curve expects a scalar before vectorization.")
        r_star = orbit.central_radius
        x, y, z = orbit.relative_position(time)
        b = jnp.sqrt(x**2 + y**2) / r_star
        r = orbit.radius / r_star
        result = light_curve(ld_u, b, r, order=order)
        return jnp.where(z > 0, result, 0)

    return jax.jit(light_curve_impl)


__all__ = [
    "greens_basis_transform",
    "light_curve",
    "limb_dark_light_curve",
    "p_integral",
    "solution_vector",
]
