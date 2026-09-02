"""Specialized direct-``u1,u2`` quadratic limb-darkening kernel.

The evaluator is an algebraic specialization of jaxoplanet 0.1.0's generic
polynomial limb-darkening implementation at degree two.  It preserves the
same contact branches and order-10 Gauss--Legendre evaluation of the linear
Green-basis term, but removes the generic coefficient transform, solution
vector construction, and final matrix product.  The inputs are the ordinary
quadratic intensity coefficients in

``I(mu) = 1 - u1 * (1 - mu) - u2 * (1 - mu)**2``.

No Kipping ``q1,q2`` transformation is used.

Portions are derived from jaxoplanet 0.1.0, Copyright (c) 2021-2024 Simons
Foundation, Inc., under the MIT License.  The required notice is retained in
``models/jaxoplanet/JAXOPLANET_LICENSE``.
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from scipy.special import roots_legendre

from jaxoplanet.light_curves.utils import vectorize
from jaxoplanet.utils import zero_safe_sqrt

from .limb_dark_streamed import kappas, s0s2


@partial(jax.jit, static_argnames=("order",))
def light_curve(u, b, r, *, order: int = 10):
    """Return the direct quadratic limb-darkened transit signal.

    The return convention matches :func:`jaxoplanet.core.limb_dark.light_curve`:
    zero out of occultation and negative during transit.
    """
    u = jnp.asarray(u)
    if u.ndim != 1 or u.shape[0] != 2:
        raise ValueError("quadratic light_curve requires u=[u1, u2].")

    separation = jnp.abs(jnp.asarray(b))
    radius = jnp.abs(jnp.asarray(r))
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
    s2 = jnp.where(inactive, jnp.zeros_like(s2), s2)

    roots, weights = roots_legendre(order)
    dtype = jnp.result_type(safe_separation, radius)
    roots = jnp.asarray(roots, dtype=dtype)
    weights = jnp.asarray(weights, dtype=dtype)
    rng = 0.5 * kappa0
    angle = rng[..., None] * roots + rng[..., None]
    cosine = jnp.cos(angle)

    omz2 = jnp.maximum(
        0,
        r2[..., None]
        + b2[..., None]
        - 2 * safe_separation[..., None] * radius[..., None] * cosine,
    )
    z2 = 1 - omz2
    small_z2 = jnp.less(z2, 10 * jnp.finfo(omz2.dtype).eps)
    safe_z2 = jnp.where(small_z2, jnp.ones_like(z2), z2)
    z3 = safe_z2 * zero_safe_sqrt(safe_z2)
    small_omz2 = jnp.less(omz2, 10 * jnp.finfo(omz2.dtype).eps)
    safe_omz2 = jnp.where(small_omz2, jnp.ones_like(z2), omz2)
    p1_integrand = (
        2
        * radius[..., None]
        * (radius[..., None] - safe_separation[..., None] * cosine)
        * (1 - z3)
        / (3 * safe_omz2)
    )
    p1_integrand = jnp.where(
        small_omz2, jnp.zeros_like(p1_integrand), p1_integrand
    )
    p1 = rng * jnp.sum(p1_integrand * weights, axis=-1)
    p1 = jnp.where(inactive, jnp.zeros_like(p1), p1)
    s1 = -p1 - 2 * (kappa1 - jnp.pi) / 3

    u1, u2 = u
    # Exact degree-two Green-basis transform, written explicitly so the
    # generic binomial matrix and solution-vector materialization disappear.
    g0 = 1 - u1 - 1.5 * u2
    g1 = u1 + 2 * u2
    g2 = -0.25 * u2
    normalization = jnp.pi * (g0 + g1 / 1.5)
    return (g0 * s0 + g1 * s1 + g2 * s2) / normalization - 1


def limb_dark_light_curve(orbit, *u, order: int = 10) -> Callable:
    """Orbit-level drop-in wrapper for direct quadratic coefficients."""
    ld_u = jnp.concatenate(
        [jnp.atleast_1d(jnp.asarray(item)) for item in u]
    )

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
