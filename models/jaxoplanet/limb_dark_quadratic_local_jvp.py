"""Exact direct-quadratic kernel with a GPU-oriented local JVP.

The primal is the jaxoplanet 0.1.0 degree-two solution at quadrature order 10.
The custom JVP exposes one compact, cadence-local linearization to XLA instead
of relying on the transpose of the full broadcast/reduction graph.  This can
substantially reduce a short-window GPU reverse pass.  It preserves ordinary
quadratic intensity coefficients

``I(mu) = 1 - u1 * (1 - mu) - u2 * (1 - mu)**2``

and never applies Kipping coordinates.

The rule is symbolic-zero aware: when ``u1,u2`` are fixed inputs, their
Jacobian block is not constructed.  If the coefficients are sampled, the full
coefficient Jacobian is evaluated, so free quadratic limb darkening remains
mathematically valid (although the GPU speedup was established for fixed LD).

Portions are derived from jaxoplanet 0.1.0, Copyright (c) 2021-2024 Simons
Foundation, Inc., under the MIT License.  The required notice is retained in
``models/jaxoplanet/JAXOPLANET_LICENSE``.
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero
from jaxoplanet.core.limb_dark import solution_vector
from jaxoplanet.light_curves.utils import vectorize


@partial(jax.jit, static_argnames=("order",))
def _quadratic_dot(u, b, r, *, order: int = 10):
    """Stock-parity direct degree-two Green-basis dot contraction."""
    u = jnp.asarray(u)
    if u.ndim != 1 or u.shape[0] != 2:
        raise ValueError("quadratic light_curve requires u=[u1, u2].")
    u1, u2 = u

    # Exact specialization of jaxoplanet's generic Green transform.  Keeping
    # the final dot is important: XLA transposes it more efficiently than a
    # handwritten sum of three independently broadcast products.
    p0 = 1 - u1 - u2
    g2 = -u2 / 4
    g0 = p0 + 2 * g2
    g1 = u1 + 2 * u2
    g = jnp.stack((g0, g1, g2))
    g = g / (jnp.pi * (g[0] + g[1] / 1.5))
    return solution_vector(2, order=order)(b, r) @ g - 1


def _scalar_light_curve(u, b, r):
    return _quadratic_dot(u, b, r, order=10)


_scalar_full = jax.value_and_grad(
    _scalar_light_curve, argnums=(0, 1, 2)
)
_scalar_geometry = jax.value_and_grad(
    _scalar_light_curve, argnums=(1, 2)
)


@jax.custom_jvp
def _light_curve(u, b, r):
    return _quadratic_dot(u, b, r, order=10)


@partial(_light_curve.defjvp, symbolic_zeros=True)
def _light_curve_jvp(primals, tangents):
    u, b, r = primals
    u_dot, b_dot, r_dot = tangents
    b_array = jnp.asarray(b)
    flat_b = jnp.ravel(b_array)

    if type(u_dot) is SymbolicZero:
        value, (db, dr) = jax.vmap(
            _scalar_geometry, in_axes=(None, 0, None)
        )(u, flat_b, r)
        du_term = jnp.zeros_like(value)
    else:
        value, (du, db, dr) = jax.vmap(
            _scalar_full, in_axes=(None, 0, None)
        )(u, flat_b, r)
        du_term = jnp.sum(du * u_dot, axis=-1)

    value = jnp.reshape(value, b_array.shape)
    db = jnp.reshape(db, b_array.shape)
    dr = jnp.reshape(dr, b_array.shape)
    b_term = (
        jnp.zeros_like(db)
        if type(b_dot) is SymbolicZero
        else db * b_dot
    )
    r_term = (
        jnp.zeros_like(dr)
        if type(r_dot) is SymbolicZero
        else dr * r_dot
    )
    tangent = jnp.reshape(du_term, b_array.shape) + b_term + r_term
    return value, tangent


@partial(jax.jit, static_argnames=("order",))
def light_curve(u, b, r, *, order: int = 10):
    """Return the exact local-JVP direct-quadratic transit signal."""
    if order != 10:
        raise ValueError("quadratic_local_jvp currently supports order=10")
    u = jnp.asarray(u)
    if u.ndim != 1 or u.shape[0] != 2:
        raise ValueError("quadratic light_curve requires u=[u1, u2].")
    r = jnp.asarray(r)
    if r.ndim != 0:
        raise ValueError("quadratic_local_jvp requires a scalar radius ratio")
    return _light_curve(u, jnp.asarray(b), r)


def limb_dark_light_curve(orbit, *u, order: int = 10) -> Callable:
    """Orbit-level drop-in wrapper for the local-JVP evaluator."""
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
