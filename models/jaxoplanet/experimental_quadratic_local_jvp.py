"""Experimental higher-order-safe local-JVP quadratic evaluator.

This isolated candidate has the same direct-``u1,u2`` (non-Kipping) primal as
``experimental_quadratic_dot``.  Its custom JVP forms the light-curve tangent
from scalar, per-cadence partial derivatives.  Reverse mode can then transpose
that compact local linear map, while forward mode and higher derivatives stay
available (unlike a ``custom_vjp``-only experiment).
"""

from functools import partial

import jax
import jax.numpy as jnp

from .experimental_quadratic_dot import light_curve as _quadratic_light_curve


def _scalar_light_curve(u, b, r):
    return _quadratic_light_curve(u, b, r, order=10)


_scalar_value_and_grad = jax.value_and_grad(
    _scalar_light_curve, argnums=(0, 1, 2)
)


@jax.custom_jvp
def _light_curve_local_jvp(u, b, r):
    return _quadratic_light_curve(u, b, r, order=10)


@_light_curve_local_jvp.defjvp
def _light_curve_jvp(primals, tangents):
    u, b, r = primals
    u_dot, b_dot, r_dot = tangents
    b_array = jnp.asarray(b)
    flat_b = jnp.ravel(b_array)
    value, (du, db, dr) = jax.vmap(
        _scalar_value_and_grad, in_axes=(None, 0, None)
    )(u, flat_b, r)
    value = jnp.reshape(value, b_array.shape)
    du = jnp.reshape(du, b_array.shape + (2,))
    db = jnp.reshape(db, b_array.shape)
    dr = jnp.reshape(dr, b_array.shape)
    tangent = (
        jnp.sum(du * u_dot, axis=-1)
        + db * b_dot
        + dr * r_dot
    )
    return value, tangent


@partial(jax.jit, static_argnames=("order",))
def light_curve(u, b, r, *, order: int = 10):
    """Evaluate the experimental direct-quadratic local-JVP kernel."""
    if order != 10:
        raise ValueError(
            "experimental_quadratic_local_jvp only supports order=10"
        )
    u = jnp.asarray(u)
    if u.ndim != 1 or u.shape[0] != 2:
        raise ValueError("quadratic light_curve requires u=[u1, u2].")
    r = jnp.asarray(r)
    if r.ndim != 0:
        raise ValueError("experimental local JVP currently requires scalar r")
    return _light_curve_local_jvp(u, jnp.asarray(b), r)


__all__ = ["light_curve"]
