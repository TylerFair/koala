"""Symbolic-zero-aware local-JVP direct-quadratic experiment.

This variant avoids constructing the limb-darkening-coefficient Jacobian when
literal ``u1,u2`` are fixed runtime inputs, while retaining the full local-JVP
path when either coefficient is differentiated.  It remains an isolated
benchmark candidate and does not use Kipping coordinates.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero

from .experimental_quadratic_dot import light_curve as _quadratic_light_curve


def _scalar_light_curve(u, b, r):
    return _quadratic_light_curve(u, b, r, order=10)


_scalar_full = jax.value_and_grad(
    _scalar_light_curve, argnums=(0, 1, 2)
)
_scalar_geometry = jax.value_and_grad(
    _scalar_light_curve, argnums=(1, 2)
)


@jax.custom_jvp
def _light_curve(u, b, r):
    return _quadratic_light_curve(u, b, r, order=10)


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
    if type(b_dot) is SymbolicZero:
        b_term = jnp.zeros_like(db)
    else:
        b_term = db * b_dot
    if type(r_dot) is SymbolicZero:
        r_term = jnp.zeros_like(dr)
    else:
        r_term = dr * r_dot
    tangent = jnp.reshape(du_term, b_array.shape) + b_term + r_term
    return value, tangent


@partial(jax.jit, static_argnames=("order",))
def light_curve(u, b, r, *, order: int = 10):
    if order != 10:
        raise ValueError(
            "experimental symbolic local JVP only supports order=10"
        )
    u = jnp.asarray(u)
    if u.ndim != 1 or u.shape[0] != 2:
        raise ValueError("quadratic light_curve requires u=[u1, u2].")
    r = jnp.asarray(r)
    if r.ndim != 0:
        raise ValueError("experimental local JVP currently requires scalar r")
    return _light_curve(u, jnp.asarray(b), r)


__all__ = ["light_curve"]
