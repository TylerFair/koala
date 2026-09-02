"""Experimental local-VJP wrapper for direct quadratic limb darkening.

This module is deliberately not wired into the production kernel router.  It
tests whether transposing the scalar light-curve Jacobian locally (one cadence
at a time) is cheaper on a GPU than letting reverse-mode AD transpose the
whole vector evaluator.  The primal and local derivatives are delegated to
the stock-parity direct-``u1,u2`` dot prototype, so this experiment introduces
no alternative limb-darkening law and no Kipping coordinates.

The custom VJP supports the production calling convention: ``u`` has shape
``(2,)``, ``b`` is an arbitrary array, and ``r`` is a scalar.  Quadrature
order 10 is intentionally the only supported order because that is the
production jaxoplanet setting being benchmarked.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .experimental_quadratic_dot import light_curve as _quadratic_light_curve


def _scalar_light_curve(u, b, r):
    """Stock-parity direct-quadratic evaluator specialized to one cadence."""
    return _quadratic_light_curve(u, b, r, order=10)


_scalar_value_and_grad = jax.value_and_grad(
    _scalar_light_curve, argnums=(0, 1, 2)
)


@jax.custom_vjp
def _light_curve_local_vjp(u, b, r):
    return _quadratic_light_curve(u, b, r, order=10)


def _light_curve_fwd(u, b, r):
    b_array = jnp.asarray(b)
    flat_b = jnp.ravel(b_array)

    # Each scalar output depends only on the matching separation.  Computing
    # that small local transpose gives per-cadence partials without asking XLA
    # to transpose the large broadcast/reduction graph made by the vector
    # evaluator.
    value, (du, db, dr) = jax.vmap(
        _scalar_value_and_grad, in_axes=(None, 0, None)
    )(u, flat_b, r)
    value = jnp.reshape(value, b_array.shape)
    residual = (
        jnp.reshape(du, b_array.shape + (2,)),
        jnp.reshape(db, b_array.shape),
        jnp.reshape(dr, b_array.shape),
    )
    return value, residual


def _light_curve_bwd(residual, cotangent):
    du, db, dr = residual
    cotangent = jnp.asarray(cotangent)
    u_bar = jnp.sum(cotangent[..., None] * du, axis=tuple(range(cotangent.ndim)))
    b_bar = cotangent * db
    r_bar = jnp.sum(cotangent * dr)
    return u_bar, b_bar, r_bar


_light_curve_local_vjp.defvjp(_light_curve_fwd, _light_curve_bwd)


@partial(jax.jit, static_argnames=("order",))
def light_curve(u, b, r, *, order: int = 10):
    """Evaluate the experimental exact direct-quadratic local-VJP kernel."""
    if order != 10:
        raise ValueError(
            "experimental_quadratic_local_vjp only supports order=10"
        )
    u = jnp.asarray(u)
    if u.ndim != 1 or u.shape[0] != 2:
        raise ValueError("quadratic light_curve requires u=[u1, u2].")
    r = jnp.asarray(r)
    if r.ndim != 0:
        raise ValueError("experimental local VJP currently requires scalar r")
    return _light_curve_local_vjp(u, jnp.asarray(b), r)


__all__ = ["light_curve"]
