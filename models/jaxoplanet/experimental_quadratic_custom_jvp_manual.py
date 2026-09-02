"""Manual-tangent custom-JVP experiment for direct quadratic LD.

This prototype differentiates the stock-parity degree-two primal once at
trace time, but presents the reverse pass with one compact tangent expression.
It is isolated from production routing and uses literal ``u1,u2`` only.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .experimental_quadratic_dot import light_curve as _primal


@jax.custom_jvp
def _custom(u, b, r):
    return _primal(u, b, r, order=10)


@_custom.defjvp
def _custom_jvp(primals, tangents):
    u, b, r = primals
    u_dot, b_dot, r_dot = tangents
    # This computes all three Jacobian blocks in one linearization.  Unlike
    # the local-JVP prototype, it leaves batching decisions to XLA.
    value, tangent = jax.jvp(
        lambda uu, bb, rr: _primal(uu, bb, rr, order=10),
        (u, b, r),
        (u_dot, b_dot, r_dot),
    )
    return value, tangent


@partial(jax.jit, static_argnames=("order",))
def light_curve(u, b, r, *, order=10):
    if order != 10:
        raise ValueError("manual custom-JVP experiment only supports order=10")
    u = jnp.asarray(u)
    if u.ndim != 1 or u.shape[0] != 2:
        raise ValueError("quadratic light_curve requires u=[u1, u2].")
    return _custom(u, jnp.asarray(b), jnp.asarray(r))


__all__ = ["light_curve"]
