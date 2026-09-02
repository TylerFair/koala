"""Experimental direct-quadratic kernel retaining the stock dot contraction.

This is an isolated benchmark candidate, not a production-routed kernel.  It
uses literal ``u1,u2`` intensity coefficients (never Kipping coordinates),
the stock jaxoplanet order-10 solution vector, and an explicit degree-two
Green-basis transform.  Retaining ``solution @ g`` lets XLA use its efficient
dot transpose instead of generating separate broadcast reductions for every
coefficient in a handwritten scalar contraction.
"""

from functools import partial

import jax
import jax.numpy as jnp

from jaxoplanet.core.limb_dark import solution_vector


@partial(jax.jit, static_argnames=("order",))
def light_curve(u, b, r, *, order: int = 10):
    """Return the exact jaxoplanet quadratic signal for direct ``u1,u2``."""
    u = jnp.asarray(u)
    if u.ndim != 1 or u.shape[0] != 2:
        raise ValueError("quadratic light_curve requires u=[u1, u2].")
    u1, u2 = u

    # Degree-two specialization of jaxoplanet's Green-basis transform.  The
    # operation order mirrors the generic recurrence where practical.
    p0 = 1 - u1 - u2
    g2 = -u2 / 4
    g0 = p0 + 2 * g2
    g1 = u1 + 2 * u2
    g = jnp.stack((g0, g1, g2))
    g = g / (jnp.pi * (g[0] + g[1] / 1.5))

    solution = solution_vector(2, order=order)(b, r)
    return solution @ g - 1


__all__ = ["light_curve"]
