"""Exact change-of-variables parameterizations for limb darkening."""

from __future__ import annotations

import jax.numpy as jnp
from numpyro.distributions import constraints
from numpyro.distributions.transforms import Transform


class Power2MaxtedTransform(Transform):
    """Map physical ``(c, alpha)`` to Maxted (2018) ``(h1, h2)``."""

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    sign = 1

    def __call__(self, x):
        c, alpha = x[..., 0], x[..., 1]
        h2 = c * jnp.exp2(-alpha)
        return jnp.stack((1.0 - c + h2, h2), axis=-1)

    def _inverse(self, y):
        h1, h2 = y[..., 0], y[..., 1]
        c = 1.0 - h1 + h2
        alpha = jnp.log(c / h2) / jnp.log(2.0)
        return jnp.stack((c, alpha), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        del x, intermediates
        return jnp.log(jnp.log(2.0) * y[..., 1])

    def tree_flatten(self):
        return (), ()

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        del aux_data, params
        return cls()


class QuadraticKippingTransform(Transform):
    """Map physical quadratic ``(u1, u2)`` to Kipping (2013) ``(q1,q2)``."""

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    sign = -1

    def __call__(self, x):
        u1, u2 = x[..., 0], x[..., 1]
        total = u1 + u2
        return jnp.stack((total ** 2, u1 / (2.0 * total)), axis=-1)

    def _inverse(self, y):
        q1, q2 = y[..., 0], y[..., 1]
        total = jnp.sqrt(q1)
        return jnp.stack((2.0 * total * q2,
                          total * (1.0 - 2.0 * q2)), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        del y, intermediates
        return jnp.zeros(jnp.shape(x)[:-1], dtype=jnp.asarray(x).dtype)

    def tree_flatten(self):
        return (), ()

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        del aux_data, params
        return cls()
