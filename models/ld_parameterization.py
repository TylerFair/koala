"""Exact change-of-variables parameterizations for limb darkening."""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import ndtr, ndtri
from numpyro.distributions import constraints
from numpyro.distributions.transforms import Transform


def gaussian_to_uniform(z, low, high):
    """Map standard-normal draws to an exact bounded uniform prior."""
    return low + (high - low) * ndtr(z)


def gaussian_to_truncated_normal(z, loc, scale, low, high):
    """Map standard-normal draws to a truncated normal by inverse CDF."""
    lower_probability = ndtr((low - loc) / scale)
    upper_probability = ndtr((high - loc) / scale)
    probability = lower_probability + (
        upper_probability - lower_probability
    ) * ndtr(z)
    return loc + scale * ndtri(probability)


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


class Power2LinearTransform(Transform):
    """Map physical power-2 coefficients to a constant-Jacobian sum/difference basis."""

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    sign = -1

    def __call__(self, x):
        c1, c2 = x[..., 0], x[..., 1]
        return jnp.stack((c1 + c2, c1 - c2), axis=-1)

    def _inverse(self, y):
        total, difference = y[..., 0], y[..., 1]
        return jnp.stack(
            ((total + difference) / 2.0, (total - difference) / 2.0),
            axis=-1,
        )

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        del y, intermediates
        return jnp.full(
            jnp.shape(x)[:-1], jnp.log(jnp.asarray(2.0, x.dtype))
        )

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
