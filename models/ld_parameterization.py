"""Exact change-of-variables parameterizations for limb darkening."""

from __future__ import annotations

import jax.numpy as jnp
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
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


class Power2SmoothTransform(Transform):
    """Map R² smoothly onto c in (0,1), alpha in (alpha_low,alpha_high).

    Write A=1-2**(-alpha_low), B=1-2**(-alpha_high),
    x=c*(1-2**(-alpha)-A)/(B-A). Then 0<x<c<1.
    x=sigmoid(z1), c=x+(1-x)*sigmoid(z2) parameterize this triangle.
    For alpha_low=0, x=(1-h1)/B is exactly a scaled Maxted coordinate.
    The shifted coordinate avoids a corner when alpha_low is positive.
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    sign = -1

    def __init__(self, alpha_low=0.0, alpha_high=1.0):
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        super().__init__()

    def _parts(self, z):
        from jax.nn import log_sigmoid
        lx, lxc = log_sigmoid(z[..., 0]), log_sigmoid(-z[..., 0])
        ls, lsc = log_sigmoid(z[..., 1]), log_sigmoid(-z[..., 1])
        lc = jnp.logaddexp(lx, lxc + ls)
        # r=x/c; retain logarithms to avoid cancellation near the boundaries.
        lr = lx - lc
        ln2 = jnp.log(2.0)
        width = -jnp.expm1(-(self.alpha_high-self.alpha_low)*ln2)
        le = -self.alpha_low*ln2 + jnp.log1p(-width*jnp.exp(lr))
        return lx, lxc, ls, lsc, lc, le

    def __call__(self, z):
        _, lxc, _, lsc, _, le = self._parts(z)
        return jnp.stack((-jnp.expm1(lxc + lsc), -le / jnp.log(2.0)), axis=-1)

    def _inverse(self, y):
        c, alpha = y[..., 0], y[..., 1]
        ln2 = jnp.log(2.0)
        width = -jnp.expm1(-(self.alpha_high-self.alpha_low)*ln2)
        r = -jnp.expm1(-(alpha-self.alpha_low)*ln2) / width
        x = c*r
        return jnp.stack((jnp.log(x)-jnp.log1p(-x),
                          jnp.log(c)+jnp.log1p(-r)-jnp.log1p(-c)), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        del y, intermediates
        lx, lxc, ls, lsc, lc, le = self._parts(x)
        ln2 = jnp.log(2.0)
        log_width = (-self.alpha_low*ln2
                     + jnp.log(-jnp.expm1(-(self.alpha_high-self.alpha_low)*ln2)))
        return log_width-jnp.log(ln2)-lc-le+lx+2*lxc+ls+lsc

    def tree_flatten(self):
        return (self.alpha_low, self.alpha_high), (('alpha_low', 'alpha_high'), {})


# Uniform power-2 prior in Maxted (2018) coordinates h1 = I(1/2) and h2 = I(1/2) - I(0).
# The support is 0 < h1 < 1 and 0 < h2 < min(1, h1 + 0.5), i.e. c = 1 - h1 + h2 < 1.5 (limb intensity down to -0.5,
# disk-integrated flux always positive) and every alpha in (0, inf). The density is flat in (h1, h2).
_MAXTED_UNIFORM_AREA = 0.875


def _maxted_uniform_top(h1):
    return jnp.minimum(1.0, h1 + 0.5)


def _maxted_from_coefficients(y):
    c, alpha = y[..., 0], y[..., 1]
    h2 = c * jnp.exp2(-alpha)
    one_minus_h1 = -c * jnp.expm1(-alpha * jnp.log(2.0))
    return one_minus_h1, h2


class Power2MaxtedUniformTransform(Transform):
    """Map R^2 smoothly onto the uniform Maxted-coordinate power-2 support, returning (c, alpha).

    h1 = sigmoid(z1), h2 = min(1, h1 + 0.5) * sigmoid(z2), c = 1 - h1 + h2, alpha = log2(c / h2).
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __call__(self, z):
        from jax.nn import sigmoid
        h1, one_minus_h1 = sigmoid(z[..., 0]), sigmoid(-z[..., 0])
        h2 = _maxted_uniform_top(h1) * sigmoid(z[..., 1])
        alpha = jnp.log1p(one_minus_h1 / h2) / jnp.log(2.0)
        return jnp.stack((one_minus_h1 + h2, alpha), axis=-1)

    def _inverse(self, y):
        one_minus_h1, h2 = _maxted_from_coefficients(y)
        h1 = 1.0 - one_minus_h1
        top = _maxted_uniform_top(h1)
        return jnp.stack((jnp.log(h1) - jnp.log(one_minus_h1), jnp.log(h2) - jnp.log(top - h2)), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        # log|d(c, alpha)/dz| = log|d(h1, h2)/dz| - log|d(h1, h2)/d(c, alpha)|, the latter being log(ln2 * h2).
        del intermediates
        from jax.nn import log_sigmoid, sigmoid
        h1 = sigmoid(x[..., 0])
        log_top = jnp.log(_maxted_uniform_top(h1))
        log_h2 = log_top + log_sigmoid(x[..., 1])
        return (log_sigmoid(x[..., 0]) + log_sigmoid(-x[..., 0]) + log_top
                + log_sigmoid(x[..., 1]) + log_sigmoid(-x[..., 1]) - jnp.log(jnp.log(2.0)) - log_h2)

    def tree_flatten(self):
        return (), ((), {})

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        del aux_data, params
        return cls()


class Power2MaxtedUniform(Distribution):
    """Prior on power-2 (c, alpha) that is uniform in the Maxted coordinates (h1, h2).

    Support: 0 < h1 < 1, 0 < h2 < min(1, h1 + 0.5). Density in (c, alpha) is ln2 * h2 / 0.875 (the Jacobian of
    (c, alpha) -> (h1, h2) divided by the support area); outside the support log_prob is -inf.
    """

    arg_constraints = {}
    support = constraints.real_vector
    reparametrized_params = []

    def __init__(self, batch_shape=(), *, validate_args=None):
        super().__init__(batch_shape=tuple(batch_shape), event_shape=(2,), validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        import jax
        shape = tuple(sample_shape) + self.batch_shape
        k1, k2 = jax.random.split(key)
        v = _MAXTED_UNIFORM_AREA * jax.random.uniform(k1, shape, dtype=jnp.float64)
        # Inverse CDF of the h1 marginal, density proportional to min(1, h1 + 0.5).
        h1 = jnp.where(v <= 0.375, -0.5 + jnp.sqrt(0.25 + 2.0 * v), v + 0.125)
        h2 = _maxted_uniform_top(h1) * jax.random.uniform(k2, shape, dtype=jnp.float64)
        c = 1.0 - h1 + h2
        return jnp.stack((c, jnp.log2(c / h2)), axis=-1)

    def log_prob(self, value):
        one_minus_h1, h2 = _maxted_from_coefficients(value)
        h1 = 1.0 - one_minus_h1
        inside = (h1 > 0.0) & (one_minus_h1 > 0.0) & (h2 > 0.0) & (h2 < _maxted_uniform_top(h1)) & (value[..., 1] > 0.0)
        safe_h2 = jnp.where(inside, h2, 1.0)
        return jnp.where(inside, jnp.log(jnp.log(2.0) * safe_h2) - jnp.log(_MAXTED_UNIFORM_AREA), -jnp.inf)


def power2_smooth_transform(mode):
    """Smooth sampling bijection of the ``ld_smooth`` site for a power-2 prior mode.

    ``uniform`` is the Maxted-coordinate uniform prior; the Gaussian modes keep the (c, alpha) box (0,1) x (0.001,1).
    """
    return Power2MaxtedUniformTransform() if mode == 'uniform' else Power2SmoothTransform(alpha_low=0.001)


def clip_into_maxted_uniform_support(coefficients, eps=1e-3):
    """Move power-2 (c, alpha) starting values strictly inside the uniform prior's support (no alpha <= 1 clip)."""
    coefficients = jnp.asarray(coefficients, dtype=jnp.float64)
    one_minus_h1, h2 = _maxted_from_coefficients(coefficients)
    h1 = jnp.clip(1.0 - jnp.nan_to_num(one_minus_h1, nan=0.5), eps, 1.0 - eps)
    top = _maxted_uniform_top(h1)
    h2 = jnp.clip(jnp.nan_to_num(h2, nan=0.5), eps * top, (1.0 - eps) * top)
    c = 1.0 - h1 + h2
    return jnp.stack((c, jnp.log1p((1.0 - h1) / h2) / jnp.log(2.0)), axis=-1)
