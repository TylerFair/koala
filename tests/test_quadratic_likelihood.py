"""The fused Gaussian score preserves the ordinary fixed-orbit likelihood."""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from models.jaxoplanet.core import orbit_separation
from models.jaxoplanet.quadratic_safe import fixed_orbit_evaluator as old_evaluator
from models.jaxoplanet.quadratic_likelihood import log_likelihood, _ordinary


def _data(columns=2, ecc=0.):
    t = jnp.linspace(59770.65, 59771.02, 401)
    x = (t - t.mean()) / jnp.ptp(t)
    design = jnp.stack([x**i for i in range(columns)], axis=1)
    theta = jnp.array([.14683, .3, .2, 59770.8339, 1.,
                       *([.002] * (columns-1)), .0002])
    y = .98 + .001*jnp.cos(x*20)
    variance = jnp.full_like(y, .0004**2)
    mask = jnp.arange(len(t)) % 13 != 0
    orbit = jnp.array([4.05528, 11.4, .45, ecc, .32])
    return theta, t, y, variance, mask, design, orbit


def _original(theta, t, y, variance, mask, design, orbit):
    period, a_rs, impact, ecc, omega = orbit
    z, front = orbit_separation(t, period, theta[3], a_rs, impact, ecc, omega)
    flux = 1+jnp.where(front, old_evaluator(True)(theta[1:3], theta[0], z), 0.)
    logp = dist.Normal(flux*(design@theta[4:-1]), jnp.sqrt(variance+theta[-1]**2)).log_prob(y)
    return jnp.sum(jnp.where(mask, logp, 0.))


@pytest.mark.parametrize('columns', [1, 2, 4])
@pytest.mark.parametrize('ecc', [0., .2])
def test_value_and_every_parameter_gradient(columns, ecc):
    args = _data(columns, ecc)
    actual = jax.jit(jax.value_and_grad(log_likelihood))(*args)
    expected = jax.jit(jax.value_and_grad(_original))(*args)
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-7, rtol=1e-12)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-7, rtol=1e-6)


def test_higher_derivatives_and_shared_geometry():
    args = _data()
    actual = jax.jit(jax.hessian(log_likelihood))(*args)
    expected = jax.jit(jax.hessian(_ordinary))(*args)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-6)
    actual = jax.jit(jax.grad(log_likelihood, argnums=(2, 6)))(*args)
    expected = jax.jit(jax.grad(_ordinary, argnums=(2, 6)))(*args)
    for left, right in zip(actual, expected):
        np.testing.assert_allclose(left, right, atol=1e-7, rtol=1e-6)


def test_masked_observations_and_batched_channels():
    args = _data()
    theta, t, y, variance, mask, design, orbit = args
    theta = jnp.stack([theta, theta.at[0].set(.15)])
    function = jax.jit(jax.vmap(jax.value_and_grad(log_likelihood),
                               in_axes=(0, None, None, None, None, None, None)))
    actual = function(theta, *args[1:])
    changed = function(theta, t, jnp.where(mask, y, 123.), variance, mask, design, orbit)
    for left, right in zip(actual, changed):
        np.testing.assert_array_equal(left, right)
