"""The default power-2 uniform prior: flat in Maxted (h1, h2), sampled through a smooth bijection."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from models.ld_parameterization import (Power2MaxtedTransform, Power2MaxtedUniform,
                                        Power2MaxtedUniformTransform)


def _h(y):
    h = Power2MaxtedTransform()(y)
    return h[..., 0], h[..., 1]


def test_transform_roundtrip_support_and_jacobian():
    z = jnp.asarray(np.random.default_rng(5).uniform(-9, 9, (4000, 2)))
    t = Power2MaxtedUniformTransform()
    y = t(z)
    np.testing.assert_allclose(t.inv(y), z, atol=1e-7, rtol=0)
    h1, h2 = _h(y)
    assert np.all((h1 > 0) & (h1 < 1) & (h2 > 0) & (h2 < np.minimum(1, h1 + .5) + 1e-12))
    assert np.all(y[:, 1] > 0) and np.all(y[:, 0] < 1.5)
    # positive disk-integrated flux everywhere on the support
    assert np.all(1 - y[:, 0] * y[:, 1] / (y[:, 1] + 2) > 0)
    jac = jax.vmap(jax.jacfwd(t))(z)
    np.testing.assert_allclose(t.log_abs_det_jacobian(z, y), jnp.linalg.slogdet(jac)[1], atol=1e-8, rtol=0)


def test_density_is_flat_in_maxted_coordinates():
    """log p(z) = log p(c, alpha) + log|J| must equal log|d(h1,h2)/dz| - log(0.875) (flat in h, area 0.875)."""
    z = jnp.asarray(np.random.default_rng(6).uniform(-6, 6, (2000, 2)))
    t, prior = Power2MaxtedUniformTransform(), Power2MaxtedUniform()
    y = t(z)
    logp_z = prior.log_prob(y) + t.log_abs_det_jacobian(z, y)
    def h_of_z(zz):
        h1 = jax.nn.sigmoid(zz[0]); return jnp.stack((h1, jnp.minimum(1., h1 + .5) * jax.nn.sigmoid(zz[1])))
    expected = jnp.linalg.slogdet(jax.vmap(jax.jacfwd(h_of_z))(z))[1] - np.log(0.875)
    np.testing.assert_allclose(logp_z, expected, atol=1e-8, rtol=0)


def test_matches_injection_recovery_arm():
    """Same z-space density as the 'maxted_h' arm validated in the HATS-6 injection-recovery test (2026-10-07)."""
    z = jnp.asarray(np.random.default_rng(7).uniform(-6, 6, (500, 2)))
    t, prior = Power2MaxtedUniformTransform(), Power2MaxtedUniform()
    y = t(z)
    h1 = jax.nn.sigmoid(z[:, 0]); top = jnp.minimum(1.0, h1 + 0.5)
    arm = (jax.nn.log_sigmoid(z[:, 0]) + jax.nn.log_sigmoid(-z[:, 0]) + jnp.log(top)
           + jax.nn.log_sigmoid(z[:, 1]) + jax.nn.log_sigmoid(-z[:, 1]))
    np.testing.assert_allclose(prior.log_prob(y) + t.log_abs_det_jacobian(z, y), arm - np.log(0.875), atol=1e-8)


def test_prior_normalised_and_sampler_uniform_in_h():
    prior = Power2MaxtedUniform()
    y = prior.sample(jax.random.PRNGKey(0), (200000,))
    h1, h2 = (np.asarray(v) for v in _h(y))
    assert np.all(np.isfinite(prior.log_prob(y)))
    # uniform on the support: the h1 marginal is proportional to min(1, h1 + .5)
    hist, edges = np.histogram(h1, bins=20, range=(0, 1), density=True)
    mid = 0.5 * (edges[1:] + edges[:-1])
    np.testing.assert_allclose(hist, np.minimum(1, mid + .5) / 0.875, rtol=0.05)
    np.testing.assert_allclose(np.mean(h2 / np.minimum(1, h1 + .5)), 0.5, atol=0.005)
    # normalisation by importance sampling from a box in (h1, h2)
    rng = np.random.default_rng(1)
    hh1, hh2 = rng.uniform(0, 1, 400000), rng.uniform(0, 1, 400000)
    c = 1 - hh1 + hh2
    yy = jnp.stack((c, jnp.log2(c / hh2)), axis=-1)
    dens_h = np.exp(np.asarray(prior.log_prob(yy))) / (np.log(2) * hh2)
    assert abs(dens_h.mean() - 1.0) < 0.01


def test_outside_support_is_minus_inf():
    prior = Power2MaxtedUniform()
    bad = jnp.asarray([[0.5, -0.1], [1.6, 0.5], [-0.2, 0.5]])
    assert np.all(np.isneginf(np.asarray(prior.log_prob(bad))))


def test_jit_and_pytree():
    t = Power2MaxtedUniformTransform()
    z = jnp.asarray([0.3, -1.2])
    np.testing.assert_allclose(jax.jit(lambda tt, x: tt(x))(t, z), t(z))
