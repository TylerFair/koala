"""Data-selected LD priors retain hard support with NumPyro validation off."""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from scipy.special import ndtr
from scipy.stats import norm
from numpyro.infer.util import log_density
from models.jaxoplanet.builder import _ld_variant_data_form, create_vectorized_model

jax.config.update('jax_enable_x64', True)


def form(law, code, point, mode):
    low, high = (np.array([-1., -2.]), np.array([2., 2.])) if code == 2 else (
        np.array([0., .001 if law == 'power2' else 0.]), np.array([1., 1.]))
    physical = np.array(point)
    if code == 1:
        c, alpha = physical
        h2 = c*2**(-alpha)
        latent = jnp.array([1-c+h2, h2])
    else:
        latent = jnp.array(physical)
    scale = np.array([.1, .1]) if mode == 'normal' else np.array([-1., -1.])
    center = np.array([.5, .3])
    coefficients, correction = _ld_variant_data_form(
        latent, jnp.array(center), jnp.array(scale), jnp.array(low), jnp.array(high), code, law)
    target = correction+jnp.sum(dist.Normal(0., 1.).log_prob(latent))
    return coefficients, target, (low, high, center, scale, np.asarray(latent))


@pytest.mark.parametrize('law,code,point', [
    ('quadratic', 0, [.4, .2]), ('power2', 0, [.6, .4]),
    ('quadratic', 2, [.5, .1]), ('power2', 1, [.6, .4]),
])
@pytest.mark.parametrize('mode', ['normal', 'uniform'])
def test_data_form_valid_coordinate_density_matches_independent_normalizers(law, code, point, mode):
    _, target, (low, high, center, scale, latent) = form(law, code, point, mode)
    if mode == 'normal':
        mass = ndtr((high-center)/scale)-ndtr((low-center)/scale)
        expected = np.sum(norm.logpdf(point, center, scale)-np.log(mass))
    else:
        expected = -np.log(high-low).sum()
    if code == 1:
        expected -= np.log(np.log(2.)*latent[1])
    np.testing.assert_allclose(target, expected, atol=2e-12, rtol=0.)
    def actual(z):
        _, correction = _ld_variant_data_form(z, jnp.array(center), jnp.array(scale),
            jnp.array(low), jnp.array(high), code, law)
        return correction+jnp.sum(dist.Normal(0., 1.).log_prob(z))
    def independent(z):
        if code == 1:
            c = 1-z[0]+z[1]
            x = np.array([c, (np.log(c)-np.log(z[1]))/np.log(2.)])
        else:
            x = z
        density = (np.sum(norm.logpdf(x, center, scale)-np.log(mass))
                   if mode == 'normal' else -np.log(high-low).sum())
        return density-np.log(np.log(2.)*z[1]) if code == 1 else density
    h = 1e-5
    finite = []
    for delta in np.eye(2)*h:
        finite.append((independent(latent-2*delta)-8*independent(latent-delta)
                       +8*independent(latent+delta)-independent(latent+2*delta))/(12*h))
    np.testing.assert_allclose(jax.grad(actual)(jnp.array(latent)), finite, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('law,code,point', [
    ('quadratic', 0, [1.2, .2]), ('quadratic', 0, [-.1, .2]),
    ('quadratic', 0, [.4, -.1]), ('power2', 0, [1.2, .4]),
    ('power2', 0, [.6, -.1]), ('power2', 0, [.6, 1.2]),
    ('quadratic', 2, [2.2, .1]), ('quadratic', 2, [.5, 2.2]),
    ('power2', 1, [1.2, .4]), ('power2', 1, [.6, 1.2]),
])
@pytest.mark.parametrize('mode', ['normal', 'uniform'])
def test_unbounded_data_form_rejects_coordinates_outside_declared_support(law, code, point, mode):
    assert not dist.Normal(0., 1.)._validate_args
    _, target, _ = form(law, code, point, mode)
    assert np.isneginf(target)


@pytest.mark.parametrize('point,inside', [([.4, .05], True), ([-.1, 0.], False),
                                        ([1.2, 0.], False), ([.4, .2], False)])
def test_sing_data_form_checks_both_limb_and_conditional_delta_support(point, inside):
    latent = jnp.array(point)
    _, correction = _ld_variant_data_form(latent, jnp.array([.5, 0.]), jnp.array([.1, .1]),
        jnp.array([0., 0.]), jnp.array([1., 1.]), 3, 'quadratic')
    target = correction+jnp.sum(dist.Normal(0., 1.).log_prob(latent))
    if inside:
        upper = (1-point[0])/4.
        expected = norm.logpdf(point[0], .5, .1)-np.log(ndtr(5.)-ndtr(-5.))
        expected += norm.logpdf(point[1], 0., .1)-np.log(ndtr(upper/.1)-ndtr(-upper/.1))
        np.testing.assert_allclose(target, expected, atol=2e-12, rtol=0.)
    else:
        assert np.isneginf(target)


@pytest.mark.parametrize('law', ['quadratic', 'power2'])
def test_actual_factory_rejects_physically_finite_ld_outside_its_declared_box(law):
    model = create_vectorized_model(param_method='a_rs',detrend_type='none', ld_mode='gaussian',
                                    ld_profile=law, ld_variant_as_data=True)
    t = jnp.linspace(-.08, .08, 35); error = jnp.full((1, 35), .001)
    low = jnp.array([[0., .001 if law == 'power2' else 0.]])
    kwargs = dict(y=jnp.ones_like(error), PERIOD=jnp.array([3.]), mu_t0=jnp.array([0.]),
        mu_a_rs=jnp.array([13.6]), mu_b=jnp.array([13.6*jnp.cos(jnp.deg2rad(88.))]),
        ld_center=jnp.array([[.5, .3]]), ld_scale=jnp.full((1, 2), .1),
        ld_low=low, ld_high=jnp.ones((1, 2)))
    # The power-2 exponent1.2 is a valid physical law, but outside the explicitly
    # declared legacy alpha<=1 prior. The quadratic test is also finite flux.
    ld = [[1.2, .2]] if law == 'quadratic' else [[.6, 1.2]]
    point = dict(rors=jnp.array([[.1]]), log_jitter=jnp.array([-9.]),
                 ld_variant_latent=jnp.array(ld))
    target = lambda p: log_density(model, (t, error), kwargs, p)[0]
    value, gradient = jax.jit(jax.value_and_grad(target))(point)
    assert np.isneginf(value)
    assert all(np.all(np.isfinite(v)) for v in gradient.values())
