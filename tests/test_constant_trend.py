"""A fitted constant baseline must not introduce a slope parameter."""
import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.handlers import seed, substitute, trace

from models.jaxoplanet import create_vectorized_model, create_whitelight_model
from models.harmonica.builder import create_whitelight_model as harmonica_white
from models.cadence_reduction import build_linear_spectro_trend_design
from models.trend_marginal import build_marginalized_trend_design
from planet_specs import default_planet_specs


@pytest.mark.parametrize('factory', [create_whitelight_model, harmonica_white])
def test_white_constant_has_only_offset(factory):
    specs = default_planet_specs(period=3., t0=0., b=0.2, rprs=0.1, duration=0.1)
    model = factory(detrend_type='constant', ld_mode='fixed',
                    ld_profile='quadratic', parameter_priors=specs)
    t = jnp.linspace(-0.2, 0.2, 21)
    sites = trace(seed(substitute(model, data={'c': 1.02}), 1)).get_trace(
        t, jnp.ones_like(t)*0.001, y=jnp.ones_like(t),
        prior_params={'u': jnp.array([0.2, 0.1]), 'period': jnp.array([3.])})
    assert sites['c']['type'] == 'sample'
    assert 'v' not in sites
    assert np.all(np.isfinite(sites['obs']['fn'].log_prob(sites['obs']['value'])))


@pytest.mark.parametrize('ld_profile', ['quadratic', 'power2'])
@pytest.mark.parametrize('trend_mode', ['free', 'fixed'])
def test_spectro_constant_equals_linear_with_zero_slope(ld_profile, trend_mode):
    t = jnp.linspace(-0.15, 0.15, 25)
    values = {'rors': jnp.array([[0.1], [0.12]]),
              'c': jnp.array([1.01, 0.99]), 'v': jnp.zeros(2),
              'log_jitter': jnp.full(2, jnp.log(1e-4))}
    kw = dict(y=jnp.ones((2,25)), rors_reference=jnp.array([.1]), mu_duration=jnp.array([0.1]),
              mu_t0=jnp.array([0.]), mu_b=jnp.array([0.2]),
              PERIOD=jnp.array([3.]), ld_fixed=jnp.array([[0.2,0.3],[0.3,0.4]]))
    densities = []
    for trend in ['constant', 'linear']:
        model = create_vectorized_model(detrend_type=trend, ld_mode='fixed',
                    ld_profile=ld_profile, trend_mode=trend_mode)
        fixed = jnp.array([[1.01], [0.99]])
        if trend == 'linear': fixed = jnp.column_stack((fixed, jnp.zeros(2)))
        sites = trace(seed(substitute(model, data=values), 1)).get_trace(
            t, jnp.full((2,25),1e-3), trend_fixed=fixed, **kw)
        if trend == 'constant': assert 'v' not in sites
        densities.append(sites['obs']['fn'].log_prob(sites['obs']['value']))
    np.testing.assert_allclose(*densities, rtol=1e-12, atol=1e-12)


def test_constant_designs_contain_only_ones():
    t = np.linspace(1., 2., 9)
    names, design = build_linear_spectro_trend_design('constant', t)
    assert names == ('c',)
    np.testing.assert_array_equal(design, np.ones((9,1)))
    design, names = build_marginalized_trend_design('constant', t, 2)
    assert names == ('c',)
    np.testing.assert_array_equal(design, np.ones((2,9,1)))
