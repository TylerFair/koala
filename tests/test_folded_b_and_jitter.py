import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers

from koala.config import parse_parameter_spec, parse_trend_priors
from models.priors import latent_init_site, sample_parameter


def _trace(spec, site, seed=0):
    model = lambda: sample_parameter(spec, site)
    return handlers.trace(handlers.seed(model, jax.random.PRNGKey(seed))).get_trace()


def test_uniform_b_from_zero_is_sampled_folded():
    spec = parse_parameter_spec('b', {'value': 0.2, 'prior': 'uniform', 'low': 0.0, 'high': 1.5})
    tr = _trace(spec, 'b_0')
    assert tr['_b_0']['type'] == 'sample' and tr['b_0']['type'] == 'deterministic'
    assert float(tr['b_0']['value']) == abs(float(tr['_b_0']['value']))
    assert latent_init_site(spec, 'b_0')[0] == '_b_0'
    draws = np.array([float(_trace(spec, 'b_0', s)['b_0']['value']) for s in range(2000)])
    assert draws.min() >= 0.0 and draws.max() <= 1.5
    # |U(-h, h)| is U(0, h): mean h/2, variance h^2/12.
    assert abs(draws.mean() - 0.75) < 0.05 and abs(draws.var() - 1.5**2 / 12) < 0.03


def test_other_b_priors_unchanged():
    for raw in ({'value': 0.5, 'prior': 'uniform', 'low': 0.1, 'high': 1.0},
                {'value': 0.2, 'prior': 'gaussian', 'sigma': 0.1, 'low': 0.0, 'high': 1.0}):
        spec = parse_parameter_spec('b', raw)
        tr = _trace(spec, 'b_0')
        assert '_b_0' not in tr and tr['b_0']['type'] == 'sample'
    rprs = parse_parameter_spec('rprs', {'value': 0.1, 'prior': 'uniform', 'low': 0.0, 'high': 0.5})
    assert latent_init_site(rprs, 'rors_0')[0] == 'rors_0'


def test_jitter_prior_configurable():
    specs = parse_trend_priors({'jitter': {'value': 0.0, 'prior': 'fixed'}})
    tr = _trace(specs['jitter'], 'jitter')
    assert tr['jitter']['type'] == 'deterministic' and float(tr['jitter']['value']) == 0.0
    specs = parse_trend_priors({'jitter': {'prior': 'log_uniform', 'low': 1e-6, 'high': 1e-3}})
    assert latent_init_site(specs['jitter'], 'jitter')[0] == 'log_jitter'
    assert 1e-6 < specs['jitter'].value < 1e-3
