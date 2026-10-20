"""Epoch aliases must preserve the declared probability measure."""
import jax.numpy as jnp
import numpy as np
import pytest
from koala.config import parse_planet_parameter_specs
from models.priors import spec_distribution


def configuration(prior):
    return dict(period=dict(value=4.,prior='fixed'),
                eclipse_time=dict(value=60001.,prior=prior,**(
                    {} if prior=='fixed' else dict(low=60000.,high=60002.,**(
                    dict(sigma=.2) if prior=='gaussian' else {})))),
                a_rs=dict(value=13.6,prior='fixed'),
                b=dict(value=.45,prior='fixed'),
                rprs=dict(value=.18,prior='fixed'))


@pytest.mark.parametrize('prior',['fixed','uniform','gaussian'])
def test_public_epoch_alias_preserves_translated_prior_and_units(prior):
    from koala.config import parse_parameter_spec
    config=configuration(prior)
    original=parse_parameter_spec('eclipse_time',config['eclipse_time'])
    converted=parse_planet_parameter_specs(config)['t0'][0]
    assert converted.name=='t0' and converted.prior==prior
    assert converted.value==original.value-2.
    assert converted.sigma==original.sigma
    for name in ['low','high']:
        original_bound=getattr(original,name)
        assert getattr(converted,name)==(None if original_bound is None else original_bound-2.)
    if prior!='fixed':
        t=jnp.array([59998.25,59999.,59999.75])
        # Unit-Jacobian translation: no change in probability density.
        np.testing.assert_allclose(spec_distribution(converted).log_prob(t),
                                   spec_distribution(original).log_prob(t+2.),atol=1e-12,rtol=0.)


def test_log_uniform_epoch_alias_fails_instead_of_changing_density():
    with pytest.raises(ValueError,match='eclipse_time.*log_uniform.*shifting'):
        parse_planet_parameter_specs(configuration('log_uniform'))


def test_explicit_native_t0_log_uniform_prior_is_preserved():
    config=configuration('log_uniform')
    config['t0']=config.pop('eclipse_time')
    spec=parse_planet_parameter_specs(config)['t0'][0]
    assert spec.prior=='log_uniform' and spec.low==60000. and spec.high==60002.
