"""Requested Gaussian LD width is an actual, exported target parameter."""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from koala.config import ParameterSpec
from koala.constants import FLAG_TIERS
from koala.white_light import _white_light_fingerprint_config
from models.model_building import ld_prior_scale
from models.limb_darkening_config import validate_gaussian_ld_width
import models.jaxoplanet.builder as jp
import models.harmonica.builder as hm

jax.config.update('jax_enable_x64',True)


@pytest.mark.parametrize('engine,stage,law',[
    ('jp','white','quadratic'),('jp','spectro','quadratic'),
    ('jp','white','power2'),('jp','spectro','power2'),
    ('hm','white','power2'),('hm','spectro','power2')])
@pytest.mark.parametrize('width',[.1,.2])
def test_actual_builder_density_uses_selected_coefficient_width(engine,stage,law,width):
    module=jp if engine=='jp' else hm
    values=dict(period=3.3,t0=0.,a_rs=13.6,b=float(13.6*np.cos(np.deg2rad(88.))),ecc=.05,omega=90.,rprs=.18)
    specs={name:(ParameterSpec(name,'fixed',value),) for name,value in values.items()}
    center=jnp.array([.4,.2] if law=='quadratic' else [.6,.4])
    options=dict(param_method='a_rs',detrend_type='none',ld_mode='gaussian',ld_profile=law,gaussian_ld_width=width)
    t=jnp.linspace(-.07,.07,13)
    if stage=='white':
        model=module.create_whitelight_model(parameter_priors=specs,**options)
        kw=dict(prior_params={'period':jnp.array([3.3]),'u':center,'u_sigma':jnp.array([.8,.9])})
        errors=jnp.full(t.size,.001)
    else:
        model=module.create_vectorized_model(**({k:v for k,v in options.items() if k!='param_method'} if engine=='hm' else options))
        errors=jnp.full((2,t.size),.001)
        kw=dict(PERIOD=jnp.array([3.3]),mu_t0=jnp.array([0.]),
                mu_a_rs=jnp.array([13.6]),mu_b=jnp.array([13.6*jnp.cos(jnp.deg2rad(88.))]),
                mu_u_ld=jnp.broadcast_to(center,(2,2)),sigma_u_ld=jnp.full((2,2),.8))
        if engine=='hm':kw.update(harmonica_a_rs=jnp.array([13.6]))
    trace=handlers.trace(handlers.seed(model,1707)).get_trace(t,errors,**kw)
    if law=='quadratic':
        value=trace['u']['value'];actual=trace['u']['fn'].log_prob(value)
        expected=dist.TruncatedNormal(center,width,low=0.,high=1.).log_prob(value).sum(-1)
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
    else:
        for index,name in enumerate(['c1','c2']):
            value=trace[name]['value'];actual=trace[name]['fn'].log_prob(value)
            expected=dist.TruncatedNormal(center[index],width,low=0. if index==0 else .001,high=1.).log_prob(value)
            np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)


def test_default_and_stellar_widths_are_preserved_and_selector_is_fingerprinted():
    mean=jnp.array([.4,.2])
    np.testing.assert_array_equal(ld_prior_scale(mean,None,'gaussian'),jnp.full(2,.2))
    np.testing.assert_array_equal(ld_prior_scale(mean,None,'gaussian',gaussian_width=1e-8),jnp.full(2,1e-8))
    np.testing.assert_array_equal(ld_prior_scale(mean,jnp.array([.03,.04]),'stellarprior',gaussian_width=.1),[.03,.04])
    assert 'ld_gaussian_sigma' in FLAG_TIERS['public']
    a=_white_light_fingerprint_config({'flags':{'ld_gaussian_sigma':.1}})
    b=_white_light_fingerprint_config({'flags':{'ld_gaussian_sigma':.2}})
    assert a!=b


@pytest.mark.parametrize('value',[True,None,0.,-.1,float('nan'),float('inf'),'invalid'])
def test_invalid_width_fails_before_model_construction(value):
    with pytest.raises(ValueError,match='ld_gaussian_sigma.*positive'):
        validate_gaussian_ld_width(value)
    for module in [jp,hm]:
        for factory in [module.create_whitelight_model,module.create_vectorized_model]:
            with pytest.raises(ValueError,match='ld_gaussian_sigma.*positive'):
                factory(gaussian_ld_width=value)
