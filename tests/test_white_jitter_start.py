"""A bounded optimizer endpoint is an invalid unconstrained start, not a prior change."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from numpyro import handlers
from numpyro.infer import init_to_value
from numpyro.infer.util import initialize_model
from numpyro.distributions.transforms import biject_to
from models.jaxoplanet import create_whitelight_model
from planet_specs import default_planet_specs
from koala.white_light import _repair_white_jitter_start


@pytest.mark.parametrize('value',[np.log(1e-5),np.log(1e-2),np.nan,np.inf])
def test_default_white_jitter_endpoint_repairs_only_initial_value(value):
    original={'log_jitter':value,'u':np.array([.2,.3])}
    actual=_repair_white_jitter_start(original,{'log_jitter':np.log(1e-4)})
    assert actual is not original
    assert actual['u'] is original['u']
    assert float(actual['log_jitter'])==np.log(1e-4)
    if np.isnan(value):assert np.isnan(original['log_jitter'])
    else:assert original['log_jitter']==value


def test_valid_and_custom_jitter_starts_remain_unchanged():
    interior={'log_jitter':np.log(1e-4)}
    assert _repair_white_jitter_start(interior,{}) is interior
    custom={'log_jitter':-30.}
    assert _repair_white_jitter_start(custom,{},jitter_spec=object()) is custom
    no_jitter={'u':np.array([.2,.3])}
    assert _repair_white_jitter_start(no_jitter,{}) is no_jitter


def test_repaired_actual_white_model_initializes_with_unchanged_jitter_density():
    specs=default_planet_specs(t0=0.,period=3.,b=.3,rprs=.1,param_method='a_rs',a_rs=9.)
    model=create_whitelight_model(detrend_type='none',param_method='a_rs',
        ld_mode='fixed',parameter_priors=specs)
    t=jnp.linspace(-.08,.08,31);args=(t,jnp.full_like(t,.001))
    kwargs=dict(y=jnp.ones_like(t),prior_params=dict(period=jnp.array([3.]),u=jnp.array([.4,.1]),ecc=jnp.array([0.]),omega=jnp.array([90.])))
    raw=dict(log_jitter=jnp.log(1e-5),rors_0=.1,t0_0=0.,_b_0=.3,log_a_rs_0=jnp.log(9.))
    tr=handlers.trace(handlers.substitute(handlers.seed(model,1),data=raw)).get_trace(*args,**kwargs)
    prior=tr['log_jitter']['fn']
    assert not np.isfinite(float(biject_to(prior.support).inv(prior.low)))
    repaired=_repair_white_jitter_start(raw,dict(log_jitter=jnp.log(1e-4)))
    info=initialize_model(jax.random.PRNGKey(1),model,init_strategy=init_to_value(values=repaired),model_args=args,model_kwargs=kwargs)
    assert np.isfinite(float(info.param_info.potential_energy))
    assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(info.param_info.z_grad))
    np.testing.assert_allclose([prior.low,prior.high],np.log([1e-5,1e-2]),rtol=0,atol=1e-14)
    np.testing.assert_allclose(prior.log_prob(repaired['log_jitter']),-np.log(np.log(1e-2)-np.log(1e-5)),atol=1e-14)
