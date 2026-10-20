"""Supported ramp compositions preserve the full multiplicative model."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers
from koala.config import _resolve_spectro_fixed_timescale_trends,_spectro_detrend_type
from koala.constants import ADVANCED_FLAGS
from models.detrend import resolve_detrend_kernel
from models.jaxoplanet.builder import create_vectorized_model
from models.jaxoplanet.core import compute_transit_model
from models.cadence_reduction import linear_spectro_trend_coefficient_names

jax.config.update('jax_enable_x64',True)


@pytest.mark.parametrize('family',['spot_spectroscopic','2spot_spectroscopic','gp_spectroscopic'])
@pytest.mark.parametrize('fixed',[False,True])
@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('law',['quadratic','power2'])
def test_ramp_compositions_values_and_all_parameter_gradients(family,fixed,reverse,law):
    ramp='explinear_spectroscopic' if fixed else 'explinear'
    trend='+'.join([ramp,family] if reverse else [family,ramp])
    model=create_vectorized_model(param_method='a_rs',detrend_type=trend,ld_profile=law,ld_mode='fixed')
    t=jnp.linspace(-.06,.06,17);dt=t-t.min()
    exp=.1+.9*jnp.exp(-dt/.04)  # supplied shape, not regenerated from fixed_tau
    spot=.01*jnp.exp(-((t-.01)/.02)**2);spot2=.02*jnp.exp(-((t+.02)/.015)**2)
    gp=.001*jnp.sin(20*t)
    ld=jnp.array([[.4,.2],[.4,.2]] if law=='quadratic' else [[.6,.4],[.6,.4]])
    point=dict(rors=jnp.array([[.18],[.19]]),log_jitter=jnp.array([-8.,-8.]),
               c=jnp.array([1.01,.99]),v=jnp.array([.002,-.003]),A=jnp.array([.02,-.01]),
               log_tau=jnp.log(jnp.array([.03,.04])),A_spot=jnp.array([.8,1.2]),
               A_spot2=jnp.array([1.1,.9]),A_gp=jnp.array([.9,1.1]))
    kwargs=dict(PERIOD=jnp.array([3.3]),mu_t0=jnp.array([0.]),mu_a_rs=jnp.array([13.6]),
                mu_b=jnp.array([13.6*jnp.cos(jnp.deg2rad(88.))]),ld_fixed=ld,spot_trend=spot,spot_trend2=spot2,
                gp_trend=gp,exp_trend=exp,fixed_tau=.031)
    weights=jnp.linspace(-1.,1.,2*len(t)).reshape(2,-1)
    def candidate(p):
        tr=handlers.trace(handlers.seed(handlers.substitute(model,data=p),0)).get_trace(
            t,jnp.full((2,len(t)),.001),**kwargs)
        if fixed:assert 'log_tau' not in tr
        return tr['obs']['fn'].loc
    def reference(p):
        signal=[]
        for lane in range(2):
            params=dict(period=jnp.array([3.3]),t0=jnp.array([0.]),a_rs=jnp.array([13.6]),
                        b=jnp.array([13.6*jnp.cos(jnp.deg2rad(88.))]),ecc=jnp.array([0.]),omega=jnp.array([0.]),
                        rors=p['rors'][lane],u=ld[lane],_ld_profile=law,
                        _jaxoplanet_kernel='quadratic_contact_safe' if law=='quadratic' else 'auto',
                        _quadratic_radius_bound=.5**.5)
            if law=='power2':params.update(_power2_exact=True,c1=ld[lane,0],c2=ld[lane,1],u=jnp.zeros(12))
            signal.append(compute_transit_model(params,t))
        shape=exp[None,:] if fixed else jnp.exp(-dt[None,:]/jnp.exp(p['log_tau'])[:,None])
        baseline=p['c'][:,None]+p['v'][:,None]*dt+p['A'][:,None]*shape
        if family=='gp_spectroscopic':baseline=baseline+p['A_gp'][:,None]*gp
        else:
            baseline=baseline+p['A_spot'][:,None]*spot
            if family=='2spot_spectroscopic':baseline=baseline+p['A_spot2'][:,None]*spot2
        return (1+jnp.stack(signal))*baseline
    actual=jax.jit(candidate)(point);expected=jax.jit(reference)(point)
    np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=0)
    a=jax.jit(jax.grad(lambda p:jnp.sum(weights*candidate(p))))(point)
    b=jax.jit(jax.grad(lambda p:jnp.sum(weights*reference(p))))(point)
    for name in point:np.testing.assert_allclose(a[name],b[name],atol=1e-7,rtol=1e-6)


def test_public_timescale_selector_and_exact_fixed_basis_contract():
    assert 'spectro_fixed_timescale_trends' in ADVANCED_FLAGS
    assert _resolve_spectro_fixed_timescale_trends({},'explinear','jaxoplanet')
    for engine in ['jaxoplanet','harmonica']:
        assert not _resolve_spectro_fixed_timescale_trends(
            {'spectro_fixed_timescale_trends':False},'explinear',engine)
    with pytest.raises(ValueError,match='false.*sampled tau'):
        _resolve_spectro_fixed_timescale_trends({},'explinear','harmonica')
    for family in ['explinear+gp','spot+explinear','2spot+explinear']:
        for fixed in [False,True]:
            assert callable(resolve_detrend_kernel(_spectro_detrend_type(family,fixed)))
    assert linear_spectro_trend_coefficient_names('spot_spectroscopic+explinear_spectroscopic')==('c','v','A','A_spot')
    assert linear_spectro_trend_coefficient_names('2spot_spectroscopic+explinear_spectroscopic')==('c','v','A','A_spot','A_spot2')


@pytest.mark.parametrize('value',[None,'false',0,1])
def test_public_timescale_selector_rejects_non_boolean_values(value):
    with pytest.raises(ValueError,match='must be Boolean'):
        _resolve_spectro_fixed_timescale_trends({'spectro_fixed_timescale_trends':value},'explinear','jaxoplanet')
