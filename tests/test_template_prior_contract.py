"""Actual template priors and multiplicative polynomial blocks, both engines."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers
from models.jaxoplanet import builder as jp
from models.harmonica import builder as hm
from models.spectro_trends import validate_spectral_trend_bounds
jax.config.update('jax_enable_x64',True)
BOUNDS={name:(-10.,10.) for name in ['c','v','v2','A','A_spot','A_spot2','A_jump','A_gp']}
BOUNDS['log_tau']=(-8.,2.)
FAMILIES=['spot_spectroscopic','2spot_spectroscopic','linear_discontinuity_spectroscopic',
          'spot_spectroscopic+linear_discontinuity_spectroscopic','linear+gp_spectroscopic','explinear']


def location(fn):
    while not hasattr(fn,'loc'):fn=fn.base_dist
    return fn.loc


@pytest.mark.parametrize('engine',['jp','hm'])
@pytest.mark.parametrize('basis',['elapsed_days','batman_normalized'])
@pytest.mark.parametrize('family',FAMILIES)
def test_actual_sampled_template_bounds_baseline_and_all_gradients(engine,basis,family):
    factory=jp.create_vectorized_model if engine=='jp' else hm.create_vectorized_model
    options=dict(ld_profile='quadratic',ld_mode='fixed')
    if engine=='hm':options['odd_parameterization']='delta_r'
    else:options['param_method']='a_rs'
    bounds={k:v for k,v in BOUNDS.items() if k!='log_tau' or family=='explinear'}
    model=factory(detrend_type=family,trend_basis=basis,trend_bounds=bounds,**options)
    bare=factory(detrend_type='none',**options)
    t=jnp.linspace(-.06,.06,17);dt=t-t.min();x=(t-t.mean())/jnp.ptp(t)
    templates=dict(spot_trend=.002*jnp.exp(-((t-.01)/.02)**2),
                   spot_trend2=.003*jnp.exp(-((t+.02)/.015)**2),
                   jump_trend=.004*jax.nn.sigmoid((t-.01)/.003),gp_trend=.001*jnp.sin(20*t))
    kw=dict(PERIOD=jnp.array([3.3]),mu_t0=jnp.array([0.]),
            mu_b=jnp.array([13.6*jnp.cos(jnp.deg2rad(88.))]),ld_fixed=jnp.array([[.4,.2],[.4,.2]]),**templates)
    if engine=='jp':kw['mu_a_rs']=jnp.array([13.6])
    else:kw['harmonica_a_rs']=jnp.array([13.6])
    prefix='trend_' if basis=='batman_normalized' else ''
    point=dict(rors=jnp.array([[.18],[.19]]),log_jitter=jnp.array([-8.,-8.]),delta_r=jnp.zeros((2,1)),
               A_spot=jnp.array([3.,2.5]),A_spot2=jnp.array([-2.,-3.]),A_jump=jnp.array([4.,3.]),
               A_gp=jnp.array([-3.,-2.]),A=jnp.array([.02,-.01]),log_tau=jnp.log(jnp.array([.03,.04])))
    point[prefix+'c']=jnp.array([1.01,.99]);point[prefix+'v']=jnp.array([.02,-.03])
    def traced(model,p):
        return handlers.trace(handlers.seed(handlers.substitute(model,data=p),708)).get_trace(t,jnp.full((2,17),.001),**kw)
    trace=traced(model,point)
    for name,pair in bounds.items():
        site=(prefix+name if name in ['c','v','v2'] else name)
        if site in trace and trace[site]['type']=='sample':
            fn=trace[site]['fn']
            while hasattr(fn,'base_dist'):fn=fn.base_dist
            np.testing.assert_array_equal([fn.low,fn.high],pair)
    def actual(p):return location(traced(model,p)['obs']['fn'])
    def expected(p):
        signal=location(traced(bare,p)['obs']['fn'])
        time=x if basis=='batman_normalized' else dt
        baseline=p[prefix+'c'][:,None]+p[prefix+'v'][:,None]*time
        for component,name,template in [('spot_spectroscopic','A_spot','spot_trend'),
             ('2spot_spectroscopic','A_spot','spot_trend'),('linear_discontinuity_spectroscopic','A_jump','jump_trend'),
             ('gp_spectroscopic','A_gp','gp_trend')]:
            if component in family.split('+'):baseline=baseline+p[name][:,None]*templates[template]
        if '2spot_spectroscopic' in family:baseline=baseline+p['A_spot2'][:,None]*templates['spot_trend2']
        if family=='explinear':baseline=baseline+p['A'][:,None]*jnp.exp(-dt/jnp.exp(p['log_tau'])[:,None])
        return signal*baseline
    np.testing.assert_allclose(jax.jit(actual)(point),jax.jit(expected)(point),atol=1e-9,rtol=0.)
    weights=jnp.linspace(-1.,1.,34).reshape(2,17)
    a=jax.jit(jax.grad(lambda p:jnp.sum(weights*actual(p))))(point)
    b=jax.jit(jax.grad(lambda p:jnp.sum(weights*expected(p))))(point)
    for name in point:np.testing.assert_allclose(a[name],b[name],atol=1e-7,rtol=1e-6)


@pytest.mark.parametrize('bad',[False,[],{'unknown':(-1.,1.)},{'A_jump':(1.,1.)},
    {'A_jump':(2.,1.)},{'A_spot':(True,2.)},{'log_tau':(float('nan'),1.)},{'A':(0.,float('inf'))}])
def test_invalid_spectral_bounds_fail_explicitly(bad):
    with pytest.raises(ValueError,match='trend_bounds'):validate_spectral_trend_bounds(bad)


def test_unsupported_fixed_ramp_and_marginal_uniform_bounds_fail():
    with pytest.raises(ValueError,match='sampled tau or the jaxoplanet'):
        hm.create_vectorized_model(detrend_type='explinear_spectroscopic',ld_profile='quadratic',ld_mode='fixed')
    with pytest.raises(ValueError,match='trend_prior_scale'):
        jp.create_vectorized_model(trend_mode='gaussian_marginalized',trend_bounds={'A_jump':(-10.,10.)})


@pytest.mark.parametrize('mode',['fixed','gaussian_marginalized'])
def test_harmonica_unsupported_trend_modes_cannot_be_silently_sampled(mode):
    with pytest.raises(ValueError,match='use jaxoplanet'):
        hm.create_vectorized_model(trend_mode=mode,ld_profile='quadratic',ld_mode='fixed')
