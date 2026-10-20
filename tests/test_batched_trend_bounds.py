"""Explicit spectral prior contracts survive augmented variant batching."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers
from koala.marginalization.batched import create_batched_model
from models.jaxoplanet.core import compute_transit_model
jax.config.update('jax_enable_x64',True)


@pytest.mark.parametrize('law',['quadratic','power2'])
@pytest.mark.parametrize('basis',['elapsed_days','batman_normalized'])
@pytest.mark.parametrize('step',[False,True])
def test_batched_explicit_bounds_mixed_degree_values_and_all_gradients(law,basis,step):
    bounds={name:(-10.,10.) for name in ['c','v','v2','A_jump']}
    model=create_batched_model(law=law,power2_parameterization='maxted_smooth' if law=='power2' else 'coefficients',stage='highres',step_trend=step,trend_basis=basis,trend_bounds=bounds)
    t=jnp.linspace(-.06,.06,17);dt=t-t.min();x=(t-t.mean())/jnp.ptp(t)
    templates=jnp.stack([.003*jax.nn.sigmoid((t-.01)/.003),.005*jax.nn.sigmoid((t+.01)/.004)])
    ld=jnp.array([[.4,.2],[.4,.2]] if law=='quadratic' else [[.6,.4],[.6,.4]])
    kwargs=dict(quadratic_trend=jnp.array([True,False]),free_ld=jnp.array([False,False]),ld_fixed=ld,
                mu_duration=jnp.array([.15]),PERIOD=jnp.array([3.3]),mu_t0=jnp.array([0.]),mu_a_rs=jnp.array([13.6]),
                mu_b=jnp.array([13.6*jnp.cos(jnp.deg2rad(88.))]),jump_template_bank=templates,step_index=jnp.array([0,1]))
    point=dict(rors=jnp.array([[.18],[.19]]),log_jitter=jnp.array([-8.,-8.]),
               ld_coordinates=jnp.array([[.4,.2],[.4,.2]]),
               trend_coordinates=jnp.array([[1.01,.02,.03],[.99,-.03,9.]]),A_jump=jnp.array([3.,-3.]))
    def trace(p):return handlers.trace(handlers.seed(handlers.substitute(model,data=p),711)).get_trace(t,jnp.full((2,17),.001),**kwargs)
    observed=trace(point);fn=observed['trend_coordinates']['fn']
    np.testing.assert_allclose(fn.log_prob(point['trend_coordinates']),jnp.full(2,-3*np.log(20.)),atol=1e-12,rtol=0.)
    if step:
        fn=observed['A_jump']['fn']
        while hasattr(fn,'base_dist'):fn=fn.base_dist
        np.testing.assert_array_equal([fn.low,fn.high],[-10.,10.])
    def actual(p):
        fn=trace(p)['obs']['fn']
        while not hasattr(fn,'loc'):fn=fn.base_dist
        return fn.loc
    def reference(p):
        signals=[]
        for lane in range(2):
            q=dict(period=jnp.array([3.3]),t0=jnp.array([0.]),a_rs=jnp.array([13.6]),
                   b=jnp.array([13.6*jnp.cos(jnp.deg2rad(88.))]),rors=p['rors'][lane],u=ld[lane],
                   _ld_profile=law,_jaxoplanet_kernel='quadratic_contact_safe' if law=='quadratic' else 'auto',
                   _quadratic_radius_bound=.5**.5)
            if law=='power2':q.update(u=jnp.zeros(12),c1=ld[lane,0],c2=ld[lane,1],_power2_exact=True)
            signals.append(1+compute_transit_model(q,t))
        time=x if basis=='batman_normalized' else dt
        beta=p['trend_coordinates'];baseline=beta[:,0,None]+beta[:,1,None]*time
        # Step templates use the ordinary linear-plus-step family; v2 is auxiliary.
        baseline=baseline+(jnp.array([True,False]) & (not step))[:,None]*beta[:,2,None]*time**2
        if step:baseline=baseline+p['A_jump'][:,None]*templates
        return jnp.stack(signals)*baseline
    np.testing.assert_allclose(jax.jit(actual)(point),jax.jit(reference)(point),atol=1e-9,rtol=0.)
    weights=jnp.linspace(-1.,1.,34).reshape(2,17)
    a=jax.jit(jax.grad(lambda p:jnp.sum(weights*actual(p))))(point)
    b=jax.jit(jax.grad(lambda p:jnp.sum(weights*reference(p))))(point)
    for name in point:np.testing.assert_allclose(a[name],b[name],atol=1e-7,rtol=1e-6)


def test_variant_groups_separate_actual_prior_bounds_and_basis():
    from koala.marginalization.batched_fitting import _spectral_trend_group
    group=lambda flags:_spectral_trend_group(SimpleNamespace(cfg={'flags':flags}))
    assert group({})!=group({'spectro_trend_basis':'batman_normalized'})
    a={'spectro_trend_bounds':{'A_jump':[-10,10],'c':[.5,1.5]}}
    b={'spectro_trend_bounds':{'c':(.5,1.5),'A_jump':(-10.,10.)}}
    c={'spectro_trend_bounds':{'c':[.5,1.5],'A_jump':[-30.,30.]}}
    assert group(a)==group(b) and group(a)!=group(c)


def test_batched_unsupported_prior_controls_fail_before_sampling():
    with pytest.raises(ValueError,match='ordinary grid'):
        create_batched_model(trend_bounds={'A_spot':[-10.,10.]})
    with pytest.raises(ValueError,match='ordinary grid'):
        create_batched_model(stage='white',trend_bounds={'c':[-10.,10.]})
