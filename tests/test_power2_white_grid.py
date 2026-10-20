"""Moving-geometry power-2 grid preserves the direct cadence target."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from models.jaxoplanet.core import compute_transit_model
from models.jaxoplanet.builder import create_whitelight_model
from koala.config import parse_planet_parameter_specs
from numpyro import handlers

jax.config.update('jax_enable_x64',True)


def evaluate(p,t,grid):
    r,c,a,b,d,t0=p
    params=dict(period=jnp.array([3.]),duration=jnp.array([d]),t0=jnp.array([t0]),
        b=jnp.array([b]),rors=jnp.array([r]),u=jnp.zeros(12),c1=c,c2=a,_power2_exact=True)
    if grid:params['_transit_grid_nodes']=4097
    return compute_transit_model(params,t)


_TIMES=jnp.linspace(-.065,.065,1001)
_WEIGHTS=jnp.linspace(.9,1.1,len(_TIMES))
_CANDIDATE=jax.jit(lambda p:(evaluate(p,_TIMES,True),jax.jacfwd(lambda p:evaluate(p,_TIMES,True))(p)))
_REFERENCE=jax.jit(lambda p:(evaluate(p,_TIMES,False),jax.jacfwd(lambda p:evaluate(p,_TIMES,False))(p)))
_REVERSE=jax.jit(jax.grad(lambda p:jnp.sum(_WEIGHTS*evaluate(p,_TIMES,True))))


@pytest.mark.parametrize('r,b',[(.01,0.),(.066,.45),(.106,.45),(.146,.45),(.186,.45),
                              (.226,.45),(.146,-.45),(.1,-.9),(.4,.2),(.1,.9),(.1,.8999999),(.1,.9000001),(.146,1.05),(.146,1.4),(.001,.45),(1.2,.1)])
def test_moving_geometry_values_forward_and_reverse_gradients(r,b):
    candidate,reference=_CANDIDATE,_REFERENCE
    for alpha in [0.,.001,.1,.6,1.]:
        p=jnp.array([r,.95,alpha,b,.12,.000123])
        actual,gradient=candidate(p);expected,expected_gradient=reference(p)
        np.testing.assert_allclose(actual,expected,atol=2e-9,rtol=0)
        np.testing.assert_allclose(gradient,expected_gradient,atol=2e-7,rtol=2e-6)
        reverse=_REVERSE(p)
        np.testing.assert_allclose(reverse,_WEIGHTS@gradient,atol=2e-7,rtol=2e-6)


def test_long_white_light_factory_uses_direct_integral_grid():
    specs=parse_planet_parameter_specs({k:dict(value=v,prior='fixed') for k,v in
        dict(period=3.,duration=.12,t0=0.,b=.45,rprs=.146).items()})
    model=create_whitelight_model(ld_mode='fixed',ld_profile='power2',parameter_priors=specs)
    t=jnp.linspace(-.08,.08,5001)
    trace=handlers.trace(handlers.seed(model,0)).get_trace(t,jnp.full(t.shape,.001),y=jnp.ones_like(t),
        prior_params=dict(u=jnp.array([.6,.6]),c=1.,v=0.))
    expected=(1+evaluate(jnp.array([.146,.6,.6,.45,.12,0.]),t,False))* (
        trace['c']['value']+trace['v']['value']*(t-jnp.min(t)))
    np.testing.assert_allclose(trace['obs']['fn'].loc,expected,atol=2e-9,rtol=0)


def test_vectorized_chain_gradient_never_uses_duration_interpolation(monkeypatch):
    import models.jaxoplanet.power2_table as module
    calls = []
    import models.jaxoplanet.power2_exact as exact
    def reject(*args, **kwargs):
        raise AssertionError('Fixed-T14 duration interpolation is not a physical orbit.')
    monkeypatch.setattr(module, 'duration_light_curve', reject)
    original = exact.light_curve
    def counted(*args, **kwargs):
        jax.debug.callback(lambda _: calls.append(True), jnp.array(0))
        return original(*args, **kwargs)
    monkeypatch.setattr(exact, 'light_curve', counted)
    times = jnp.linspace(-.065, .065, 81)
    points = jnp.array([[.146,.6,.6,.45,.12,0.], [.106,.8,.1,.45,.12,0.],
                        [.186,.7,.3,.45,.12,0.], [.01,.9,.001,.45,.12,0.]])
    candidate = jax.jit(jax.vmap(jax.value_and_grad(
        lambda p: evaluate(p, times, True).sum())))
    reference = jax.jit(jax.vmap(jax.value_and_grad(
        lambda p: evaluate(p, times, False).sum())))
    actual = candidate(points)
    jax.block_until_ready(actual)
    jax.effects_barrier()
    assert calls, 'Exact physical quadrature must run even with legacy grid metadata.'
    expected = reference(points)
    np.testing.assert_allclose(actual[0], expected[0], atol=81*2e-9, rtol=0)
    np.testing.assert_allclose(actual[1], expected[1], atol=81*2e-7, rtol=2e-6)
    jax.effects_barrier()
    calls.clear()
    outside = points.at[0,0].set(.001)
    actual = candidate(outside)
    jax.block_until_ready(actual)
    jax.effects_barrier()
    assert calls, 'A small-radius chain failed to execute physical quadrature.'
    expected = reference(outside)
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-10, rtol=1e-10)


def test_long_direct_cadence_values_and_gradients_match_vector_integral():
    from models.jaxoplanet.power2_exact import light_curve
    from models.jaxoplanet.core import build_transit_phase_offsets
    times=jnp.linspace(-.065,.065,5003)
    point=jnp.array([.146,.8,.1,.45,.12,.000123])
    def reference(p):
        r,c,a,b,d,t0=p
        phase,mask=build_transit_phase_offsets(times,jnp.array([3.]),
                                              jnp.array([t0]),jnp.array([d]))
        axis_squared=b*b+((1+r)**2-b*b)/jnp.sin(jnp.pi*d/3.)**2
        separation=jnp.sqrt(b*b+(axis_squared-b*b)*jnp.sin(2*jnp.pi*(times-t0)/3.)**2)
        return light_curve(c,a,separation,r)
    candidate=lambda p:evaluate(p,times,False)
    actual=jax.jit(candidate)(point)
    expected=jax.jit(reference)(point)
    np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=0)
    weights=jnp.linspace(.9,1.1,len(times))
    actual_grad=jax.jit(jax.grad(lambda p:weights@candidate(p)))(point)
    expected_grad=jax.jit(jax.grad(lambda p:weights@reference(p)))(point)
    np.testing.assert_allclose(actual_grad,expected_grad,atol=2e-7,rtol=2e-6)
