"""Table accuracy and posterior-target safeguards for fixed PRISM geometry."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from models.jaxoplanet.power2_exact import light_curve
from models.jaxoplanet.power2_table import prepare_table, table_node_flux, duration_light_curve
from models.jaxoplanet.transit_grid import build_duration_transit_grid

jax.config.update('jax_enable_x64', True)


@pytest.fixture(scope='module')
def table():
    return prepare_table(impact=.45, radius_bounds=(.016, .276), num_nodes=769)


def test_table_nodes_and_gradients(table):
    rng = np.random.default_rng(265)
    params = np.column_stack((rng.uniform(.016, .276, 25), rng.uniform(0., 1., 25),
                              rng.uniform(0., 1., 25)))
    params[:4] = [[.016, .95, .001], [.276, .8, .1], [.146, 1., 0.], [.146, .6, 1.]]
    def direct(p):
        r,c,a = p
        z = build_duration_transit_grid(impact=.45, radius_ratio=r, num_nodes=769)
        return light_curve(c,a,z,r).at[-1].set(0.)
    candidate = lambda p: table_node_flux(table,*p)
    actual, actual_grad = jax.jit(jax.vmap(lambda p: (candidate(p),jax.jacfwd(candidate)(p))))(params)
    expected, expected_grad = jax.jit(jax.vmap(lambda p: (direct(p),jax.jacfwd(direct)(p))))(params)
    print('table max flux error ppm',np.max(np.abs(actual-expected))*1e6,
          'max gradient error',np.max(np.abs(actual_grad-expected_grad)))
    np.testing.assert_allclose(actual,expected,atol=2e-9,rtol=0)
    np.testing.assert_allclose(actual_grad,expected_grad,atol=2e-7,rtol=2e-6)


@pytest.mark.slow
def test_direct_fallback_preserves_prior(table):
    phases = jnp.linspace(-.06,.06,71)
    mask = jnp.abs(phases)<.06
    def evaluate(p,table):
        return duration_light_curve(*p,phases,mask,duration=.12,impact=.45,num_nodes=769,table=table)
    for p in [jnp.array([.0032,.8,.1]),jnp.array([.4,.7,.5]),jnp.array([.7,.6,.1]),
              jnp.array([.146,.5,3.])]:
        actual = jax.jit(lambda p: (evaluate(p,table),jax.jacfwd(lambda p:evaluate(p,table))(p)))(p)
        expected = jax.jit(lambda p: (evaluate(p,None),jax.jacfwd(lambda p:evaluate(p,None))(p)))(p)
        np.testing.assert_allclose(actual[0],expected[0],atol=1e-12,rtol=0)
        np.testing.assert_allclose(actual[1],expected[1],atol=1e-11,rtol=1e-10)


@pytest.mark.slow
def test_batched_lazy_fallback_and_gradients(table, monkeypatch):
    import models.jaxoplanet.power2_table as module
    calls = []
    original = module.light_curve
    def counted(*args, **kwargs):
        jax.debug.callback(lambda _: calls.append(int(args[2].size)), jnp.array(0))
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'light_curve', counted)
    phases = jnp.linspace(-.06,.06,31)
    mask = jnp.abs(phases)<.06
    def loss(p, table):
        return duration_light_curve(*p,phases,mask,duration=.12,impact=.45,num_nodes=769,table=table).sum()
    parameters = jnp.array([[.14,.7,.6],[.16,.8,.1]])
    candidate = jax.jit(jax.vmap(jax.value_and_grad(lambda p:loss(p,table))))
    actual = candidate(parameters)
    jax.block_until_ready(actual)
    jax.effects_barrier()
    assert not calls or max(calls) <= 16, 'Full-series fallback ran while every lane was in the table.'
    expected = jax.jit(jax.vmap(jax.value_and_grad(lambda p:loss(p,None))))(parameters)
    np.testing.assert_allclose(actual[0],expected[0],atol=1e-8,rtol=0)
    np.testing.assert_allclose(actual[1],expected[1],atol=2e-7,rtol=2e-6)
    calls.clear()
    outside = parameters.at[1,0].set(.4)
    actual = candidate(outside)
    jax.block_until_ready(actual)
    jax.effects_barrier()
    assert calls and max(calls) > 16, 'An out-of-domain lane did not invoke the direct fallback.'
    expected = jax.jit(jax.vmap(jax.value_and_grad(lambda p:loss(p,None))))(outside)
    np.testing.assert_allclose(actual[0],expected[0],atol=1e-12,rtol=0)
    np.testing.assert_allclose(actual[1],expected[1],atol=1e-11,rtol=1e-10)


def test_core_table_stays_lazy_across_chain_and_channel_axes(table, monkeypatch):
    import models.jaxoplanet.power2_table as module
    from models.jaxoplanet.core import compute_transit_model, build_transit_phase_offsets
    calls = []
    original = module.light_curve
    def counted(*args, **kwargs):
        jax.debug.callback(lambda _: calls.append(int(args[2].size)), jnp.array(0))
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'light_curve', counted)
    times = jnp.linspace(-.065, .065, 81)
    phase, mask = build_transit_phase_offsets(times, jnp.array([3.]), jnp.array([0.]), jnp.array([.12]))
    points = jnp.tile(jnp.array([[.14,.7,.6], [.16,.8,.1], [.12,.6,.3]])[None], (4,1,1))
    def loss(parameters, accelerated):
        within = jnp.all((parameters[:,0] >= .016) & (parameters[:,0] <= .276)
                         & (parameters[:,2] >= 0.) & (parameters[:,2] <= 1.))
        def lane(p):
            r,c,alpha = p
            params = dict(period=jnp.array([3.]), duration=jnp.array([.12]),
                          t0=jnp.array([0.]), b=jnp.array([.45]), rors=jnp.array([r]),
                          u=jnp.zeros(12), c1=c, c2=alpha, _power2_exact=True,
                          _transit_phase_offsets=phase, _transit_phase_mask=mask)
            if accelerated:
                params.update(_transit_grid_nodes=2049, _power2_table=table,
                              _power2_use_table=within)
            return compute_transit_model(params, times).sum()
        return jax.vmap(lane)(parameters).sum()
    candidate = jax.jit(jax.vmap(jax.value_and_grad(lambda p:loss(p, True))))
    reference = jax.jit(jax.vmap(jax.value_and_grad(lambda p:loss(p, False))))
    actual = candidate(points)
    jax.block_until_ready(actual)
    jax.effects_barrier()
    assert not calls or max(calls) <= 16, 'Full-series fallback ran inside nested chain/channel batching.'
    expected = reference(points)
    np.testing.assert_allclose(actual[0], expected[0], atol=3*81*2e-9, rtol=0)
    np.testing.assert_allclose(actual[1], expected[1], atol=81*2e-7, rtol=2e-6)


@pytest.mark.parametrize('impact,radius',[(0.,.146),(.45,.146),(.45,.01),(.8,.3),
                                        (.9,.1),(.900001,.1),(1.05,.146),(1.4,.146)])
def test_cadence_interpolation_against_direct_disk(impact,radius):
    from models.jaxoplanet.power2_exact import light_curve
    phases=jnp.concatenate((jnp.linspace(-.065,.065,501),jnp.array([-.06,0.,.06])))
    mask=jnp.abs(phases)<.06
    speed_squared=4*jnp.maximum(0.,(1+radius)**2-impact**2)/.12**2
    squared=impact**2+speed_squared*phases**2
    z=jnp.where(squared>0,jnp.sqrt(jnp.where(squared>0,squared,1.)),0.)
    for alpha in [.001,.1,.6,1.]:
        actual=jax.jit(lambda p:duration_light_curve(radius,.95,alpha,p,mask,
            duration=.12,impact=impact,num_nodes=2049))(phases)
        expected=jnp.where(mask,light_curve(.95,alpha,z,radius),0.)
        np.testing.assert_allclose(actual,expected,atol=2e-9,rtol=0)


@pytest.mark.slow
def test_validated_smaller_table_grid_and_larger_direct_fallback():
    from models.jaxoplanet.power2_table import validated_grid_size
    nodes = validated_grid_size(impact=.4498,radius_bounds=(.066,.226))
    assert 1025 <= nodes <= 4097
    table = prepare_table(impact=.4498,radius_bounds=(.066,.226),num_nodes=nodes)
    phases = jnp.linspace(-.06+1e-10,.06-1e-10,1001)
    mask = jnp.ones(phases.shape,dtype=bool)
    def candidate(p):
        return duration_light_curve(*p,phases,mask,duration=.12,impact=.4498,
            num_nodes=2049,table=table)
    def reference(p):
        r,c,alpha=p
        z=jnp.sqrt(.4498**2+4*((1+r)**2-.4498**2)/.12**2*phases**2)
        return light_curve(c,alpha,z,r)
    rng=np.random.default_rng(265)
    parameters=np.column_stack((rng.uniform(.066,.226,12),rng.uniform(0.,1.,12),
                                rng.uniform(0.,1.,12)))
    actual,gradient=jax.jit(jax.vmap(lambda p:(candidate(p),jax.jacfwd(candidate)(p))))(parameters)
    expected,expected_gradient=jax.jit(jax.vmap(lambda p:(reference(p),jax.jacfwd(reference)(p))))(parameters)
    np.testing.assert_allclose(actual,expected,atol=2e-9,rtol=0)
    np.testing.assert_allclose(gradient,expected_gradient,atol=2e-7,rtol=2e-6)
    outside=jnp.array([.7,.6,.1])
    exact=lambda p:duration_light_curve(*p,phases,mask,duration=.12,impact=.4498,num_nodes=2049)
    np.testing.assert_allclose(candidate(outside),exact(outside),atol=1e-12,rtol=0)
    np.testing.assert_allclose(jax.jacfwd(candidate)(outside),jax.jacfwd(exact)(outside),atol=1e-11,rtol=1e-10)


def test_compression_preserves_uniform_coefficient_error_bound(table):
    if 'moment_left' not in table:
        pytest.skip('This table does not benefit from compression.')
    left=table['moment_left'].reshape(-1,table['moment_left'].shape[-1])
    original=table['coefficients'].reshape(left.shape[0],-1)
    error=np.max(np.sum(np.abs(original-left@table['moment_right']),axis=0))
    assert error <= 5.1e-14
    assert left.size+table['moment_right'].size < original.size


def test_fallback_cadence_values_and_gradients_across_prior(table):
    phases=jnp.linspace(-.06+1e-10,.06-1e-10,501)
    mask=jnp.ones(phases.shape,dtype=bool)
    def candidate(p):
        return duration_light_curve(*p,phases,mask,duration=.12,impact=.45,
                                    num_nodes=2049,table=table)
    def reference(p):
        r,c,alpha=p
        z=jnp.sqrt(.45**2+4*jnp.maximum(0.,(1+r)**2-.45**2)*phases**2/.12**2)
        return light_curve(c,alpha,z,r)
    parameters=np.array([(r,.95,a) for r in (.0032,.4,.5499999,.55,.5500001,.7)
                         for a in (.001,.6,3.)])
    evaluate_candidate=jax.jit(lambda p:(candidate(p),jax.jacfwd(candidate)(p)))
    evaluate_reference=jax.jit(lambda p:(reference(p),jax.jacfwd(reference)(p)))
    # Evaluate separately: one coalescing-contact lane deliberately routes an
    # entire vmap to direct integration, hiding errors in the grid fallback.
    for p in parameters:
        actual,gradient=evaluate_candidate(p)
        expected,expected_gradient=evaluate_reference(p)
        np.testing.assert_allclose(actual,expected,atol=2e-9,rtol=0)
        np.testing.assert_allclose(gradient,expected_gradient,atol=2e-7,rtol=2e-6)
