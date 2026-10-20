"""Shared table banks preserve geometry, physical fallbacks and ownership."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from models.harmonica.core import harmonica_a_rs_from_duration
from koala.marginalization.tables import pack_tables
from koala.marginalization.batched_fitting import _lane_kwargs
from models.jaxoplanet.power2_table import prepare_table, table_node_flux, duration_light_curve


def test_selected_table_stencils_preserve_full_curve_and_parameter_jacobian():
    from models.jaxoplanet.power2_table import _table_cadence_flux
    from models.jaxoplanet.transit_grid import interpolate_duration_transit_grid
    table = prepare_table(impact=.71, radius_bounds=(.07, .13), num_nodes=4097)
    phase = jnp.asarray(np.unique(np.r_[np.linspace(-.051, .051, 79), -.05, 0., .05]))
    mask = jnp.abs(phase) <= .05
    def full(p):
        return interpolate_duration_transit_grid(table_node_flux(table, *p), phase, mask,
                                                 duration=.1, impact=.71, radius_ratio=p[0])
    def selected(p):
        return _table_cadence_flux(table, *p, phase, mask, duration=.1, impact=.71)
    values = jnp.array([[.07, 0., 0.], [.1, .6, .001], [.13, 1., 1.]])
    full_eval = jax.jit(jax.vmap(lambda p:(full(p), jax.jacfwd(full)(p))))
    local_eval = jax.jit(jax.vmap(lambda p:(selected(p), jax.jacfwd(selected)(p))))
    for actual, expected in zip(local_eval(values), full_eval(values)):
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
from models.jaxoplanet.power2_exact import light_curve


def test_bank_padding_preserves_each_compressed_table_and_invalid_domain():
    tables = [prepare_table(impact=.2, radius_bounds=(.08, .12), num_nodes=129,
                            radius_degree=6, exponent_degree=4),
              prepare_table(impact=.71, radius_bounds=(.07, .13), num_nodes=129), None]
    bank, shape = pack_tables(tables, 129)
    assert bank['impact'][2] < 0
    assert shape[-1] == 129
    for index, table in enumerate(tables[:2]):
        restored = {k:jnp.asarray(v[index]) for k,v in bank.items()}
        restored['coefficients'] = jnp.zeros(shape, dtype=jnp.uint8)
        point = jnp.array([.1, .6, .4])
        original = lambda p: table_node_flux(table, *p)
        shared = lambda p: table_node_flux(restored, *p)
        np.testing.assert_allclose(shared(point), original(point), atol=2e-12, rtol=0)
        np.testing.assert_allclose(jax.jacfwd(shared)(point), jax.jacfwd(original)(point),
                                   atol=2e-10, rtol=0)


def test_strict_outside_domain_fallback_is_direct_at_contacts_and_grazing():
    table = prepare_table(impact=.71, radius_bounds=(.07, .13), num_nodes=129)
    phases = jnp.array([-.5, -.5+1e-10, -.3, 0., .3, .5-1e-10, .5])
    mask = jnp.abs(phases) < .5
    def direct(p):
        r,c,alpha = p
        squared = .71**2 + 4*jnp.maximum(0., (1+r)**2-.71**2)*phases**2
        return jnp.where(mask, light_curve(c, alpha, jnp.sqrt(squared), r), 0.)
    def fallback(p):
        return duration_light_curve(*p, phases, mask, duration=1., impact=.71,
            num_nodes=129, table=table, strict_table_domain=True)
    for point in (jnp.array([.01,.6,.01]), jnp.array([.29,.6,.4]), jnp.array([.5,1.,1.])):
        np.testing.assert_allclose(fallback(point), direct(point), atol=2e-12, rtol=0)
        np.testing.assert_allclose(jax.jacrev(fallback)(point), jax.jacfwd(direct)(point),
                                   atol=2e-7, rtol=2e-6)


def test_chunks_keep_shared_bank_once_even_when_bank_count_equals_lane_count():
    bank = {'impact': np.array([.2, .3]), 'matrix': np.ones((2, 3, 4))}
    kwargs = dict(power2_table_bank=bank, table_index=np.array([0, 1]), free_ld=np.array([True, False]))
    sliced = _lane_kwargs(kwargs, slice(1, 2))
    assert sliced['power2_table_bank'] is bank
    np.testing.assert_array_equal(sliced['table_index'], [1])
    np.testing.assert_array_equal(sliced['free_ld'], [False])


@pytest.mark.parametrize('missing', [False, True])
@pytest.mark.parametrize('radii', [(.1, .1), (.1, .3)])
def test_model_bank_preserves_likelihood_with_distinct_lane_domains(missing, radii):
    from koala.marginalization.batched import create_batched_model
    from numpyro import handlers
    table = prepare_table(impact=.71, radius_bounds=(.07, .13), num_nodes=4097)
    bank, shape = pack_tables([table, None if missing else table], 4097)
    model = create_batched_model(law='power2', power2_grid_nodes=4097, power2_table_shape=shape)
    reference = create_batched_model(law='power2')
    t = jnp.linspace(-.06, .06, 31)
    errors, observed = jnp.full((2, len(t)), .001), jnp.ones((2, len(t)))
    kwargs = dict(quadratic_trend=jnp.array([False, True]), free_ld=jnp.array([True, False]),
        ld_fixed=jnp.array([[.5, .4], [.6, .7]]), PERIOD=jnp.full((2, 1), 3.),
        mu_duration=jnp.full((2, 1), .1), mu_t0=jnp.zeros((2, 1)),
        mu_b=jnp.full((2, 1), .71),
        mu_a_rs=jnp.full((2,1), harmonica_a_rs_from_duration(3., .1, .71, .1)))
    def prediction(radius):
        values = dict(ld_coordinates=jnp.array([[.5, .4], [.6, .7]]),
            trend_coordinates=jnp.array([[1., .003, 0.], [1., -.002, .004]]),
            rors=radius[:, None], log_jitter=jnp.full(2, -9.))
        def evaluate(model, extra):
            trace = handlers.trace(handlers.seed(handlers.substitute(model, data=values), 0))\
                .get_trace(t, errors, y=observed, **kwargs, **extra)
            return trace['obs']['fn'].loc
        return jnp.stack((evaluate(model, dict(power2_table_bank=bank, table_index=jnp.array([0, 1]))),
                          evaluate(reference, {})))
    radius = jnp.array(radii)
    predictions = jax.jit(prediction)(radius)
    np.testing.assert_allclose(predictions[0], predictions[1], atol=2e-12, rtol=0)
    gradient = jax.jit(jax.jacrev(prediction))(radius)
    np.testing.assert_allclose(gradient[0], gradient[1], atol=2e-7, rtol=2e-6)
