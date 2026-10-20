"""An exact internal contact must retain movement across the grid segments."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from models.jaxoplanet.power2_exact import light_curve
from models.jaxoplanet.power2_table import duration_light_curve, prepare_table

jax.config.update('jax_enable_x64', True)


def test_nested_domain_decision_keeps_the_fallback_lazy():
    from models.jaxoplanet.power2_table import _all_lanes_in_domain
    calls = []
    def lane(x):
        def fallback(_):
            jax.debug.callback(lambda _:calls.append(True), jnp.array(0))
            return x-1
        return jax.lax.cond(_all_lanes_in_domain(x > 0),lambda _:x+1,fallback,None)
    evaluate = jax.jit(jax.vmap(jax.vmap(lane)))
    values = jnp.ones((4,3))
    actual = evaluate(values)
    jax.block_until_ready(actual)
    jax.effects_barrier()
    assert not calls
    np.testing.assert_array_equal(actual,values+1)
    values = values.at[1,0].set(-1)
    actual = evaluate(values)
    jax.block_until_ready(actual)
    jax.effects_barrier()
    assert calls
    np.testing.assert_array_equal(actual,values-1)


@pytest.mark.slow
@pytest.mark.parametrize('use_table', [False, True])
@pytest.mark.parametrize('impact,radius', [(0., .5), (.4498, .1457), (.8, .1)])
def test_internal_contact_radius_derivative_matches_physics_and_finite_difference(use_table, impact, radius):
    contact = .5*np.sqrt(((1-radius)**2-impact**2)/((1+radius)**2-impact**2))
    phases = jnp.asarray([-contact, -contact+1e-10, 0., contact-1e-10, contact])
    table = prepare_table(impact=impact, radius_bounds=(radius-.02, radius+.02), num_nodes=4097) if use_table else None
    def reference(p):
        r,c,a = p
        squared = impact**2+4*((1+r)**2-impact**2)*phases**2
        z = jnp.where(squared > 0., jnp.sqrt(jnp.where(squared > 0., squared, 1.)), 0.)
        return light_curve(c,a,z,r)
    def candidate(p):
        return duration_light_curve(*p, phases, jnp.ones(phases.shape, dtype=bool),
            duration=1., impact=impact, num_nodes=4097, table=table, strict_table_domain=True)
    for c,a in [(0.,0.), (1.,.001), (.6,.4), (1.,1.)]:
        p = jnp.asarray([radius,c,a])
        expected = jax.jit(jax.jacfwd(reference))(p)
        actual = jax.jit(jax.jacfwd(candidate))(p)
        reverse = jax.jit(jax.jacrev(candidate))(p)
        np.testing.assert_allclose(candidate(p), reference(p), atol=2e-9, rtol=0)
        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)
        np.testing.assert_allclose(reverse, expected, atol=2e-7, rtol=2e-6)
    # alpha=1 regularizes the limb sufficiently for a central difference.
    p = jnp.asarray([radius,1.,1.])
    delta = jnp.asarray([1e-6,0.,0.])
    finite_difference = (reference(p+delta)-reference(p-delta))/(2e-6)
    np.testing.assert_allclose(jax.jacfwd(candidate)(p)[:,0], finite_difference,
                               atol=2e-7, rtol=2e-6)
