"""A curved geometry coordinate must preserve the full posterior density."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from koala.white_light import _FoldedImpactNUTS, _sample_white_light, _white_light_geometry_coordinates
from models.independent_nuts import LaplaceMetricPreparation


def _preparation(width=.2):
    center = {'_b_0': jnp.array(.5), 'vector': jnp.array([1.25, 0.])}
    def potential(position):
        b, vector = position['_b_0'], position['vector']
        return .5*((b-.5)/width)**2 + .5*((vector[0]-5*b*b)/.1)**2 + .5*(vector[1]/.3)**2
    variance = width**2
    covariance = jnp.array([[variance, 5*variance, 0.],
                            [5*variance, 25*variance+.01, 0.], [0., 0., .09]])
    return LaplaceMetricPreparation(
        center, covariance, jnp.array(0.), jnp.array(0.), jnp.array(0),
        jnp.array(1.), jnp.array(1.), potential_fn=potential,
        postprocess_fn=lambda position: position,
    )


def test_quadratic_shear_has_unit_jacobian_and_independent_gaussian_density():
    original = _preparation()
    transformed, active, diagnostics = _white_light_geometry_coordinates(original)
    assert active and diagnostics['laplace_b_quadratic_shear']
    flat, unravel = ravel_pytree(original.unconstrained_map)
    for b in (-.3, 0., .2, .7):
        point = {'_b_0': jnp.array(b), 'vector': jnp.array([1.3, -.2])}
        mapped = transformed.postprocess_fn(point)
        np.testing.assert_allclose(transformed.potential_fn(point), original.potential_fn(mapped), atol=1e-12)
        expected = .5*((b-.5)/.2)**2 + .5*((1.3-1.25)/.1)**2 + .5*(-.2/.3)**2
        np.testing.assert_allclose(transformed.potential_fn(point), expected, atol=1e-12)
        jacobian = jax.jacfwd(lambda x: ravel_pytree(transformed.postprocess_fn(unravel(x)))[0])(
            ravel_pytree(point)[0])
        np.testing.assert_allclose(jnp.linalg.det(jacobian), 1., atol=1e-12)
    np.testing.assert_allclose(transformed.inverse_mass_matrix, jnp.diag(jnp.array([.04, .01, .09])), atol=1e-14)
    assert transformed.unconstrained_map is original.unconstrained_map


def test_narrow_geometry_keeps_original_preparation():
    original = _preparation(width=.01)
    actual, active, _ = _white_light_geometry_coordinates(original)
    assert not active and actual is original


def test_production_white_caller_recovers_curved_posterior(tmp_path):
    def model(t, error, y=None, prior_params=None):
        b = numpyro.sample('_b_0', dist.Normal(.5, .2))
        numpyro.sample('vector', dist.Normal(jnp.array([5*b*b, 0.]), jnp.array([.1, .3])).to_event(1))
    context = SimpleNamespace(
        cfg={'flags': {}}, ld_profile='quadratic', ld_prior_mode='uniform',
        ld_uniform_basis='uplus_uminus', transit_engine='jaxoplanet',
        NUTS_KWARGS={}, key_master=jax.random.PRNGKey(73), uses_surface_model=False,
        whitelight_laplace_options=dict(mass_matrix='laplace', hessian_method='finite_difference',
            trust_radius=5., warmup=200, target_accept=.9, max_tree_depth=10),
        data=SimpleNamespace(wl_time=jnp.arange(3.), wl_flux_err=jnp.ones(3), wl_flux=jnp.zeros(3)),
        output_dir=str(tmp_path),
    )
    samples = _sample_white_light(
        context=context, hyper_params_wl={}, science_sites={'b': '_b_0', 'vector': 'vector'},
        soln={'_b_0': .5, 'vector': jnp.array([1.25, 0.])},
        whitelight_mcmc_kwargs=dict(num_chains=4, chain_method='vectorized', num_samples=1000, num_warmup=200),
        whitelight_model_for_run=model,
    )
    # E[b^2] = E[b]^2 + Var[b]; the returned vector must be in original
    # coordinates, not the Gaussian coordinates used internally by NUTS.
    np.testing.assert_allclose(np.mean(samples['_b_0']), .5, atol=.02)
    np.testing.assert_allclose(np.mean(samples['vector'], axis=0), [1.45, 0.], atol=.08)
    import json
    diagnostics = json.loads((tmp_path/'whitelight_mcmc_diagnostics.json').read_text())
    assert diagnostics['laplace_b_quadratic_shear']
    assert diagnostics['whitelight_quality_gate_passed']
    assert diagnostics['num_divergences'] == 0
    assert not diagnostics.get('whitelight_laplace_fell_back', False)


@pytest.mark.parametrize('chain_method', ['vectorized', 'sequential'])
def test_reflections_mix_both_signs_without_changing_physical_moments(chain_method):
    def potential(position):
        b = position['_b_0']
        return -jax.scipy.special.logsumexp(-.5*((b-jnp.array([-1., 1.]))/.03)**2) + .5*position['x']**2

    sampler = numpyro.infer.MCMC(
        _FoldedImpactNUTS(potential_fn=potential, dense_mass=True,
            inverse_mass_matrix=jnp.diag(jnp.array([.03**2, 1.])), target_accept_prob=.9),
        num_warmup=100, num_samples=400, num_chains=4,
        chain_method=chain_method, progress_bar=False,
    )
    sampler.run(jax.random.PRNGKey(19), init_params={'_b_0': jnp.ones(4), 'x': jnp.zeros(4)},
                extra_fields=('diverging',))
    b = np.asarray(sampler.get_samples(group_by_chain=True)['_b_0'])
    # Every chain starts in the positive mode. The negative copy is many
    # local standard deviations away and must be reached by a valid move.
    assert np.all((np.mean(b > 0, axis=1) > .4) & (np.mean(b > 0, axis=1) < .6))
    np.testing.assert_allclose(np.mean(b**2), 1.0009, atol=.01)
    assert int(np.sum(sampler.get_extra_fields()['diverging'])) == 0
