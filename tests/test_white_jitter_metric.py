"""A conditional metric slice must preserve the original posterior target."""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from scipy.integrate import quad
from scipy.special import expit

from koala.white_light import _white_light_jitter_metric
from models.independent_nuts import prepare_laplace_metric


def test_jitter_metric_matches_independent_integral_and_preserves_target():
    low, high = np.log(1e-5), np.log(1e-2)
    def model():
        log_jitter = numpyro.sample('log_jitter', dist.Uniform(low, high))
        numpyro.sample('a_vector', dist.Normal(0., 1.).expand([2]).to_event(1))
        numpyro.factor('likelihood', -.5 * (jnp.exp(log_jitter) / 3e-5)**2)

    original = prepare_laplace_metric(
        model, jax.random.PRNGKey(3), {'log_jitter': np.log(3e-5), 'a_vector': jnp.zeros(2)},
        hessian_method='exact', gradient_tolerance=1e-8, decrement_tolerance=1e-8,
    )
    calibrated, diagnostics = _white_light_jitter_metric(original)
    def density(z):
        fraction = expit(z)
        return fraction * expit(-z) * np.exp(-.5 * (np.exp(low + (high-low)*fraction)/3e-5)**2)
    normalization = quad(density, -35., 20., epsabs=1e-12)[0]
    mean = quad(lambda z: z*density(z), -35., 20., epsabs=1e-12)[0] / normalization
    variance = quad(lambda z: (z-mean)**2*density(z), -35., 20., epsabs=1e-12)[0] / normalization
    np.testing.assert_allclose(diagnostics['laplace_jitter_slice_variance'], variance, rtol=5e-5)
    assert diagnostics['laplace_jitter_metric_variance'] > diagnostics['laplace_jitter_local_variance']
    assert calibrated.potential_fn is original.potential_fn
    assert calibrated.unconstrained_map is original.unconstrained_map
    assert np.linalg.eigvalsh(calibrated.inverse_mass_matrix).min() > 0.
    # Jitter is the second site but the third flat coordinate.
    np.testing.assert_array_equal(calibrated.inverse_mass_matrix[:2, :2], original.inverse_mass_matrix[:2, :2])
    for z in (-8., -2., 1.):
        point = {**original.unconstrained_map, 'log_jitter': jnp.asarray(z)}
        np.testing.assert_array_equal(calibrated.potential_fn(point), original.potential_fn(point))


def test_model_without_default_jitter_retains_preparation():
    def model():
        numpyro.sample('x', dist.Normal(0., 1.))
    preparation = prepare_laplace_metric(model, jax.random.PRNGKey(1), {'x': .1})
    actual, diagnostics = _white_light_jitter_metric(preparation)
    assert actual is preparation
    assert diagnostics == {}


def test_joint_member_jitter_sites_are_each_widened_like_a_single_member():
    low, high = np.log(1e-5), np.log(1e-2)
    def member(scale):
        log_jitter = numpyro.sample('log_jitter', dist.Uniform(low, high))
        numpyro.sample('a', dist.Normal(0., 1.))
        numpyro.factor('likelihood', -.5 * (jnp.exp(log_jitter) / scale)**2)
    def single():
        member(3e-5)
    def joint():
        with numpyro.handlers.scope(prefix='member0'):
            member(3e-5)
        with numpyro.handlers.scope(prefix='member1'):
            member(5e-5)
    start = {'log_jitter': np.log(3e-5), 'a': 0.}
    alone = prepare_laplace_metric(single, jax.random.PRNGKey(3), start, hessian_method='exact',
                                   gradient_tolerance=1e-8, decrement_tolerance=1e-8)
    _, single_diag = _white_light_jitter_metric(alone)
    both = prepare_laplace_metric(
        joint, jax.random.PRNGKey(3),
        {'member0/log_jitter': np.log(3e-5), 'member0/a': 0., 'member1/log_jitter': np.log(5e-5), 'member1/a': 0.},
        hessian_method='exact', gradient_tolerance=1e-8, decrement_tolerance=1e-8)
    calibrated, diagnostics = _white_light_jitter_metric(both, ['member0/log_jitter', 'member1/log_jitter'])
    # The two members are independent here, so member0 must match the single-member slice.
    np.testing.assert_allclose(diagnostics['laplace_jitter_slice_variance[member0]'],
                               single_diag['laplace_jitter_slice_variance'], rtol=1e-10)
    for name in ('member0', 'member1'):
        assert diagnostics[f'laplace_jitter_metric_variance[{name}]'] > diagnostics[f'laplace_jitter_local_variance[{name}]']
    assert calibrated.potential_fn is both.potential_fn
    assert np.linalg.eigvalsh(calibrated.inverse_mass_matrix).min() > 0.
    # Only the listed sites change; a missing site is ignored.
    unchanged, none = _white_light_jitter_metric(both, ['member2/log_jitter'])
    assert unchanged is both and none == {}


def test_joint_jitter_sites_follow_member_limb_darkening():
    from types import SimpleNamespace
    from koala.joint import joint_jitter_metric_sites
    contexts = [SimpleNamespace(ld_profile='power2', ld_prior_mode='stellarprior', trend_priors=None),
                SimpleNamespace(ld_profile='quadratic', ld_prior_mode='uniform', trend_priors=None),
                SimpleNamespace(ld_profile='power2', ld_prior_mode='stellarprior', trend_priors={'jitter': {'prior': 'uniform', 'low': -12., 'high': -4.}})]
    assert joint_jitter_metric_sites(contexts) == ['member0/log_jitter']
