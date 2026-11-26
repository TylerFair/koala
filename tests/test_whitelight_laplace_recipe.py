"""White-light Laplace checks: converged maximum, and scaled adaptive coordinates."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from koala.white_light import _laplace_metric_is_usable, _laplace_scaled_coordinates
from models.independent_nuts import prepare_laplace_metric

jax.config.update('jax_enable_x64', True)


def _preparation(**values):
    return SimpleNamespace(**{'newton_decrement': 1e-6, 'gradient_norm': 1e-6,
                              'hessian_min_eigenvalue': 1.0, **values})


def test_only_a_converged_maximum_keeps_the_fixed_laplace_metric():
    assert _laplace_metric_is_usable(_preparation()) == (True, True, True)
    # Out of iterations far from the mode (the WASP-17 joint case).
    assert _laplace_metric_is_usable(_preparation(newton_decrement=3e3, gradient_norm=4e4))[:2] == (False, False)
    # Converged onto a saddle.
    assert _laplace_metric_is_usable(_preparation(hessian_min_eigenvalue=-10.2)) == (False, True, False)
    # A non-finite optimum is never usable.
    assert not _laplace_metric_is_usable(_preparation(newton_decrement=np.nan, gradient_norm=np.nan))[0]


def tiny_scale_model(y=None):
    # Variances of 1e-10 and 1e-8, far below NumPyro's 1e-3 shrinkage target.
    t0 = numpyro.sample('t0', dist.Normal(60000.1, 1.))
    depth = numpyro.sample('depth', dist.Uniform(0., .1))
    numpyro.sample('obs_t0', dist.Normal(t0, 1e-5), obs=y[0])
    numpyro.sample('obs_depth', dist.Normal(depth, 1e-4), obs=y[1])


def test_scaled_adaptive_coordinates_keep_the_posterior_and_unit_metric():
    y = jnp.array([60000.1002, .0123])
    preparation = prepare_laplace_metric(tiny_scale_model, jax.random.PRNGKey(0),
                                         {'t0': 60000.1, 'depth': .01}, model_kwargs={'y': y},
                                         hessian_method='exact', eigenvalue_floor=1e-12)
    potential, postprocess, covariance, to_scaled = _laplace_scaled_coordinates(preparation)
    np.testing.assert_allclose(np.diag(covariance), 1., rtol=1e-10)
    scaled_map = to_scaled(preparation.unconstrained_map)
    # The rescaled potential differs from the original only by a constant Jacobian.
    offsets = [potential({k: v + d for k, v in scaled_map.items()}) - preparation.potential_fn(
        {k: v + d * np.sqrt(float(np.diag(preparation.inverse_mass_matrix)[i]))
         for i, (k, v) in enumerate(sorted(preparation.unconstrained_map.items()))})
        for d in (-1., 0., .7)]
    # t0 near 6e4 with a 1e-5 width leaves ~1e-8 relative round-off.
    np.testing.assert_allclose(offsets, offsets[1], rtol=1e-7, atol=0)
    mcmc = MCMC(NUTS(None, potential_fn=potential, dense_mass=True, inverse_mass_matrix=covariance,
                     regularize_mass_matrix=True, target_accept_prob=.9),
                num_warmup=300, num_samples=2000, num_chains=1, progress_bar=False, postprocess_fn=postprocess)
    mcmc.run(jax.random.PRNGKey(1), init_params=scaled_map, extra_fields=('num_steps',))
    samples = mcmc.get_samples()
    np.testing.assert_allclose(np.mean(samples['t0']), 60000.1002, rtol=0, atol=2e-6)
    np.testing.assert_allclose(np.std(samples['t0']), 1e-5, rtol=.1)
    np.testing.assert_allclose(np.mean(samples['depth']), .0123, rtol=0, atol=2e-5)
    np.testing.assert_allclose(np.std(samples['depth']), 1e-4, rtol=.1)
    # Unit scales: the regularized adaptation keeps short trajectories.
    assert np.mean(mcmc.get_extra_fields()['num_steps']) < 8
