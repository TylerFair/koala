import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from models.independent_nuts import prepare_laplace_metric


@pytest.mark.parametrize("sequential", [False, True])
def test_laplace_evaluation_modes_match_analytic_correlated_posterior(sequential):
    design = jnp.array([[1., .3], [.7, 1.], [-.2, .8]])
    observed = jnp.array([.4, -.1, -.2])
    error = .2

    def model():
        x = numpyro.sample("x", dist.Normal(0., 1.).expand([2]).to_event(1))
        numpyro.sample("obs", dist.Normal(design @ x, error), obs=observed)

    prepared = prepare_laplace_metric(model, jax.random.PRNGKey(7), {"x": jnp.zeros(2)},
                                      max_iterations=10, sequential_evaluations=sequential)
    covariance = np.linalg.inv(np.eye(2) + np.asarray(design).T @ np.asarray(design) / error**2)
    mean = covariance @ np.asarray(design).T @ np.asarray(observed) / error**2
    np.testing.assert_allclose(prepared.inverse_mass_matrix, covariance, rtol=2e-7, atol=1e-9)
    np.testing.assert_allclose(prepared.unconstrained_map["x"], mean, rtol=2e-7, atol=1e-9)
