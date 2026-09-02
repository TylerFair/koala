import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

from models.ld_parameterization import (
    Power2MaxtedTransform,
    QuadraticKippingTransform,
)


def test_transformed_prior_is_exact_in_physical_coefficients():
    key = jax.random.PRNGKey(71)
    base = dist.TruncatedNormal(
        jnp.asarray([0.45, 0.55]), jnp.asarray([0.2, 0.2]),
        low=jnp.asarray([0.0, 0.001]), high=1.0,
    ).to_event(1)
    direct = base.sample(key, (20000,))
    h = dist.TransformedDistribution(base, Power2MaxtedTransform()).sample(
        key, (20000,)
    )
    recovered = Power2MaxtedTransform().inv(h)
    assert jnp.allclose(direct, recovered, rtol=0.0, atol=5e-13)

    quadratic_base = dist.Uniform(0.0, 1.0).expand([2]).to_event(1)
    direct_q = quadratic_base.sample(key, (20000,))
    q = dist.TransformedDistribution(
        quadratic_base, QuadraticKippingTransform()
    ).sample(key, (20000,))
    recovered_q = QuadraticKippingTransform().inv(q)
    assert jnp.allclose(direct_q, recovered_q, rtol=0.0, atol=2e-15)


def _coefficient_model(observed=None):
    coeff = numpyro.sample(
        "coeff", dist.TruncatedNormal(
            jnp.asarray([0.45, 0.55]), jnp.asarray([0.2, 0.2]),
            low=jnp.asarray([0.0, 0.001]), high=1.0,
        ).to_event(1)
    )
    numpyro.deterministic("c1", coeff[0])
    numpyro.deterministic("c2", coeff[1])
    numpyro.sample("obs", dist.Normal(coeff[0] - 0.3 * coeff[1], 0.08),
                   obs=observed)


def _decorrelated_model(observed=None):
    base = dist.TruncatedNormal(
        jnp.asarray([0.45, 0.55]), jnp.asarray([0.2, 0.2]),
        low=jnp.asarray([0.0, 0.001]), high=1.0,
    ).to_event(1)
    h = numpyro.sample(
        "h", dist.TransformedDistribution(base, Power2MaxtedTransform())
    )
    coeff = Power2MaxtedTransform().inv(h)
    numpyro.deterministic("c1", coeff[0])
    numpyro.deterministic("c2", coeff[1])
    numpyro.sample("obs", dist.Normal(coeff[0] - 0.3 * coeff[1], 0.08),
                   obs=observed)


def test_decorrelated_posterior_matches_coefficients_within_mc_scatter():
    summaries = []
    for seed, model in enumerate((_coefficient_model, _decorrelated_model), 81):
        mcmc = MCMC(
            NUTS(model, target_accept_prob=0.9), num_warmup=300,
            num_samples=800, progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(seed), observed=0.23)
        draws = Predictive(model, posterior_samples=mcmc.get_samples())(
            jax.random.PRNGKey(seed + 1000), observed=0.23
        )
        summaries.append(jnp.asarray([
            jnp.mean(draws["c1"]), jnp.std(draws["c1"]),
            jnp.mean(draws["c2"]), jnp.std(draws["c2"]),
        ]))
    assert jnp.allclose(summaries[0], summaries[1], rtol=0.08, atol=0.02)
