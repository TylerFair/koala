import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.util import log_density

from models.ld_parameterization import (
    Power2LinearTransform,
    Power2MaxtedTransform,
    gaussian_to_truncated_normal,
    gaussian_to_uniform,
)
from models.jaxoplanet.builder import _enforce_decorrelated_coefficient_support


def test_decorrelated_support_factor_rejects_invalid_inverse_images():
    def model(coefficients):
        _enforce_decorrelated_coefficient_support(coefficients, 0.0, 1.0)

    valid, _ = log_density(model, (jnp.asarray([0.2, 0.8]),), {}, {})
    invalid, _ = log_density(model, (jnp.asarray([0.2, -0.1]),), {}, {})
    assert valid == 0.0
    assert invalid < -1.0e90

    def sanitized_model(coefficients):
        return _enforce_decorrelated_coefficient_support(
            coefficients, 0.0, 1.0
        )

    assert jnp.all(jnp.isfinite(sanitized_model(jnp.asarray([jnp.nan, -1.0]))))


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

    linear = dist.TransformedDistribution(base, Power2LinearTransform()).sample(
        key, (20000,)
    )
    recovered_linear = Power2LinearTransform().inv(linear)
    assert jnp.allclose(direct, recovered_linear, rtol=0.0, atol=2e-15)
    assert jnp.allclose(
        Power2LinearTransform().log_abs_det_jacobian(direct, linear),
        jnp.log(2.0),
    )


def test_latent_gaussian_prior_quantiles_match_original_priors():
    z = jax.random.normal(jax.random.PRNGKey(72), (20000, 2))
    keys = jax.random.split(jax.random.PRNGKey(73), 2)
    loc = jnp.asarray([0.45, 0.55])
    scale = jnp.asarray([0.2, 0.2])
    low = jnp.asarray([0.0, 0.001])
    latent_tn = gaussian_to_truncated_normal(z, loc, scale, low, 1.0)
    direct_tn = dist.TruncatedNormal(loc, scale, low=low, high=1.0).sample(
        keys[0], (20000,)
    )
    latent_uniform = gaussian_to_uniform(z, 0.0, 1.0)
    direct_uniform = dist.Uniform(0.0, 1.0).sample(keys[1], (20000, 2))
    probabilities = jnp.asarray([0.01, 0.1, 0.5, 0.9, 0.99])
    assert jnp.allclose(
        jnp.quantile(latent_tn, probabilities, axis=0),
        jnp.quantile(direct_tn, probabilities, axis=0),
        atol=0.012,
    )
    assert jnp.allclose(
        jnp.quantile(latent_uniform, probabilities, axis=0),
        jnp.quantile(direct_uniform, probabilities, axis=0),
        atol=0.012,
    )

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


def _latent_gaussian_model(observed=None):
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([2]).to_event(1))
    coeff = gaussian_to_truncated_normal(
        z,
        jnp.asarray([0.45, 0.55]),
        jnp.asarray([0.2, 0.2]),
        jnp.asarray([0.0, 0.001]),
        1.0,
    )
    numpyro.deterministic("c1", coeff[0])
    numpyro.deterministic("c2", coeff[1])
    numpyro.sample("obs", dist.Normal(coeff[0] - 0.3 * coeff[1], 0.08),
                   obs=observed)


def test_decorrelated_posterior_matches_coefficients_within_mc_scatter():
    summaries = []
    for seed, model in enumerate(
        (_coefficient_model, _decorrelated_model, _latent_gaussian_model), 81
    ):
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
    assert jnp.allclose(summaries[0], summaries[2], rtol=0.08, atol=0.02)
