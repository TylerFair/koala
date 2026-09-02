import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpy as np

from diagnose_nuts_gradient import diagnose_gradient_quality


jax.config.update("jax_enable_x64", True)


def _linear_gaussian_model(t, yerr, *, y, prior_params):
    slope = numpyro.sample(
        "slope", dist.Normal(prior_params["mean"], prior_params["sigma"])
    )
    offset = numpyro.sample("offset", dist.Normal(0.0, 1.0))
    numpyro.sample("obs", dist.Normal(offset + slope * t, yerr), obs=y)


def test_gradient_diagnostic_matches_directional_finite_differences():
    t = jnp.linspace(-1.0, 1.0, 21, dtype=jnp.float64)
    yerr = jnp.full_like(t, 0.1)
    y = 0.2 + 0.4 * t
    result = diagnose_gradient_quality(
        _linear_gaussian_model,
        {"slope": jnp.asarray(0.35), "offset": jnp.asarray(0.15)},
        t,
        yerr,
        y,
        {"mean": 0.0, "sigma": 1.0},
    )
    assert result["passed"]
    assert result["all_finite"]
    assert result["parameter_count"] == 2
    assert set(result["sites"]) == {"offset", "slope"}
    assert result["max_directional_relative_error"] < 1.0e-7
    assert np.isfinite(result["gradient_l2"])


def _absolute_time_model(t, yerr, *, y, prior_params):
    del t, yerr, y
    epoch = numpyro.sample(
        "epoch", dist.Normal(prior_params["center"], 0.01)
    )
    scaled = (epoch - prior_params["center"]) / 1.0e-3
    numpyro.factor("narrow_nonlinear_term", -0.25 * scaled**4)


def test_gradient_step_is_not_scaled_by_large_bjd_coordinate_origin():
    center = 60_000.0
    result = diagnose_gradient_quality(
        _absolute_time_model,
        {"epoch": jnp.asarray(center + 3.0e-4)},
        jnp.asarray([0.0]),
        jnp.asarray([1.0]),
        jnp.asarray([0.0]),
        {"center": center},
    )
    assert result["passed"]
    assert result["finite_difference_step"] < 1.0e-5


def test_gradient_diagnostic_accepts_generic_spectroscopic_model_kwargs():
    def factorized_model(t, yerr, *, y, prior_params):
        slope = numpyro.sample(
            "slope", dist.Normal(prior_params["mean"], prior_params["sigma"])
        )
        offset = numpyro.sample("offset", dist.Normal(jnp.zeros(2), 1.0))
        prediction = offset[:, None] + slope[:, None] * t
        numpyro.sample("obs", dist.Normal(prediction, yerr), obs=y)

    t = jnp.linspace(-0.5, 0.5, 11, dtype=jnp.float64)
    yerr = jnp.full((2, t.size), 0.05, dtype=jnp.float64)
    y = jnp.stack((0.2 + 0.3 * t, -0.1 - 0.2 * t))
    result = diagnose_gradient_quality(
        factorized_model,
        {"slope": jnp.asarray([0.3, -0.2]), "offset": jnp.asarray([0.2, -0.1])},
        t,
        yerr,
        y,
        model_kwargs={
            "prior_params": {
                "mean": jnp.zeros(2),
                "sigma": jnp.ones(2),
            }
        },
    )
    assert result["passed"]
    assert result["all_finite"]
    assert result["parameter_count"] == 4
