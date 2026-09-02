import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.handlers import seed, substitute, trace

from models.jaxoplanet import build_transit_window_indices, create_vectorized_model
from models.trend_marginal import (
    build_marginalized_trend_design,
    marginalized_trend_coefficient_names,
    materialize_marginalized_trend_samples,
)


def test_trend_designs_cover_production_linear_families():
    t = jnp.linspace(0.0, 1.0, 6)
    design, names = build_marginalized_trend_design("quartic", t, 2)
    assert names == ("c", "v", "v2", "v3", "v4")
    assert design.shape == (2, 6, 5)

    template = jnp.linspace(-0.2, 0.3, 6)
    design, names = build_marginalized_trend_design(
        "spot_spectroscopic+linear_discontinuity_spectroscopic",
        t,
        3,
        spot_trend=template,
        jump_trend=-template,
    )
    assert names == ("c", "A_spot", "A_jump")
    assert design.shape == (3, 6, 3)

    design, names = build_marginalized_trend_design(
        "explinear", t, 2, tau=jnp.array([0.1, 0.2])
    )
    assert names == ("c", "v", "A")
    assert design.shape == (2, 6, 3)

    design, names = build_marginalized_trend_design(
        "spot_spectroscopic+explinear",
        t,
        2,
        tau=jnp.array([0.1, 0.2]),
        spot_trend=template,
    )
    assert names == ("c", "v", "A", "A_spot")
    assert design.shape == (2, 6, 4)

    design, names = build_marginalized_trend_design(
        "2spot_spectroscopic+explinear",
        t,
        2,
        tau=jnp.array([0.1, 0.2]),
        spot_trend=template,
        spot_trend2=-template,
    )
    assert names == ("c", "v", "A", "A_spot", "A_spot2")
    assert design.shape == (2, 6, 5)


def test_materialize_conditional_samples_is_reproducible_and_named():
    mean = jnp.arange(12.0).reshape(2, 3, 2)
    factor = jnp.broadcast_to(jnp.eye(2) * 0.1, (2, 3, 2, 2))
    samples = {"rors": jnp.ones((2, 3, 1)), "trend_beta_mean": mean,
               "trend_beta_factor": factor}
    first = materialize_marginalized_trend_samples(
        samples, ("c", "v"), jax.random.PRNGKey(4)
    )
    second = materialize_marginalized_trend_samples(
        samples, ("c", "v"), jax.random.PRNGKey(4)
    )
    np.testing.assert_array_equal(first["c"], second["c"])
    assert first["c"].shape == (2, 3)
    assert first["v"].shape == (2, 3)
    assert "trend_beta_mean" not in first
    assert "trend_beta_factor" not in first


def test_vectorized_power2_model_marginalizes_linear_sites():
    channels, times = 2, 13
    t = jnp.linspace(-0.12, 0.12, times)
    yerr = jnp.full((channels, times), 1.0e-3)
    y = jnp.ones_like(yerr)
    period = jnp.array([3.0])
    duration = jnp.array([0.1])
    t0 = jnp.array([0.0])
    impact = jnp.array([0.3])
    indices = build_transit_window_indices(t, period, t0, duration)
    model = create_vectorized_model(
        detrend_type="linear",
        ld_mode="fixed",
        trend_mode="gaussian_marginalized",
        n_planets=1,
        ld_profile="power2",
        param_method="duration",
        transit_window="auto",
        transit_window_indices=indices,
    )
    values = {
        "rors": jnp.full((channels, 1), 0.1),
        "log_jitter": jnp.full((channels,), jnp.log(2.0e-4)),
    }
    model_trace = trace(seed(substitute(model, data=values), 2)).get_trace(
        t,
        yerr,
        y=y,
        mu_duration=duration,
        mu_t0=t0,
        mu_b=impact,
        mu_depths=jnp.full((channels, 1), 0.01),
        PERIOD=period,
        ld_fixed=jnp.tile(jnp.array([[0.4, 0.5]]), (channels, 1)),
        trend_prior_mean=jnp.tile(jnp.array([[1.0, 0.0]]), (channels, 1)),
        trend_prior_scale=jnp.tile(jnp.array([[0.1, 0.1]]), (channels, 1)),
    )
    assert "c" not in model_trace
    assert "v" not in model_trace
    assert model_trace["trend_beta_mean"]["value"].shape == (channels, 2)
    assert model_trace["trend_beta_factor"]["value"].shape == (channels, 2, 2)
    factor_site = model_trace["obs_marginalized"]
    assert jnp.isfinite(factor_site["fn"].log_factor)
