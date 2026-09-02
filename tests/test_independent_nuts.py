import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest

import fit_jwst
from models.independent_nuts import (
    build_independent_nuts_runner,
    get_samples_independent,
)
from models.jaxoplanet import build_transit_window_indices, create_vectorized_model


def _normal_channels(t, yerr, y=None, prior_loc=None):
    num_channels = yerr.shape[0]
    x = numpyro.sample(
        "x",
        dist.Normal(prior_loc, 2.0),
    )
    prediction = x[:, None] + jnp.zeros_like(t)
    numpyro.deterministic("twice_x", 2.0 * x)
    numpyro.sample("obs", dist.Normal(prediction, yerr), obs=y)


def _bounded_channels(t, yerr, y=None, prior_loc=None):
    num_channels = yerr.shape[0]
    radius = numpyro.sample(
        "radius",
        dist.Uniform(0.01, 0.5).expand([num_channels]),
    )
    c1 = numpyro.sample(
        "c1",
        dist.TruncatedNormal(
            prior_loc,
            0.2,
            low=0.0,
            high=1.0,
        ),
    )
    c2 = numpyro.sample(
        "c2",
        dist.TruncatedNormal(
            0.5,
            0.2,
            low=0.001,
            high=1.0,
        ).expand([num_channels]),
    )
    prediction = (radius + 0.1 * c1 + 0.05 * c2)[:, None]
    numpyro.sample("obs", dist.Normal(prediction, yerr), obs=y)


def _one_channel_with_shared_geometry(
    t,
    yerr,
    y=None,
    prior_loc=None,
    period=None,
    mu_t0=None,
):
    if period.ndim != 1 or mu_t0.ndim != 1:
        raise ValueError("Shared one-planet geometry gained a channel axis.")
    x = numpyro.sample("x", dist.Normal(prior_loc, 1.0))
    prediction = x[:, None] + 0.0 * period[0] + 0.0 * mu_t0[0]
    numpyro.sample("obs", dist.Normal(prediction, yerr), obs=y)


def test_independent_nuts_matches_analytic_channels_and_padding():
    num_channels = 3
    num_times = 5
    t = jnp.linspace(-1.0, 1.0, num_times)
    means = jnp.array([-1.5, 0.25, 2.0])
    y = jnp.broadcast_to(means[:, None], (num_channels, num_times))
    yerr = jnp.ones_like(y)
    prior_loc = jnp.array([-0.5, 0.0, 0.5])

    samples, diagnostics = get_samples_independent(
        _normal_channels,
        jax.random.PRNGKey(13),
        t,
        yerr,
        y,
        {"x": prior_loc},
        prior_loc=prior_loc,
        lane_width=4,
        num_warmup=150,
        num_samples=500,
        channel_varying_kwargs=("prior_loc",),
        return_diagnostics=True,
    )

    # Normal(0, 2) prior generalized to the channel-specific prior location.
    expected = (num_times * means + 0.25 * prior_loc) / (num_times + 0.25)
    assert samples["x"].shape == (500, num_channels)
    assert samples["twice_x"].shape == (500, num_channels)
    assert jnp.allclose(samples["x"].mean(axis=0), expected, atol=0.12)
    assert jnp.allclose(samples["twice_x"], 2.0 * samples["x"])
    assert diagnostics.step_size.shape == (num_channels,)
    assert diagnostics.num_steps.shape == (500, num_channels)
    assert diagnostics.diverging.shape == (500, num_channels)
    assert not jnp.any(diagnostics.diverging)
    assert jnp.all(diagnostics.step_size > 0)


def test_independent_nuts_bounded_sites_and_fit_aliases_are_constrained():
    t = jnp.arange(4.0)
    yerr = jnp.full((2, 4), 0.1)
    y = jnp.array([[0.2] * 4, [0.4] * 4])
    prior_loc = jnp.array([0.3, 0.7])

    samples = get_samples_independent(
        _bounded_channels,
        jax.random.PRNGKey(4),
        t,
        yerr,
        y,
        {
            "radius": jnp.array([0.15, 0.35]),
            # fit_jwst supplies power-2 coefficients under the legacy "u" key.
            "u": jnp.array([[0.3, 0.4], [0.7, 0.6]]),
        },
        prior_loc=prior_loc,
        lane_width=3,
        num_warmup=100,
        num_samples=200,
        channel_varying_kwargs=("prior_loc",),
    )

    assert jnp.all((samples["radius"] > 0.01) & (samples["radius"] < 0.5))
    assert jnp.all((samples["c1"] > 0.0) & (samples["c1"] < 1.0))
    assert jnp.all((samples["c2"] > 0.001) & (samples["c2"] < 1.0))
    assert jnp.all(jnp.isfinite(samples["radius"]))
    assert jnp.all(jnp.isfinite(samples["c1"]))
    assert jnp.all(jnp.isfinite(samples["c2"]))


def test_independent_nuts_rejects_shared_matched_latent_initialization():
    t = jnp.arange(3.0)
    y = jnp.zeros((2, 3))
    yerr = jnp.ones_like(y)

    with pytest.raises(ValueError, match="factorize by channel"):
        get_samples_independent(
            _normal_channels,
            jax.random.PRNGKey(0),
            t,
            yerr,
            y,
            {"x": jnp.array(0.0)},
            prior_loc=jnp.zeros(2),
            num_warmup=2,
            num_samples=2,
            channel_varying_kwargs=("prior_loc",),
        )


def test_independent_nuts_validates_lane_width():
    t = jnp.arange(3.0)
    y = jnp.zeros((2, 3))
    yerr = jnp.ones_like(y)

    with pytest.raises(ValueError, match="cannot hold"):
        get_samples_independent(
            _normal_channels,
            jax.random.PRNGKey(0),
            t,
            yerr,
            y,
            {"x": jnp.zeros(2)},
            prior_loc=jnp.zeros(2),
            lane_width=1,
            num_warmup=2,
            num_samples=2,
            channel_varying_kwargs=("prior_loc",),
        )


def test_laplace_mass_matrix_reports_map_diagnostics():
    t = jnp.linspace(-1.0, 1.0, 5)
    yerr = jnp.full((2, 5), 0.3)
    y = jnp.asarray([[-0.7] * 5, [1.1] * 5])
    prior_loc = jnp.asarray([-0.5, 0.8])

    samples, diagnostics = get_samples_independent(
        _normal_channels,
        jax.random.PRNGKey(30),
        t,
        yerr,
        y,
        {"x": jnp.asarray([0.0, 0.0])},
        prior_loc=prior_loc,
        channel_varying_kwargs=("prior_loc",),
        lane_width=2,
        num_samples=20,
        nuts_kwargs={
            "mass_matrix": "laplace",
            "laplace_warmup": 10,
            "laplace_target_accept": 0.9,
            "laplace_max_tree_depth": 5,
            "laplace_start_at_map": True,
            "laplace_map_iterations": 4,
        },
        return_diagnostics=True,
    )

    assert samples["x"].shape == (20, 2)
    assert diagnostics.num_steps.shape == (20, 2)
    assert diagnostics.map_gradient_norm.shape == (2,)
    assert diagnostics.map_newton_decrement.shape == (2,)
    assert diagnostics.map_iterations.shape == (2,)
    assert diagnostics.map_condition_number.shape == (2,)
    assert diagnostics.map_hessian_min_eigenvalue.shape == (2,)
    assert jnp.all(jnp.isfinite(diagnostics.map_gradient_norm))
    assert jnp.all(diagnostics.map_gradient_norm < 1.0e-6)
    assert jnp.all(diagnostics.map_condition_number >= 1.0)
    assert not jnp.any(diagnostics.diverging)


def test_finite_difference_hessian_matches_exact_normal_metric():
    t = jnp.arange(4.0)
    yerr = jnp.full((2, 4), 0.2)
    y = jnp.asarray([[0.7] * 4, [-1.2] * 4])
    prior_loc = jnp.asarray([0.0, -0.5])

    _, diagnostics = get_samples_independent(
        _normal_channels,
        jax.random.PRNGKey(33),
        t,
        yerr,
        y,
        {"x": prior_loc},
        prior_loc=prior_loc,
        channel_varying_kwargs=("prior_loc",),
        num_samples=4,
        nuts_kwargs={
            "mass_matrix": "laplace",
            "laplace_hessian_method": "finite_difference",
            "laplace_compare_exact_hessian": True,
            "laplace_map_method": "diagonal",
            "laplace_line_search_steps": 4,
            "laplace_fuse_program": True,
            "laplace_map_iterations": 3,
            "laplace_warmup": 4,
        },
        return_diagnostics=True,
    )

    assert diagnostics.map_hessian_relative_error.shape == (2,)
    assert jnp.all(diagnostics.map_hessian_relative_error < 1.0e-8)


def test_larger_trust_radius_reaches_distant_mode_and_exits_early():
    t = jnp.arange(4.0)
    yerr = jnp.full((2, 4), 0.2)
    y = jnp.asarray([[40.0] * 4, [-35.0] * 4])
    prior_loc = jnp.asarray([40.0, -35.0])

    _, diagnostics = get_samples_independent(
        _normal_channels,
        jax.random.PRNGKey(34),
        t,
        yerr,
        y,
        {"x": jnp.zeros(2)},
        prior_loc=prior_loc,
        channel_varying_kwargs=("prior_loc",),
        num_samples=2,
        nuts_kwargs={
            "mass_matrix": "laplace",
            "laplace_hessian_method": "finite_difference",
            "laplace_trust_radius": 20.0,
            "laplace_map_iterations": 16,
            "laplace_warmup": 2,
        },
        return_diagnostics=True,
    )

    assert jnp.all(diagnostics.map_gradient_norm < 1.0e-6)
    assert jnp.all(diagnostics.map_iterations < 16)


def test_laplace_runner_reuses_compiled_programs_without_stale_values():
    t = jnp.linspace(-1.0, 1.0, 4)
    yerr = jnp.full((2, 4), 0.25)
    prior_loc = jnp.asarray([1.5, 2.0])
    y = jnp.asarray([[1.7] * 4, [2.2] * 4])
    init = {"x": jnp.asarray([1.0, 1.0])}
    nuts_kwargs = {
        "mass_matrix": "laplace",
        "laplace_warmup": 6,
        "laplace_max_tree_depth": 5,
        "laplace_map_iterations": 4,
        "laplace_fuse_program": True,
    }
    common = {
        "nuts_kwargs": nuts_kwargs,
        "num_samples": 8,
        "lane_width": 2,
        "channel_varying_kwargs": ("prior_loc",),
    }
    runner = build_independent_nuts_runner(
        _normal_channels,
        **common,
    )
    first = get_samples_independent(
        _normal_channels,
        jax.random.PRNGKey(31),
        t,
        yerr,
        -y,
        {"x": -init["x"]},
        prior_loc=-prior_loc,
        _runner=runner,
        **common,
    )
    reused = get_samples_independent(
        _normal_channels,
        jax.random.PRNGKey(32),
        t,
        yerr,
        y,
        init,
        prior_loc=prior_loc,
        _runner=runner,
        **common,
    )
    fresh = get_samples_independent(
        _normal_channels,
        jax.random.PRNGKey(32),
        t,
        yerr,
        y,
        init,
        prior_loc=prior_loc,
        **common,
    )

    assert runner.program_build_count == 1
    assert float(jnp.mean(first["x"])) < 0.0
    assert float(jnp.mean(reused["x"])) > 0.0
    assert jnp.array_equal(reused["x"], fresh["x"])


def test_pipeline_resolves_laplace_spectro_options_by_stage():
    resolved = fit_jwst._resolve_jaxoplanet_spectro_nuts_kwargs(
        {
            "spectro_mass_matrix": "laplace",
            "spectro_laplace_warmup": 150,
            "spectro_laplace_target_accept": 0.95,
            "spectro_laplace_max_tree_depth": 10,
            "lowres_laplace_warmup": 75,
            "lowres_laplace_start_at_map": True,
            "lowres_laplace_trust_radius": 5.0,
        },
        "lowres",
        {"dense_mass": False, "target_accept_prob": 0.8},
        independent=True,
    )

    assert resolved["mass_matrix"] == "laplace"
    assert resolved["laplace_warmup"] == 75
    assert resolved["laplace_target_accept"] == 0.95
    assert resolved["laplace_max_tree_depth"] == 10
    assert resolved["laplace_start_at_map"] is True
    assert resolved["laplace_trust_radius"] == 5.0
    assert resolved["dense_mass"] is True


def test_one_channel_keeps_planet_geometry_shared():
    t = jnp.arange(3.0)
    y = jnp.full((1, 3), 0.2)
    yerr = jnp.ones_like(y)

    samples = get_samples_independent(
        _one_channel_with_shared_geometry,
        jax.random.PRNGKey(8),
        t,
        yerr,
        y,
        {"x": jnp.array([0.1])},
        prior_loc=jnp.array([0.0]),
        period=jnp.array([3.2]),
        mu_t0=jnp.array([0.0]),
        channel_varying_kwargs=("prior_loc",),
        lane_width=2,
        num_warmup=20,
        num_samples=20,
    )
    assert samples["x"].shape == (20, 1)


def test_independent_nuts_runs_real_windowed_power2_model():
    num_channels, num_times = 2, 15
    t = jnp.linspace(-0.12, 0.12, num_times)
    yerr = jnp.full((num_channels, num_times), 1.0e-3)
    y = jnp.ones_like(yerr)
    period = jnp.array([3.0])
    duration = jnp.array([0.1])
    t0 = jnp.array([0.0])
    impact = jnp.array([0.3])
    indices = build_transit_window_indices(t, period, t0, duration)
    model = create_vectorized_model(
        detrend_type="linear",
        ld_mode="informed",
        trend_mode="free",
        n_planets=1,
        ld_profile="power2",
        param_method="duration",
        transit_window="auto",
        transit_window_indices=indices,
    )
    mu_u = jnp.tile(jnp.array([[0.4, 0.5]]), (num_channels, 1))
    model_kwargs = {
        "mu_duration": duration,
        "mu_t0": t0,
        "mu_b": impact,
        "mu_depths": jnp.full((num_channels, 1), 0.01),
        "PERIOD": period,
        "mu_u_ld": mu_u,
        "sigma_u_ld": jnp.full((num_channels, 2), 0.1),
        "precomputed_yerr_per_lc": jnp.full((num_channels,), 1.0e-3),
    }
    samples, diagnostics = get_samples_independent(
        model,
        jax.random.PRNGKey(2),
        t,
        yerr,
        y,
        {
            "rors": jnp.full((num_channels, 1), 0.1),
            "u": mu_u,
            "c": jnp.ones(num_channels),
            "v": jnp.zeros(num_channels),
        },
        num_warmup=2,
        num_samples=2,
        lane_width=2,
        channel_varying_kwargs=(
            "mu_depths",
            "mu_u_ld",
            "sigma_u_ld",
            "precomputed_yerr_per_lc",
        ),
        return_diagnostics=True,
        **model_kwargs,
    )
    assert samples["rors"].shape == (2, num_channels, 1)
    assert samples["c1"].shape == (2, num_channels)
    assert samples["total_error"].shape == (2, num_channels)
    assert diagnostics.num_steps.shape == (2, num_channels)
    assert jnp.all(jnp.isfinite(samples["rors"]))
