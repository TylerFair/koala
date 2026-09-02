import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

import models.laplace_is as laplace_module
import fit_jwst
from models.independent_nuts import get_samples_independent
from models.jaxoplanet import build_transit_window_indices, create_vectorized_model
from models.laplace_is import get_samples_laplace_is, psis_smooth_log_weights


def _normal_channels(t, yerr, y=None, prior_loc=None):
    x = numpyro.sample("x", dist.Normal(prior_loc, 2.0))
    prediction = x[:, None] + jnp.zeros_like(t)
    numpyro.deterministic("twice_x", 2.0 * x)
    numpyro.sample("obs", dist.Normal(prediction, yerr), obs=y)


def _normal_problem(num_channels=3, num_times=5):
    t = jnp.linspace(-1.0, 1.0, num_times)
    means = jnp.linspace(-1.25, 1.5, num_channels)
    y = jnp.broadcast_to(means[:, None], (num_channels, num_times))
    yerr = jnp.ones_like(y)
    prior_loc = jnp.linspace(-0.5, 0.5, num_channels)
    return t, yerr, y, prior_loc, means


def test_psis_matches_arviz_random_weights():
    rng = np.random.default_rng(20260901)
    raw = rng.normal(size=(5, 513)) * 1.7
    actual_lw, actual_k = psis_smooth_log_weights(jnp.asarray(raw))
    expected_lw, expected_k = az.psislw(raw)
    np.testing.assert_allclose(actual_lw, expected_lw, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual_k, expected_k, rtol=1.0e-12, atol=1.0e-12)


def test_laplace_is_recovers_gaussian_and_padding_deterministics():
    t, yerr, y, prior_loc, means = _normal_problem()
    samples, diagnostics = get_samples_laplace_is(
        _normal_channels,
        jax.random.PRNGKey(13),
        t,
        yerr,
        y,
        {"x": prior_loc},
        prior_loc=prior_loc,
        lane_width=4,
        num_warmup=50,
        num_samples=800,
        channel_varying_kwargs=("prior_loc",),
        laplace_is_num_draws=1024,
        laplace_is_rounds=1,
        laplace_is_draw_chunk_size=128,
        laplace_is_imh_thin=2,
        laplace_is_fallback=False,
        return_diagnostics=True,
    )
    expected_mean = (5.0 * means + 0.25 * prior_loc) / 5.25
    expected_sigma = np.sqrt(1.0 / 5.25)
    assert samples["x"].shape == (800, 3)
    assert samples["twice_x"].shape == (800, 3)
    np.testing.assert_allclose(samples["x"].mean(0), expected_mean, atol=0.08)
    np.testing.assert_allclose(samples["x"].std(0), expected_sigma, atol=0.08)
    np.testing.assert_allclose(samples["twice_x"], 2.0 * samples["x"])
    assert np.all(np.asarray(diagnostics.pareto_k) < 0.7)
    assert np.all(np.asarray(diagnostics.imh_acceptance) > 0.75)
    assert np.all(np.asarray(diagnostics.converged))
    assert np.max(np.asarray(diagnostics.map_newton_decrement)) <= 1.0e-8


def test_draw_chunk_size_is_large_for_soss_and_memory_bounded_for_prism():
    assert laplace_module._effective_draw_chunk_size(256, 40, 225, 4096) == 256
    assert laplace_module._effective_draw_chunk_size(256, 21, 40780, 4096) == 64


def _tiny_power2_problem(num_channels=2, num_times=15):
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
    init = {
        "rors": jnp.full((num_channels, 1), 0.1),
        "u": mu_u,
        "c": jnp.ones(num_channels),
        "v": jnp.zeros(num_channels),
    }
    varying = (
        "mu_depths",
        "mu_u_ld",
        "sigma_u_ld",
        "precomputed_yerr_per_lc",
    )
    return model, t, yerr, y, init, model_kwargs, varying


def test_real_windowed_power2_model_agrees_with_short_independent_nuts():
    model, t, yerr, y, init, model_kwargs, varying = _tiny_power2_problem()
    laplace, diagnostics = get_samples_laplace_is(
        model,
        jax.random.PRNGKey(21),
        t,
        yerr,
        y,
        init,
        lane_width=2,
        num_warmup=50,
        num_samples=120,
        channel_varying_kwargs=varying,
        laplace_is_num_draws=512,
        laplace_is_rounds=2,
        laplace_is_draw_chunk_size=16,
        laplace_is_map_maxiter=60,
        laplace_is_fallback=False,
        return_diagnostics=True,
        **model_kwargs,
    )
    nuts = get_samples_independent(
        model,
        jax.random.PRNGKey(22),
        t,
        yerr,
        y,
        init,
        lane_width=2,
        num_warmup=80,
        num_samples=120,
        channel_varying_kwargs=varying,
        **model_kwargs,
    )
    for name in ("rors", "c1", "c2", "total_error"):
        assert laplace[name].shape == nuts[name].shape
        assert np.all(np.isfinite(np.asarray(laplace[name])))
    # This deliberately small comparison is a regression check, not the
    # literature fidelity study.  It catches transform/postprocess failures.
    nuts_sigma = np.std(np.asarray(nuts["rors"]), axis=0)
    standardized = np.abs(
        np.median(np.asarray(laplace["rors"]), axis=0)
        - np.median(np.asarray(nuts["rors"]), axis=0)
    ) / np.maximum(nuts_sigma, 1.0e-6)
    assert np.max(standardized) < 1.5
    assert diagnostics.pareto_k.shape == (2,)


def test_failed_gate_splices_independent_fallback(monkeypatch):
    t, yerr, y, prior_loc, _ = _normal_problem(num_channels=2)
    calls = []

    def fake_fallback(model, key, t_arg, err, obs, init, **kwargs):
        calls.append((err.shape, kwargs["lane_width"], kwargs["nuts_kwargs"]))
        draws = kwargs["num_samples"]
        channels = err.shape[0]
        x = jnp.full((draws, channels), 7.0)
        return {"x": x, "twice_x": 2.0 * x}

    monkeypatch.setattr(laplace_module, "get_samples_independent", fake_fallback)
    samples, diagnostics = get_samples_laplace_is(
        _normal_channels,
        jax.random.PRNGKey(8),
        t,
        yerr,
        y,
        {"x": prior_loc},
        prior_loc=prior_loc,
        lane_width=4,
        num_warmup=2,
        num_samples=12,
        channel_varying_kwargs=("prior_loc",),
        laplace_is_output="resample",
        laplace_is_num_draws=32,
        laplace_is_rounds=0,
        laplace_is_draw_chunk_size=8,
        laplace_is_khat_threshold=-10.0,
        return_diagnostics=True,
    )
    # The fallback now retains the fixed parent width so subsequent chunks
    # reuse the same compiled runner even when their failed-lane count differs.
    assert calls[0][:2] == ((2, 5), 4)
    assert calls[0][2]["mass_matrix"] == "laplace"
    assert calls[0][2]["laplace_hessian_method"] == "finite_difference"
    assert calls[0][2]["laplace_warmup"] == 150
    assert calls[0][2]["laplace_target_accept"] == 0.95
    assert calls[0][2]["laplace_max_tree_depth"] == 6
    assert calls[0][2]["laplace_trust_radius"] == 5.0
    assert calls[0][2]["laplace_start_at_map"] is True
    np.testing.assert_allclose(samples["x"], 7.0)
    np.testing.assert_allclose(samples["twice_x"], 14.0)
    assert np.all(np.asarray(diagnostics.fell_back))
    assert not np.any(np.asarray(diagnostics.gate_passed))


def test_fit_chunked_laplace_dispatch_reuses_runner(monkeypatch):
    builders = []
    sampler_calls = []
    runner_token = object()

    def fake_builder(model, **kwargs):
        builders.append(kwargs)
        return runner_token

    def fake_sampler(model, key, t, err, obs, init, **kwargs):
        sampler_calls.append(kwargs)
        assert kwargs["_runner"] is runner_token
        return {"x": jnp.zeros((kwargs["mcmc_kwargs"]["num_samples"], err.shape[0]))}

    monkeypatch.setattr(laplace_module, "build_laplace_is_runner", fake_builder)
    monkeypatch.setattr(laplace_module, "get_samples_laplace_is", fake_sampler)
    samples = fit_jwst.get_samples_chunked(
        _normal_channels,
        jax.random.PRNGKey(4),
        jnp.arange(3.0),
        jnp.ones((3, 3)),
        jnp.zeros((3, 3)),
        {"x": jnp.zeros(3)},
        2,
        mcmc_kwargs={"num_warmup": 1, "num_samples": 5},
        sampler_backend="laplace_is",
        laplace_is_kwargs={
            "laplace_is_num_draws": 32,
            "laplace_is_fallback": False,
        },
    )
    assert samples["x"].shape == (5, 3)
    assert len(builders) == 1
    assert len(sampler_calls) == 2
    assert all(call["lane_width"] == 2 for call in sampler_calls)
    assert all(call["laplace_is_num_draws"] == 32 for call in sampler_calls)


def test_laplace_stage_options_parse_string_boolean():
    resolved = fit_jwst._resolve_laplace_is_stage_kwargs(
        {
            "spectro_laplace_is_num_draws": 128,
            "spectro_laplace_is_imh_thin": 3,
            "lowres_laplace_is_fallback": "false",
        },
        "lowres",
    )
    assert resolved["laplace_is_num_draws"] == 128
    assert resolved["laplace_is_imh_thin"] == 3
    assert resolved["laplace_is_fallback"] is False


def test_laplace_stage_production_defaults():
    resolved = fit_jwst._resolve_laplace_is_stage_kwargs({}, "highres")
    assert resolved["laplace_is_imh_thin"] == 8
    assert resolved["laplace_is_map_maxiter"] == 200
    assert resolved["laplace_is_map_tol"] == 1.0e-4
    assert resolved["laplace_is_trust_radius"] == 5.0
    assert resolved["laplace_is_khat_threshold"] == 0.7
    assert resolved["laplace_is_min_imh_acceptance"] == 0.2
    assert resolved["laplace_is_force"] is False


def test_wide_ld_routes_whole_stage_to_laplace_nuts(monkeypatch, capsys):
    calls = []

    def fake_chunked(*args, **kwargs):
        calls.append(kwargs)
        return {"x": np.zeros((4, 2))}

    monkeypatch.setattr(fit_jwst, "get_samples_chunked", fake_chunked)
    t, yerr, y, prior_loc, _ = _normal_problem(num_channels=2)
    fit_jwst._run_sampling_stage(
        _normal_channels,
        jax.random.PRNGKey(0),
        t,
        yerr,
        y,
        {"x": prior_loc},
        mcmc_kwargs={"num_warmup": 10, "num_samples": 4},
        use_chunked=True,
        chunk_size=2,
        sampler_backend="laplace_is",
        laplace_is_kwargs={"laplace_is_force": False},
        checkpoint_signature={"ld_mode": "widegaussian"},
        channel_varying_kwargs=("prior_loc",),
        prior_loc=prior_loc,
    )
    assert calls[0]["sampler_backend"] == "independent_nuts"
    assert calls[0]["nuts_kwargs"]["mass_matrix"] == "laplace"
    assert calls[0]["nuts_kwargs"]["laplace_max_tree_depth"] == 6
    assert "routing the whole chunk" in capsys.readouterr().out


def test_force_keeps_laplace_is_for_wide_ld(monkeypatch):
    calls = []

    def fake_chunked(*args, **kwargs):
        calls.append(kwargs)
        return {"x": np.zeros((4, 2))}

    monkeypatch.setattr(fit_jwst, "get_samples_chunked", fake_chunked)
    t, yerr, y, prior_loc, _ = _normal_problem(num_channels=2)
    fit_jwst._run_sampling_stage(
        _normal_channels,
        jax.random.PRNGKey(0),
        t,
        yerr,
        y,
        {"x": prior_loc},
        use_chunked=True,
        chunk_size=2,
        sampler_backend="laplace_is",
        laplace_is_kwargs={"laplace_is_force": True},
        checkpoint_signature={"ld_mode": "uniform"},
        channel_varying_kwargs=("prior_loc",),
        prior_loc=prior_loc,
    )
    assert calls[0]["sampler_backend"] == "laplace_is"
    assert "laplace_is_force" not in calls[0]["laplace_is_kwargs"]
