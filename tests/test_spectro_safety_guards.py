import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest
import numpyro
import numpyro.distributions as dist

import fit_jwst
from models.channel_batching import (
    DifficultyConfig,
    PilotDiagnostics,
    build_difficulty_batch_plan,
)


@pytest.mark.parametrize("prior", ["free", "widegaussian", "uniform"])
def test_wide_power2_ld_defaults_to_jacobian_corrected_coordinates(prior):
    assert fit_jwst._resolve_ld_parameterization(
        {}, "spectro_ld_parameterization", prior, "power2"
    ) == "decorrelated"
    assert fit_jwst._resolve_ld_parameterization(
        {}, "whitelight_ld_parameterization", prior, "power2"
    ) == "decorrelated"


@pytest.mark.parametrize("prior", ["free", "widegaussian", "uniform"])
def test_wide_quadratic_ld_keeps_coefficient_default(prior):
    assert fit_jwst._resolve_ld_parameterization(
        {}, "spectro_ld_parameterization", prior, "quadratic"
    ) == "coefficients"
    assert fit_jwst._resolve_ld_parameterization(
        {}, "whitelight_ld_parameterization", prior, "quadratic"
    ) == "coefficients"


@pytest.mark.parametrize("prior", ["fixed", "informed", "sing"])
def test_other_ld_priors_keep_coefficient_default(prior):
    assert fit_jwst._resolve_ld_parameterization(
        {}, "spectro_ld_parameterization", prior
    ) == "coefficients"


def test_explicit_ld_parameterization_overrides_prior_default():
    assert fit_jwst._resolve_ld_parameterization(
        {"spectro_ld_parameterization": "coefficients"},
        "spectro_ld_parameterization",
        "widegaussian",
    ) == "coefficients"
    with pytest.raises(ValueError, match="does not support decorrelated"):
        fit_jwst._resolve_ld_parameterization(
            {"spectro_ld_parameterization": "decorrelated"},
            "spectro_ld_parameterization", "uniform", "quadratic",
        )


def _unprovenanced_plan():
    pilot = PilotDiagnostics(
        channel_indices=(0, 1),
        mean_num_steps=(4.0, 8.0),
        max_num_steps=(7.0, 15.0),
        num_divergences=(0, 0),
        step_size=(0.2, 0.1),
        num_draws=10,
    )
    return build_difficulty_batch_plan(
        pilot,
        nominal_width=2,
        config=DifficultyConfig(
            pathology_step_ratio=1.0e12,
            pathology_step_size_ratio=1.0e12,
            pathology_score_z=1.0e12,
            quarantine_divergences=False,
        ),
    )


@pytest.mark.parametrize(
    ("engine", "detrend"),
    [
        ("harmonica", "linear"),
        ("jaxoplanet", "quadratic"),
        ("jaxoplanet", "explinear"),
        ("jaxoplanet", "spot"),
        ("jaxoplanet", "linear_discontinuity"),
    ],
)
def test_interpolated_trend_rejects_incomplete_fixed_models(engine, detrend):
    with pytest.raises(ValueError, match="currently supports only"):
        fit_jwst._validate_interpolated_trend_mode(
            True,
            transit_engine=engine,
            detrending_type=detrend,
        )


def test_interpolated_linear_jaxoplanet_trend_is_supported():
    fit_jwst._validate_interpolated_trend_mode(
        True,
        transit_engine="jaxoplanet",
        detrending_type="linear",
    )
    # Disabled interpolation is valid for every ordinary detrend family.
    fit_jwst._validate_interpolated_trend_mode(
        False,
        transit_engine="harmonica",
        detrending_type="quartic+gp",
    )


def test_sampling_stage_rejects_legacy_batch_plan_before_sampling():
    with pytest.raises(ValueError, match="no workload provenance"):
        fit_jwst._run_sampling_stage(
            lambda *args, **kwargs: None,
            jax.random.PRNGKey(1),
            jnp.arange(3.0),
            jnp.ones((2, 3)),
            jnp.ones((2, 3)),
            {"theta": jnp.zeros(2)},
            use_chunked=True,
            chunk_size=2,
            channel_batch_plan=_unprovenanced_plan(),
            gradient_diagnostic_mode="off",
        )


def test_sampling_stage_rejects_plan_from_another_workload():
    legacy = _unprovenanced_plan()
    plan = build_difficulty_batch_plan(
        legacy.pilot,
        nominal_width=legacy.nominal_width,
        config=legacy.config,
        provenance={
            "artifact_kind": "jwst_spectro_channel_batch_plan",
            "sampling_workload_fingerprint_sha256": "0" * 64,
        },
    )
    with pytest.raises(ValueError, match="different data, model, prior"):
        fit_jwst._run_sampling_stage(
            lambda *args, **kwargs: None,
            jax.random.PRNGKey(2),
            jnp.arange(3.0),
            jnp.ones((2, 3)),
            jnp.ones((2, 3)),
            {"theta": jnp.zeros(2)},
            use_chunked=True,
            chunk_size=2,
            channel_batch_plan=plan,
            gradient_diagnostic_mode="off",
        )


def test_sampler_swap_order_is_directional_and_ends_adaptive():
    assert fit_jwst._spectro_sampler_swap_order("independent_nuts") == (
        "independent_hmc", "joint_nuts"
    )
    assert fit_jwst._spectro_sampler_swap_order("independent_hmc") == (
        "independent_nuts", "joint_nuts"
    )


def _tiny_radius_model(t, yerr, y=None):
    rors = numpyro.sample("rors", dist.Uniform(0.02, 0.3))
    numpyro.sample("obs", dist.Normal(rors[:, None], yerr), obs=y)


def test_tiny_exact_model_selectively_swaps_nuts_to_hmc(tmp_path, monkeypatch):
    gate_calls = []

    def forced_gate(samples, diagnostics_path, min_depth_ess, max_divergences):
        gate_calls.append(diagnostics_path)
        failed = jnp.asarray([0], dtype=int) if len(gate_calls) == 1 else jnp.asarray([], dtype=int)
        return failed, {
            "depth_ess_per_channel": [999.0],
            "num_divergences_per_channel": [0],
        }

    monkeypatch.setattr(fit_jwst, "_spectro_failed_lanes", forced_gate)
    t = jnp.arange(4.0)
    samples = fit_jwst.get_samples_chunked(
        _tiny_radius_model, jax.random.PRNGKey(92), t,
        jnp.full((1, 4), 0.08), jnp.full((1, 4), 0.12),
        {"rors": jnp.asarray([0.1])}, chunk_size=1,
        nuts_kwargs={"mass_matrix": "laplace", "laplace_warmup": 10,
                     "laplace_map_iterations": 4},
        mcmc_kwargs={"num_warmup": 10, "num_samples": 20},
        output_dir=str(tmp_path), checkpoint_prefix="tiny",
        sampler_backend="independent_nuts", spectro_min_depth_ess=1,
    )
    assert samples["rors"].shape == (20, 1)
    assert samples.sampler_used == ["independent_hmc"]
    checkpoint = next((tmp_path / "chunks").glob("*.pkl"))
    loaded = fit_jwst._load_chunk_samples(checkpoint)
    assert loaded.sampler_used == ["independent_hmc"]
