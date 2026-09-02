import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

import fit_jwst
from models.channel_batching import (
    DifficultyConfig,
    PilotDiagnostics,
    build_difficulty_batch_plan,
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
