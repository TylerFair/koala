"""``flags.spectro_joint_geometry``: shared transit geometry across channels."""

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpyro import handlers

from koala.config import (
    _resolve_spectro_joint_geometry,
    _resolve_spectro_joint_geometry_chunk_size,
    _resolve_spectro_joint_geometry_sampler,
)
from koala.constants import ADVANCED_FLAGS
from koala.geometry import _joint_geometry_sites, _whitelight_geometry_prior
from koala.outputs import save_detailed_fit_results, summarize_joint_geometry
from models.jaxoplanet.builder import NUTS_KWARGS, create_vectorized_model


jax.config.update("jax_enable_x64", True)

T = jnp.linspace(0.0, 0.12, 31)
NUM_CHANNELS = 3


def _model_kwargs(joint=False, param_method="duration"):
    kwargs = dict(
        t=T,
        yerr=jnp.full((NUM_CHANNELS, T.size), 1.0e-4),
        y=jnp.ones((NUM_CHANNELS, T.size)),
        mu_t0=jnp.array([0.05]),
        mu_b=jnp.array([0.3]),
        mu_depths=jnp.full((NUM_CHANNELS, 1), 0.01),
        PERIOD=jnp.array([3.0]),
        ld_fixed=jnp.tile(jnp.array([[0.4, 0.2]]), (NUM_CHANNELS, 1)),
    )
    if param_method == "duration":
        kwargs["mu_duration"] = jnp.array([0.08])
    else:
        kwargs["mu_a_rs"] = jnp.array([8.0])
    if joint:
        kwargs.update(sigma_t0=jnp.array([3.0e-4]), sigma_b=jnp.array([0.03]))
        if param_method == "duration":
            kwargs["sigma_duration"] = jnp.array([1.5e-3])
        else:
            kwargs["sigma_a_rs"] = jnp.array([0.3])
    return kwargs


def _trace(model, **kwargs):
    return handlers.trace(handlers.seed(model, jax.random.PRNGKey(3))).get_trace(**kwargs)


def _sample_sites(trace):
    return {
        name for name, site in trace.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    }


def test_flags_are_documented_advanced_keys():
    assert {"spectro_joint_geometry", "spectro_joint_geometry_prior_inflation"} <= ADVANCED_FLAGS


def test_fixed_geometry_trace_has_no_shared_sites():
    model = create_vectorized_model(detrend_type="linear", ld_mode="fixed")
    trace = _trace(model, **_model_kwargs())
    assert not {"t0", "b", "duration", "a_rs"} & set(trace)
    assert trace["rors"]["value"].shape == (NUM_CHANNELS, 1)


@pytest.mark.parametrize("param_method", ["duration", "a_rs"])
def test_joint_geometry_trace_has_single_shared_sites_with_whitelight_prior(param_method):
    model = create_vectorized_model(
        detrend_type="linear", ld_mode="fixed", joint_geometry=True,
        param_method=param_method,
    )
    kwargs = _model_kwargs(joint=True, param_method=param_method)
    trace = _trace(model, **kwargs)
    expected = set(_joint_geometry_sites(param_method))
    assert expected <= _sample_sites(trace)
    absent = {"duration", "a_rs"} - expected
    assert not absent & set(trace)
    for name in expected:
        site = trace[name]
        # One draw per planet, not per channel: the site is shared.
        assert site["value"].shape == (1,)
        assert type(site["fn"]).__name__ == "Normal"
        np.testing.assert_allclose(site["fn"].loc, kwargs[f"mu_{name}"])
        np.testing.assert_allclose(site["fn"].scale, kwargs[f"sigma_{name}"])
    # Per-channel science sites keep their channel axis.
    assert trace["rors"]["value"].shape == (NUM_CHANNELS, 1)
    assert trace["c"]["value"].shape == (NUM_CHANNELS,)


def test_joint_geometry_requires_prior_widths_and_fit_geometry():
    model = create_vectorized_model(detrend_type="linear", ld_mode="fixed", joint_geometry=True)
    kwargs = _model_kwargs(joint=True)
    kwargs.pop("sigma_b")
    with pytest.raises(ValueError, match="sigma_b"):
        _trace(model, **kwargs)
    with pytest.raises(ValueError, match="fit_geometry"):
        create_vectorized_model(
            detrend_type="linear", ld_mode="fixed", joint_geometry=True,
            param_method="a_rs",
            surface_config={"model": "transit", "spots": (), "fit_geometry": False},
        )


def test_joint_geometry_disables_static_geometry_accelerations():
    indices = np.arange(4, 27)
    model = create_vectorized_model(
        detrend_type="linear", ld_mode="fixed", joint_geometry=True,
        transit_window="auto", transit_window_indices=indices,
        cadence_reduction="auto", transit_grid="auto",
        transit_grid_non_grazing=True, transit_grid_outer_contact_safe=True,
    )
    trace = _trace(model, **_model_kwargs(joint=True))
    # The reduced-likelihood factors are the fingerprint of the fixed-geometry
    # accelerations; the joint model must use the full observation site.
    assert "obs" in trace
    assert "obs_active" not in trace and "obs_out_of_window" not in trace


def test_flag_resolution_defaults_and_validation():
    assert _resolve_spectro_joint_geometry({}) == (False, 3.0)
    assert _resolve_spectro_joint_geometry({"spectro_joint_geometry": True}) == (True, 3.0)
    assert _resolve_spectro_joint_geometry(
        {"spectro_joint_geometry": "true", "spectro_joint_geometry_prior_inflation": 1.5}
    ) == (True, 1.5)
    with pytest.raises(ValueError, match="prior_inflation"):
        _resolve_spectro_joint_geometry({"spectro_joint_geometry_prior_inflation": 0})


def test_sampler_override_warns_and_uses_joint_nuts_defaults():
    laplace_kwargs = {"dense_mass": True, "mass_matrix": "laplace", "laplace_warmup": 150}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _resolve_spectro_joint_geometry_sampler(
            "independent_nuts", False, laplace_kwargs
        ) == ("independent_nuts", laplace_kwargs)
        assert _resolve_spectro_joint_geometry_sampler(
            "joint_nuts", True, {"dense_mass": False}
        ) == ("joint_nuts", {"dense_mass": False})
    for backend in ("independent_nuts", "independent_hmc"):
        with pytest.warns(UserWarning, match=f"overriding flags.spectro_sampler='{backend}'"):
            sampler, nuts_kwargs = _resolve_spectro_joint_geometry_sampler(
                backend, True, laplace_kwargs, stage_name="highres"
            )
        assert sampler == "joint_nuts"
        assert nuts_kwargs == dict(NUTS_KWARGS)
        assert "mass_matrix" not in nuts_kwargs


def test_chunking_guard_keeps_every_channel_in_one_chunk():
    assert _resolve_spectro_joint_geometry_chunk_size(40, 120, False) == 40
    assert _resolve_spectro_joint_geometry_chunk_size(None, 12, True) == 12
    assert _resolve_spectro_joint_geometry_chunk_size("auto", 12, True) == 12
    assert _resolve_spectro_joint_geometry_chunk_size(50, 12, True) == 12
    with pytest.raises(ValueError, match="would split the 12 channels"):
        _resolve_spectro_joint_geometry_chunk_size(5, 12, True)


def _whitelight_table(**overrides):
    row = {
        "t0": 59787.05675, "t0_err": 1.4e-5,
        "b": 0.4526, "b_err": 0.0021,
        "duration": 0.11695, "duration_err": 9.5e-5,
        "a_rs": 11.389, "a_rs_err": 0.0218,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _handoff():
    return {"geometry": {
        "t0": [59787.05675], "b": [0.4526], "duration": [0.11695], "a_rs": [11.389],
    }}


@pytest.mark.parametrize("param_method", ["duration", "a_rs"])
def test_handoff_prior_carries_inflated_whitelight_std(param_method):
    table = _whitelight_table()
    prior = _whitelight_geometry_prior(
        table, _handoff(), param_method=param_method, prior_inflation=3.0
    )
    for name in _joint_geometry_sites(param_method):
        np.testing.assert_allclose(prior[name], table[name].values)
        np.testing.assert_allclose(prior[f"sigma_{name}"], 3.0 * table[f"{name}_err"].values)
    absent = {"duration", "a_rs"} - set(_joint_geometry_sites(param_method))
    assert not any(f"sigma_{name}" in prior for name in absent)


def test_handoff_prior_rejects_missing_or_degenerate_std():
    with pytest.raises(ValueError, match="b_err"):
        _whitelight_geometry_prior(
            _whitelight_table().drop(columns=["b_err"]), _handoff(),
            param_method="duration", prior_inflation=3.0,
        )
    with pytest.raises(ValueError, match="must be > 0"):
        _whitelight_geometry_prior(
            _whitelight_table(t0_err=0.0), _handoff(),
            param_method="duration", prior_inflation=3.0,
        )


def test_joint_geometry_posterior_is_written(tmp_path):
    rng = np.random.default_rng(0)
    draws = 200
    samples = {
        "rors": np.full((draws, 2, 1), 0.1) + rng.normal(0, 1e-4, (draws, 2, 1)),
        "t0": 0.05 + rng.normal(0, 1e-4, (draws, 1)),
        "b": 0.3 + rng.normal(0, 0.01, (draws, 1)),
        "duration": 0.08 + rng.normal(0, 1e-3, (draws, 1)),
        "u": np.tile(np.array([[0.4, 0.2]]), (draws, 2, 1)),
        "c": np.ones((draws, 2)),
        "v": np.zeros((draws, 2)),
    }
    summary = {
        name: {
            "prior_center": np.array([center]),
            "prior_sigma": np.array([3.0 * std]),
            "whitelight_std": np.array([std]),
        }
        for name, center, std in (
            ("t0", 0.05, 1e-4), ("b", 0.3, 0.01), ("duration", 0.08, 1e-3)
        )
    }
    columns, rows = summarize_joint_geometry(samples, summary)
    assert {"t0", "t0_err", "b_err_low", "duration_err_high"} <= set(columns)
    assert [row["parameter"] for row in rows] == ["t0", "b", "duration"]
    prefix = str(tmp_path / "stage")
    params_df, _ = save_detailed_fit_results(
        np.linspace(0.0, 0.12, 5), np.ones((2, 5)), np.full((2, 5), 1e-4),
        np.array([1.0, 2.0]), np.array([0.1, 0.1]), samples, {}, {},
        "linear", prefix, joint_geometry=summary,
    )
    joint = pd.read_csv(f"{prefix}_joint_geometry.csv")
    assert list(joint["parameter"]) == ["t0", "b", "duration"]
    np.testing.assert_allclose(joint["whitelight_median"], [0.05, 0.3, 0.08])
    np.testing.assert_allclose(joint["prior_sigma"], [3e-4, 0.03, 3e-3])
    assert np.all(np.abs(joint["shift_in_whitelight_sigma"]) < 1.0)
    written = pd.read_csv(f"{prefix}_bestfit_params.csv")
    assert len(written) == 2
    assert np.allclose(written["t0"], joint.loc[0, "median"])
    assert np.allclose(written["duration_err"], joint.loc[2, "std"])
