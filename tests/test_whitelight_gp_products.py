import os
import sys
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

# The koala311 numerical test environment intentionally omits the Stage 4
# reduction package; these tests only import post-fit helpers and never call it.
if "exotedrf.stage4" not in sys.modules:
    exotedrf = types.ModuleType("exotedrf")
    stage4 = types.ModuleType("exotedrf.stage4")
    stage4.bin_at_resolution = None
    stage4.bin_at_pixel = None
    exotedrf.stage4 = stage4
    sys.modules.setdefault("exotedrf", exotedrf)
    sys.modules["exotedrf.stage4"] = stage4

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

jax.config.update("jax_enable_x64", True)

import models.gp as kgp
from koala.outputs import save_whitelight_timeseries
from koala.white_light import (
    _compute_whitelight_gp_products,
    _save_whitelight_gp_database,
    _white_light_fingerprint_config,
)


def _base_params():
    return dict(
        GP_log_rho=jnp.log(0.12), GP_log_sigma=jnp.log(0.015),
        c=jnp.array(0.98), v=jnp.array(0.04),
    )


def _jaxoplanet_params():
    return {
        **_base_params(),
        "period": jnp.array([4.0]), "duration": jnp.array([0.16]),
        "t0": jnp.array([1.0]), "b": jnp.array([0.3]),
        "rors": jnp.array([0.1]), "u": jnp.array([0.2, 0.1]),
    }


def _harmonica_params():
    return {
        **_base_params(),
        "period": jnp.array([4.0]), "t0": jnp.array([1.0]),
        "b": jnp.array([0.3]), "rors": jnp.array([0.1]),
        "a_rs": jnp.array([10.0]), "c_ld": jnp.array(0.55),
        "alpha_ld": jnp.array(0.7),
    }


@pytest.mark.parametrize("params_factory", [_jaxoplanet_params, _harmonica_params])
@pytest.mark.parametrize("scalar_error", [False, True])
def test_white_light_products_match_legacy_training_condition(params_factory,
                                                               scalar_error):
    params = params_factory()
    t = jnp.linspace(0.88, 1.16, 31)
    error = jnp.array(8e-4) if scalar_error else jnp.linspace(7e-4, 9e-4, t.size)
    mean = kgp.compute_lc_linear_gp_mean(params, t, t_ref=jnp.min(t))
    y = mean + 4e-4 * jnp.sin(jnp.linspace(0.0, 4.0, t.size))
    products = _compute_whitelight_gp_products(
        params, t, error, y, detrend_type="linear+gp", gp_solver="serial"
    )
    gp = kgp.build_gp_linear(
        params, t, error, gp_solver="serial", assume_sorted=True
    )
    legacy = gp.condition(y).gp
    np.testing.assert_allclose(products["mu"], legacy.loc, rtol=0, atol=1e-9)
    np.testing.assert_allclose(products["var"], legacy.variance, rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        products["gp_stochastic_component"],
        products["mu"] - mean,
        rtol=0,
        atol=1e-12,
    )


def test_saved_white_light_csv_columns_and_lengths_are_unchanged(tmp_path):
    n = 7
    values = np.linspace(0.9, 1.1, n)
    path = tmp_path / "white_light.csv"
    save_whitelight_timeseries(
        time=values,
        flux=np.ones(n),
        flux_err=np.full(n, 1e-3),
        bestfit_model=np.ones(n),
        output_csv=path,
        t0_reference=1.0,
        transit_model=np.ones(n),
        trend_model=np.zeros(n),
        residual=np.zeros(n),
        outlier_mask=np.zeros(n, dtype=bool),
        gp_flux=np.ones(n),
        gp_err=np.full(n, 2e-4),
        gp_trend=np.zeros(n),
    )
    frame = pd.read_csv(path)
    assert list(frame.columns) == [
        "time_bjd", "time_from_t0_hr", "flux", "flux_err",
        "bestfit_model", "residual", "residual_ppm", "is_outlier",
        "transit_model", "trend_model", "detrended_flux", "gp_flux",
        "gp_err", "gp_trend",
    ]
    assert len(frame) == n


def test_gp_database_columns_and_fingerprint_ignore_solver(tmp_path):
    n = 6
    path = tmp_path / "gp.csv"
    products = {
        "mu": jnp.ones(n),
        "var": jnp.full(n, 4e-8),
        "gp_stochastic_component": jnp.arange(n, dtype=float),
    }
    _save_whitelight_gp_database(path, jnp.linspace(0.99, 1.01, n), products)
    frame = pd.read_csv(path, index_col=0)
    assert list(frame.columns) == ["wl_flux", "gp_flux", "gp_err", "gp_trend"]
    assert len(frame) == n
    base = {"instrument": "NIRSPEC/PRISM", "gp_solver": "serial"}
    changed = {**base, "gp_solver": "parallel"}
    assert _white_light_fingerprint_config(base) == {
        "instrument": "NIRSPEC/PRISM"
    }
    assert _white_light_fingerprint_config(base) == _white_light_fingerprint_config(changed)


def test_repair_gp_support_edges_moves_edge_starts_inside():
    from koala.white_light import _repair_gp_support_edges
    from models.gp import GP_HYPERPARAMETER_BOUNDS

    sigma_lo, sigma_hi = GP_HYPERPARAMETER_BOUNDS["GP_log_sigma"]
    rho_lo, rho_hi = GP_HYPERPARAMETER_BOUNDS["GP_log_rho"]
    soln = {
        "GP_log_sigma": jnp.asarray(sigma_lo + 1e-12),  # roundoff-level edge
        "GP_log_rho": jnp.asarray(rho_hi),  # exactly on the upper edge
        "c": jnp.asarray(1.0),
    }
    repaired = _repair_gp_support_edges(soln)
    assert soln["GP_log_sigma"] == sigma_lo + 1e-12  # input untouched
    assert sigma_lo < float(repaired["GP_log_sigma"]) < sigma_lo + 0.02 * (sigma_hi - sigma_lo)
    assert rho_hi - 0.02 * (rho_hi - rho_lo) < float(repaired["GP_log_rho"]) < rho_hi
    assert float(repaired["c"]) == 1.0

    interior = {"GP_log_sigma": jnp.asarray(-9.0), "GP_log_rho": jnp.asarray(-2.5)}
    unchanged = _repair_gp_support_edges(interior)
    assert float(unchanged["GP_log_sigma"]) == -9.0 and float(unchanged["GP_log_rho"]) == -2.5

    nonfinite = _repair_gp_support_edges({"GP_log_sigma": jnp.asarray(jnp.nan)})
    assert sigma_lo < float(nonfinite["GP_log_sigma"]) < sigma_hi
