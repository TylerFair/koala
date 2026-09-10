"""Koala's systematics are multiplicative: F(t) = (1 + transit(t)) * trend(t)."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import models.gp as gp_module
import models.trends as trends_module
from models.common import apply_systematics
from models.linear_marginalization import conditional_coefficients
from models.trend_marginal import build_marginalized_trend_design
from koala.outputs import save_whitelight_timeseries


T = jnp.linspace(0.0, 0.3, 40)


def _fake_signal(params, t):
    # A box-like dip: 0 out of transit, -0.02 in the middle third.
    t = jnp.asarray(t)
    return jnp.where((t > 0.1) & (t < 0.2), -0.02, 0.0)


def test_apply_systematics_is_multiplicative():
    lc = jnp.array([0.0, -0.02, 0.0])
    trend = jnp.array([1.0, 1.01, 0.99])
    np.testing.assert_allclose(apply_systematics(lc, trend), (1.0 + lc) * trend)


@pytest.mark.parametrize("name, params, extra", [
    ("linear", {"c": 1.003, "v": -0.02}, ()),
    ("quadratic", {"c": 1.003, "v": -0.02, "v2": 0.05}, ()),
    ("explinear", {"c": 1.003, "v": -0.02, "A": 0.004, "tau": 0.05}, ()),
    ("spot", {"c": 1.003, "v": -0.02, "spot_amp": 0.002, "spot_mu": 0.15, "spot_sigma": 0.01}, ()),
    ("linear_discontinuity", {"c": 1.003, "v": -0.02, "jump": 0.001, "t_jump": 0.12, "width": 0.005}, ()),
])
def test_jaxoplanet_trend_kernels_multiply_the_transit(monkeypatch, name, params, extra):
    monkeypatch.setattr(trends_module, "compute_transit_model_auto", _fake_signal)
    kernel = getattr(trends_module, f"compute_lc_{name}")
    model = np.asarray(kernel(params, T, *extra))
    t_norm = np.asarray(T - jnp.min(T))
    trend = params["c"] + params["v"] * t_norm
    if "v2" in params:
        trend = trend + params["v2"] * t_norm**2
    if "A" in params:
        trend = trend + params["A"] * np.exp(-t_norm / params["tau"])
    if "spot_amp" in params:
        trend = trend + params["spot_amp"] * np.exp(-0.5 * (np.asarray(T) - params["spot_mu"]) ** 2 / params["spot_sigma"] ** 2)
    if "jump" in params:
        trend = trend + params["jump"] / (1.0 + np.exp(-(np.asarray(T) - params["t_jump"]) / params["width"]))
    expected = (1.0 + np.asarray(_fake_signal(params, T))) * trend
    np.testing.assert_allclose(model, expected, rtol=1e-12, atol=1e-14)
    # In-transit points differ from the additive form by transit * (trend - 1).
    assert not np.allclose(model, np.asarray(_fake_signal(params, T)) + trend)


def test_spectroscopic_template_kernels_keep_templates_inside_the_trend(monkeypatch):
    monkeypatch.setattr(trends_module, "compute_transit_model_auto", _fake_signal)
    template = jnp.sin(T * 30.0) * 1e-3
    params = {"c": 0.999, "A_spot": 1.2}
    model = trends_module.compute_lc_spot_spectroscopic(params, T, template)
    expected = (1.0 + _fake_signal(params, T)) * (params["c"] + params["A_spot"] * template)
    np.testing.assert_allclose(np.asarray(model), np.asarray(expected), rtol=1e-12)


def test_gp_mean_and_spectroscopic_gp_kernels_are_multiplicative(monkeypatch):
    monkeypatch.setattr(gp_module, "compute_transit_model_auto", _fake_signal)
    params = {"c": 1.002, "v": -0.01, "A_gp": 0.9}
    gp_template = jnp.cos(T * 20.0) * 2e-4
    mean = gp_module.compute_lc_linear_gp_mean(params, T)
    expected_mean = (1.0 + _fake_signal(params, T)) * (params["c"] + params["v"] * (T - jnp.min(T)))
    np.testing.assert_allclose(np.asarray(mean), np.asarray(expected_mean), rtol=1e-12)
    spectro = gp_module.compute_lc_linear_gp_spectroscopic(params, T, gp_template)
    expected = (1.0 + _fake_signal(params, T)) * (
        params["c"] + params["v"] * (T - jnp.min(T)) + params["A_gp"] * gp_template
    )
    np.testing.assert_allclose(np.asarray(spectro), np.asarray(expected), rtol=1e-12)


def test_marginalized_trend_recovers_multiplicative_coefficients():
    """With a broad prior the conditional mean is the least-squares fit of
    y = (1 + transit) * (c + v t), i.e. the design scaled by the transit."""
    transit = np.asarray(_fake_signal({}, T))
    t_norm = np.asarray(T - jnp.min(T))
    true_beta = np.array([1.004, -0.03])
    y = (1.0 + transit) * (true_beta[0] + true_beta[1] * t_norm)
    design, names = build_marginalized_trend_design(
        "linear", T, 1, transit_factor=1.0 + transit,
    )
    assert names == ("c", "v")
    np.testing.assert_allclose(np.asarray(design[0, :, 0]), 1.0 + transit)
    conditional = conditional_coefficients(
        jnp.asarray(y)[None, :], design, jnp.full((1, T.size), 1e-4),
        jnp.array([[1.0, 0.0]]), prior_scale=jnp.array([[10.0, 10.0]]),
    )
    np.testing.assert_allclose(np.asarray(conditional.mean[0]), true_beta, rtol=1e-6)


def test_whitelight_timeseries_detrends_by_division(tmp_path):
    time = np.asarray(T)
    trend = 1.002 - 0.01 * (time - time.min())
    transit = 1.0 + np.asarray(_fake_signal({}, T))
    flux = transit * trend
    out = tmp_path / "wl.csv"
    save_whitelight_timeseries(time, flux, np.full_like(flux, 1e-4), flux, str(out),
                               transit_model=transit, trend_model=trend)
    import pandas as pd
    frame = pd.read_csv(out)
    np.testing.assert_allclose(frame["detrended_flux"].to_numpy(), transit, rtol=1e-12)
