"""Configurable white-light trend priors (the top-level ``trend_priors`` block)."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, substitute, trace
from numpyro.infer import Predictive
from numpyro.infer.util import log_density

from koala.config import parse_trend_priors, trend_prior_tau_range
from koala.surface import _scan_tau_grid
from koala.white_light import _apply_trend_prior_starts
from models.jaxoplanet.builder import create_whitelight_model
from models.model_building import sample_whitelight_trend
from models.trends import compute_lc_explinear
from planet_specs import default_planet_specs

jax.config.update("jax_enable_x64", True)

SPECS = default_planet_specs(period=3.0, t0=1.05, b=0.2, rprs=0.1, duration=0.12)
TIME = jnp.linspace(1.0, 1.1, 64)
PRIOR = {"period": jnp.array([3.0]), "u": jnp.array([0.2, 0.1])}
PAPER = {
    "c": {"prior": "uniform", "low": 0.9, "high": 1.1},
    "v": {"prior": "gaussian", "value": 0.0, "sigma": 0.01},
    "A": {"prior": "uniform", "low": -0.01, "high": 0.01},
    "tau": {"prior": "uniform", "low": 0.0, "high": 0.55},
}
# The built-in starts used by koala.white_light for an explinear fit.
DEFAULT_INIT = {"c": 1.0, "v": 0.0, "A": 0.001, "log_tau": jnp.log(1e-2)}
GEOMETRY = {"t0_0": 1.05, "rors_0": 0.1, "_b_0": 0.2, "logD_0": jnp.log(0.12),
            "log_jitter": jnp.log(1e-4)}


def _trend_trace(trend_priors=None):
    def model():
        sample_whitelight_trend(TIME, {}, {"explinear"}, "physical", "free",
                                trend_priors=trend_priors)
    return trace(seed(model, jax.random.PRNGKey(0))).get_trace()


def _model(trend_priors=None):
    kwargs = dict(detrend_type="explinear", ld_mode="fixed", ld_profile="quadratic",
                  parameter_priors=SPECS)
    if trend_priors is not None:
        kwargs["trend_priors"] = trend_priors
    return create_whitelight_model(**kwargs)


def _data():
    params = {"period": jnp.array([3.0]), "t0": jnp.array([1.05]), "duration": jnp.array([0.12]),
              "b": jnp.array([0.2]), "rors": jnp.array([0.1]), "u": jnp.array([0.2, 0.1]),
              "c": 1.0, "v": 0.002, "A": 0.004, "tau": 0.02}
    try:
        flux = compute_lc_explinear(params, TIME)
    except Exception:  # pragma: no cover - kernel needs builder-only keys
        flux = 1.0 + 0.002 * (TIME - TIME[0]) + 0.004 * jnp.exp(-(TIME - TIME[0]) / 0.02)
    return jnp.asarray(flux), jnp.full_like(TIME, 2e-4)


def _log_density(model, values):
    y, yerr = _data()
    return log_density(model, (TIME, yerr), {"y": y, "prior_params": PRIOR}, values)[0]


def test_absent_block_keeps_builtin_sites_and_priors():
    assert parse_trend_priors(None) == {}
    assert parse_trend_priors({}) == {}
    sites = _trend_trace()
    assert {name for name, site in sites.items() if site["type"] == "sample"} == {
        "c", "v", "A", "log_tau"}
    assert sites["tau"]["type"] == "deterministic"
    expected = {"c": (0.9, 1.1), "v": (-0.1, 0.1), "A": (-0.1, 0.1),
                "log_tau": (np.log(1e-3), np.log(1e-1))}
    for name, (low, high) in expected.items():
        fn = sites[name]["fn"]
        assert isinstance(fn, dist.Uniform)
        assert float(fn.low) == low and float(fn.high) == high
    np.testing.assert_allclose(sites["tau"]["value"], jnp.exp(sites["log_tau"]["value"]), rtol=0)


def test_default_log_density_unchanged():
    point = {"c": 1.01, "v": 0.01, "A": 0.003, "log_tau": jnp.log(0.02)}
    sites = trace(substitute(seed(lambda: sample_whitelight_trend(
        TIME, {}, {"explinear"}, "physical", "free"), jax.random.PRNGKey(0)), point)).get_trace()
    total = sum(site["fn"].log_prob(site["value"]) for site in sites.values()
                if site["type"] == "sample")
    old = (-np.log(0.2) - np.log(0.2) - np.log(0.2) - np.log(np.log(1e-1) - np.log(1e-3)))
    np.testing.assert_allclose(float(total), old, rtol=1e-14)

    # A full model with no trend_priors keyword and an explicit restatement of
    # the defaults give the same sites and the same joint log-density.
    restated = parse_trend_priors({
        "c": {"prior": "uniform", "low": 0.9, "high": 1.1},
        "v": {"prior": "uniform", "low": -0.1, "high": 0.1},
        "A": {"prior": "uniform", "low": -0.1, "high": 0.1},
        "tau": {"prior": "log_uniform", "low": 1e-3, "high": 1e-1},
    })
    assert {name: spec.value for name, spec in restated.items()} == {
        "c": 1.0, "v": 0.0, "A": 0.001, "tau": 1e-2}
    assert set(_trend_trace(restated)) == set(_trend_trace())
    values = {**point, **GEOMETRY}
    reference = _log_density(_model(), values)
    np.testing.assert_allclose(float(_log_density(_model(restated), values)), float(reference),
                               rtol=1e-14)
    init = dict(DEFAULT_INIT)
    _apply_trend_prior_starts(init, restated)
    assert set(init) == set(DEFAULT_INIT)
    np.testing.assert_allclose(float(init["log_tau"]), float(DEFAULT_INIT["log_tau"]), rtol=0)


def test_paper_priors_build_sample_and_start_finite():
    specs = parse_trend_priors(PAPER)
    assert specs["tau"].prior == "uniform" and specs["tau"].value == 1e-2
    assert specs["A"].value == 0.001 and specs["c"].value == 1.0
    sites = _trend_trace(specs)
    assert "log_tau" not in sites
    assert sites["tau"]["type"] == "sample"
    assert isinstance(sites["v"]["fn"], dist.Normal) and float(sites["v"]["fn"].scale) == 0.01
    assert float(sites["A"]["fn"].low) == -0.01 and float(sites["tau"]["fn"].high) == 0.55

    model = _model(specs)
    y, yerr = _data()
    draws = Predictive(model, num_samples=200)(
        jax.random.PRNGKey(1), TIME, yerr, prior_params=PRIOR)
    assert np.all((draws["tau"] > 0.0) & (draws["tau"] < 0.55))
    assert np.all(np.abs(draws["A"]) <= 0.01)
    assert np.all(np.isfinite(draws["obs"]))

    init = dict(DEFAULT_INIT)
    _apply_trend_prior_starts(init, specs)
    assert "log_tau" not in init and init["tau"] == 1e-2
    start = {k: jnp.asarray(v) for k, v in {**init, **GEOMETRY}.items()}
    value, grad = jax.value_and_grad(lambda p: _log_density(model, p))(start)
    assert np.isfinite(float(value))
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree_util.tree_leaves(grad))

    # tau -> 0 stays finite thanks to the kernel floor.
    tiny = {**start, "tau": jnp.asarray(1e-12)}
    assert np.isfinite(float(_log_density(model, tiny)))


def test_start_falls_back_to_midpoint_and_errors_name_block():
    specs = parse_trend_priors({"tau": {"prior": "uniform", "low": 0.1, "high": 0.5},
                                "A": {"prior": "log_uniform", "low": 1e-4, "high": 1e-2}})
    assert specs["tau"].value == pytest.approx(0.3)
    assert specs["A"].value == pytest.approx(1e-3)
    with pytest.raises(ValueError, match="trend_priors: unknown keys"):
        parse_trend_priors({"tau0": {"prior": "fixed", "value": 0.1}})
    with pytest.raises(ValueError, match="trend_priors.tau"):
        parse_trend_priors({"tau": {"prior": "uniform", "low": 0.0}})
    with pytest.raises(ValueError, match="trend_priors.tau"):
        parse_trend_priors({"tau": {"prior": "fixed", "value": 0.0}})


def test_eclipse_scan_tau_grid_covers_configured_range():
    np.testing.assert_array_equal(_scan_tau_grid(trend_prior_tau_range({})),
                                  np.geomspace(1e-3, 1e-1, 25))
    low, high = trend_prior_tau_range(parse_trend_priors(PAPER))
    grid = _scan_tau_grid((low, high))
    assert low > 0.0 and grid[0] == pytest.approx(low) and grid[-1] == pytest.approx(0.55)
