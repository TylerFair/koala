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


def test_every_registered_kernel_multiplies_transit_and_new_entries_are_guarded(monkeypatch):
    """Walk real registries/resolvers, including future entries and composites."""
    import inspect
    import models.detrend as detrend_module

    params = dict(c=1.013, v=-0.023, v2=0.041, v3=-0.017, v4=0.009,
                  A=0.007, tau=0.043, spot_amp=0.004, spot_mu=0.14,
                  spot_sigma=0.019, spot_amp2=0.003, spot_mu2=0.18,
                  spot_sigma2=0.016, jump=0.006, t_jump=0.13, width=0.008,
                  A_spot=0.8, A_spot2=1.3, A_jump=0.7, A_gp=1.1,
                  transit_amplitude=0.027)

    def transit(values, t):
        return values['transit_amplitude'] * jnp.where(
            (t > 0.1) & (t < 0.2), -1.0, 0.0,
        )

    monkeypatch.setattr(trends_module, 'compute_transit_model_auto', transit)
    monkeypatch.setattr(gp_module, 'compute_transit_model_auto', transit)
    templates = dict(spot_trend=0.003 * jnp.sin(19 * T) + 0.001,
                     spot_trend2=0.002 * jnp.cos(23 * T) + 0.001,
                     jump_trend=0.004 * jax.nn.sigmoid((T - 0.13) / 0.008),
                     gp_trend=0.002 * jnp.cos(17 * T),
                     exp_trend=jnp.exp(-(T - T.min()) / 0.043),
                     t_ref=T.min())
    registries = (
        (detrend_module.COMPUTE_KERNELS, detrend_module.resolve_detrend_kernel,
         lambda key: key),
        (detrend_module.COMPOSITE_KERNELS, detrend_module.resolve_detrend_kernel,
         lambda key: '+'.join(sorted(key))),
        (gp_module.GP_MEAN_FUNCTIONS, gp_module.resolve_gp_mean_function,
         lambda key: key),
    )

    def check_all():
        for registry, resolve, public_name in registries:
            for key, registered in registry.items():
                name = public_name(key)
                kernel = resolve(name)
                assert kernel is registered, name
                extras = list(inspect.signature(kernel).parameters.values())[2:]
                # Unknown arguments must fail rather than silently omit a new kernel.
                assert all(arg.name in templates for arg in extras), name
                kwargs = {arg.name: templates[arg.name] for arg in extras}
                # Exercise optional exponential templates both supplied and absent.
                variants = [kwargs]
                if any(arg.name == 'exp_trend' and arg.default is None for arg in extras):
                    variants.append({k: v for k, v in kwargs.items() if k != 'exp_trend'})
                for variant in variants:
                    with_transit = np.asarray(kernel(params, T, **variant))
                    baseline = np.asarray(kernel(dict(params, transit_amplitude=0.0), T,
                                                 **variant))
                    assert np.all(np.isfinite(with_transit)), name
                    np.testing.assert_allclose(
                        with_transit, (1.0 + np.asarray(transit(params, T))) * baseline,
                        rtol=0.0, atol=1e-12, err_msg=name,
                    )

    check_all()

    def additive_mutant(values, t):
        return transit(values, t) + values['c'] + values['v'] * (t - t.min())

    # Prove newly registered additive kernels are picked up in every registry.
    for registry, _, _ in registries:
        key = (frozenset({'unguarded', 'new_kernel'})
               if registry is detrend_module.COMPOSITE_KERNELS else 'unguarded_new_kernel')
        with monkeypatch.context() as mutation:
            mutation.setitem(registry, key, additive_mutant)
            with pytest.raises(AssertionError, match='unguarded'):
                check_all()
