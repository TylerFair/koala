"""Short quadratic free-epoch fits use the exact reduced, fused likelihood."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model

from models.cadence_reduction import (
    build_linear_oot_statistics,
    build_linear_spectro_trend_design,
)
from models.jaxoplanet import build_transit_window_indices, create_vectorized_model
from models.jaxoplanet import quadratic_cuda
from models.jaxoplanet.core import compute_transit_model


jax.config.update("jax_enable_x64", True)

HALF_WIDTH = 1 / 48
TRENDS = (
    "linear",
    "spot_spectroscopic",
    "explinear_spectroscopic",
    "linear_discontinuity_spectroscopic",
    "quadratic+2spot_spectroscopic",
)
VALUES = dict(c=1.001, v=-.002, v2=.004, A=.003, A_spot=1.04, A_spot2=.93, A_jump=1.02)


def _problem(detrend_type, n=801):
    t = np.linspace(-.3, .3, n)
    bases = dict(
        exp_trend=np.exp(-(t - t.min()) / .03),
        spot_trend=7e-4 * np.exp(-.5 * ((t - .02) / .01) ** 2),
        spot_trend2=-5e-4 * np.exp(-.5 * ((t + .03) / .008) ** 2),
        jump_trend=4e-4 * (t > .05),
    )
    names, design = build_linear_spectro_trend_design(detrend_type, t, **bases)
    beta = np.array([[VALUES[k] for k in names], [VALUES[k] * 1.01 for k in names]])
    rors, dt0, u = [.12, .121], [5 / 86400, -3 / 86400], [[.3, .2], [.32, .18]]
    transit = np.stack([np.asarray(compute_transit_model(dict(
        period=jnp.array([4.]), t0=jnp.array([offset]), a_rs=jnp.array([9.]),
        b=jnp.array([.35]), rors=jnp.array([r]), u=jnp.asarray(coefficients),
        _jaxoplanet_kernel="quadratic_contact_safe"), jnp.asarray(t)))
        for r, offset, coefficients in zip(rors, dt0, u)])
    y = (1 + transit) * (beta @ design.T) + 1e-4 * np.sin(np.arange(n))[None]
    # Constant errors in one channel, a repeating pattern in the other.
    yerr = np.stack([np.full(n, 3e-4), np.tile([3e-4, 3.3e-4, 2.9e-4], n)[:n]])
    mask = np.ones_like(y, dtype=bool)
    mask[0, [3, n - 5]] = False
    window = build_transit_window_indices(t, np.array([4.]), np.array([0.]), .2 + 2 * HALF_WIDTH)
    initial = dict(
        rors=jnp.array([[.1201], [.1208]]), dt0=jnp.array([[4.], [-2.]]) / 86400,
        u=jnp.array([[.31, .19], [.31, .19]]), log_jitter=jnp.full(2, -9.),
        **{k: jnp.asarray(beta[:, i] * (1 + 1e-4)) for i, k in enumerate(names)},
    )
    statistics = build_linear_oot_statistics(t, y, yerr, window, design, beta, mask)
    kwargs = dict(
        y=jnp.asarray(y), likelihood_mask=jnp.asarray(mask), PERIOD=jnp.array([4.]),
        mu_t0=jnp.array([0.]), mu_a_rs=jnp.array([9.]), mu_b=jnp.array([.35]),
        mu_duration=jnp.array([.15]), mu_u_ld=jnp.array([[.3, .2], [.32, .18]]),
        sigma_u_ld=jnp.array([[.05, .05], [.05, .05]]), fixed_tau=.03,
        **{k: jnp.asarray(v) for k, v in bases.items()},
    )
    return t, yerr, window, design, statistics, kwargs, initial


def _model(detrend_type, window, cadence_reduction):
    return create_vectorized_model(
        detrend_type=detrend_type, ld_mode="gaussian", trend_mode="free",
        ld_profile="quadratic", param_method="a_rs", transit_window="auto",
        transit_window_indices=window, cadence_reduction=cadence_reduction,
        surface_config={"model": "transit", "spectro_free_t0": True,
                        "spectro_t0_half_width_days": HALF_WIDTH},
    )


def _potential(detrend_type, reduced, accelerated=True):
    t, yerr, window, design, statistics, kwargs, initial = _problem(detrend_type)
    if reduced:
        kwargs = {**kwargs, **statistics, "trend_design": jnp.asarray(design)}
    original = quadratic_cuda.accelerated
    quadratic_cuda.accelerated = lambda: accelerated
    try:
        model = _model(detrend_type, window, "auto" if reduced else "off")
    finally:
        quadratic_cuda.accelerated = original
    trace = handlers.trace(handlers.substitute(handlers.seed(model, 0), data=initial)).get_trace(
        jnp.asarray(t), jnp.asarray(yerr), **kwargs)
    info = initialize_model(
        jax.random.PRNGKey(3), model, model_args=(jnp.asarray(t), jnp.asarray(yerr)),
        model_kwargs=kwargs, init_strategy=init_to_value(values=initial), validate_grad=False,
    )
    value, gradient = jax.jit(jax.value_and_grad(info.potential_fn))(info.param_info.z)
    return trace, value, jax.flatten_util.ravel_pytree(gradient)[0]


@pytest.mark.parametrize("detrend_type", TRENDS)
def test_short_free_epoch_reduction_matches_full_likelihood(monkeypatch, detrend_type):
    calls = []
    fused = quadratic_cuda.log_likelihood
    monkeypatch.setattr(quadratic_cuda, "log_likelihood",
                        lambda *args: calls.append(args[5].shape) or fused(*args))
    full_trace, full_value, full_gradient = _potential(detrend_type, False)
    trace, value, gradient = _potential(detrend_type, True)
    assert "obs" in full_trace and "obs" not in trace
    assert {"obs_active", "obs_out_of_window"} <= set(trace)
    columns = len(build_linear_spectro_trend_design(
        detrend_type, np.zeros(3), exp_trend=np.zeros(3), spot_trend=np.zeros(3),
        spot_trend2=np.zeros(3), jump_trend=np.zeros(3))[0])
    assert calls and all(shape[-1] == columns for shape in calls)
    assert abs(float(value - full_value)) <= 1e-9 * max(1., abs(float(full_value)))
    assert (np.linalg.norm(np.asarray(gradient - full_gradient))
            <= 1e-9 * np.linalg.norm(np.asarray(full_gradient)))


def test_short_light_curves_keep_full_likelihood_without_cuda():
    trace, value, gradient = _potential("spot_spectroscopic", True, accelerated=False)
    assert "obs" in trace and "obs_active" not in trace


@pytest.mark.parametrize("free_t0,accelerated,expected",
                         [(True, True, True), (True, False, False), (False, True, False)])
def test_short_light_curves_prepare_statistics_only_for_fused_quadratic(
        monkeypatch, free_t0, accelerated, expected):
    import koala.spectroscopy as spectroscopy
    monkeypatch.setattr(quadratic_cuda, "accelerated", lambda: accelerated)
    t, yerr, window, design, statistics, kwargs, initial = _problem("linear")
    model = SimpleNamespace(transit_engine="jaxoplanet", ld_profile="quadratic",
                            _engine_spectro_kw={"cadence_reduction": "auto"})
    inputs = SimpleNamespace(flags={"spectro_free_t0": free_t0})
    fit = SimpleNamespace(time=t, trend_mode="free", transit_window_indices=window,
                          detrend_type="linear",
                          init_params={"c": initial["c"], "v": initial["v"]})
    model_kwargs = {}
    spectroscopy._prepare_trend_likelihood(
        model, inputs, fit, np.asarray(kwargs["y"]), yerr, model_kwargs)
    assert ("oot_reference_beta" in model_kwargs) is expected
    assert ("trend_design" in model_kwargs) is expected
