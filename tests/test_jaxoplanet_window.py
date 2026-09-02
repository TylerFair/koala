import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("jaxoplanet")
numpyro = pytest.importorskip("numpyro")

jax.config.update("jax_enable_x64", True)

from models.jaxoplanet.core import (
    build_transit_window_indices,
    compute_transit_model,
)
from models.jaxoplanet.builder import create_vectorized_model


def _params(**updates):
    params = {
        "period": jnp.array([4.05528043], dtype=jnp.float64),
        "duration": jnp.array([0.11693087083333333], dtype=jnp.float64),
        "t0": jnp.array([59787.055], dtype=jnp.float64),
        "b": jnp.array([0.4498], dtype=jnp.float64),
        "rors": jnp.array([0.1457], dtype=jnp.float64),
        "u": jnp.array([0.2, 0.1], dtype=jnp.float64),
    }
    params.update(updates)
    return params


def _windowed(params, times):
    indices = build_transit_window_indices(
        times, params["period"], params["t0"], params["duration"]
    )
    return {
        **params,
        "_transit_window_indices": jnp.asarray(indices, dtype=jnp.int32),
    }


def test_windowed_flux_matches_full_model_exactly():
    times = jnp.linspace(59786.87, 59787.21, 525, dtype=jnp.float64)
    params = _params()
    full = compute_transit_model(params, times)
    windowed = compute_transit_model(_windowed(params, times), times)
    np.testing.assert_array_equal(np.asarray(windowed), np.asarray(full))


def test_windowed_gradients_match_full_model():
    times = jnp.linspace(59786.87, 59787.21, 525, dtype=jnp.float64)
    base = _params()
    indices = build_transit_window_indices(
        times, base["period"], base["t0"], base["duration"]
    )

    def loss(rors, use_window):
        params = {**base, "rors": jnp.atleast_1d(rors)}
        if use_window:
            params["_transit_window_indices"] = jnp.asarray(indices)
        model = compute_transit_model(params, times)
        return jnp.sum(jnp.square(model))

    full_grad = jax.grad(loss)(base["rors"], False)
    window_grad = jax.grad(loss)(base["rors"], True)
    np.testing.assert_allclose(window_grad, full_grad, rtol=1e-11, atol=1e-11)


def test_window_builder_returns_union_for_multiple_planets():
    times = np.linspace(-0.3, 0.3, 601)
    indices = build_transit_window_indices(
        times,
        period=np.array([2.0, 3.0]),
        t0=np.array([-0.1, 0.15]),
        duration=np.array([0.04, 0.06]),
    )
    selected = times[indices]
    assert np.any(np.abs(selected + 0.1) <= 0.02 + 1e-12)
    assert np.any(np.abs(selected - 0.15) <= 0.03 + 1e-12)
    assert not np.any((selected > -0.07) & (selected < 0.11))


def test_window_builder_conservatively_keeps_duration_boundaries():
    times = np.array([-0.05, 0.0, 0.05], dtype=np.float64)
    indices = build_transit_window_indices(
        times, period=1.0, t0=0.0, duration=0.1
    )
    np.testing.assert_array_equal(indices, np.array([0, 1, 2], dtype=np.int32))


def test_empty_window_returns_zero_signal():
    times = jnp.linspace(-0.2, 0.2, 11, dtype=jnp.float64)
    params = _params(
        period=jnp.array([1.0]),
        duration=jnp.array([0.0]),
        t0=jnp.array([10.5]),
        _transit_window_indices=jnp.array([], dtype=jnp.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(compute_transit_model(params, times)),
        np.zeros(times.shape, dtype=np.float64),
    )


def test_keplerian_path_ignores_duration_window_indices():
    times = jnp.linspace(-0.08, 0.08, 81, dtype=jnp.float64)
    params = _params(
        period=jnp.array([3.0]),
        t0=jnp.array([0.0]),
        b=jnp.array([0.3]),
        rors=jnp.array([0.1]),
        a_rs=jnp.array([8.0]),
        ecc=jnp.array([0.0]),
        omega=jnp.array([0.0]),
    )
    expected = compute_transit_model(params, times)
    actual = compute_transit_model(
        {**params, "_transit_window_indices": jnp.array([40], dtype=jnp.int32)},
        times,
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def _trace_vectorized_model(model, times, yerr, **kwargs):
    seeded = numpyro.handlers.seed(model, rng_seed=123)
    return numpyro.handlers.trace(seeded).get_trace(
        times,
        yerr,
        y=jnp.ones_like(yerr),
        **kwargs,
    )


def test_vectorized_auto_window_and_precomputed_errors_match_full_path():
    times = jnp.linspace(-0.2, 0.2, 101, dtype=jnp.float64)
    yerr = jnp.stack(
        [
            jnp.linspace(1.0e-4, 2.0e-4, times.size),
            jnp.linspace(2.0e-4, 3.0e-4, times.size),
        ]
    )
    geometry = {
        "mu_duration": jnp.array([0.1]),
        "mu_t0": jnp.array([0.0]),
        "mu_b": jnp.array([0.4]),
        "mu_depths": jnp.array([[0.01], [0.01]]),
        "PERIOD": jnp.array([2.0]),
        "ld_fixed": jnp.array([[0.2, 0.1], [0.25, 0.05]]),
    }
    indices = build_transit_window_indices(
        times, geometry["PERIOD"], geometry["mu_t0"], geometry["mu_duration"]
    )
    full_model = create_vectorized_model(
        detrend_type="none",
        ld_mode="fixed",
        trend_mode="free",
        n_planets=1,
        ld_profile="quadratic",
        transit_window="off",
        transit_window_indices=np.array([times.size // 2]),
    )
    window_model = create_vectorized_model(
        detrend_type="none",
        ld_mode="fixed",
        trend_mode="free",
        n_planets=1,
        ld_profile="quadratic",
        transit_window="auto",
        transit_window_indices=indices,
    )
    full_trace = _trace_vectorized_model(full_model, times, yerr, **geometry)
    window_trace = _trace_vectorized_model(
        window_model,
        times,
        yerr,
        precomputed_yerr_per_lc=jnp.nanmedian(yerr, axis=1),
        **geometry,
    )

    np.testing.assert_array_equal(
        np.asarray(window_trace["obs"]["fn"].loc),
        np.asarray(full_trace["obs"]["fn"].loc),
    )
    np.testing.assert_array_equal(
        np.asarray(window_trace["obs"]["fn"].scale),
        np.asarray(full_trace["obs"]["fn"].scale),
    )
    scale = np.asarray(window_trace["obs"]["fn"].scale)
    jitter = np.exp(np.asarray(window_trace["log_jitter"]["value"]))
    expected_scale = np.sqrt(np.asarray(yerr) ** 2 + jitter[:, None] ** 2)
    np.testing.assert_allclose(scale, expected_scale, rtol=2e-15, atol=0.0)
    # The varying cadence uncertainties must not be collapsed to one value.
    assert np.ptp(scale[0]) > 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"period": [1.0, 2.0], "t0": [0.0, 0.1, 0.2], "duration": 0.1},
        {"period": 0.0, "t0": 0.0, "duration": 0.1},
        {"period": 1.0, "t0": 0.0, "duration": -0.1},
    ],
)
def test_window_builder_rejects_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        build_transit_window_indices(np.linspace(-1.0, 1.0, 9), **kwargs)
