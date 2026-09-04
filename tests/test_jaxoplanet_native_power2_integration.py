import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("jaxoplanet")
numpyro = pytest.importorskip("numpyro")

jax.config.update("jax_enable_x64", True)

from models.jaxoplanet import builder
from models.jaxoplanet.core import (
    build_transit_phase_offsets,
    build_transit_window_indices,
    compute_transit_model,
    resolve_jaxoplanet_kernel,
)
from models.jaxoplanet.experimental_power2_native import light_curve as native_light_curve


def _direct_params():
    return {
        "period": jnp.asarray([3.1], dtype=jnp.float64),
        "duration": jnp.asarray([0.12], dtype=jnp.float64),
        "t0": jnp.asarray([0.0], dtype=jnp.float64),
        "b": jnp.asarray([0.42], dtype=jnp.float64),
        "rors": jnp.asarray([0.12], dtype=jnp.float64),
        "c1": jnp.float64(0.61),
        "c2": jnp.float64(0.73),
        "_jaxoplanet_kernel": "native_power2",
        "_ld_profile": "power2",
    }


def test_native_power2_core_uses_direct_coefficients_with_and_without_window():
    times = jnp.linspace(-0.18, 0.18, 121, dtype=jnp.float64)
    params = _direct_params()
    dt, mask = build_transit_phase_offsets(
        times, params["period"], params["t0"], params["duration"]
    )
    speed = 2.0 * jnp.sqrt(
        (1.0 + params["rors"][0]) ** 2 - params["b"][0] ** 2
    ) / params["duration"][0]
    separation = jnp.sqrt((speed * dt[0]) ** 2 + params["b"][0] ** 2)
    expected = jnp.where(
        mask[0],
        native_light_curve(
            params["c1"], params["c2"], separation, params["rors"][0], order=16
        ),
        0.0,
    )

    actual = compute_transit_model(params, times)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)

    indices = build_transit_window_indices(
        np.asarray(times),
        np.asarray(params["period"]),
        np.asarray(params["t0"]),
        np.asarray(params["duration"]),
    )
    accelerated = compute_transit_model(
        {
            **params,
            "_transit_phase_offsets": dt,
            "_transit_phase_mask": mask,
            "_transit_window_indices": jnp.asarray(indices, dtype=jnp.int32),
        },
        times,
    )
    np.testing.assert_allclose(accelerated, actual, rtol=0.0, atol=2.0e-15)


def test_native_power2_routing_and_unsupported_builder_modes():
    assert resolve_jaxoplanet_kernel(
        "native_power2", ld_profile="power2", degree=None
    ) == "native_power2"
    assert resolve_jaxoplanet_kernel(
        "native_power2", ld_profile="power2", degree=12, keplerian=True
    ) == "stock"
    assert resolve_jaxoplanet_kernel(
        "native_power2", ld_profile="quadratic", degree=2
    ) == "stock"

    with pytest.raises(ValueError, match="ld_profile='power2'"):
        builder.create_vectorized_model(
            ld_profile="quadratic", jaxoplanet_kernel="native_power2"
        )
    with pytest.raises(ValueError, match="param_method='duration'"):
        builder.create_vectorized_model(
            ld_profile="power2",
            param_method="a_rs",
            jaxoplanet_kernel="native_power2",
        )
    with pytest.raises(ValueError, match="interpolated"):
        builder.create_vectorized_model(
            ld_profile="power2",
            ld_mode="interpolated",
            jaxoplanet_kernel="native_power2",
        )


def test_native_power2_refuses_float32_execution():
    was_enabled = bool(jax.config.x64_enabled)
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="requires JAX float64"):
            resolve_jaxoplanet_kernel(
                "native_power2", ld_profile="power2", degree=None
            )
    finally:
        jax.config.update("jax_enable_x64", was_enabled)


def test_native_power2_builder_skips_polynomial_projection(monkeypatch, capsys):
    def projection_must_not_run(*args, **kwargs):
        raise AssertionError("degree-12 Power-2 projection was called")

    monkeypatch.setattr(builder, "_prepare_power2_poly", projection_must_not_run)
    model = builder.create_vectorized_model(
        detrend_type="none",
        ld_mode="fixed",
        n_planets=1,
        ld_profile="power2",
        param_method="duration",
        jaxoplanet_kernel="native_power2",
    )
    assert callable(model)
    builder_log = capsys.readouterr().out
    assert "kernel=" not in builder_log
    assert "param_method=" not in builder_log

    times = jnp.linspace(-0.09, 0.09, 31, dtype=jnp.float64)
    yerr = jnp.full((2, times.size), 1.5e-4, dtype=jnp.float64)
    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model, rng_seed=321)
    ).get_trace(
        times,
        yerr,
        y=jnp.ones_like(yerr),
        mu_duration=jnp.asarray([0.12], dtype=jnp.float64),
        mu_t0=jnp.asarray([0.0], dtype=jnp.float64),
        mu_b=jnp.asarray([0.43], dtype=jnp.float64),
        mu_depths=jnp.asarray([[0.014], [0.014]], dtype=jnp.float64),
        PERIOD=jnp.asarray([3.1], dtype=jnp.float64),
        ld_fixed=jnp.asarray([[0.58, 0.72], [0.63, 0.67]], dtype=jnp.float64),
        precomputed_yerr_per_lc=jnp.full((2,), 1.5e-4, dtype=jnp.float64),
    )
    assert "c1" in trace and "c2" in trace
    assert "u" not in trace
    assert bool(jnp.all(jnp.isfinite(trace["obs"]["fn"].loc)))

    white_model = builder.create_whitelight_model(
        detrend_type="none",
        ld_mode="fixed",
        n_planets=1,
        ld_profile="power2",
        param_method="duration",
        jaxoplanet_kernel="native_power2",
    )
    assert callable(white_model)
    white_builder_log = capsys.readouterr().out
    assert "kernel=" not in white_builder_log
    assert "param_method=" not in white_builder_log
    white_trace = numpyro.handlers.trace(
        numpyro.handlers.seed(white_model, rng_seed=123)
    ).get_trace(
        times,
        yerr[0],
        y=jnp.ones_like(times),
        prior_params={
            "period": jnp.asarray([3.1], dtype=jnp.float64),
            "u": jnp.asarray([0.61, 0.73], dtype=jnp.float64),
        },
    )
    assert "c1" in white_trace and "c2" in white_trace
    assert "u" not in white_trace
    assert bool(jnp.all(jnp.isfinite(white_trace["obs"]["fn"].loc)))


def test_native_power2_requires_direct_coefficients():
    params = _direct_params()
    params.pop("c2")
    with pytest.raises(ValueError, match="params\['c1'\].*params\['c2'\]"):
        compute_transit_model(params, jnp.asarray([0.0], dtype=jnp.float64))
