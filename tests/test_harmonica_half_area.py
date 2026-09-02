import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from numpyro import handlers
from unittest.mock import patch

from fit_jwst import (
    _harmonica_checkpoint_prefix,
    _harmonica_coefficients_to_half_area_init,
    _harmonica_model_radius_samples,
    _harmonica_spectro_artifact_stem,
    _validate_harmonica_spectro_parameterization,
)
from models.harmonica.builder import (
    _sample_harmonica_half_area,
    create_vectorized_model,
)
from models.harmonica.core import (
    HARMONICA_HALF_AREA_CONTRAST_FACTOR,
    HARMONICA_HALF_AREA_CONVEX_Q_LIMIT,
    HARMONICA_SPECTRO_RORS_MIN,
    HARMONICA_SPECTRO_RORS_MAX,
    compute_transit_model_harmonica,
    harmonica_half_area_area_radius_and_q,
    harmonica_half_area_coefficients_from_area_radius,
    harmonica_half_area_depths,
    harmonica_half_area_q_from_ratio,
    harmonica_half_area_ratio_from_q,
)


def test_half_area_forward_inverse_round_trip_including_circular_limit():
    ratios = jnp.asarray([-0.49, -0.2, -1e-12, 0.0, 1e-12, 0.2, 0.49])
    q = harmonica_half_area_q_from_ratio(ratios)
    recovered = harmonica_half_area_ratio_from_q(q)
    np.testing.assert_allclose(recovered, ratios, rtol=2e-14, atol=2e-14)

    a0 = jnp.asarray([0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14])
    a1 = a0 * ratios
    area_radius, coefficient_q = harmonica_half_area_area_radius_and_q(a0, a1)
    recovered_a0, recovered_a1 = harmonica_half_area_coefficients_from_area_radius(
        area_radius, coefficient_q
    )
    np.testing.assert_allclose(recovered_a0, a0, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(recovered_a1, a1, rtol=3e-14, atol=3e-14)


def test_half_area_depths_are_exact_requested_representative_areas():
    area_radius = jnp.asarray([0.08, 0.10, 0.12, 0.14])
    q = jnp.asarray([-0.4, -0.05, 0.0, 0.4])
    a0, a1 = harmonica_half_area_coefficients_from_area_radius(area_radius, q)
    total, evening, morning = harmonica_half_area_depths(a0, a1)

    expected_total = area_radius**2
    np.testing.assert_allclose(total, expected_total, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(
        evening, expected_total * (1.0 + q), rtol=3e-14, atol=3e-14
    )
    np.testing.assert_allclose(
        morning, expected_total * (1.0 - q), rtol=3e-14, atol=3e-14
    )


def test_half_area_numpyro_trace_has_bounded_latent_and_physical_outputs():
    area_radius = jnp.asarray([[0.09], [0.11], [0.13]])
    frac_sigma = 0.07

    def model():
        _sample_harmonica_half_area(area_radius, frac_sigma=frac_sigma)

    trace = handlers.trace(handlers.seed(model, random.PRNGKey(17))).get_trace()
    assert trace["q"]["type"] == "sample"
    for name in (
        "a0",
        "a1",
        "depth_total_area",
        "depth_evening",
        "depth_morning",
    ):
        assert trace[name]["type"] == "deterministic"
        assert trace[name]["value"].shape == area_radius.shape

    q = np.asarray(trace["q"]["value"])
    assert np.all(np.abs(q) < HARMONICA_HALF_AREA_CONVEX_Q_LIMIT)
    np.testing.assert_allclose(
        np.asarray(trace["q"]["fn"].base_dist.scale),
        HARMONICA_HALF_AREA_CONTRAST_FACTOR * frac_sigma,
        rtol=1e-14,
        atol=1e-14,
    )
    total = np.asarray(trace["depth_total_area"]["value"])
    np.testing.assert_allclose(total, np.asarray(area_radius) ** 2)
    np.testing.assert_allclose(
        trace["depth_evening"]["value"], total * (1.0 + q)
    )
    np.testing.assert_allclose(
        trace["depth_morning"]["value"], total * (1.0 - q)
    )


def test_half_area_convex_bound_maps_strictly_inside_a1_over_a0_half():
    q = jnp.asarray(
        [-1.0, 1.0]
    ) * HARMONICA_HALF_AREA_CONVEX_Q_LIMIT * (1.0 - 1e-10)
    ratio = np.asarray(harmonica_half_area_ratio_from_q(q))
    assert np.all(np.abs(ratio) < 0.5)

    a0, a1 = harmonica_half_area_coefficients_from_area_radius(0.1, q)
    assert np.all(np.asarray(a0) > 2.0 * np.abs(np.asarray(a1)))


def test_half_area_fit_validation_initialization_and_radius_selection():
    assert _validate_harmonica_spectro_parameterization("half_area", 1) == "half_area"
    with pytest.raises(ValueError, match="requires.*max_order: 1"):
        _validate_harmonica_spectro_parameterization("half_area", 3)
    with pytest.raises(ValueError, match="must be one of"):
        _validate_harmonica_spectro_parameterization("unknown", 1)
    with pytest.raises(ValueError, match="only supports max_harmonic_order=1"):
        create_vectorized_model(
            max_harmonic_order=3, odd_parameterization="half_area"
        )

    wl_a0 = np.asarray([0.10, 0.12])
    wl_a1 = np.asarray([0.01, -0.02])
    init_radius, init_q = _harmonica_coefficients_to_half_area_init(wl_a0, wl_a1)
    recovered_a0, recovered_a1 = harmonica_half_area_coefficients_from_area_radius(
        init_radius, init_q
    )
    np.testing.assert_allclose(recovered_a0, wl_a0, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(recovered_a1, wl_a1, rtol=3e-14, atol=3e-14)

    # Even an unphysical WL coefficient is clipped to a finite point strictly
    # inside the latent support, never to the inverse-transform endpoint.
    _, clipped_q = _harmonica_coefficients_to_half_area_init(0.1, 0.08)
    assert np.isfinite(clipped_q)
    assert abs(float(clipped_q)) < HARMONICA_HALF_AREA_CONVEX_Q_LIMIT

    tiny_radius, _ = _harmonica_coefficients_to_half_area_init(0.0, 0.0)
    huge_radius, _ = _harmonica_coefficients_to_half_area_init(10.0, 1.0)
    assert HARMONICA_SPECTRO_RORS_MIN < float(tiny_radius)
    assert float(huge_radius) < HARMONICA_SPECTRO_RORS_MAX

    samples = {"rors": jnp.asarray([0.101]), "a0": jnp.asarray([0.099])}
    np.testing.assert_array_equal(
        _harmonica_model_radius_samples(samples, "half_area"), samples["a0"]
    )
    np.testing.assert_array_equal(
        _harmonica_model_radius_samples(samples, "delta_r"), samples["rors"]
    )
    assert _harmonica_checkpoint_prefix("run_R50", "harmonica", "half_area") == (
        "run_R50_half_area"
    )
    assert _harmonica_checkpoint_prefix("run_R50", "harmonica", "delta_r") == (
        "run_R50"
    )
    assert _harmonica_spectro_artifact_stem(
        "run_R50", "harmonica", "half_area"
    ) == "run_R50_half_area"


def test_vectorized_half_area_forward_consumes_a0_but_retains_total_area_rors():
    num_channels, n_planets, num_times = 3, 2, 7
    times = jnp.linspace(-0.03, 0.03, num_times)
    yerr = jnp.full((num_channels, num_times), 1e-4)
    area_radius = jnp.asarray(
        [[0.090, 0.070], [0.100, 0.075], [0.110, 0.080]]
    )
    q = jnp.asarray(
        [[0.20, -0.10], [0.15, -0.05], [0.10, 0.05]]
    )
    expected_a0, expected_a1 = harmonica_half_area_coefficients_from_area_radius(
        area_radius, q
    )
    captured = {}

    def fake_forward(params, t):
        captured.update(params)
        return jnp.zeros((num_channels, num_times), dtype=jnp.float64)

    model = create_vectorized_model(
        detrend_type="none",
        ld_mode="fixed",
        trend_mode="free",
        n_planets=n_planets,
        max_harmonic_order=1,
        odd_parameterization="half_area",
        fit_jitter=False,
        ld_profile="power2",
    )
    conditioned = handlers.condition(model, data={"rors": area_radius, "q": q})
    with patch(
        "models.harmonica.builder.compute_transit_model_harmonica_batched",
        side_effect=fake_forward,
    ):
        trace = handlers.trace(handlers.seed(conditioned, random.PRNGKey(23))).get_trace(
            times,
            yerr,
            y=jnp.ones_like(yerr),
            mu_duration=jnp.asarray([0.08, 0.09]),
            mu_t0=jnp.asarray([0.0, 0.01]),
            mu_b=jnp.asarray([0.2, 0.3]),
            mu_cos_i=jnp.asarray([0.02, 0.03]),
            mu_depths=area_radius**2,
            PERIOD=jnp.asarray([2.0, 3.0]),
            harmonica_a_rs=jnp.asarray([10.0, 12.0]),
            ld_fixed=jnp.asarray(
                [[0.5, 0.6], [0.52, 0.62], [0.54, 0.64]]
            ),
        )

    # The sampled/saved transmission radius is sqrt(total silhouette area).
    np.testing.assert_array_equal(trace["rors"]["value"], area_radius)
    np.testing.assert_allclose(trace["depths"]["value"], area_radius**2)
    np.testing.assert_allclose(trace["depth_total_area"]["value"], area_radius**2)

    # The actual Harmonica transmission string must instead receive a0.
    np.testing.assert_allclose(trace["a0"]["value"], expected_a0)
    np.testing.assert_allclose(trace["a1"]["value"], expected_a1)
    np.testing.assert_allclose(captured["rors"], expected_a0)
    np.testing.assert_allclose(captured["a1"], expected_a1)
    assert captured["rors"].shape == (num_channels, n_planets)


@pytest.mark.parametrize("ld_profile", ["power2", "quadratic"])
def test_real_half_area_builder_flux_matches_equivalent_fractional_model(ld_profile):
    """Exercise the real Harmonica forward, not a mocked parameter capture."""
    num_channels, num_times = 2, 31
    times = jnp.linspace(-0.035, 0.035, num_times)
    yerr = jnp.full((num_channels, num_times), 1e-4)
    area_radius = jnp.asarray([[0.095], [0.108]])
    q = jnp.asarray([[0.22], [-0.17]])
    a0, a1 = harmonica_half_area_coefficients_from_area_radius(area_radius, q)
    shared = dict(
        detrend_type="none",
        ld_mode="fixed",
        trend_mode="free",
        n_planets=1,
        max_harmonic_order=1,
        fit_jitter=False,
        ld_profile=ld_profile,
    )
    model_args = dict(
        mu_duration=jnp.asarray([0.09]),
        mu_t0=jnp.asarray([0.0]),
        mu_b=jnp.asarray([0.25]),
        mu_cos_i=jnp.asarray([0.025]),
        mu_depths=area_radius**2,
        PERIOD=jnp.asarray([2.75]),
        harmonica_a_rs=jnp.asarray([10.0]),
        ld_fixed=(
            jnp.asarray([[0.52, 0.68], [0.61, 0.73]])
            if ld_profile == "power2"
            else jnp.asarray([[0.27, 0.18], [0.34, 0.12]])
        ),
    )

    half_model = create_vectorized_model(
        odd_parameterization="half_area", **shared
    )
    fractional_model = create_vectorized_model(
        odd_parameterization="fractional", **shared
    )
    half_trace = handlers.trace(
        handlers.seed(
            handlers.condition(
                half_model, data={"rors": area_radius, "q": q}
            ),
            random.PRNGKey(51),
        )
    ).get_trace(times, yerr, y=jnp.ones_like(yerr), **model_args)
    fractional_trace = handlers.trace(
        handlers.seed(
            handlers.condition(
                fractional_model,
                data={"rors": a0, "a1_frac": a1 / a0},
            ),
            random.PRNGKey(52),
        )
    ).get_trace(times, yerr, y=jnp.ones_like(yerr), **model_args)

    np.testing.assert_allclose(
        np.asarray(half_trace["obs"]["fn"].loc),
        np.asarray(fractional_trace["obs"]["fn"].loc),
        rtol=2e-13,
        atol=2e-13,
    )


@pytest.mark.parametrize("ld_profile", ["power2", "quadratic"])
def test_real_half_area_forward_has_finite_and_finite_difference_gradients(ld_profile):
    times = jnp.linspace(-0.028, 0.031, 25, dtype=jnp.float64)
    weights = jnp.linspace(0.7, 1.3, times.size, dtype=jnp.float64)

    def weighted_signal(area_radius, q):
        a0, a1 = harmonica_half_area_coefficients_from_area_radius(
            area_radius, q
        )
        params = {
            "period": jnp.asarray([2.75]),
            "t0": jnp.asarray([0.0]),
            "b": jnp.asarray([0.25]),
            "a_rs": jnp.asarray([10.0]),
            "ecc": jnp.asarray([0.0]),
            "omega": jnp.asarray([0.0]),
            "rors": jnp.atleast_1d(a0),
            "a1": jnp.atleast_1d(a1),
        }
        if ld_profile == "power2":
            params.update({"c_ld": 0.55, "alpha_ld": 0.70})
        else:
            params.update({"u1_ld": 0.30, "u2_ld": 0.18})
        return jnp.vdot(weights, compute_transit_model_harmonica(params, times))

    area0 = jnp.asarray(0.102, dtype=jnp.float64)
    q0 = jnp.asarray(0.18, dtype=jnp.float64)
    grad_area, grad_q = jax.grad(weighted_signal, argnums=(0, 1))(area0, q0)
    assert np.isfinite(float(grad_area))
    assert np.isfinite(float(grad_q))

    area_step = 2e-6
    q_step = 2e-5
    fd_area = (
        weighted_signal(area0 + area_step, q0)
        - weighted_signal(area0 - area_step, q0)
    ) / (2 * area_step)
    fd_q = (
        weighted_signal(area0, q0 + q_step)
        - weighted_signal(area0, q0 - q_step)
    ) / (2 * q_step)
    np.testing.assert_allclose(grad_area, fd_area, rtol=3e-5, atol=2e-8)
    np.testing.assert_allclose(grad_q, fd_q, rtol=3e-5, atol=2e-8)

    edge = HARMONICA_HALF_AREA_CONVEX_Q_LIMIT * (1.0 - 1e-8)
    for edge_q in (-edge, 0.0, edge):
        edge_grad = jax.grad(weighted_signal, argnums=1)(area0, edge_q)
        assert np.isfinite(float(edge_grad))
