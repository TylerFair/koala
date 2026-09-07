import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model

from models.common import get_I_power2
from models.cadence_reduction import build_explinear_oot_statistics
from models.detrend import _prepare_power2_poly
from models.jaxoplanet import build_transit_window_indices, create_vectorized_model
from models.jaxoplanet.core import (
    build_transit_phase_offsets,
    compute_transit_model,
)
from models.jaxoplanet.limb_dark_streamed import light_curve
from models.jaxoplanet.transit_grid import (
    _power2_duration_transit_primal,
    interpolate_power2_duration_transit,
)


jax.config.update("jax_enable_x64", True)


def _problem(cadence_count):
    time = np.linspace(-0.3, 0.3, cadence_count)
    period = np.array([2.0])
    t0 = np.array([0.0])
    duration = np.array([0.1])
    impact = np.array([0.3])
    window = build_transit_window_indices(time, period, t0, duration)
    exp_trend = np.exp(-(time - time.min()) / 0.03)
    error_pattern = np.array([0.0037, 0.0040, 0.0044, 0.0040])
    yerr = np.tile(error_pattern, cadence_count // 4 + 1)[:cadence_count][None, :]
    mus, projection = _prepare_power2_poly(degree=12)
    u = projection @ (1.0 - get_I_power2(0.61, 0.71, mus))
    phase, phase_mask = build_transit_phase_offsets(
        jnp.asarray(time), period, t0, duration
    )
    transit = compute_transit_model(
        {
            "period": jnp.asarray(period),
            "t0": jnp.asarray(t0),
            "b": jnp.asarray(impact),
            "duration": jnp.asarray(duration),
            "rors": jnp.asarray([0.1]),
            "u": u,
            "_ld_profile": "power2",
            "_jaxoplanet_kernel": "streamed",
            "_transit_phase_offsets": phase,
            "_transit_phase_mask": phase_mask,
            "_transit_window_indices": window,
        },
        jnp.asarray(time),
    )
    y = (
        1.001
        - 0.002 * (time - time.min())
        + 0.003 * exp_trend
        + np.asarray(transit)
        + 1.0e-4 * np.sin(np.arange(cadence_count))
    )[None, :]
    mask = np.ones_like(y, dtype=bool)
    mask[0, [0, cadence_count - 2]] = False
    common = {
        "y": jnp.asarray(y),
        "mu_duration": jnp.asarray(duration),
        "mu_t0": jnp.asarray(t0),
        "mu_b": jnp.asarray(impact),
        "mu_depths": jnp.array([[0.01]]),
        "PERIOD": jnp.asarray(period),
        "mu_u_ld": jnp.array([[0.61, 0.71]]),
        "sigma_u_ld": jnp.array([[0.02, 0.02]]),
        "precomputed_yerr_per_lc": jnp.median(jnp.asarray(yerr), axis=1),
        "exp_trend": jnp.asarray(exp_trend),
        "fixed_tau": 0.03,
        "likelihood_mask": jnp.asarray(mask),
    }
    initial = {
        "rors": jnp.array([[0.1]]),
        "c1": jnp.array([0.61]),
        "c2": jnp.array([0.71]),
        "c": jnp.array([1.001]),
        "v": jnp.array([-0.002]),
        "A": jnp.array([0.003]),
        "log_jitter": jnp.log(jnp.array([1.0e-3])),
    }
    reference_beta = np.column_stack((
        np.asarray(initial["c"]),
        np.asarray(initial["v"]),
        np.asarray(initial["A"]),
    ))
    statistics = build_explinear_oot_statistics(
        time, y, yerr, window, exp_trend, reference_beta, mask
    )
    return time, yerr, window, common, initial, statistics


def _potential(
    cadence_count,
    cadence_mode,
    grid_mode,
    *,
    non_grazing=True,
    grid_nodes=769,
):
    time, yerr, window, common, initial, statistics = _problem(cadence_count)
    model = create_vectorized_model(
        detrend_type="explinear_spectroscopic",
        ld_mode="stellarprior",
        trend_mode="free",
        ld_profile="power2",
        transit_window="auto",
        transit_window_indices=window,
        cadence_reduction=cadence_mode,
        transit_grid=grid_mode,
        transit_grid_nodes=grid_nodes,
        transit_grid_non_grazing=non_grazing,
    )
    info = initialize_model(
        jax.random.PRNGKey(1),
        model,
        model_args=(jnp.asarray(time), jnp.asarray(yerr)),
        model_kwargs={**common, **statistics},
        init_strategy=init_to_value(values=initial),
        validate_grad=False,
    )
    return jax.jit(jax.value_and_grad(info.potential_fn))(info.param_info.z)


def _assert_relative_potential_gradient_parity(candidate_modes):
    baseline_value, baseline_gradient = _potential(5001, "off", "off")
    reduced_value, reduced_gradient = _potential(5001, *candidate_modes)
    baseline_flat, _ = jax.flatten_util.ravel_pytree(baseline_gradient)
    reduced_flat, _ = jax.flatten_util.ravel_pytree(reduced_gradient)
    value_relative = abs(float(reduced_value - baseline_value)) / abs(
        float(baseline_value)
    )
    gradient_relative = np.linalg.norm(
        np.asarray(reduced_flat - baseline_flat)
    ) / np.linalg.norm(np.asarray(baseline_flat))
    assert value_relative <= 1.0e-9
    assert gradient_relative <= 1.0e-9


def test_long_exact_cadence_reduction_matches_direct_likelihood():
    _assert_relative_potential_gradient_parity(("auto", "off"))


def test_long_transit_grid_matches_direct_potential_and_gradient():
    _assert_relative_potential_gradient_parity(("off", "auto"))


def test_long_combined_reductions_match_direct_potential_and_gradient():
    _assert_relative_potential_gradient_parity(("auto", "auto"))


def test_short_auto_path_is_bit_identical_to_off():
    baseline_value, baseline_gradient = _potential(1025, "off", "off")
    reduced_value, reduced_gradient = _potential(1025, "auto", "auto")
    baseline_flat, _ = jax.flatten_util.ravel_pytree(baseline_gradient)
    reduced_flat, _ = jax.flatten_util.ravel_pytree(reduced_gradient)
    np.testing.assert_array_equal(reduced_value, baseline_value)
    np.testing.assert_array_equal(reduced_flat, baseline_flat)


def test_grid_auto_is_inactive_without_non_grazing_handoff():
    baseline_value, baseline_gradient = _potential(5001, "off", "off")
    candidate_value, candidate_gradient = _potential(
        5001, "off", "auto", non_grazing=False
    )
    baseline_flat, _ = jax.flatten_util.ravel_pytree(baseline_gradient)
    candidate_flat, _ = jax.flatten_util.ravel_pytree(candidate_gradient)
    np.testing.assert_array_equal(candidate_value, baseline_value)
    np.testing.assert_array_equal(candidate_flat, baseline_flat)


def test_forward_sensitivity_vjp_matches_ordinary_grid_reverse():
    phase = jnp.linspace(-0.0499, 0.0499, 401)
    mask = jnp.ones_like(phase, dtype=bool)
    weights = 0.75 + 0.25 * jnp.cos(jnp.arange(phase.size) * 0.017)
    theta = jnp.array([0.103, 0.53, 0.44])
    mus, projection = _prepare_power2_poly()

    def coefficients(values):
        return projection @ (
            1.0 - get_I_power2(values[1], values[2], mus)
        )

    def reference(values):
        flux = _power2_duration_transit_primal(
            light_curve,
            values,
            coefficients(values),
            phase,
            mask,
            duration=jnp.float64(0.1),
            impact=jnp.float64(0.12),
            num_nodes=769,
        )
        return jnp.sum(weights * flux)

    def candidate(values):
        flux = interpolate_power2_duration_transit(
            light_curve,
            values[1],
            values[2],
            coefficients(values),
            phase,
            mask,
            duration=jnp.float64(0.1),
            impact=jnp.float64(0.12),
            radius_ratio=values[0],
            num_nodes=769,
        )
        return jnp.sum(weights * flux)

    reference_value, reference_gradient = jax.value_and_grad(reference)(theta)
    candidate_value, candidate_gradient = jax.value_and_grad(candidate)(theta)
    value_relative = abs(float(candidate_value - reference_value)) / abs(
        float(reference_value)
    )
    gradient_relative = np.linalg.norm(
        np.asarray(candidate_gradient - reference_gradient)
    ) / np.linalg.norm(np.asarray(reference_gradient))
    assert value_relative <= 1.0e-9
    assert gradient_relative <= 1.0e-9
