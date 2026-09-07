import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model

from models.common import get_I_power2
from models.cadence_reduction import (
    build_explinear_oot_statistics,
    build_linear_oot_statistics,
    build_linear_spectro_trend_design,
    linear_spectro_trend_coefficient_names,
)
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


def _problem(cadence_count, impact_value=0.3):
    time = np.linspace(-0.3, 0.3, cadence_count)
    period = np.array([2.0])
    t0 = np.array([0.0])
    duration = np.array([0.1])
    impact = np.array([impact_value])
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
    impact=0.3,
):
    time, yerr, window, common, initial, statistics = _problem(
        cadence_count, impact
    )
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
        transit_grid_outer_contact_safe=(
            abs(impact) < 1.0 + np.sqrt(1.0e-5) - 1.0e-5
        ),
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


def _assert_value_gradient_close(reference, candidate):
    reference_value, reference_gradient = reference
    candidate_value, candidate_gradient = candidate
    reference_flat, _ = jax.flatten_util.ravel_pytree(reference_gradient)
    candidate_flat, _ = jax.flatten_util.ravel_pytree(candidate_gradient)
    value_relative = abs(float(candidate_value - reference_value)) / max(
        1.0, abs(float(reference_value))
    )
    gradient_relative = np.linalg.norm(
        np.asarray(candidate_flat - reference_flat)
    ) / np.linalg.norm(np.asarray(reference_flat))
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


def test_grid_auto_handles_grazing_handoff():
    candidate_value, candidate_gradient = _potential(
        5001, "off", "auto", non_grazing=False, impact=0.95
    )
    baseline_value, baseline_gradient = _potential(
        5001, "off", "off", non_grazing=False, impact=0.95
    )
    baseline_flat, _ = jax.flatten_util.ravel_pytree(baseline_gradient)
    candidate_flat, _ = jax.flatten_util.ravel_pytree(candidate_gradient)
    value_relative = abs(float(candidate_value - baseline_value)) / abs(
        float(baseline_value)
    )
    gradient_relative = np.linalg.norm(
        np.asarray(candidate_flat - baseline_flat)
    ) / np.linalg.norm(np.asarray(baseline_flat))
    assert value_relative <= 1.0e-9
    assert gradient_relative <= 1.0e-9


def test_grid_auto_keeps_outer_contact_unsafe_handoff_bit_identical_to_off():
    candidate_value, candidate_gradient = _potential(
        5001, "off", "auto", non_grazing=False, impact=1.01
    )
    baseline_value, baseline_gradient = _potential(
        5001, "off", "off", non_grazing=False, impact=1.01
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


_EXACT_TREND_TYPES = (
    "linear",
    "quadratic",
    "cubic",
    "quartic",
    "explinear_spectroscopic",
    "spot_spectroscopic",
    "quadratic+spot_spectroscopic",
    "2spot_spectroscopic",
    "linear_discontinuity_spectroscopic",
    "spot_spectroscopic+linear_discontinuity_spectroscopic",
)


def _fixed_trend_bases(time):
    time = np.asarray(time)
    centered = time - time.min()
    return {
        "exp_trend": np.exp(-centered / 0.03),
        "spot_trend": np.exp(-0.5 * ((time + 0.08) / 0.017) ** 2),
        "spot_trend2": np.exp(-0.5 * ((time - 0.11) / 0.021) ** 2),
        "jump_trend": 1.0 / (1.0 + np.exp(-(time - 0.04) / 0.002)),
    }


@pytest.mark.parametrize("detrend_type", _EXACT_TREND_TYPES)
def test_all_fixed_shape_linear_trend_statistics_are_exact(detrend_type):
    cadence_count = 5001
    time = np.linspace(-0.3, 0.3, cadence_count)
    bases = _fixed_trend_bases(time)
    names, design = build_linear_spectro_trend_design(
        detrend_type, time, **bases
    )
    coefficient_values = {
        "c": 1.002,
        "v": -0.004,
        "v2": 0.007,
        "v3": -0.009,
        "v4": 0.011,
        "A": 0.003,
        "A_spot": 1.04,
        "A_spot2": 0.93,
        "A_jump": 1.02,
    }
    reference_beta = np.array([coefficient_values[name] for name in names])
    truth_beta = reference_beta + np.linspace(-2.0e-4, 2.0e-4, len(names))
    reported_error = np.resize(
        np.array([0.0037, 0.0040, 0.0044, 0.0040]), cadence_count
    )
    flux = design @ truth_beta + 1.0e-4 * np.sin(np.arange(cadence_count))
    mask = np.ones(cadence_count, dtype=bool)
    mask[[0, 901, cadence_count - 2]] = False
    window = build_transit_window_indices(
        time, np.array([2.0]), np.array([0.0]), np.array([0.1])
    )
    statistics = build_linear_oot_statistics(
        time,
        flux[None, :],
        reported_error[None, :],
        window,
        design,
        reference_beta[None, :],
        mask[None, :],
    )
    oot_mask = mask.copy()
    oot_mask[window] = False

    def direct(parameters):
        beta, jitter = parameters[:-1], parameters[-1]
        residual = flux - design @ beta
        variance = reported_error**2 + jitter**2
        terms = -0.5 * (
            residual**2 / variance + jnp.log(2.0 * jnp.pi * variance)
        )
        return jnp.sum(jnp.where(oot_mask, terms, 0.0))

    def reduced(parameters):
        beta, jitter = parameters[:-1], parameters[-1]
        delta = beta - jnp.asarray(statistics["oot_reference_beta"][0])
        sse = (
            jnp.asarray(statistics["oot_group_reference_sse"][0])
            - 2.0 * jnp.einsum(
                "p,gp->g",
                delta,
                jnp.asarray(
                    statistics["oot_group_x_reference_residual"][0]
                ),
            )
            + jnp.einsum(
                "p,gpq,q->g",
                delta,
                jnp.asarray(statistics["oot_group_xx"][0]),
                delta,
            )
        )
        count = jnp.asarray(statistics["oot_group_count"][0])
        variance = (
            jnp.asarray(statistics["oot_group_yerr"][0]) ** 2 + jitter**2
        )
        return jnp.sum(
            -0.5 * (sse / variance + count * jnp.log(2.0 * jnp.pi * variance))
        )

    point = jnp.concatenate((jnp.asarray(truth_beta + 1.0e-5), jnp.array([8e-4])))
    _assert_value_gradient_close(
        jax.value_and_grad(direct)(point),
        jax.value_and_grad(reduced)(point),
    )


def _trend_reduction_potential(detrend_type, cadence_mode):
    time, yerr, window, common, base_initial, _ = _problem(5001)
    bases = _fixed_trend_bases(time)
    names, design = build_linear_spectro_trend_design(
        detrend_type, time, **bases
    )
    initial = {
        key: value for key, value in base_initial.items()
        if key in {"rors", "c1", "c2", "log_jitter"}
    }
    initial.update({"c": jnp.array([1.001]), "v": jnp.array([-0.002])})
    for name, value in {
        "v2": 0.004,
        "v3": -0.003,
        "v4": 0.002,
        "A": 0.003,
        "A_spot": 1.04,
        "A_spot2": 0.93,
        "A_jump": 1.02,
    }.items():
        if name in names:
            initial[name] = jnp.array([value])
    reference_beta = np.column_stack([
        np.asarray(initial[name]) for name in names
    ])
    statistics = build_linear_oot_statistics(
        time,
        np.asarray(common["y"]),
        yerr,
        window,
        design,
        reference_beta,
        np.asarray(common["likelihood_mask"]),
    )
    model = create_vectorized_model(
        detrend_type=detrend_type,
        ld_mode="stellarprior",
        trend_mode="free",
        ld_profile="power2",
        transit_window="auto",
        transit_window_indices=window,
        cadence_reduction=cadence_mode,
        transit_grid="off",
    )
    model_kwargs = {
        **common,
        **statistics,
        **{key: jnp.asarray(value) for key, value in bases.items()},
        "trend_design": jnp.asarray(design),
    }
    info = initialize_model(
        jax.random.PRNGKey(17),
        model,
        model_args=(jnp.asarray(time), jnp.asarray(yerr)),
        model_kwargs=model_kwargs,
        init_strategy=init_to_value(values=initial),
        validate_grad=False,
    )
    return jax.jit(jax.value_and_grad(info.potential_fn))(info.param_info.z)


@pytest.mark.parametrize(
    "detrend_type",
    (
        "linear",
        "quartic",
        "quadratic+spot_spectroscopic",
        "2spot_spectroscopic",
        "linear_discontinuity_spectroscopic",
        "spot_spectroscopic+linear_discontinuity_spectroscopic",
    ),
)
def test_long_generic_trend_reduction_matches_direct_potential_and_gradient(
    detrend_type,
):
    _assert_value_gradient_close(
        _trend_reduction_potential(detrend_type, "off"),
        _trend_reduction_potential(detrend_type, "auto"),
    )


@pytest.mark.parametrize(
    "detrend_type",
    ("explinear", "linear+gp_spectroscopic"),
)
def test_nonlinear_or_gp_trend_is_not_eligible_for_exact_collapse(detrend_type):
    assert linear_spectro_trend_coefficient_names(detrend_type) is None


def _quadratic_ld_potential(ld_case, grid_mode):
    time, yerr, window, common, base_initial, _ = _problem(5001)
    names, design = build_linear_spectro_trend_design("linear", time)
    initial = {
        "rors": base_initial["rors"],
        "c": base_initial["c"],
        "v": base_initial["v"],
        "log_jitter": base_initial["log_jitter"],
    }
    common = dict(common)
    common["mu_u_ld"] = jnp.array([[0.4, 0.2]])
    common["sigma_u_ld"] = jnp.array([[0.05, 0.04]])
    builder_kwargs = {
        "ld_mode": ld_case,
        "ld_uniform_basis": "uplus_uminus",
    }
    if ld_case in {"gaussian", "stellarprior"}:
        initial["u"] = jnp.array([[0.4, 0.2]])
    elif ld_case == "uniform_coefficients":
        builder_kwargs.update(
            ld_mode="uniform", ld_uniform_basis="coefficients"
        )
        initial["u"] = jnp.array([[0.4, 0.2]])
    elif ld_case == "uniform_uplus_uminus":
        builder_kwargs["ld_mode"] = "uniform"
        initial["ld_uplus_uminus"] = jnp.array([[0.6, 0.2]])
    elif ld_case == "sing":
        common["mu_u_ld"] = jnp.array([[0.4, 0.05]])
        common["sigma_u_ld"] = jnp.array([[0.05, 0.02]])
        initial["limb_l"] = jnp.array([0.4])
        initial["limb_delta"] = jnp.array([0.05])
    elif ld_case == "fixed":
        common["ld_fixed"] = jnp.array([[0.4, 0.2]])
    else:
        raise AssertionError(ld_case)

    reference_beta = np.column_stack([
        np.asarray(initial[name]) for name in names
    ])
    statistics = build_linear_oot_statistics(
        time,
        np.asarray(common["y"]),
        yerr,
        window,
        design,
        reference_beta,
        np.asarray(common["likelihood_mask"]),
    )
    model = create_vectorized_model(
        detrend_type="linear",
        trend_mode="free",
        ld_profile="quadratic",
        transit_window="auto",
        transit_window_indices=window,
        cadence_reduction="auto",
        transit_grid=grid_mode,
        transit_grid_nodes=769,
        transit_grid_non_grazing=True,
        **builder_kwargs,
    )
    info = initialize_model(
        jax.random.PRNGKey(29),
        model,
        model_args=(jnp.asarray(time), jnp.asarray(yerr)),
        model_kwargs={
            **common,
            **statistics,
            "trend_design": jnp.asarray(design),
        },
        init_strategy=init_to_value(values=initial),
        validate_grad=False,
    )
    return jax.jit(jax.value_and_grad(info.potential_fn))(info.param_info.z)


@pytest.mark.parametrize(
    "ld_case",
    (
        "gaussian",
        "stellarprior",
        "uniform_coefficients",
        "uniform_uplus_uminus",
        "sing",
        "fixed",
    ),
)
def test_quadratic_ld_parameterizations_match_direct_potential_and_gradient(
    ld_case,
):
    _assert_value_gradient_close(
        _quadratic_ld_potential(ld_case, "off"),
        _quadratic_ld_potential(ld_case, "auto"),
    )
