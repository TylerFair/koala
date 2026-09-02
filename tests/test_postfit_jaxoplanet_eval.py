import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from fit_jwst import (
    _attach_jaxoplanet_eval_metadata,
    _posterior_quadratic_ld_median,
    _select_transit_eval_params,
)
from models.jaxoplanet.core import build_transit_window_indices
from plotting import _single_curve_transit_signal


def test_quadratic_postfit_prefers_sampled_joint_u_over_legacy_sites_and_prior():
    sampled_u = jnp.asarray(
        [
            [[0.10, 0.20], [0.30, 0.40]],
            [[0.20, 0.30], [0.40, 0.50]],
            [[0.90, 0.80], [0.70, 0.60]],
        ],
        dtype=jnp.float64,
    )
    samples = {
        "u": sampled_u,
        "u1": jnp.full((3, 2), 0.99),
        "u2": jnp.full((3, 2), 0.98),
    }
    fallback = jnp.full((2, 2), 0.01)

    actual = _posterior_quadratic_ld_median(samples, fallback)

    np.testing.assert_array_equal(
        np.asarray(actual), np.asarray(jnp.nanmedian(sampled_u, axis=0))
    )


def test_quadratic_postfit_supports_harmonica_split_sites_and_partial_fallback():
    samples = {
        "u1": jnp.asarray([[0.20, 0.30], [0.40, 0.50], [0.30, 0.40]])
    }
    fallback = jnp.asarray([[0.11, 0.12], [0.13, 0.14]])

    actual = _posterior_quadratic_ld_median(samples, fallback)

    np.testing.assert_allclose(np.asarray(actual[:, 0]), [0.30, 0.40])
    np.testing.assert_allclose(np.asarray(actual[:, 1]), [0.12, 0.14])


def test_quadratic_postfit_rejects_nonquadratic_joint_site():
    with pytest.raises(ValueError, match="final dimension 2"):
        _posterior_quadratic_ld_median(
            {"u": jnp.ones((4, 3, 12))}, jnp.ones((3, 2))
        )


def _duration_map_params():
    return {
        "period": jnp.asarray([2.75]),
        "duration": jnp.asarray([0.12]),
        "t0": jnp.asarray([0.0]),
        "b": jnp.asarray([0.31]),
        "rors": jnp.asarray([[0.096], [0.104]]),
        "u": jnp.asarray([[0.27, 0.18], [0.34, 0.12]]),
    }


def test_postfit_metadata_is_rebuilt_for_the_current_clipped_time_axis():
    full_time = jnp.linspace(-0.20, 0.20, 101)
    clipped_time = full_time[7:-9]
    params = _attach_jaxoplanet_eval_metadata(
        _duration_map_params(),
        full_time,
        jaxoplanet_kernel="quadratic_specialized",
        ld_profile="quadratic",
        transit_window_optimization="auto",
    )

    rebuilt = _attach_jaxoplanet_eval_metadata(
        params,
        clipped_time,
        jaxoplanet_kernel="quadratic_specialized",
        ld_profile="quadratic",
        transit_window_optimization="auto",
    )

    assert rebuilt["_transit_phase_offsets"].shape == (1, clipped_time.size)
    assert rebuilt["_transit_phase_mask"].shape == (1, clipped_time.size)
    expected_indices = build_transit_window_indices(
        np.asarray(clipped_time),
        np.asarray(rebuilt["period"]),
        np.asarray(rebuilt["t0"]),
        np.asarray(rebuilt["duration"]),
    )
    np.testing.assert_array_equal(
        np.asarray(rebuilt["_transit_window_indices"]), expected_indices
    )


def test_select_eval_params_preserves_duration_route_and_optimized_metadata():
    times = jnp.linspace(-0.08, 0.08, 33)
    raw = {**_duration_map_params(), "a_rs": jnp.asarray([10.4])}

    selected = _select_transit_eval_params(
        raw,
        transit_engine="jaxoplanet",
        param_method="duration",
        t=times,
        jaxoplanet_kernel="quadratic_specialized",
        ld_profile="quadratic",
        transit_window_optimization="auto",
    )

    assert "a_rs" not in selected
    assert selected["_jaxoplanet_kernel"] == "quadratic_specialized"
    assert selected["_ld_profile"] == "quadratic"
    assert selected["_transit_phase_offsets"].shape[-1] == times.size
    assert "_transit_window_indices" in selected


def test_plotting_forwards_postfit_jaxoplanet_metadata_unchanged():
    times = jnp.linspace(-0.08, 0.08, 33)
    map_params = _attach_jaxoplanet_eval_metadata(
        _duration_map_params(),
        times,
        jaxoplanet_kernel="quadratic_specialized",
        ld_profile="quadratic",
        transit_window_optimization="auto",
    )
    expected = jnp.zeros_like(times)

    with patch("plotting.compute_transit_model_auto", return_value=expected) as model:
        actual = _single_curve_transit_signal(
            np.asarray(times),
            map_params,
            {"period": np.asarray([2.75]), "transit_engine": "jaxoplanet", "param_method": "duration"},
            1,
        )

    forwarded = model.call_args.args[0]
    assert forwarded["_jaxoplanet_kernel"] == "quadratic_specialized"
    assert forwarded["_ld_profile"] == "quadratic"
    np.testing.assert_array_equal(
        np.asarray(forwarded["_transit_phase_offsets"]),
        np.asarray(map_params["_transit_phase_offsets"]),
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_plotting_forwards_native_power2_coefficients_without_polynomial_u():
    times = jnp.linspace(-0.08, 0.08, 33)
    map_params = _attach_jaxoplanet_eval_metadata(
        {
            **_duration_map_params(),
            "c1": jnp.asarray([0.61, 0.64]),
            "c2": jnp.asarray([0.72, 0.68]),
        },
        times,
        jaxoplanet_kernel="native_power2",
        ld_profile="power2",
        transit_window_optimization="auto",
    )
    map_params.pop("u")
    expected = jnp.zeros_like(times)

    with patch("plotting.compute_transit_model_auto", return_value=expected) as model:
        actual = _single_curve_transit_signal(
            np.asarray(times),
            map_params,
            {
                "period": np.asarray([2.75]),
                "transit_engine": "jaxoplanet",
                "param_method": "duration",
            },
            1,
        )

    forwarded = model.call_args.args[0]
    assert "u" not in forwarded
    np.testing.assert_array_equal(np.asarray(forwarded["c1"]), 0.64)
    np.testing.assert_array_equal(np.asarray(forwarded["c2"]), 0.68)
    assert forwarded["_jaxoplanet_kernel"] == "native_power2"
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
