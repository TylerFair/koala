import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import numpy as np
import pytest

from models.common import compute_transit_model_auto
from plotting import _resolve_plot_geometry_mode, _single_curve_transit_signal


_TIMES = np.linspace(-0.075, 0.075, 41)
_TRANSIT_PARAMS = {"period": np.array([2.75])}


def _base_map_params():
    """Representative two-channel, one-planet MAP products from fit_jwst."""
    return {
        "b": np.array([0.31]),
        "t0": np.array([0.0]),
        "a_rs": np.array([10.4]),
        "ecc": np.array([0.02]),
        "omega": np.array([0.4]),
        "rors": np.array([[0.096], [0.104]]),
        # Exercise every supported odd cosine coefficient and channel slicing.
        "a1": np.array([[0.0015], [-0.0030]]),
        "a3": np.array([[-0.0004], [0.0008]]),
        "a5": np.array([[0.0002], [-0.0005]]),
    }


def _direct_channel_params(map_params, idx, ld_keys):
    params = {
        "period": jnp.asarray(_TRANSIT_PARAMS["period"]),
        "t0": jnp.asarray(map_params["t0"]),
        "b": jnp.asarray(map_params["b"]),
        "a_rs": jnp.asarray(map_params["a_rs"]),
        "ecc": jnp.asarray(map_params["ecc"]),
        "omega": jnp.asarray(map_params["omega"]),
        "rors": jnp.asarray(map_params["rors"][idx]),
    }
    for name in ("a1", "a3", "a5"):
        params[name] = jnp.asarray(np.atleast_1d(map_params[name][idx]))
    for name in ld_keys:
        params[name] = jnp.asarray(map_params[name][idx])
    return params


@pytest.mark.parametrize(
    ("profile", "ld_keys"),
    [
        ("power2", ("c_ld", "alpha_ld")),
        ("quadratic", ("u1_ld", "u2_ld")),
    ],
)
def test_single_curve_harmonica_matches_direct_model_with_all_odd_coefficients(
    profile, ld_keys
):
    map_params = _base_map_params()
    if profile == "power2":
        map_params.update(
            {
                "c_ld": np.array([0.52, 0.61]),
                "alpha_ld": np.array([0.68, 0.73]),
                # fit_jwst also retains this plotting-compatible approximation.
                "u": np.array([[0.2, 0.1], [0.25, 0.12]]),
            }
        )
    else:
        u1 = np.array([0.27, 0.34])
        u2 = np.array([0.18, 0.12])
        map_params.update(
            {
                "u1_ld": u1,
                "u2_ld": u2,
                # This key is present in real quadratic Harmonica MAP products;
                # it must not make plotting fall back to circular jaxoplanet.
                "u": np.stack((u1, u2), axis=-1),
            }
        )

    idx = 1
    plotted = _single_curve_transit_signal(
        _TIMES, map_params, _TRANSIT_PARAMS, idx
    )
    expected = compute_transit_model_auto(
        _direct_channel_params(map_params, idx, ld_keys), jnp.asarray(_TIMES)
    )

    np.testing.assert_allclose(
        plotted, np.asarray(expected), rtol=2e-13, atol=2e-13
    )


@pytest.mark.parametrize(
    ("map_params", "expected_engine"),
    [
        ({"c_ld": 0.5, "alpha_ld": 0.7}, "harmonica"),
        ({"u1_ld": 0.3, "u2_ld": 0.2}, "harmonica"),
        ({"u": np.array([0.3, 0.2])}, "jaxoplanet"),
        # A partial LD pair is not a valid Harmonica profile.
        ({"u": np.array([0.3, 0.2]), "u1_ld": 0.3}, "jaxoplanet"),
    ],
)
def test_plot_engine_fallback_recognizes_complete_harmonica_profiles(
    map_params, expected_engine
):
    engine, param_method = _resolve_plot_geometry_mode(
        {"a_rs": 10.0, **map_params}, {}
    )

    assert engine == expected_engine
    assert param_method == "a_rs"


def test_plot_engine_and_geometry_explicit_values_override_fallbacks():
    engine, param_method = _resolve_plot_geometry_mode(
        {"a_rs": 10.0, "u1_ld": 0.3, "u2_ld": 0.2},
        {"transit_engine": "jaxoplanet", "param_method": "duration"},
    )

    assert engine == "jaxoplanet"
    assert param_method == "duration"
