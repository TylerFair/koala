import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from unittest.mock import patch

import jax.numpy as jnp

from models.common import compute_transit_model_auto


def test_auto_dispatches_harmonica_for_power2_limb_darkening():
    params = {"a_rs": 10.0, "c_ld": 0.5, "alpha_ld": 0.6}
    expected = jnp.asarray([0.99])
    with patch(
        "models.harmonica.core.compute_transit_model_harmonica",
        return_value=expected,
    ) as harmonica, patch(
        "models.jaxoplanet.core.compute_transit_model"
    ) as circular:
        result = compute_transit_model_auto(params, jnp.asarray([0.0]))
    assert result is expected
    harmonica.assert_called_once()
    circular.assert_not_called()


def test_auto_dispatches_harmonica_for_quadratic_limb_darkening():
    params = {"a_rs": 10.0, "u1_ld": 0.3, "u2_ld": 0.2, "a1": 1e-3}
    expected = jnp.asarray([0.98])
    with patch(
        "models.harmonica.core.compute_transit_model_harmonica",
        return_value=expected,
    ) as harmonica, patch(
        "models.jaxoplanet.core.compute_transit_model"
    ) as circular:
        result = compute_transit_model_auto(params, jnp.asarray([0.0]))
    assert result is expected
    harmonica.assert_called_once()
    circular.assert_not_called()


def test_auto_keeps_circular_jaxoplanet_dispatch_without_harmonica_ld_keys():
    params = {"a_rs": 10.0, "u": jnp.asarray([0.3, 0.2])}
    expected = jnp.asarray([0.97])
    with patch(
        "models.harmonica.core.compute_transit_model_harmonica"
    ) as harmonica, patch(
        "models.jaxoplanet.core.compute_transit_model", return_value=expected
    ) as circular:
        result = compute_transit_model_auto(params, jnp.asarray([0.0]))
    assert result is expected
    harmonica.assert_not_called()
    circular.assert_called_once()
