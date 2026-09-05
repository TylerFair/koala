import jax.numpy as jnp
import numpy as np

from fit_jwst import (
    _power2_ld_initial_sites,
    _power2_ld_optimization_sites,
    _validate_whitelight_optimized_start,
)
from models.ld_parameterization import Power2MaxtedTransform


def test_decorrelated_power2_initialization_uses_real_latent_site():
    coefficients = jnp.array([0.37111993, 0.33026019])
    initial = _power2_ld_initial_sites(coefficients, "decorrelated")

    assert set(initial) == {"ld_decorrelated"}
    assert _power2_ld_optimization_sites("decorrelated") == ["ld_decorrelated"]
    recovered = Power2MaxtedTransform().inv(initial["ld_decorrelated"])
    np.testing.assert_allclose(recovered, coefficients, rtol=0, atol=1e-12)
    assert np.isfinite(np.asarray(initial["ld_decorrelated"])).all()


def test_decorrelated_power2_initialization_supports_spectroscopic_batches():
    coefficients = jnp.array([[0.37, 0.33], [0.41, 0.28], [0.35, 0.36]])
    initial = _power2_ld_initial_sites(coefficients, "decorrelated")

    assert initial["ld_decorrelated"].shape == coefficients.shape
    recovered = Power2MaxtedTransform().inv(initial["ld_decorrelated"])
    np.testing.assert_allclose(recovered, coefficients, rtol=0, atol=1e-12)


def test_power2_initialization_sites_match_each_parameterization():
    coefficients = jnp.array([0.4, 0.3])
    assert set(_power2_ld_initial_sites(coefficients, "coefficients")) == {"c1", "c2"}
    assert set(_power2_ld_initial_sites(coefficients, "decorrelated_linear")) == {"ld_decorrelated"}
    assert set(_power2_ld_initial_sites(coefficients, "latent_gaussian")) == {"ld_latent"}


def test_whitelight_optimizer_rejects_observed_nonphysical_maxted_solution():
    invalid = {
        "ld_decorrelated": jnp.array([316.73443542, 0.44949058]),
        "_b_0": jnp.array(2.0),
        "rors_0": jnp.array(0.70710678),
    }
    valid, reasons = _validate_whitelight_optimized_start(
        invalid, ld_profile="power2", ld_parameterization="decorrelated",
        ld_prior_mode="gaussian", n_planets=1,
    )
    assert not valid
    assert any("non-finite" in reason for reason in reasons)
    assert any("rprs" in reason for reason in reasons)
    assert any("violates" in reason for reason in reasons)


def test_whitelight_optimizer_accepts_transformed_physical_prior_start():
    coefficients = jnp.array([0.37111993, 0.33026019])
    valid, reasons = _validate_whitelight_optimized_start(
        {
            **_power2_ld_initial_sites(coefficients, "decorrelated"),
            "_b_0": jnp.array(0.4498),
            "rors_0": jnp.array(0.1457),
        },
        ld_profile="power2", ld_parameterization="decorrelated",
        ld_prior_mode="gaussian", n_planets=1,
    )
    assert valid
    assert reasons == []
