import jax.numpy as jnp
import numpy as np

from fit_jwst import _power2_ld_initial_sites, _power2_ld_optimization_sites
from models.ld_parameterization import Power2MaxtedTransform


def test_decorrelated_power2_initialization_uses_real_latent_site():
    coefficients = jnp.array([0.37111993, 0.33026019])
    initial = _power2_ld_initial_sites(coefficients, "decorrelated")

    assert set(initial) == {"ld_decorrelated"}
    assert _power2_ld_optimization_sites("decorrelated") == ["ld_decorrelated"]
    recovered = Power2MaxtedTransform().inv(initial["ld_decorrelated"])
    np.testing.assert_allclose(recovered, coefficients, rtol=0, atol=1e-12)
    assert np.isfinite(np.asarray(initial["ld_decorrelated"])).all()


def test_power2_initialization_sites_match_each_parameterization():
    coefficients = jnp.array([0.4, 0.3])
    assert set(_power2_ld_initial_sites(coefficients, "coefficients")) == {"c1", "c2"}
    assert set(_power2_ld_initial_sites(coefficients, "decorrelated_linear")) == {"ld_decorrelated"}
    assert set(_power2_ld_initial_sites(coefficients, "latent_gaussian")) == {"ld_latent"}
