import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers

import fit_jwst
from models.jaxoplanet import create_vectorized_model, create_whitelight_model
from tools.spectro_stage_inputs import _rebuild_model


def _spectro_trace(basis="uplus_uminus"):
    model = create_vectorized_model(
        ld_mode="uniform", ld_profile="quadratic",
        ld_uniform_basis=basis, transit_window="off",
    )
    return handlers.trace(handlers.seed(model, jax.random.PRNGKey(19))).get_trace(
        jnp.linspace(-0.03, 0.03, 13), jnp.full((2, 13), 1e-3),
        y=jnp.ones((2, 13)), mu_duration=jnp.array([0.06]),
        mu_t0=jnp.array([0.0]), mu_b=jnp.array([0.3]),
        mu_depths=jnp.full((2, 1), 0.01), PERIOD=jnp.array([3.0]),
    )


def test_quadratic_uniform_default_is_wide_uplus_uminus():
    trace = _spectro_trace()
    assert trace["ld_uplus_uminus"]["type"] == "sample"
    low = np.asarray(trace["ld_uplus_uminus"]["fn"].base_dist.low)
    high = np.asarray(trace["ld_uplus_uminus"]["fn"].base_dist.high)
    np.testing.assert_allclose(low[0], [-1.0, -2.0])
    np.testing.assert_allclose(high[0], [2.0, 2.0])
    sumdiff = np.asarray(trace["ld_uplus_uminus"]["value"])
    u = np.asarray(trace["u"]["value"])
    np.testing.assert_allclose(u[:, 0], (sumdiff[:, 0] + sumdiff[:, 1]) / 2)
    np.testing.assert_allclose(u[:, 1], (sumdiff[:, 0] - sumdiff[:, 1]) / 2)
    np.testing.assert_allclose(trace["l"]["value"], 1 - sumdiff[:, 0])
    np.testing.assert_allclose(trace["delta"]["value"],
                               (sumdiff[:, 0] - sumdiff[:, 1]) / 8)


def test_quadratic_uniform_coefficients_preserves_legacy_prior():
    trace = _spectro_trace("coefficients")
    assert trace["u"]["type"] == "sample"
    assert "ld_uplus_uminus" not in trace
    base = trace["u"]["fn"]
    while hasattr(base, "base_dist"):
        base = base.base_dist
    low = np.asarray(base.low)
    high = np.asarray(base.high)
    np.testing.assert_allclose(low, 0.0)
    np.testing.assert_allclose(high, 1.0)


def test_quadratic_uniform_coefficients_accept_wide_configured_bounds():
    model = create_vectorized_model(
        ld_mode="uniform", ld_profile="quadratic",
        ld_uniform_basis="coefficients",
        ld_uniform_coefficient_bounds=(-2.0, 2.0), transit_window="off",
    )
    trace = handlers.trace(handlers.seed(model, jax.random.PRNGKey(23))).get_trace(
        jnp.linspace(-0.03, 0.03, 13), jnp.full((2, 13), 1e-3),
        y=jnp.ones((2, 13)), mu_duration=jnp.array([0.06]),
        mu_t0=jnp.array([0.0]), mu_b=jnp.array([0.3]),
        mu_depths=jnp.full((2, 1), 0.01), PERIOD=jnp.array([3.0]),
    )
    base = trace["u"]["fn"]
    while hasattr(base, "base_dist"):
        base = base.base_dist
    np.testing.assert_allclose(np.asarray(base.low), -2.0)
    np.testing.assert_allclose(np.asarray(base.high), 2.0)


def test_uniform_basis_is_part_of_builder_fingerprint_inputs():
    model = fit_jwst._build_spectroscopic_model(
        create_vectorized_model,
        ld_mode="uniform", ld_profile="quadratic",
        ld_uniform_basis="uplus_uminus")
    assert model.__sampler_input_builder__["kwargs"]["ld_uniform_basis"] == "uplus_uminus"


def test_quadratic_uniform_initializer_matches_new_coordinates():
    coefficients = np.array([[0.4, 0.1], [0.3, -0.05]])
    sites = fit_jwst._quadratic_uniform_initial_sites(
        coefficients, "uplus_uminus")
    np.testing.assert_allclose(
        sites["ld_uplus_uminus"], [[0.5, 0.3], [0.25, 0.35]])
    legacy = fit_jwst._quadratic_uniform_initial_sites(
        coefficients, "coefficients")
    np.testing.assert_allclose(legacy["u"], coefficients)


def test_whitelight_default_emits_legacy_output_sites():
    from planet_specs import default_planet_specs
    specs = default_planet_specs(period=3.0, t0=0.0, b=0.3, rprs=0.1, duration=0.06)
    model = create_whitelight_model(ld_mode="uniform", ld_profile="quadratic",
                                    parameter_priors=specs)
    trace = handlers.trace(handlers.seed(model, jax.random.PRNGKey(2))).get_trace(
        jnp.linspace(-0.03, 0.03, 13), jnp.full(13, 1e-3), y=jnp.ones(13),
        prior_params={"period": np.array([3.0]), "ecc": np.array([0.0]),
                      "omega": np.array([0.0]), "u": np.array([0.4, 0.1]),
                      "parameter_priors": specs},
    )
    for name in ("u", "u1", "u2", "l", "delta"):
        assert trace[name]["type"] == "deterministic"


def test_legacy_dump_without_basis_rebuilds_coefficient_prior():
    model = _rebuild_model({
        "identity": "models.jaxoplanet.builder.create_vectorized_model",
        "kwargs": {"ld_mode": "uniform", "ld_profile": "quadratic"},
        "callable_is_model": False,
    })
    trace = handlers.trace(handlers.seed(model, jax.random.PRNGKey(7))).get_trace(
        jnp.linspace(-0.03, 0.03, 13), jnp.full((1, 13), 1e-3),
        y=jnp.ones((1, 13)), mu_duration=jnp.array([0.06]),
        mu_t0=jnp.array([0.0]), mu_b=jnp.array([0.3]),
        mu_depths=jnp.full((1, 1), 0.01), PERIOD=jnp.array([3.0]),
    )
    assert trace["u"]["type"] == "sample"
    assert "ld_uplus_uminus" not in trace
