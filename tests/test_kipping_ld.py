"""Kipping (2013) (q1, q2) sampling of the quadratic limb-darkening law."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers

from koala.config import parse_planet_parameter_specs
from koala.limb_darkening import _quadratic_uniform_initial_sites
from models.jaxoplanet import create_vectorized_model, create_whitelight_model
from models.limb_darkening_config import LD_PROFILES, resolve_ld_prior, validate_ld_prior, validate_ld_profile
from models.model_building import kipping_q_from_u, kipping_q_sigma, kipping_u_from_q

jax.config.update("jax_enable_x64", True)


def test_mapping_round_trips():
    q = np.random.default_rng(0).uniform(0.01, 0.99, size=(500, 2))
    u = np.asarray(kipping_u_from_q(q))
    np.testing.assert_allclose(kipping_q_from_u(u), q, rtol=1e-12, atol=1e-12)
    # Known point: q1 = 1, q2 = 1/2 is u = (1, 0); q2 = 0 is pure u2.
    np.testing.assert_allclose(kipping_u_from_q([1.0, 0.5]), [1.0, 0.0])
    np.testing.assert_allclose(kipping_u_from_q([0.25, 0.0]), [0.0, 0.5])
    u_typical = np.array([[0.4, 0.2], [0.1, 0.3], [0.9, -0.1]])
    np.testing.assert_allclose(kipping_u_from_q(kipping_q_from_u(u_typical)), u_typical, atol=1e-12)


def test_uniform_q_covers_exactly_the_physical_triangle():
    q = np.random.default_rng(1).uniform(size=(200_000, 2))
    u1, u2 = np.asarray(kipping_u_from_q(q)).T
    assert np.all(u1 >= 0.0) and np.all(u1 + u2 <= 1.0 + 1e-12) and np.all(u1 + 2 * u2 >= -1e-12)
    # The triangle (0,0), (0,1), (2,-1) has unit area and the induced prior is uniform on it:
    # the u1 < 1 part has area 3/4.
    assert abs(np.mean(u1 < 1.0) - 0.75) < 5e-3
    # Every physical point maps back inside the unit square (surjectivity).
    w = np.random.default_rng(2).dirichlet(np.ones(3), size=50_000)
    pts = w @ np.array([[0.0, 0.0], [0.0, 1.0], [2.0, -1.0]])
    back = np.asarray(kipping_q_from_u(pts))
    assert np.all((back >= 0.0) & (back <= 1.0 + 1e-12))


def test_sigma_propagation_matches_jacobian():
    u, s = jnp.array([0.35, 0.22]), jnp.array([0.2, 0.2])
    jac = jax.jacfwd(kipping_q_from_u)(u)
    expected = jnp.sqrt((jac ** 2) @ (s ** 2))
    np.testing.assert_allclose(kipping_q_sigma(u, s), expected, rtol=1e-10)


_T = jnp.linspace(0.88, 1.12, 150)
_YERR = jnp.full(_T.shape, 5e-4)


def _wl_model(basis, mode="uniform"):
    specs = parse_planet_parameter_specs({
        'period': {'value': 3.0, 'prior': 'fixed'},
        't0': {'value': 1.0, 'prior': 'uniform', 'low': 0.98, 'high': 1.02},
        'duration': {'value': 0.12, 'prior': 'fixed'},
        'b': {'value': 0.3, 'prior': 'fixed'},
        'rprs': {'value': 0.1, 'prior': 'uniform', 'low': 0.05, 'high': 0.2},
    })
    return create_whitelight_model(
        detrend_type="linear", ld_mode=mode, ld_profile="quadratic",
        ld_uniform_basis=basis, parameter_priors=specs,
    )


def _wl_trace(model, seed=0, data=None):
    if data is not None:
        model = handlers.condition(model, data=data)
    return handlers.trace(handlers.seed(model, jax.random.PRNGKey(seed))).get_trace(
        _T, _YERR, y=jnp.ones(_T.shape),
        prior_params={"period": jnp.array([3.0]), "u": jnp.array([0.3, 0.2])},
    )


def _latents(trace, exclude):
    return {k: v["value"] for k, v in trace.items()
            if v["type"] == "sample" and not v["is_observed"] and k not in exclude}


def test_whitelight_kipping_matches_quadratic_with_converted_u_exactly():
    kip = _wl_trace(_wl_model("kipping"), seed=4)
    assert kip["ld_q"]["type"] == "sample"
    q = np.asarray(kip["ld_q"]["value"])
    np.testing.assert_allclose([kip["q1"]["value"], kip["q2"]["value"]], q)
    u = np.asarray(kipping_u_from_q(q))
    np.testing.assert_allclose(kip["u"]["value"], u)
    np.testing.assert_allclose([kip["u1"]["value"], kip["u2"]["value"]], u)
    data = _latents(kip, {"ld_q"})
    data["u"] = jnp.asarray(u)
    quad = _wl_trace(_wl_model("coefficients"), seed=4, data=data)
    np.testing.assert_array_equal(np.asarray(kip["obs"]["fn"].loc), np.asarray(quad["obs"]["fn"].loc))
    # The q-space prior is the unit square.
    base = kip["ld_q"]["fn"]
    while hasattr(base, "base_dist"):
        base = base.base_dist
    np.testing.assert_allclose(base.low, 0.0)
    np.testing.assert_allclose(base.high, 1.0)


def test_whitelight_kipping_gaussian_is_truncated_normal_in_q():
    trace = _wl_trace(_wl_model("kipping", mode="gaussian"), seed=5)
    fn = trace["ld_q"]["fn"].base_dist
    u_mean = jnp.array([0.3, 0.2])
    np.testing.assert_allclose(fn.base_dist.loc, kipping_q_from_u(u_mean))
    np.testing.assert_allclose(fn.base_dist.scale, kipping_q_sigma(u_mean, 0.2))
    np.testing.assert_allclose(fn.low, 0.0)
    np.testing.assert_allclose(fn.high, 1.0)


def _spectro_trace(basis, data=None):
    model = create_vectorized_model(
        ld_mode="uniform", ld_profile="quadratic", ld_uniform_basis=basis, transit_window="off",
    )
    if data is not None:
        model = handlers.condition(model, data=data)
    return handlers.trace(handlers.seed(model, jax.random.PRNGKey(19))).get_trace(
        jnp.linspace(-0.03, 0.03, 13), jnp.full((2, 13), 1e-3),
        y=jnp.ones((2, 13)), rors_reference=jnp.array([.1]), mu_duration=jnp.array([0.06]),
        mu_t0=jnp.array([0.0]), mu_b=jnp.array([0.3]),
        mu_depths=jnp.full((2, 1), 0.01), PERIOD=jnp.array([3.0]),
    )


def test_spectroscopic_kipping_matches_quadratic_with_converted_u_exactly():
    kip = _spectro_trace("kipping")
    q = np.asarray(kip["ld_q"]["value"])
    assert q.shape == (2, 2)
    np.testing.assert_allclose(kip["q1"]["value"], q[:, 0])
    np.testing.assert_allclose(kip["q2"]["value"], q[:, 1])
    data = _latents(kip, {"ld_q"})
    data["u"] = kipping_u_from_q(q)
    quad = _spectro_trace("coefficients", data=data)
    np.testing.assert_array_equal(np.asarray(kip["obs"]["fn"].loc), np.asarray(quad["obs"]["fn"].loc))


def test_initial_sites_use_q_coordinates():
    u = np.array([[0.4, 0.1], [0.3, 0.25]])
    sites = _quadratic_uniform_initial_sites(u, "kipping")
    np.testing.assert_allclose(kipping_u_from_q(sites["ld_q"]), u, atol=1e-12)


def test_config_validation_accepts_kipping():
    assert "kipping" in LD_PROFILES
    assert validate_ld_profile("kipping") == "kipping"
    for prior in ("uniform", "gaussian", "fixed"):
        assert validate_ld_prior(prior, ld_profile="kipping") == prior
    assert resolve_ld_prior(None, ld_profile="kipping") == "gaussian"
    with pytest.raises(ValueError, match="'sing'.*'quadratic'"):
        validate_ld_prior("sing", ld_profile="kipping")
    with pytest.raises(ValueError, match="'stellarprior'.*'power2'"):
        validate_ld_prior("stellarprior", ld_profile="kipping")
