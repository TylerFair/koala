from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import models.gp as kgp


def _base_params(**updates):
    params = dict(GP_log_rho=jnp.array(-0.7), GP_log_sigma=jnp.array(-1.5),
                  c=jnp.array(0.9), v=jnp.array(0.08), v2=jnp.array(-0.01),
                  v3=jnp.array(0.004), v4=jnp.array(-0.001),
                  A=jnp.array(0.03), tau=jnp.array(0.4))
    params.update(updates)
    return params


@pytest.mark.parametrize("name", list(kgp.GP_MEAN_FUNCTIONS))
def test_all_mean_functions_full_array_matches_scalar_vmap(monkeypatch, name):
    monkeypatch.setattr(kgp, "compute_transit_model_auto",
                        lambda params, t: -0.01 * jnp.cos(2.0 * t))
    t = jnp.array([3.0, 3.07, 3.2, 3.55, 4.0])
    function = kgp.resolve_gp_mean_function(name)
    bound = partial(function, _base_params(), t_ref=jnp.min(t))
    full = bound(t)
    scalar = jax.vmap(bound)(t)
    np.testing.assert_allclose(scalar, full, rtol=0, atol=2e-15)


def test_builder_binds_training_origin_and_nonconstant_mean(monkeypatch):
    monkeypatch.setattr(kgp, "compute_transit_model_auto",
                        lambda params, t: jnp.zeros_like(t))
    params = _base_params()
    t = jnp.array([8.0, 8.2, 8.7, 9.4])
    error = jnp.full(t.shape, 0.03)
    gp = kgp.build_gp_explinear(params, t, error, gp_solver="serial",
                                assume_sorted=True)
    scalar = jax.vmap(gp.mean_function)(t)
    expected = kgp.compute_lc_explinear_gp_mean(params, t, t_ref=t.min())
    np.testing.assert_allclose(scalar, gp.loc, rtol=0, atol=2e-15)
    np.testing.assert_allclose(scalar, expected, rtol=0, atol=2e-15)
    assert not np.allclose(np.asarray(scalar), scalar[0])
    later = jnp.array([10.0, 10.4])
    np.testing.assert_allclose(jax.vmap(gp.mean_function)(later),
                               kgp.compute_lc_explinear_gp_mean(
                                   params, later, t_ref=t.min()))


def _jaxoplanet_params():
    return _base_params(
        period=jnp.array([4.0]), duration=jnp.array([0.16]),
        t0=jnp.array([1.0]), b=jnp.array([0.3]), rors=jnp.array([0.1]),
        u=jnp.array([0.2, 0.1]),
    )


def _harmonica_params():
    return _base_params(
        period=jnp.array([4.0]), t0=jnp.array([1.0]), b=jnp.array([0.3]),
        rors=jnp.array([0.1]), a_rs=jnp.array([10.0]),
        c_ld=jnp.array(0.55), alpha_ld=jnp.array(0.7),
    )


@pytest.mark.parametrize("factory", [_jaxoplanet_params, _harmonica_params])
@pytest.mark.parametrize("name", list(kgp.GP_MEAN_FUNCTIONS))
def test_real_transit_engines_are_scalar_vmap_consistent(factory, name):
    pytest.importorskip("jaxoplanet")
    params = factory()
    t = jnp.array([0.91, 0.97, 1.0, 1.05, 1.13])
    function = kgp.resolve_gp_mean_function(name)
    bound = partial(function, params, t_ref=jnp.min(t))
    full = bound(t)
    scalar = jax.vmap(bound)(t)
    np.testing.assert_allclose(scalar, full, rtol=2e-10, atol=2e-10)


def test_legacy_none_tref_uses_argument_min(monkeypatch):
    monkeypatch.setattr(kgp, "compute_transit_model_auto",
                        lambda params, t: jnp.zeros_like(t))
    params = _base_params()
    t = jnp.array([4.0, 4.5, 5.0])
    expected = params["c"] + params["v"] * (t - 4.0)
    np.testing.assert_allclose(kgp.compute_lc_linear_gp_mean(params, t), expected)

