from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import models.gp as kgp


def _mean(params, t, t_ref=None):
    x = t - t_ref
    return params["c"] + params["v"] * x + params["A"] * jnp.exp(-x / params["tau"])


def _case(n=19):
    t = jnp.asarray(np.sort(np.random.default_rng(91).uniform(4.0, 5.0, n)))
    error = jnp.linspace(0.025, 0.045, n)
    params = dict(GP_log_rho=jnp.array(-0.8), GP_log_sigma=jnp.array(-1.7),
                  c=jnp.array(0.95), v=jnp.array(0.04), A=jnp.array(0.03),
                  tau=jnp.array(0.3))
    y = _mean(params, t, jnp.min(t)) + 0.015 * jnp.sin(15 * t)
    return params, t, error, y


def _modes():
    return ["serial"] + (["parallel"] if kgp.gp_parallel_supported() else [])


@pytest.mark.parametrize("solver", _modes())
@pytest.mark.parametrize("diag", [None, 0.0, 2.5e-8])
@pytest.mark.parametrize("include_mean", [True, False])
def test_training_prediction_matches_dense_and_tinygp(solver, diag, include_mean):
    params, t, error, y = _case()
    gp = kgp.build_gp_model(params, t, error, mean_function=_mean,
                            gp_solver=solver, assume_sorted=True)
    actual_mean, actual_var = kgp.predict_gp_training_points(
        gp, y, diag=diag, include_mean=include_mean
    )
    kernel = np.asarray(gp.kernel(t, t))
    covariance = kernel + np.diag(np.asarray(error) ** 2)
    alpha = np.linalg.solve(covariance, np.asarray(y - gp.loc))
    expected_mean = np.asarray(gp.loc) + kernel @ alpha
    if not include_mean:
        expected_mean -= np.asarray(gp.loc)
    diag_pred = np.sqrt(np.finfo(np.float64).eps) if diag is None else diag
    expected_var = np.diag(kernel - kernel @ np.linalg.solve(covariance, kernel)) + diag_pred
    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual_var, expected_var, rtol=1e-9, atol=1e-11)

    legacy = gp.condition(y, diag=diag, include_mean=include_mean).gp
    np.testing.assert_allclose(actual_mean, legacy.loc, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual_var, legacy.variance, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("solver", _modes())
def test_return_mean_only_and_broadcast_noise(solver):
    params, t, _, y = _case(7)
    error = jnp.array(0.04)
    gp = kgp.build_gp_model(params, t, error, mean_function=_mean,
                            gp_solver=solver, assume_sorted=True)
    mean, _ = kgp.predict_gp_training_points(gp, y)
    only_mean = kgp.predict_gp_training_points(gp, y, return_var=False)
    np.testing.assert_array_equal(mean, only_mean)


def test_filter_jit_prediction_accepts_static_metadata(monkeypatch):
    params, t, error, y = _case(11)
    params = {**params, "_surface_model": "transit", "_stellar_spots": (),
              "unused_none": None}
    monkeypatch.setattr(kgp, "compute_transit_model_auto",
                        lambda p, x: jnp.zeros_like(x))
    mean, variance = kgp.compute_gp_training_prediction(
        params, t, error, y, detrend_type="explinear+gp", gp_solver="serial",
        assume_sorted=True, diag=0.0,
    )
    assert mean.shape == t.shape
    assert variance.shape == t.shape
    assert jnp.all(jnp.isfinite(mean))


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
