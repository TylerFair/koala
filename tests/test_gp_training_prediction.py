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

