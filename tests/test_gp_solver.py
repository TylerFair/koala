import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import solve_triangular

jax.config.update("jax_enable_x64", True)

import models.gp as kgp


def _mean(params, t, t_ref=None):
    return params["c"] + params["v"] * (t - t_ref)


def _params(theta=None):
    if theta is None:
        theta = jnp.array([-0.7, -1.5, 0.9, 0.03])
    return dict(GP_log_rho=theta[0], GP_log_sigma=theta[1], c=theta[2], v=theta[3])


def _data(n=33):
    t = jnp.asarray(np.sort(np.random.default_rng(41).uniform(0, 2, n)))
    error = 0.035 + 0.01 * jnp.arange(n) / max(n - 1, 1)
    y = 0.9 + 0.02 * t + 0.03 * jnp.sin(4.1 * t)
    return t, error, y


def _gp(theta, t, error, solver):
    return kgp.build_gp_model(
        _params(theta), t, error, mean_function=_mean, gp_solver=solver,
        assume_sorted=True,
    )


def _dense_cov(theta, t, error):
    distance = jnp.abs(t[:, None] - t[None, :])
    q = jnp.sqrt(3.0) * distance / jnp.exp(theta[0])
    kernel = jnp.exp(2.0 * theta[1]) * (1.0 + q) * jnp.exp(-q)
    return kernel + jnp.diag(error**2)


def _dense_logp(theta, t, error, y):
    loc = theta[2] + theta[3] * (t - jnp.min(t))
    residual = y - loc
    covariance = _dense_cov(theta, t, error)
    sign, logdet = jnp.linalg.slogdet(covariance)
    return -0.5 * (
        residual @ jnp.linalg.solve(covariance, residual)
        + logdet + t.size * jnp.log(2 * jnp.pi)
    )


def _modes():
    return ["serial"] + (["parallel"] if kgp.gp_parallel_supported() else [])


def test_solver_selection_and_precedence(monkeypatch):
    monkeypatch.setenv(kgp.GP_SOLVER_ENV, "parallel")
    assert kgp.resolve_gp_solver("serial", device="gpu") == "serial"
    monkeypatch.setenv(kgp.GP_SOLVER_ENV, "serial")
    assert kgp.resolve_gp_solver(device="gpu") == "serial"
    monkeypatch.delenv(kgp.GP_SOLVER_ENV)
    assert kgp.resolve_gp_solver(device="cpu") == "serial"
    with pytest.raises(ValueError, match="Unknown GP solver"):
        kgp.resolve_gp_solver("bogus")


def test_legacy_parallel_error_is_actionable(monkeypatch):
    monkeypatch.setattr(kgp, "gp_parallel_supported", lambda: False)
    with pytest.raises(kgp.GPSolverUnavailableError) as exc:
        kgp.resolve_gp_solver("parallel")
    message = str(exc.value)
    assert "Python >= 3.10" in message
    assert (
        'pip install "tinygp @ git+https://github.com/TylerFair/tinygp.git@'
        'koala-parallel-v1"'
    ) in message
    with pytest.warns(RuntimeWarning, match="falling back"):
        assert kgp.resolve_gp_solver("auto", device="gpu") == "serial"


@pytest.mark.parametrize("bad", [[], [[1.0]], [0.0, np.nan], [1.0, 0.0]])
def test_validate_gp_times_rejects_invalid_input(bad):
    with pytest.raises(ValueError, match="GP times"):
        kgp.validate_gp_times(bad)


def test_validate_gp_times_accepts_duplicates():
    value = kgp.validate_gp_times([0.0, 0.2, 0.2, 1.0])
    np.testing.assert_array_equal(value, [0.0, 0.2, 0.2, 1.0])


@pytest.mark.parametrize("solver", _modes())
def test_log_probability_gradient_hessian_and_triangular_dense_oracle(solver):
    t, error, y = _data(17)
    theta = jnp.array([-0.7, -1.5, 0.9, 0.03])
    objective = lambda value: _gp(value, t, error, solver).log_probability(y)
    np.testing.assert_allclose(objective(theta), _dense_logp(theta, t, error, y),
                               rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(jax.grad(objective)(theta),
                               jax.grad(_dense_logp)(theta, t, error, y),
                               rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(jax.hessian(objective)(theta),
                               jax.hessian(_dense_logp)(theta, t, error, y),
                               rtol=2e-7, atol=2e-7)

    gp = _gp(theta, t, error, solver)
    dense_l = np.linalg.cholesky(np.asarray(_dense_cov(theta, t, error)))
    rhs = np.asarray(jnp.cos(t))
    np.testing.assert_allclose(gp.solver.solve_triangular(rhs),
                               solve_triangular(dense_l, rhs, lower=True),
                               rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(gp.solver.solve_triangular(rhs, transpose=True),
                               solve_triangular(dense_l.T, rhs, lower=False),
                               rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("solver", _modes())
@pytest.mark.parametrize("n", [1, 2, 7, 33, 1025])
def test_irregular_odd_sizes_under_jit(solver, n):
    t, error, y = _data(n)
    value = jax.jit(lambda yy: _gp(jnp.array([-0.7, -1.5, 0.9, 0.03]),
                                     t, error, solver).log_probability(yy))(y)
    assert jnp.isfinite(value)


@pytest.mark.parametrize("solver", _modes())
def test_duplicate_timestamps_and_vmap_hyperparameters(solver):
    t = jnp.array([0.0, 0.03, 0.03, 0.4, 0.9, 1.1, 1.1])
    error = jnp.full(t.shape, 0.04)
    y = 0.9 + 0.01 * t
    values = jnp.array([[-0.7, -1.5, 0.9, 0.03], [-0.2, -1.2, 0.91, -0.01]])
    result = jax.jit(jax.vmap(lambda theta: _gp(theta, t, error, solver)
                              .log_probability(y)))(values)
    assert result.shape == (2,)
    assert jnp.all(jnp.isfinite(result))


@pytest.mark.parametrize("solver", _modes())
def test_sampling_and_dot_triangular_statistics(solver):
    n, draws = 64, 3000
    t, error, _ = _data(n)
    theta = jnp.array([-0.2, -1.7, 0.9, 0.03])
    gp = _gp(theta, t, error, solver)
    samples = gp.sample(jax.random.PRNGKey(2), shape=(draws,))
    covariance = np.asarray(_dense_cov(theta, t, error))
    empirical_mean = np.asarray(samples).mean(axis=0)
    empirical_cov = np.cov(np.asarray(samples), rowvar=False)
    np.testing.assert_allclose(empirical_mean, np.asarray(gp.loc), atol=1.3e-2)
    np.testing.assert_allclose(np.diag(empirical_cov), np.diag(covariance),
                               rtol=0.12, atol=2e-3)
    np.testing.assert_allclose(empirical_cov, covariance, rtol=0.35, atol=2.5e-3)
    z = jax.random.normal(jax.random.PRNGKey(3), (n, draws))
    transformed = np.asarray(gp.solver.dot_triangular(z)).T
    transformed_cov = np.cov(transformed, rowvar=False)
    np.testing.assert_allclose(transformed_cov, covariance,
                               rtol=0.35, atol=2.5e-3)


def test_parallel_mode_skips_cleanly_or_is_available():
    if kgp.gp_parallel_supported():
        assert kgp.resolve_gp_solver("parallel") == "parallel"
    else:
        with pytest.raises(kgp.GPSolverUnavailableError):
            kgp.resolve_gp_solver("parallel")
