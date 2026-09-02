import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from models.linear_marginalization import (
    conditional_coefficients,
    marginalized_log_likelihood,
    marginalized_log_likelihood_and_conditional,
    sample_conditional,
)


def _dense_reference(residual, design, noise_scale, prior_mean, prior_covariance):
    """Observation-space reference used only by these small unit tests."""

    residual = np.asarray(residual, dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)
    noise_scale = np.asarray(noise_scale, dtype=np.float64)
    prior_mean = np.asarray(prior_mean, dtype=np.float64)
    prior_covariance = np.asarray(prior_covariance, dtype=np.float64)

    centered = residual - design @ prior_mean
    data_covariance = (
        np.diag(noise_scale**2)
        + design @ prior_covariance @ design.T
    )
    sign, logdet = np.linalg.slogdet(data_covariance)
    assert sign == 1.0
    solved_centered = np.linalg.solve(data_covariance, centered)
    log_likelihood = -0.5 * (
        residual.size * np.log(2.0 * np.pi)
        + logdet
        + centered @ solved_centered
    )

    cross_covariance = prior_covariance @ design.T
    posterior_mean = (
        prior_mean + cross_covariance @ solved_centered
    )
    posterior_covariance = prior_covariance - cross_covariance @ np.linalg.solve(
        data_covariance, design @ prior_covariance
    )
    posterior_covariance = 0.5 * (
        posterior_covariance + posterior_covariance.T
    )
    return log_likelihood, posterior_mean, posterior_covariance


def _example_problem():
    time = np.linspace(-0.8, 1.1, 9, dtype=np.float64)
    design = np.stack(
        (np.ones_like(time), time, time**2 - np.mean(time**2)), axis=-1
    )
    residual = 0.17 - 0.23 * time + 0.08 * time**2 + 0.03 * np.sin(4 * time)
    noise_scale = np.linspace(0.07, 0.16, time.size, dtype=np.float64)
    prior_mean = np.array([0.11, -0.04, 0.02], dtype=np.float64)
    prior_scale = np.array([0.45, 0.31, 0.22], dtype=np.float64)
    return residual, design, noise_scale, prior_mean, prior_scale


def test_diagonal_prior_matches_dense_marginal_and_conditional():
    residual, design, noise_scale, prior_mean, prior_scale = _example_problem()
    expected = _dense_reference(
        residual,
        design,
        noise_scale,
        prior_mean,
        np.diag(prior_scale**2),
    )

    actual_log_likelihood, actual_conditional = (
        marginalized_log_likelihood_and_conditional(
            residual,
            design,
            noise_scale,
            prior_mean,
            prior_scale=prior_scale,
        )
    )

    np.testing.assert_allclose(actual_log_likelihood, expected[0], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(actual_conditional.mean, expected[1], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        actual_conditional.covariance, expected[2], rtol=4e-12, atol=4e-12
    )
    np.testing.assert_allclose(
        actual_conditional.factor @ actual_conditional.factor.T,
        actual_conditional.covariance,
        rtol=2e-14,
        atol=2e-14,
    )
    assert actual_log_likelihood.dtype == jnp.float64
    assert actual_conditional.mean.dtype == jnp.float64


def test_general_covariance_batches_jit_and_vmap_match_dense_reference():
    rng = np.random.default_rng(98271)
    num_batches, num_observations, num_coefficients = 4, 8, 3
    time = np.linspace(-1.0, 1.0, num_observations, dtype=np.float64)
    # This basis is intentionally shared across all channels.
    design = np.stack((np.ones_like(time), time, np.exp(time)), axis=-1)
    residual = rng.normal(0.0, 0.2, size=(num_batches, num_observations))
    noise_scale = rng.uniform(0.04, 0.13, size=(num_batches, num_observations))
    prior_mean = rng.normal(0.0, 0.1, size=(num_batches, num_coefficients))
    covariance_roots = rng.normal(
        0.0, 0.15, size=(num_batches, num_coefficients, num_coefficients)
    )
    prior_covariance = np.matmul(
        covariance_roots, np.swapaxes(covariance_roots, -1, -2)
    ) + 0.025 * np.eye(num_coefficients)[None, :, :]

    compiled = jax.jit(
        lambda r, s, m, c: marginalized_log_likelihood_and_conditional(
            r,
            design,
            s,
            m,
            prior_covariance=c,
        )
    )
    batched_log_likelihood, batched_conditional = compiled(
        residual, noise_scale, prior_mean, prior_covariance
    )

    expected = [
        _dense_reference(
            residual[index],
            design,
            noise_scale[index],
            prior_mean[index],
            prior_covariance[index],
        )
        for index in range(num_batches)
    ]
    np.testing.assert_allclose(
        batched_log_likelihood,
        np.asarray([item[0] for item in expected]),
        rtol=3e-12,
        atol=3e-12,
    )
    np.testing.assert_allclose(
        batched_conditional.mean,
        np.asarray([item[1] for item in expected]),
        rtol=4e-12,
        atol=4e-12,
    )
    np.testing.assert_allclose(
        batched_conditional.covariance,
        np.asarray([item[2] for item in expected]),
        rtol=8e-12,
        atol=8e-12,
    )

    per_channel = lambda r, s, m, c: marginalized_log_likelihood_and_conditional(
        r, design, s, m, prior_covariance=c
    )
    vmapped_log_likelihood, vmapped_conditional = jax.jit(jax.vmap(per_channel))(
        residual, noise_scale, prior_mean, prior_covariance
    )
    np.testing.assert_array_equal(vmapped_log_likelihood, batched_log_likelihood)
    np.testing.assert_allclose(
        vmapped_conditional.mean, batched_conditional.mean, rtol=2e-14, atol=2e-14
    )
    np.testing.assert_allclose(
        vmapped_conditional.covariance,
        batched_conditional.covariance,
        rtol=2e-14,
        atol=2e-14,
    )

    # Residual-only batching is a useful broadcasting edge case: JAX's raw
    # triangular_solve does not itself broadcast an unbatched KxK factor.
    shared_system_log_likelihood = marginalized_log_likelihood(
        residual,
        design,
        noise_scale[0],
        prior_mean[0],
        prior_covariance=prior_covariance[0],
    )
    shared_system_expected = [
        _dense_reference(
            channel_residual,
            design,
            noise_scale[0],
            prior_mean[0],
            prior_covariance[0],
        )[0]
        for channel_residual in residual
    ]
    np.testing.assert_allclose(
        shared_system_log_likelihood,
        shared_system_expected,
        rtol=3e-12,
        atol=3e-12,
    )


def test_gradient_matches_dense_observation_space_reference():
    time = jnp.linspace(-0.7, 0.9, 11, dtype=jnp.float64)
    measured = 1.0 - 0.011 * jnp.exp(-0.5 * (time / 0.19) ** 2) + 0.003 * time
    reported_error = jnp.linspace(8.0e-4, 1.3e-3, time.size, dtype=jnp.float64)
    prior_mean = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)

    def problem(theta):
        depth, log_jitter, log_tau, log_prior_scale = theta
        nonlinear_model = -depth * jnp.exp(-0.5 * (time / 0.21) ** 2)
        residual = measured - nonlinear_model
        noise_scale = jnp.sqrt(reported_error**2 + jnp.exp(2.0 * log_jitter))
        tau = jnp.exp(log_tau)
        shifted_time = time - jnp.min(time)
        design = jnp.stack(
            (jnp.ones_like(time), shifted_time, jnp.exp(-shifted_time / tau)),
            axis=-1,
        )
        prior_scale = jnp.exp(
            log_prior_scale + jnp.array([0.0, -0.4, -0.7], dtype=jnp.float64)
        )
        return residual, design, noise_scale, prior_scale

    def coefficient_space(theta):
        residual, design, noise_scale, prior_scale = problem(theta)
        return marginalized_log_likelihood(
            residual,
            design,
            noise_scale,
            prior_mean,
            prior_scale=prior_scale,
        )

    def dense_observation_space(theta):
        residual, design, noise_scale, prior_scale = problem(theta)
        centered = residual - design @ prior_mean
        scaled_design = design * prior_scale[None, :]
        covariance = jnp.diag(noise_scale**2) + scaled_design @ scaled_design.T
        _, logdet = jnp.linalg.slogdet(covariance)
        return -0.5 * (
            time.size * jnp.log(2.0 * jnp.pi)
            + logdet
            + centered @ jnp.linalg.solve(covariance, centered)
        )

    theta = jnp.array([0.012, -7.0, -1.8, -1.2], dtype=jnp.float64)
    actual_value, actual_gradient = jax.jit(jax.value_and_grad(coefficient_space))(theta)
    expected_value, expected_gradient = jax.jit(
        jax.value_and_grad(dense_observation_space)
    )(theta)

    np.testing.assert_allclose(actual_value, expected_value, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(
        actual_gradient, expected_gradient, rtol=3e-8, atol=3e-8
    )
    assert np.all(np.isfinite(np.asarray(actual_gradient)))


def test_conditional_draws_have_requested_shape_and_moments():
    residual, design, noise_scale, prior_mean, prior_scale = _example_problem()
    conditional = conditional_coefficients(
        residual,
        design,
        noise_scale,
        prior_mean,
        prior_scale=prior_scale,
    )
    draw = jax.jit(
        lambda key: sample_conditional(key, conditional, sample_shape=(40_000,))
    )
    samples = np.asarray(draw(jax.random.PRNGKey(471)))

    assert samples.shape == (40_000, design.shape[-1])
    np.testing.assert_allclose(
        samples.mean(axis=0), conditional.mean, rtol=0.0, atol=1.5e-3
    )
    np.testing.assert_allclose(
        np.cov(samples, rowvar=False),
        conditional.covariance,
        rtol=0.025,
        atol=2.5e-4,
    )

    batched_conditional = conditional_coefficients(
        np.stack((residual, residual + 0.02)),
        design,
        np.stack((noise_scale, 1.2 * noise_scale)),
        np.stack((prior_mean, prior_mean)),
        prior_scale=np.stack((prior_scale, prior_scale)),
    )
    batched_draws = jax.jit(
        lambda key: sample_conditional(key, batched_conditional, sample_shape=5)
    )(jax.random.PRNGKey(99))
    assert batched_draws.shape == (5, 2, design.shape[-1])


def test_broad_prior_small_noise_remains_finite():
    time = jnp.linspace(-1.0, 1.0, 80, dtype=jnp.float64)
    design = jnp.stack(
        (jnp.ones_like(time), time, time**2, time**3), axis=-1
    )
    coefficients = jnp.array([1.0, -0.2, 0.03, 0.01], dtype=jnp.float64)
    residual = design @ coefficients + 2.0e-6 * jnp.sin(17.0 * time)
    noise_scale = jnp.full(time.shape, 1.0e-5, dtype=jnp.float64)

    log_likelihood, conditional = marginalized_log_likelihood_and_conditional(
        residual,
        design,
        noise_scale,
        jnp.zeros(4, dtype=jnp.float64),
        prior_scale=jnp.full((4,), 1.0e3, dtype=jnp.float64),
    )
    assert jnp.isfinite(log_likelihood)
    assert jnp.all(jnp.isfinite(conditional.mean))
    assert jnp.all(jnp.isfinite(conditional.covariance))
    assert jnp.all(jnp.linalg.eigvalsh(conditional.covariance) > 0.0)


def test_likelihood_jaxpr_never_materializes_observation_covariance():
    num_observations, num_coefficients = 17, 3
    residual = jnp.linspace(-0.1, 0.2, num_observations, dtype=jnp.float64)
    design = jnp.reshape(
        jnp.linspace(-1.0, 1.0, num_observations * num_coefficients),
        (num_observations, num_coefficients),
    )
    noise_scale = jnp.full((num_observations,), 0.1, dtype=jnp.float64)
    prior_mean = jnp.zeros((num_coefficients,), dtype=jnp.float64)
    prior_scale = jnp.ones((num_coefficients,), dtype=jnp.float64)

    closed_jaxpr = jax.make_jaxpr(
        lambda r, x, s, m, p: marginalized_log_likelihood(
            r, x, s, m, prior_scale=p
        )
    )(residual, design, noise_scale, prior_mean, prior_scale)
    all_shapes = []
    for equation in closed_jaxpr.jaxpr.eqns:
        for variable in equation.outvars:
            shape = getattr(getattr(variable, "aval", None), "shape", None)
            if shape is not None:
                all_shapes.append(tuple(shape))

    assert (num_observations, num_observations) not in all_shapes
    assert (num_coefficients, num_coefficients) in all_shapes


def test_prior_choice_and_static_shapes_are_validated():
    residual, design, noise_scale, prior_mean, prior_scale = _example_problem()
    prior_covariance = np.diag(prior_scale**2)

    with pytest.raises(ValueError, match="exactly one"):
        marginalized_log_likelihood(
            residual, design, noise_scale, prior_mean
        )
    with pytest.raises(ValueError, match="exactly one"):
        marginalized_log_likelihood(
            residual,
            design,
            noise_scale,
            prior_mean,
            prior_scale=prior_scale,
            prior_covariance=prior_covariance,
        )
    with pytest.raises(ValueError, match="disagree on N"):
        marginalized_log_likelihood(
            residual[:-1],
            design,
            noise_scale,
            prior_mean,
            prior_scale=prior_scale,
        )
    with pytest.raises(ValueError, match="broadcast-compatible"):
        marginalized_log_likelihood(
            np.broadcast_to(residual, (2, residual.size)),
            design,
            np.broadcast_to(noise_scale, (3, noise_scale.size)),
            prior_mean,
            prior_scale=prior_scale,
        )
