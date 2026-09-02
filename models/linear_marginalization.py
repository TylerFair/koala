r"""Analytic marginalization of linear light-curve coefficients.

This module evaluates the likelihood for

.. math::

    r = X\beta + \epsilon,\qquad
    \epsilon \sim \mathcal{N}(0, \operatorname{diag}(\sigma^2)),\qquad
    \beta \sim \mathcal{N}(m, C),

where ``r`` is the data minus the nonlinear light-curve model.  It works in
the (usually very small) coefficient space: the only square matrices formed
have shape ``(..., K, K)``, never ``(..., N, N)``.

All array arguments may have broadcast-compatible leading batch dimensions.
The final dimensions are ``N`` for observations and ``K`` for linear
coefficients.  In particular, a shared ``(N, K)`` design matrix can be used
with channel-batched ``(channels, N)`` residuals and noise scales.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


class LinearGaussianConditional(NamedTuple):
    """Conditional Gaussian distribution of the marginalized coefficients.

    ``factor`` is a square root of ``covariance`` (it is not necessarily
    triangular), so ``factor @ factor.T == covariance`` up to roundoff.
    Keeping the factor avoids another Cholesky decomposition when generating
    conditional coefficient draws.
    """

    mean: jax.Array
    covariance: jax.Array
    factor: jax.Array


class _WhitenedSystem(NamedTuple):
    log_likelihood: jax.Array
    prior_mean: jax.Array
    prior_cholesky: jax.Array
    precision_cholesky: jax.Array
    latent_mean: jax.Array


def _require_float64() -> None:
    if not jax.config.x64_enabled:
        raise RuntimeError(
            "Linear-trend marginalization requires JAX 64-bit mode. Set "
            "JAX_ENABLE_X64=1 before importing JAX or call "
            "jax.config.update('jax_enable_x64', True) during startup."
        )


def _as_float64(value) -> jax.Array:
    return jnp.asarray(value, dtype=jnp.float64)


def _validate_shapes(
    residual: jax.Array,
    design_matrix: jax.Array,
    noise_scale: jax.Array,
    prior_mean: jax.Array,
    prior_scale: jax.Array | None,
    prior_covariance: jax.Array | None,
) -> tuple[int, ...]:
    if residual.ndim < 1:
        raise ValueError("residual must have shape (..., N).")
    if design_matrix.ndim < 2:
        raise ValueError("design_matrix must have shape (..., N, K).")
    if noise_scale.ndim < 1:
        raise ValueError("noise_scale must have shape (..., N).")
    if prior_mean.ndim < 1:
        raise ValueError("prior_mean must have shape (..., K).")

    num_observations, num_coefficients = design_matrix.shape[-2:]
    if num_observations == 0 or num_coefficients == 0:
        raise ValueError("N and K must both be nonzero.")
    if residual.shape[-1] != num_observations:
        raise ValueError(
            "residual and design_matrix disagree on N: "
            f"{residual.shape[-1]} != {num_observations}."
        )
    if noise_scale.shape[-1] != num_observations:
        raise ValueError(
            "noise_scale and design_matrix disagree on N: "
            f"{noise_scale.shape[-1]} != {num_observations}."
        )
    if prior_mean.shape[-1] != num_coefficients:
        raise ValueError(
            "prior_mean and design_matrix disagree on K: "
            f"{prior_mean.shape[-1]} != {num_coefficients}."
        )

    if prior_scale is not None:
        if prior_scale.ndim < 1 or prior_scale.shape[-1] != num_coefficients:
            raise ValueError(
                "prior_scale must have shape (..., K), with K matching "
                "design_matrix."
            )
        prior_batch_shape = prior_scale.shape[:-1]
    else:
        assert prior_covariance is not None
        if (
            prior_covariance.ndim < 2
            or prior_covariance.shape[-2:] != (num_coefficients, num_coefficients)
        ):
            raise ValueError(
                "prior_covariance must have shape (..., K, K), with K "
                "matching design_matrix."
            )
        prior_batch_shape = prior_covariance.shape[:-2]

    batch_shapes = (
        residual.shape[:-1],
        design_matrix.shape[:-2],
        noise_scale.shape[:-1],
        prior_mean.shape[:-1],
        prior_batch_shape,
    )
    try:
        batch_shape = jnp.broadcast_shapes(*batch_shapes)
    except ValueError as exc:
        raise ValueError(
            "Leading batch dimensions are not broadcast-compatible: "
            + ", ".join(str(shape) for shape in batch_shapes)
        ) from exc
    return batch_shape


def _prior_cholesky(
    num_coefficients: int,
    *,
    prior_scale: jax.Array | None,
    prior_covariance: jax.Array | None,
) -> jax.Array:
    if (prior_scale is None) == (prior_covariance is None):
        raise ValueError(
            "Provide exactly one of prior_scale or prior_covariance."
        )
    if prior_scale is not None:
        identity = jnp.eye(num_coefficients, dtype=jnp.float64)
        return identity * prior_scale[..., None, :]
    return jnp.linalg.cholesky(prior_covariance)


def _build_whitened_system(
    residual,
    design_matrix,
    noise_scale,
    prior_mean,
    *,
    prior_scale=None,
    prior_covariance=None,
) -> _WhitenedSystem:
    _require_float64()

    residual = _as_float64(residual)
    design_matrix = _as_float64(design_matrix)
    noise_scale = _as_float64(noise_scale)
    prior_mean = _as_float64(prior_mean)
    prior_scale = (
        None if prior_scale is None else _as_float64(prior_scale)
    )
    prior_covariance = (
        None
        if prior_covariance is None
        else _as_float64(prior_covariance)
    )
    if (prior_scale is None) == (prior_covariance is None):
        raise ValueError(
            "Provide exactly one of prior_scale or prior_covariance."
        )

    batch_shape = _validate_shapes(
        residual,
        design_matrix,
        noise_scale,
        prior_mean,
        prior_scale,
        prior_covariance,
    )
    num_observations, num_coefficients = design_matrix.shape[-2:]
    # JAX's triangular solve requires identical (rather than merely
    # broadcast-compatible) batch ranks, so make the inexpensive broadcasts
    # explicit before constructing the coefficient-space system.
    residual = jnp.broadcast_to(
        residual, batch_shape + (num_observations,)
    )
    design_matrix = jnp.broadcast_to(
        design_matrix, batch_shape + (num_observations, num_coefficients)
    )
    noise_scale = jnp.broadcast_to(
        noise_scale, batch_shape + (num_observations,)
    )
    prior_mean = jnp.broadcast_to(
        prior_mean, batch_shape + (num_coefficients,)
    )
    if prior_scale is not None:
        prior_scale = jnp.broadcast_to(
            prior_scale, batch_shape + (num_coefficients,)
        )
    else:
        prior_covariance = jnp.broadcast_to(
            prior_covariance,
            batch_shape + (num_coefficients, num_coefficients),
        )
    prior_cholesky = _prior_cholesky(
        num_coefficients,
        prior_scale=prior_scale,
        prior_covariance=prior_covariance,
    )

    prior_prediction = jnp.einsum(
        "...nk,...k->...n", design_matrix, prior_mean
    )
    whitened_residual = (residual - prior_prediction) / noise_scale
    noise_whitened_design = design_matrix / noise_scale[..., :, None]
    # beta = prior_mean + prior_cholesky @ z, with z ~ Normal(0, I).
    whitened_design = jnp.matmul(noise_whitened_design, prior_cholesky)

    identity = jnp.eye(num_coefficients, dtype=jnp.float64)
    latent_precision = identity + jnp.matmul(
        jnp.swapaxes(whitened_design, -1, -2), whitened_design
    )
    precision_cholesky = jnp.linalg.cholesky(latent_precision)
    information = jnp.einsum(
        "...nk,...n->...k", whitened_design, whitened_residual
    )
    solved_lower = solve_triangular(
        precision_cholesky, information[..., None], lower=True
    )
    latent_mean = solve_triangular(
        jnp.swapaxes(precision_cholesky, -1, -2),
        solved_lower,
        lower=False,
    )[..., 0]

    # This equals e.T @ e - q.T @ precision^-1 @ q, but evaluating the
    # completed-square form avoids catastrophic cancellation for broad priors.
    conditional_residual = whitened_residual - jnp.einsum(
        "...nk,...k->...n", whitened_design, latent_mean
    )
    quadratic = (
        jnp.sum(conditional_residual**2, axis=-1)
        + jnp.sum(latent_mean**2, axis=-1)
    )
    logdet_noise = 2.0 * jnp.sum(jnp.log(noise_scale), axis=-1)
    logdet_latent_precision = 2.0 * jnp.sum(
        jnp.log(jnp.diagonal(precision_cholesky, axis1=-2, axis2=-1)),
        axis=-1,
    )
    normalization = num_observations * math.log(2.0 * math.pi)
    log_likelihood = -0.5 * (
        normalization + logdet_noise + logdet_latent_precision + quadratic
    )

    return _WhitenedSystem(
        log_likelihood=log_likelihood,
        prior_mean=prior_mean,
        prior_cholesky=prior_cholesky,
        precision_cholesky=precision_cholesky,
        latent_mean=latent_mean,
    )


def _conditional_from_system(
    system: _WhitenedSystem,
) -> LinearGaussianConditional:
    num_coefficients = system.precision_cholesky.shape[-1]
    identity = jnp.broadcast_to(
        jnp.eye(num_coefficients, dtype=jnp.float64),
        system.precision_cholesky.shape,
    )
    latent_factor = solve_triangular(
        jnp.swapaxes(system.precision_cholesky, -1, -2),
        identity,
        lower=False,
    )
    coefficient_factor = jnp.matmul(system.prior_cholesky, latent_factor)
    coefficient_mean = system.prior_mean + jnp.einsum(
        "...ij,...j->...i", system.prior_cholesky, system.latent_mean
    )
    coefficient_covariance = jnp.matmul(
        coefficient_factor, jnp.swapaxes(coefficient_factor, -1, -2)
    )
    return LinearGaussianConditional(
        mean=coefficient_mean,
        covariance=coefficient_covariance,
        factor=coefficient_factor,
    )


def marginalized_log_likelihood(
    residual,
    design_matrix,
    noise_scale,
    prior_mean,
    *,
    prior_scale=None,
    prior_covariance=None,
) -> jax.Array:
    """Return the fully normalized likelihood with ``beta`` integrated out.

    Parameters
    ----------
    residual
        Data minus the nonlinear model, with shape ``(..., N)``.
    design_matrix
        Linear basis vectors, with shape ``(..., N, K)``.
    noise_scale
        Strictly positive per-observation Gaussian standard deviations, with
        shape ``(..., N)``.
    prior_mean
        Gaussian coefficient-prior mean, with shape ``(..., K)``.
    prior_scale
        Strictly positive independent-prior standard deviations, with shape
        ``(..., K)``.  This is the cheapest preferred prior representation.
    prior_covariance
        Positive-definite general prior covariance with shape
        ``(..., K, K)``.  Exactly one prior representation must be supplied.

    Returns
    -------
    jax.Array
        One scalar log likelihood per broadcast batch element.
    """

    return _build_whitened_system(
        residual,
        design_matrix,
        noise_scale,
        prior_mean,
        prior_scale=prior_scale,
        prior_covariance=prior_covariance,
    ).log_likelihood


def conditional_coefficients(
    residual,
    design_matrix,
    noise_scale,
    prior_mean,
    *,
    prior_scale=None,
    prior_covariance=None,
) -> LinearGaussianConditional:
    """Return ``p(beta | residual, nonlinear parameters)`` analytically."""

    system = _build_whitened_system(
        residual,
        design_matrix,
        noise_scale,
        prior_mean,
        prior_scale=prior_scale,
        prior_covariance=prior_covariance,
    )
    return _conditional_from_system(system)


def marginalized_log_likelihood_and_conditional(
    residual,
    design_matrix,
    noise_scale,
    prior_mean,
    *,
    prior_scale=None,
    prior_covariance=None,
) -> tuple[jax.Array, LinearGaussianConditional]:
    """Return the marginal likelihood and conditional without duplicate work."""

    system = _build_whitened_system(
        residual,
        design_matrix,
        noise_scale,
        prior_mean,
        prior_scale=prior_scale,
        prior_covariance=prior_covariance,
    )
    return system.log_likelihood, _conditional_from_system(system)


def sample_conditional(
    key: jax.Array,
    conditional: LinearGaussianConditional,
    sample_shape: Sequence[int] | int = (),
) -> jax.Array:
    """Draw coefficient samples, preserving any leading conditional batches.

    The returned shape is ``sample_shape + conditional.mean.shape``.  An
    integer ``sample_shape`` is accepted as shorthand for a one-dimensional
    sample batch.
    """

    _require_float64()
    if isinstance(sample_shape, int):
        sample_shape = (sample_shape,)
    else:
        sample_shape = tuple(sample_shape)
    standard_normal = jax.random.normal(
        key,
        shape=sample_shape + conditional.mean.shape,
        dtype=jnp.float64,
    )
    offsets = jnp.einsum(
        "...ij,...j->...i", conditional.factor, standard_normal
    )
    return conditional.mean + offsets
