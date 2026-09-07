"""Pareto-smoothed importance sampling utilities."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jax.scipy as jsp


def _gpdfit(sorted_excesses):
    """Zhang-Stephens empirical-Bayes GPD fit used by ArviZ PSIS."""
    ary = jnp.asarray(sorted_excesses, dtype=jnp.float64)
    n = ary.shape[-1]
    prior_bs = 3.0
    prior_k = 10.0
    m_est = 30 + int(math.sqrt(n))
    indices = jnp.arange(1, m_est + 1, dtype=jnp.float64)
    b_ary = 1.0 - jnp.sqrt(m_est / (indices - 0.5))
    quartile_index = int(n / 4.0 + 0.5) - 1
    b_ary = b_ary / (prior_bs * ary[quartile_index]) + 1.0 / ary[-1]
    log_terms = jnp.log1p(-b_ary[:, None] * ary[None, :])
    k_ary = jnp.mean(log_terms, axis=1)
    len_scale = n * (jnp.log(-(b_ary / k_ary)) - k_ary - 1.0)
    # ArviZ/Zhang-Stephens use
    # 1 / sum_j exp(len_scale[j] - len_scale[i]) = softmax(len_scale)[i].
    weights = jax.nn.softmax(len_scale)
    b_post = jnp.sum(b_ary * weights)
    k_post = jnp.mean(jnp.log1p(-b_post * ary))
    sigma = -k_post / b_post
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)
    return k_post, sigma


def _gpinv(probabilities, kappa, sigma):
    probabilities = jnp.asarray(probabilities, dtype=jnp.float64)
    exponential = -jnp.log1p(-probabilities)
    pareto = jnp.expm1(-kappa * jnp.log1p(-probabilities)) / kappa
    standardized = jnp.where(jnp.abs(kappa) < jnp.finfo(jnp.float64).eps,
                             exponential, pareto)
    return standardized * sigma


def _psis_one(log_weights):
    """Pareto-smooth and normalize one fixed-length log-weight vector."""
    x = jnp.asarray(log_weights, dtype=jnp.float64)
    n_samples = x.shape[0]
    tail_size = int(math.ceil(min(n_samples / 5.0, 3.0 * math.sqrt(n_samples))))
    max_x = jnp.max(x)
    centered = x - max_x
    order = jnp.argsort(centered)
    sorted_x = centered[order]
    cutoff = jnp.maximum(
        sorted_x[-tail_size - 1],
        jnp.log(jnp.finfo(jnp.float64).tiny),
    )
    tail = sorted_x[-tail_size:]
    excess = jnp.exp(tail) - jnp.exp(cutoff)

    if tail_size <= 4:
        k_hat = jnp.asarray(jnp.inf, dtype=jnp.float64)
        smooth_tail = tail
    else:
        k_hat, sigma = _gpdfit(excess)
        probabilities = (
            jnp.arange(tail_size, dtype=jnp.float64) + 0.5
        ) / tail_size
        fitted = jnp.log(_gpinv(probabilities, k_hat, sigma) + jnp.exp(cutoff))
        valid = jnp.isfinite(k_hat) & jnp.isfinite(sigma) & (sigma > 0.0)
        smooth_tail = jnp.where(valid, fitted, tail)
        smooth_tail = jnp.minimum(smooth_tail, 0.0)

    sorted_out = sorted_x.at[-tail_size:].set(smooth_tail)
    out = jnp.empty_like(sorted_out).at[order].set(sorted_out)
    out = out - jsp.special.logsumexp(out)
    return out, k_hat


def psis_smooth_log_weights(log_weights):
    """JAX PSIS along the final axis, returning normalized log weights and k."""
    values = jnp.asarray(log_weights, dtype=jnp.float64)
    if values.ndim < 1 or values.shape[-1] < 6:
        raise ValueError("PSIS requires at least six weights along the final axis.")
    flat = values.reshape((-1, values.shape[-1]))
    smoothed, k_hat = jax.vmap(_psis_one)(flat)
    return (
        smoothed.reshape(values.shape),
        k_hat.reshape(values.shape[:-1]),
    )
