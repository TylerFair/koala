"""Laplace-centred adaptive importance sampling for factorized light curves.

Each wavelength channel is optimized and sampled independently in NumPyro's
unconstrained coordinates.  A repaired inverse Hessian initializes a
multivariate Student-t proposal, Pareto-smoothed importance weights drive a
small number of moment-matching rounds, and the final proposal is used either
for systematic resampling or an independence Metropolis-Hastings (IMH) chain.

The public entry point mirrors :func:`models.independent_nuts.get_samples_independent`.
It is valid only for models whose latent sample sites factorize by channel.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model

from models.independent_nuts import (
    _asarray_f64,
    _enrich_initial_values,
    _first_lane_initial_values,
    _pad_first_axis,
    _partition_model_kwargs,
    _prepare_unconstrained_initial_values,
    _split_dynamic_and_static_kwargs,
    _static_kwargs_equal,
    build_independent_nuts_runner,
    _squeeze_internal_channel_axis,
    get_samples_independent,
)


@dataclass(frozen=True)
class LaplaceISDiagnostics:
    """Quality and engineering diagnostics for every real channel."""

    converged: jax.Array
    pareto_k: jax.Array
    is_ess: jax.Array
    imh_acceptance: jax.Array
    max_rejection_run: jax.Array
    map_gradient_norm: jax.Array
    map_newton_decrement: jax.Array
    hessian_condition_number: jax.Array
    hessian_min_eigenvalue: jax.Array
    gate_passed: jax.Array
    fell_back: jax.Array
    map_iterations: jax.Array
    effective_draw_chunk_size: jax.Array
    map_seconds: jax.Array
    adaptation_seconds: jax.Array
    importance_round_seconds: jax.Array
    sampling_seconds: jax.Array
    postprocess_seconds: jax.Array
    fallback_seconds: jax.Array


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


def _resolve_laplace_options(
    *,
    nuts_kwargs,
    mcmc_kwargs,
    num_warmup,
    num_samples,
    laplace_is_output,
    laplace_is_num_draws,
    laplace_is_rounds,
    laplace_is_draw_chunk_size,
    laplace_is_student_df,
    laplace_is_scale_inflation,
    laplace_is_wide_fraction,
    laplace_is_wide_scale,
    laplace_is_map_maxiter,
    laplace_is_map_tol,
    laplace_is_trust_radius,
    laplace_is_khat_threshold,
    laplace_is_min_ess,
    laplace_is_min_ess_fraction,
    laplace_is_min_imh_acceptance,
    laplace_is_imh_thin,
    laplace_is_fallback,
):
    del nuts_kwargs  # NUTS options are consumed only if fallback is needed.
    mcmc = dict(mcmc_kwargs or {})
    supported_mcmc = {
        "num_warmup", "num_samples", "progress_bar", "jit_model_args"
    }
    unknown = set(mcmc) - supported_mcmc
    if unknown:
        raise ValueError(
            "Unsupported Laplace-IS MCMC options: " + ", ".join(sorted(unknown))
        )
    options = {
        "num_warmup": int(
            num_warmup if num_warmup is not None else mcmc.get("num_warmup", 1000)
        ),
        "num_samples": int(
            num_samples if num_samples is not None else mcmc.get("num_samples", 1000)
        ),
        "output": str(laplace_is_output).lower(),
        "n_draws": int(laplace_is_num_draws),
        "rounds": int(laplace_is_rounds),
        "draw_chunk_size": int(laplace_is_draw_chunk_size),
        "student_df": float(laplace_is_student_df),
        "scale_inflation": float(laplace_is_scale_inflation),
        "wide_fraction": float(laplace_is_wide_fraction),
        "wide_scale": float(laplace_is_wide_scale),
        "map_maxiter": int(laplace_is_map_maxiter),
        "map_tol": float(laplace_is_map_tol),
        "trust_radius": float(laplace_is_trust_radius),
        "khat_threshold": float(laplace_is_khat_threshold),
        "min_ess": float(laplace_is_min_ess),
        "min_ess_fraction": float(laplace_is_min_ess_fraction),
        "min_imh_acceptance": float(laplace_is_min_imh_acceptance),
        "imh_thin": int(laplace_is_imh_thin),
        "fallback": bool(laplace_is_fallback),
    }
    if options["output"] not in {"imh", "resample"}:
        raise ValueError("laplace_is_output must be 'imh' or 'resample'.")
    if options["num_warmup"] < 0 or options["num_samples"] < 1:
        raise ValueError("num_warmup must be >= 0 and num_samples must be >= 1.")
    if options["n_draws"] < 6 or options["rounds"] < 0:
        raise ValueError("laplace_is_num_draws >= 6 and laplace_is_rounds >= 0 are required.")
    if options["draw_chunk_size"] < 1 or options["map_maxiter"] < 1:
        raise ValueError("draw chunk size and MAP iteration budget must be positive.")
    if options["student_df"] <= 2.0 or options["scale_inflation"] <= 0.0:
        raise ValueError("Student df must exceed 2 and scale inflation must be positive.")
    if not 0.0 <= options["wide_fraction"] < 1.0:
        raise ValueError("laplace_is_wide_fraction must lie in [0, 1).")
    if options["wide_scale"] < 1.0:
        raise ValueError("laplace_is_wide_scale must be >= 1.")
    if options["map_tol"] <= 0.0 or options["trust_radius"] <= 0.0:
        raise ValueError("MAP tolerance and trust radius must be positive.")
    if not 0.0 <= options["min_ess_fraction"] <= 1.0:
        raise ValueError("laplace_is_min_ess_fraction must lie in [0, 1].")
    if not 0.0 <= options["min_imh_acceptance"] <= 1.0:
        raise ValueError("laplace_is_min_imh_acceptance must lie in [0, 1].")
    if options["imh_thin"] < 1:
        raise ValueError("laplace_is_imh_thin must be at least one.")
    return options


def _repair_positive_matrix(matrix, *, inverse=False):
    """Symmetrize/eigen-clip a small matrix and optionally invert it."""
    matrix = 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    spectral_scale = jnp.maximum(jnp.max(jnp.abs(eigenvalues)), 1.0)
    floor = jnp.maximum(1.0e-10, spectral_scale * 1.0e-8)
    clipped = jnp.maximum(eigenvalues, floor)
    values = 1.0 / clipped if inverse else clipped
    repaired = (eigenvectors * values[None, :]) @ eigenvectors.T
    condition = jnp.max(clipped) / jnp.min(clipped)
    return 0.5 * (repaired + repaired.T), condition


def _effective_draw_chunk_size(requested, lane_width, num_cadences, n_draws):
    """Choose a large static draw batch under a conservative 4 GiB budget."""
    requested = min(int(requested), int(n_draws))
    # Reverse-engineered transit intermediates need several float64 arrays per
    # draw/lane/cadence.  Budgeting 64 bytes per element keeps the large PRISM
    # workload below about 4 GiB while allowing 256-draw SOSS/G395H batches.
    memory_limited = int(
        (4 * 1024**3) // max(64 * int(lane_width) * int(num_cadences), 1)
    )
    selected = max(1, min(requested, memory_limited))
    # Power-of-two batches give stable XLA shapes and avoid an odd PRISM tail.
    return 1 << int(math.floor(math.log2(selected)))


def _student_scale_cholesky(covariance, student_df, inflation):
    covariance, _ = _repair_positive_matrix(covariance)
    scale = covariance * ((student_df - 2.0) / student_df) * inflation**2
    scale, _ = _repair_positive_matrix(scale)
    return jnp.linalg.cholesky(scale)


def _mixture_base_inflation(inflation, wide_fraction, wide_scale):
    """Keep the two-component proposal's aggregate covariance calibrated.

    Both Student-t components share a centre.  Without this correction a
    nominal 10% component at 3x scale nearly doubles the proposal covariance,
    defeating the moment-matching update.  ``inflation`` therefore controls
    the covariance of the complete mixture rather than only its narrow core.
    """
    variance_multiplier = (
        1.0 - wide_fraction + wide_fraction * wide_scale**2
    )
    return inflation / math.sqrt(variance_multiplier)


def _student_logpdf(samples, location, cholesky, student_df):
    delta = samples - location
    solved = jsp.linalg.solve_triangular(cholesky, delta, lower=True)
    dimension = samples.shape[-1]
    quadratic = jnp.sum(solved**2, axis=-1)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))
    normalizer = (
        jsp.special.gammaln((student_df + dimension) / 2.0)
        - jsp.special.gammaln(student_df / 2.0)
        - 0.5 * (dimension * jnp.log(student_df * jnp.pi) + logdet)
    )
    return normalizer - 0.5 * (student_df + dimension) * jnp.log1p(
        quadratic / student_df
    )


def _sample_student(key, location, cholesky, student_df, n_draws):
    normal_key, chi_key = jax.random.split(key)
    lanes, dimension = location.shape
    normal = jax.random.normal(normal_key, (n_draws, lanes, dimension), dtype=jnp.float64)
    chi2 = 2.0 * jax.random.gamma(
        chi_key,
        student_df / 2.0,
        shape=(n_draws, lanes),
        dtype=jnp.float64,
    )
    standardized = normal * jnp.sqrt(student_df / chi2)[..., None]
    return location[None, ...] + jnp.einsum("lij,slj->sli", cholesky, standardized)


def _student_mixture_logpdf(
    samples,
    location,
    cholesky,
    student_df,
    wide_fraction,
    wide_scale,
):
    base = _student_logpdf(samples, location, cholesky, student_df)
    if wide_fraction == 0.0:
        return base
    wide = _student_logpdf(
        samples, location, cholesky * wide_scale, student_df
    )
    return jnp.logaddexp(
        jnp.log1p(-wide_fraction) + base,
        jnp.log(wide_fraction) + wide,
    )


def _sample_student_mixture(
    key,
    location,
    cholesky,
    student_df,
    n_draws,
    wide_fraction,
    wide_scale,
):
    if wide_fraction == 0.0:
        return _sample_student(
            key, location, cholesky, student_df, n_draws
        )
    base_key, wide_key, choice_key = jax.random.split(key, 3)
    base = _sample_student(
        base_key, location, cholesky, student_df, n_draws
    )
    wide = _sample_student(
        wide_key,
        location,
        cholesky * wide_scale,
        student_df,
        n_draws,
    )
    choose_wide = jax.random.bernoulli(
        choice_key,
        wide_fraction,
        shape=(n_draws, location.shape[0], 1),
    )
    return jnp.where(choose_wide, wide, base)


def _save_diagnostics(path, diagnostics, *, output, n_draws, rounds):
    if path is None:
        return
    host = jax.device_get(diagnostics)
    payload = {
        "backend": "laplace_is",
        "output": output,
        "importance_draws_per_round": int(n_draws),
        "moment_matching_rounds": int(rounds),
        "num_channels": int(host.pareto_k.shape[0]),
        "num_fallback_channels": int(host.fell_back.sum()),
    }
    for field in host.__dataclass_fields__:
        value = getattr(host, field)
        payload[field + "_per_channel"] = value.tolist()
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)


class LaplaceISRunner:
    """Reusable, shape-specialized Laplace/IS program for one lane width."""

    def __init__(
        self,
        model,
        *,
        options,
        lane_width,
        channel_varying_kwargs=(),
    ):
        self.model = model
        self.options = dict(options)
        self.lane_width = int(lane_width)
        self.channel_varying_kwargs = tuple(channel_varying_kwargs)
        self.program_build_count = 0
        self._static_shared_kwargs = None
        self._map_program = None
        self._importance_program = None
        self._resample_program = None
        self._imh_program = None
        self._postprocess_program = None
        # Failed-lane counts vary by chunk.  Compile only the active fallback
        # width, then reuse that runner when a later chunk has the same count.
        self._fallback_runners = {}
        self._effective_draw_chunk_size = None

    def validate_configuration(self, model, *, options, lane_width, channel_varying_kwargs):
        if model is not self.model:
            raise ValueError("Laplace-IS runner belongs to another model.")
        if dict(options) != self.options:
            raise ValueError("Laplace-IS runner options differ from this call.")
        if int(lane_width) != self.lane_width:
            raise ValueError("Laplace-IS runner lane width differs from this call.")
        if tuple(channel_varying_kwargs) != self.channel_varying_kwargs:
            raise ValueError("Laplace-IS runner channel-varying arguments differ from this call.")

    @staticmethod
    def _merge_lane_kwargs(static_shared, dynamic_shared, varying_lane):
        return {**static_shared, **dynamic_shared, **varying_lane}

    def _build_programs(
        self,
        model_info,
        static_shared_kwargs,
        unravel,
        draw_chunk_size,
    ):
        self._static_shared_kwargs = dict(static_shared_kwargs)
        self._effective_draw_chunk_size = int(draw_chunk_size)
        static_shared = self._static_shared_kwargs
        options = self.options
        potential_gen = model_info.potential_fn
        varying_axes = {name: 0 for name in self.channel_varying_kwargs}
        line_search_scales = jnp.power(
            jnp.asarray(0.5, dtype=jnp.float64),
            jnp.arange(12, dtype=jnp.float64),
        )

        def objective_one(flat, t, err, obs, varying, dynamic_shared):
            kwargs = self._merge_lane_kwargs(static_shared, dynamic_shared, varying)
            return potential_gen(t, err, y=obs, **kwargs)(unravel(flat))

        def repair_hessian(hessian):
            hessian = 0.5 * (hessian + hessian.T)
            hessian = jnp.nan_to_num(
                hessian, nan=0.0, posinf=1.0e12, neginf=-1.0e12
            )
            eigenvalues, eigenvectors = jnp.linalg.eigh(hessian)
            scale = jnp.maximum(jnp.max(jnp.abs(eigenvalues)), 1.0)
            floor = jnp.maximum(1.0e-10, scale * 1.0e-8)
            repaired = jnp.maximum(eigenvalues, floor)
            repaired_hessian = (
                eigenvectors * repaired[None, :]
            ) @ eigenvectors.T
            covariance = (eigenvectors / repaired[None, :]) @ eigenvectors.T
            covariance = 0.5 * (covariance + covariance.T)
            condition = jnp.max(repaired) / jnp.min(repaired)
            return (
                covariance,
                repaired_hessian,
                condition,
                jnp.min(eigenvalues),
            )

        def map_one(flat0, err, obs, varying, t, dynamic_shared):
            def objective(flat):
                return objective_one(
                    flat, t, err, obs, varying, dynamic_shared
                )

            value_and_grad = jax.value_and_grad(objective)
            hessian_fn = jax.hessian(objective)

            def derivatives(flat):
                value, gradient = value_and_grad(flat)
                (
                    covariance,
                    repaired_hessian,
                    condition,
                    minimum_eigenvalue,
                ) = repair_hessian(hessian_fn(flat))
                newton_solution = jnp.linalg.solve(
                    repaired_hessian, gradient
                )
                raw_decrement2 = gradient @ newton_solution
                decrement = jnp.sqrt(jnp.maximum(raw_decrement2, 0.0))
                decrement = jnp.where(
                    jnp.isfinite(decrement), decrement, jnp.inf
                )
                return (
                    value,
                    gradient,
                    covariance,
                    repaired_hessian,
                    decrement,
                    condition,
                    minimum_eigenvalue,
                )

            (
                value0,
                gradient0,
                covariance0,
                repaired_hessian0,
                decrement0,
                condition0,
                minimum_eigenvalue0,
            ) = derivatives(flat0)

            def take_step(state):
                (
                    flat,
                    value,
                    gradient,
                    covariance,
                    repaired_hessian,
                    decrement,
                    _,
                    _,
                    iterations,
                ) = state
                newton = -jnp.linalg.solve(repaired_hessian, gradient)
                newton_norm = jnp.linalg.norm(newton)
                # A trust radius protects repaired nearly-flat directions;
                # backtracking restores the full quadratic step near the MAP.
                newton = newton * jnp.minimum(
                    1.0,
                    options["trust_radius"]
                    / jnp.maximum(newton_norm, 1.0e-12),
                )
                gradient_norm = jnp.linalg.norm(gradient)
                gradient_step = -gradient * (
                    0.1 / jnp.maximum(gradient_norm, 1.0e-12)
                )
                directions = jnp.concatenate(
                    (
                        line_search_scales[:, None] * newton[None, :],
                        line_search_scales[:, None] * gradient_step[None, :],
                    ),
                    axis=0,
                )
                candidates = flat[None, :] + directions
                candidate_values = jax.vmap(objective)(candidates)
                candidate_values = jnp.where(
                    jnp.isfinite(candidate_values), candidate_values, jnp.inf
                )
                best_index = jnp.argmin(candidate_values)
                best_value = candidate_values[best_index]
                accepted = best_value < value
                # Once the Newton decrement is tiny, the predicted objective
                # change can be below the accumulated likelihood's rounding
                # scale even though the Newton equation still improves the
                # first-order residual. Take the finite full Newton step in
                # that local regime instead of stalling on a strict f test.
                near_solution = decrement < 1.0e-2
                full_newton_flat = flat + newton
                full_newton_value = objective(full_newton_flat)
                take_full_newton = near_solution & jnp.isfinite(
                    full_newton_value
                )
                next_flat = jnp.where(
                    take_full_newton,
                    full_newton_flat,
                    jnp.where(accepted, candidates[best_index], flat),
                )
                (
                    next_value,
                    next_gradient,
                    next_covariance,
                    next_repaired_hessian,
                    next_decrement,
                    next_condition,
                    next_minimum_eigenvalue,
                ) = derivatives(next_flat)
                return (
                    next_flat,
                    next_value,
                    next_gradient,
                    next_covariance,
                    next_repaired_hessian,
                    next_decrement,
                    next_condition,
                    next_minimum_eigenvalue,
                    iterations + jnp.asarray(1, dtype=jnp.int32),
                )

            def done(state):
                value = state[1]
                gradient = state[2]
                decrement = state[5]
                return (
                    jnp.isfinite(value)
                    & jnp.all(jnp.isfinite(gradient))
                    & jnp.isfinite(decrement)
                    & (decrement <= options["map_tol"])
                )

            initial_state = (
                    flat0,
                    value0,
                    gradient0,
                    covariance0,
                    repaired_hessian0,
                    decrement0,
                    condition0,
                    minimum_eigenvalue0,
                    jnp.asarray(0, dtype=jnp.int32),
                )
            state = jax.lax.while_loop(
                lambda current: (
                    (current[8] < options["map_maxiter"])
                    & ~done(current)
                ),
                take_step,
                initial_state,
            )
            (
                flat,
                value,
                gradient,
                covariance,
                _,
                decrement,
                condition,
                minimum_eigenvalue,
                iterations,
            ) = state
            gradient_norm = jnp.linalg.norm(gradient)
            converged = (
                jnp.isfinite(value)
                & jnp.isfinite(decrement)
                & (decrement <= options["map_tol"])
            )
            return (
                flat,
                covariance,
                gradient_norm,
                decrement,
                iterations,
                condition,
                minimum_eigenvalue,
                converged,
            )

        map_lanes = jax.vmap(
            map_one,
            in_axes=(0, 0, 0, varying_axes, None, None),
        )
        self._map_program = jax.jit(map_lanes)

        padded_draws = (
            math.ceil(options["n_draws"] / draw_chunk_size)
            * draw_chunk_size
        )
        n_blocks = padded_draws // draw_chunk_size

        def proposal_logpdf(z, mu, scale_chol):
            return _student_mixture_logpdf(
                z,
                mu,
                scale_chol,
                options["student_df"],
                options["wide_fraction"],
                options["wide_scale"],
            )

        def proposal_sample(key, mean, chol, n_draws):
            return _sample_student_mixture(
                key,
                mean,
                chol,
                options["student_df"],
                n_draws,
                options["wide_fraction"],
                options["wide_scale"],
            )

        def target_lanes(flat_lanes, t, errors, observations, varying, dynamic_shared):
            def target_one(flat, err, obs, varying_lane):
                kwargs = self._merge_lane_kwargs(
                    static_shared, dynamic_shared, varying_lane
                )
                return -potential_gen(t, err, y=obs, **kwargs)(unravel(flat))

            return jax.vmap(
                target_one, in_axes=(0, 0, 0, varying_axes)
            )(flat_lanes, errors, observations, varying)

        def evaluate(draws, t, errors, observations, varying, dynamic_shared):
            missing = padded_draws - options["n_draws"]
            if missing:
                draws = jnp.concatenate(
                    (draws, jnp.repeat(draws[-1:], missing, axis=0)), axis=0
                )
            blocks = draws.reshape(
                (n_blocks, draw_chunk_size, self.lane_width, draws.shape[-1])
            )

            def evaluate_block(block):
                return jax.vmap(
                    target_lanes,
                    in_axes=(0, None, None, None, None, None),
                )(block, t, errors, observations, varying, dynamic_shared)

            values = jax.lax.map(evaluate_block, blocks)
            return values.reshape((padded_draws, self.lane_width))[: options["n_draws"]]

        def moment_match(draws, log_target, mean, chol):
            log_q = jax.vmap(
                lambda draw: jax.vmap(
                    proposal_logpdf
                )(draw, mean, chol)
            )(draws)
            log_weights = log_target - log_q
            psis_lw, k_hat = psis_smooth_log_weights(log_weights.T)
            weights = jnp.exp(psis_lw.T)
            updated_mean = jnp.einsum("sl,sld->ld", weights, draws)
            centered = draws - updated_mean[None, ...]
            covariance = jnp.einsum("sl,sli,slj->lij", weights, centered, centered)
            effective_sample_size = 1.0 / jnp.sum(weights**2, axis=0)

            def repair_and_cholesky(cov):
                repaired, _ = _repair_positive_matrix(cov)
                return _student_scale_cholesky(
                    repaired,
                    options["student_df"],
                    _mixture_base_inflation(
                        options["scale_inflation"],
                        options["wide_fraction"],
                        options["wide_scale"],
                    ),
                )

            updated_chol = jax.vmap(repair_and_cholesky)(covariance)
            return (
                updated_mean,
                updated_chol,
                psis_lw.T,
                k_hat,
                effective_sample_size,
                log_q,
            )

        def importance_round(
            key, mean, chol, t, errors, observations, varying, dynamic_shared
        ):
            draws = proposal_sample(key, mean, chol, options["n_draws"])
            target = evaluate(
                draws, t, errors, observations, varying, dynamic_shared
            )
            updated_mean, updated_chol, psis_lw, k_hat, ess, _ = (
                moment_match(draws, target, mean, chol)
            )
            return (
                draws,
                target,
                updated_mean,
                updated_chol,
                psis_lw,
                k_hat,
                ess,
            )

        # Sampling, batched forward evaluation, PSIS, and moment matching are
        # one reusable executable for all adaptive and final rounds.
        self._importance_program = jax.jit(importance_round)

        def systematic_resample(key, draws, psis_lw):
            offsets = jax.random.uniform(
                key, (self.lane_width,), dtype=jnp.float64
            ) / options["num_samples"]
            positions = (
                jnp.arange(options["num_samples"], dtype=jnp.float64)[:, None]
                / options["num_samples"]
                + offsets[None, :]
            )
            cdf = jnp.cumsum(jnp.exp(psis_lw), axis=0)
            cdf = cdf.at[-1].set(jnp.ones((self.lane_width,), dtype=jnp.float64))

            def resample_lane(cdf_lane, lane_draws, lane_positions):
                indices = jnp.searchsorted(cdf_lane, lane_positions, side="right")
                return lane_draws[jnp.minimum(indices, cdf_lane.shape[0] - 1)]

            return jax.vmap(
                resample_lane, in_axes=(1, 1, 1), out_axes=1
            )(cdf, draws, positions)

        self._resample_program = jax.jit(systematic_resample)

        def imh_chain(
            key, draws, log_target_draws, psis_lw, mean, chol,
            t, errors, observations, varying, dynamic_shared,
        ):
            key_init, key_chain = jax.random.split(key)
            init_keys = jax.random.split(key_init, self.lane_width)
            init_indices = jax.vmap(jax.random.categorical)(init_keys, psis_lw.T)
            lane_indices = jnp.arange(self.lane_width)
            initial = draws[init_indices, lane_indices]
            initial_target = log_target_draws[init_indices, lane_indices]
            initial_q = jax.vmap(
                proposal_logpdf
            )(initial, mean, chol)

            def transition(carry, transition_key):
                current, target_current, q_current, accepted_count, run, max_run = carry
                proposal_key, uniform_key = jax.random.split(transition_key)
                proposal = proposal_sample(proposal_key, mean, chol, 1)[0]
                target_proposal = target_lanes(
                    proposal, t, errors, observations, varying, dynamic_shared
                )
                q_proposal = jax.vmap(
                    proposal_logpdf
                )(proposal, mean, chol)
                log_alpha = target_proposal - q_proposal - target_current + q_current
                uniforms = jax.random.uniform(
                    uniform_key, (self.lane_width,), dtype=jnp.float64
                )
                accepted = jnp.log(uniforms) < jnp.minimum(log_alpha, 0.0)
                current = jnp.where(accepted[:, None], proposal, current)
                target_current = jnp.where(accepted, target_proposal, target_current)
                q_current = jnp.where(accepted, q_proposal, q_current)
                run = jnp.where(accepted, 0, run + 1)
                max_run = jnp.maximum(max_run, run)
                accepted_count = accepted_count + accepted.astype(jnp.int32)
                return (
                    current, target_current, q_current, accepted_count, run, max_run
                ), current

            zero = jnp.zeros((self.lane_width,), dtype=jnp.int32)
            state = (initial, initial_target, initial_q, zero, zero, zero)
            warmup_key, sample_key = jax.random.split(key_chain)
            if options["num_warmup"]:
                warmup_keys = jax.random.split(warmup_key, options["num_warmup"])
                state, _ = jax.lax.scan(transition, state, warmup_keys)
            state = (state[0], state[1], state[2], zero, zero, zero)
            sample_transitions = options["num_samples"] * options["imh_thin"]
            sample_keys = jax.random.split(sample_key, sample_transitions)
            state, samples = jax.lax.scan(transition, state, sample_keys)
            acceptance = state[3].astype(jnp.float64) / sample_transitions
            retained = samples[options["imh_thin"] - 1::options["imh_thin"]]
            return retained, acceptance, state[5]

        self._imh_program = jax.jit(imh_chain)

        postprocess_gen = model_info.postprocess_fn

        def postprocess_lane(flat, t, err, obs, varying, dynamic_shared):
            kwargs = self._merge_lane_kwargs(static_shared, dynamic_shared, varying)
            return postprocess_gen(t, err, y=obs, **kwargs)(unravel(flat))

        postprocess_lanes = jax.vmap(
            postprocess_lane,
            in_axes=(0, None, 0, 0, varying_axes, None),
        )
        self._postprocess_program = jax.jit(
            jax.vmap(postprocess_lanes, in_axes=(0, None, None, None, None, None))
        )
        self.program_build_count += 1

    def run_raw(self, key, t, yerr, y, init_params, model_kwargs):
        t = _asarray_f64(t)
        yerr = _asarray_f64(yerr)
        y = _asarray_f64(y)
        if yerr.ndim != 2 or y.ndim != 2 or yerr.shape != y.shape:
            raise ValueError("yerr and y must have identical [channel, time] shapes.")
        num_channels = int(yerr.shape[0])
        if num_channels < 1:
            raise ValueError("At least one channel is required.")
        if self.lane_width < num_channels:
            raise ValueError(
                f"lane_width={self.lane_width} cannot hold {num_channels} channels."
            )

        padded_yerr = _pad_first_axis(yerr, self.lane_width)[:, None, :]
        padded_y = _pad_first_axis(y, self.lane_width)[:, None, :]
        varying, shared = _partition_model_kwargs(
            model_kwargs,
            self.channel_varying_kwargs,
            num_channels,
            self.lane_width,
        )
        dynamic_shared, static_shared = _split_dynamic_and_static_kwargs(shared)
        if self._static_shared_kwargs is not None and not _static_kwargs_equal(
            static_shared, self._static_shared_kwargs
        ):
            raise ValueError(
                "A reused Laplace-IS runner received different non-array/static model arguments."
            )

        enriched = _enrich_initial_values(init_params)
        first_init = _first_lane_initial_values(enriched, num_channels)
        first_varying = {name: value[0] for name, value in varying.items()}
        first_kwargs = self._merge_lane_kwargs(
            static_shared, dynamic_shared, first_varying
        )
        key_model, key_work = jax.random.split(key)
        model_info = initialize_model(
            key_model,
            self.model,
            init_strategy=init_to_value(values=first_init),
            dynamic_args=True,
            model_args=(t, padded_yerr[0]),
            model_kwargs={"y": padded_y[0], **first_kwargs},
        )
        batched_init = _prepare_unconstrained_initial_values(
            enriched, model_info.model_trace, num_channels, self.lane_width
        )
        first_flat, unravel = ravel_pytree(
            jax.tree.map(lambda value: value[0], batched_init)
        )
        del first_flat
        flat_init = jax.vmap(lambda tree: ravel_pytree(tree)[0])(batched_init)
        effective_draw_chunk_size = _effective_draw_chunk_size(
            self.options["draw_chunk_size"],
            self.lane_width,
            int(t.shape[0]),
            self.options["n_draws"],
        )
        if self._map_program is None:
            self._build_programs(
                model_info,
                static_shared,
                unravel,
                effective_draw_chunk_size,
            )
        elif effective_draw_chunk_size != self._effective_draw_chunk_size:
            raise ValueError(
                "A reused Laplace-IS runner received a cadence shape that "
                "requires a different effective draw chunk size."
            )

        map_start = time.perf_counter()
        (
            map_values,
            covariance,
            gradient_norm,
            newton_decrement,
            iterations,
            condition,
            minimum_eigenvalue,
            converged,
        ) = self._map_program(
            flat_init,
            padded_yerr,
            padded_y,
            varying,
            t,
            dynamic_shared,
        )
        covariance.block_until_ready()
        map_seconds = time.perf_counter() - map_start

        chol = jax.vmap(
            lambda cov: _student_scale_cholesky(
                cov,
                self.options["student_df"],
                _mixture_base_inflation(
                    self.options["scale_inflation"],
                    self.options["wide_fraction"],
                    self.options["wide_scale"],
                ),
            )
        )(covariance)
        mean = map_values
        work_keys = jax.random.split(key_work, self.options["rounds"] + 3)
        importance_round_seconds = []
        for round_index in range(self.options["rounds"] + 1):
            round_start = time.perf_counter()
            (
                draws,
                target,
                updated_mean,
                updated_chol,
                psis_lw,
                pareto_k,
                is_ess,
            ) = self._importance_program(
                work_keys[round_index],
                mean,
                chol,
                t,
                padded_yerr,
                padded_y,
                varying,
                dynamic_shared,
            )
            is_ess.block_until_ready()
            importance_round_seconds.append(
                time.perf_counter() - round_start
            )
            if round_index < self.options["rounds"]:
                mean, chol = updated_mean, updated_chol
        adaptation_seconds = sum(importance_round_seconds)

        sampling_start = time.perf_counter()
        output_key = work_keys[self.options["rounds"] + 1]
        if self.options["output"] == "imh":
            flat_samples, imh_acceptance, max_rejection_run = self._imh_program(
                output_key,
                draws,
                target,
                psis_lw,
                mean,
                chol,
                t,
                padded_yerr,
                padded_y,
                varying,
                dynamic_shared,
            )
        else:
            flat_samples = self._resample_program(output_key, draws, psis_lw)
            imh_acceptance = jnp.full(
                (self.lane_width,), jnp.nan, dtype=jnp.float64
            )
            max_rejection_run = jnp.full(
                (self.lane_width,), -1, dtype=jnp.int32
            )
        flat_samples.block_until_ready()
        sampling_seconds = time.perf_counter() - sampling_start

        postprocess_start = time.perf_counter()
        samples = self._postprocess_program(
            flat_samples, t, padded_yerr, padded_y, varying, dynamic_shared
        )
        samples = _squeeze_internal_channel_axis(samples)
        samples = jax.tree.map(lambda value: value[:, :num_channels], samples)
        jax.tree.leaves(samples)[0].block_until_ready()
        postprocess_seconds = time.perf_counter() - postprocess_start

        min_ess = max(
            self.options["min_ess"],
            self.options["min_ess_fraction"] * self.options["n_draws"],
        )
        gate = (
            jnp.isfinite(pareto_k)
            & (pareto_k < self.options["khat_threshold"])
            & jnp.isfinite(is_ess)
            & (is_ess >= min_ess)
        )
        if self.options["output"] == "imh":
            gate = gate & (imh_acceptance >= self.options["min_imh_acceptance"])
        gate = gate & jnp.all(jnp.isfinite(flat_samples), axis=(0, 2))

        def repeated(value):
            return jnp.full((self.lane_width,), value, dtype=jnp.float64)

        repeated_round_seconds = jnp.broadcast_to(
            jnp.asarray(importance_round_seconds, dtype=jnp.float64)[None, :],
            (self.lane_width, len(importance_round_seconds)),
        )

        raw = {
            "converged": converged[:num_channels],
            "pareto_k": pareto_k[:num_channels],
            "is_ess": is_ess[:num_channels],
            "imh_acceptance": imh_acceptance[:num_channels],
            "max_rejection_run": max_rejection_run[:num_channels],
            "map_gradient_norm": gradient_norm[:num_channels],
            "map_newton_decrement": newton_decrement[:num_channels],
            "hessian_condition_number": condition[:num_channels],
            "hessian_min_eigenvalue": minimum_eigenvalue[:num_channels],
            "gate_passed": gate[:num_channels],
            "fell_back": jnp.zeros((num_channels,), dtype=jnp.bool_),
            "map_iterations": iterations[:num_channels],
            "effective_draw_chunk_size": jnp.full(
                (num_channels,),
                effective_draw_chunk_size,
                dtype=jnp.int32,
            ),
            # These are batch-stage wall times, repeated so the diagnostics
            # remain channel aligned; the JSON labels them per channel.
            "map_seconds": repeated(map_seconds)[:num_channels],
            "adaptation_seconds": repeated(adaptation_seconds)[:num_channels],
            "importance_round_seconds": repeated_round_seconds[:num_channels],
            "sampling_seconds": repeated(sampling_seconds)[:num_channels],
            "postprocess_seconds": repeated(postprocess_seconds)[:num_channels],
            "fallback_seconds": repeated(0.0)[:num_channels],
        }
        return samples, raw


def build_laplace_is_runner(
    model,
    *,
    nuts_kwargs=None,
    mcmc_kwargs=None,
    num_warmup=None,
    num_samples=None,
    lane_width,
    channel_varying_kwargs=(),
    laplace_is_output="imh",
    laplace_is_num_draws=4096,
    laplace_is_rounds=2,
    laplace_is_draw_chunk_size=256,
    laplace_is_student_df=3.0,
    laplace_is_scale_inflation=1.5,
    laplace_is_wide_fraction=0.0,
    laplace_is_wide_scale=3.0,
    laplace_is_map_maxiter=200,
    laplace_is_map_tol=1.0e-4,
    laplace_is_trust_radius=5.0,
    laplace_is_khat_threshold=0.7,
    laplace_is_min_ess=400.0,
    laplace_is_min_ess_fraction=0.2,
    laplace_is_min_imh_acceptance=0.2,
    laplace_is_imh_thin=8,
    laplace_is_fallback=True,
):
    """Build a lazily compiled Laplace/IS runner reusable at one lane width."""
    options = _resolve_laplace_options(
        nuts_kwargs=nuts_kwargs,
        mcmc_kwargs=mcmc_kwargs,
        num_warmup=num_warmup,
        num_samples=num_samples,
        laplace_is_output=laplace_is_output,
        laplace_is_num_draws=laplace_is_num_draws,
        laplace_is_rounds=laplace_is_rounds,
        laplace_is_draw_chunk_size=laplace_is_draw_chunk_size,
        laplace_is_student_df=laplace_is_student_df,
        laplace_is_scale_inflation=laplace_is_scale_inflation,
        laplace_is_wide_fraction=laplace_is_wide_fraction,
        laplace_is_wide_scale=laplace_is_wide_scale,
        laplace_is_map_maxiter=laplace_is_map_maxiter,
        laplace_is_map_tol=laplace_is_map_tol,
        laplace_is_trust_radius=laplace_is_trust_radius,
        laplace_is_khat_threshold=laplace_is_khat_threshold,
        laplace_is_min_ess=laplace_is_min_ess,
        laplace_is_min_ess_fraction=laplace_is_min_ess_fraction,
        laplace_is_min_imh_acceptance=laplace_is_min_imh_acceptance,
        laplace_is_imh_thin=laplace_is_imh_thin,
        laplace_is_fallback=laplace_is_fallback,
    )
    return LaplaceISRunner(
        model,
        options=options,
        lane_width=lane_width,
        channel_varying_kwargs=channel_varying_kwargs,
    )


def get_samples_laplace_is(
    model: Callable,
    key: jax.Array,
    t,
    yerr,
    y,
    init_params: Mapping[str, Any],
    nuts_kwargs: Mapping[str, Any] | None = None,
    mcmc_kwargs: Mapping[str, Any] | None = None,
    diagnostics_path: str | None = None,
    lane_width: int | None = None,
    channel_varying_kwargs: tuple[str, ...] = (),
    _runner: LaplaceISRunner | None = None,
    *,
    num_warmup: int | None = None,
    num_samples: int | None = None,
    laplace_is_output: str = "imh",
    laplace_is_num_draws: int = 4096,
    laplace_is_rounds: int = 2,
    laplace_is_draw_chunk_size: int = 256,
    laplace_is_student_df: float = 3.0,
    laplace_is_scale_inflation: float = 1.5,
    laplace_is_wide_fraction: float = 0.0,
    laplace_is_wide_scale: float = 3.0,
    laplace_is_map_maxiter: int = 200,
    laplace_is_map_tol: float = 1.0e-4,
    laplace_is_trust_radius: float = 5.0,
    laplace_is_khat_threshold: float = 0.7,
    laplace_is_min_ess: float = 400.0,
    laplace_is_min_ess_fraction: float = 0.2,
    laplace_is_min_imh_acceptance: float = 0.2,
    laplace_is_imh_thin: int = 8,
    laplace_is_fallback: bool = True,
    return_diagnostics: bool = False,
    **model_kwargs,
):
    """Sample channel-factorized posteriors with adaptive Laplace/IS."""
    yerr_array = _asarray_f64(yerr)
    if not hasattr(yerr_array, "ndim") or yerr_array.ndim != 2:
        raise ValueError("yerr and y must have shape [channel, time].")
    num_channels = int(yerr_array.shape[0])
    resolved_width = num_channels if lane_width is None else int(lane_width)
    options = _resolve_laplace_options(
        nuts_kwargs=nuts_kwargs,
        mcmc_kwargs=mcmc_kwargs,
        num_warmup=num_warmup,
        num_samples=num_samples,
        laplace_is_output=laplace_is_output,
        laplace_is_num_draws=laplace_is_num_draws,
        laplace_is_rounds=laplace_is_rounds,
        laplace_is_draw_chunk_size=laplace_is_draw_chunk_size,
        laplace_is_student_df=laplace_is_student_df,
        laplace_is_scale_inflation=laplace_is_scale_inflation,
        laplace_is_wide_fraction=laplace_is_wide_fraction,
        laplace_is_wide_scale=laplace_is_wide_scale,
        laplace_is_map_maxiter=laplace_is_map_maxiter,
        laplace_is_map_tol=laplace_is_map_tol,
        laplace_is_trust_radius=laplace_is_trust_radius,
        laplace_is_khat_threshold=laplace_is_khat_threshold,
        laplace_is_min_ess=laplace_is_min_ess,
        laplace_is_min_ess_fraction=laplace_is_min_ess_fraction,
        laplace_is_min_imh_acceptance=laplace_is_min_imh_acceptance,
        laplace_is_imh_thin=laplace_is_imh_thin,
        laplace_is_fallback=laplace_is_fallback,
    )
    if resolved_width < num_channels:
        raise ValueError(
            f"lane_width={resolved_width} cannot hold {num_channels} channels."
        )
    if _runner is None:
        runner = LaplaceISRunner(
            model,
            options=options,
            lane_width=resolved_width,
            channel_varying_kwargs=channel_varying_kwargs,
        )
    else:
        runner = _runner
        runner.validate_configuration(
            model,
            options=options,
            lane_width=resolved_width,
            channel_varying_kwargs=channel_varying_kwargs,
        )

    laplace_key, fallback_key = jax.random.split(key)
    samples, raw = runner.run_raw(
        laplace_key, t, yerr_array, y, init_params, model_kwargs
    )
    failed = ~jnp.asarray(raw["gate_passed"])
    failed_indices = jnp.flatnonzero(failed, size=num_channels, fill_value=-1)
    n_failed = int(jax.device_get(jnp.sum(failed)))
    if options["fallback"] and n_failed:
        selected = jax.device_get(failed_indices[:n_failed])
        fallback_started = time.perf_counter()

        def subset_channel_value(value):
            try:
                array = _asarray_f64(value)
            except (TypeError, ValueError):
                return value
            if (
                hasattr(array, "ndim")
                and array.ndim > 0
                and array.shape[0] == num_channels
            ):
                return array[selected]
            return value

        fallback_init = {
            name: subset_channel_value(value) for name, value in init_params.items()
        }
        fallback_kwargs = dict(model_kwargs)
        for name in channel_varying_kwargs:
            fallback_kwargs[name] = subset_channel_value(fallback_kwargs[name])
        fallback_nuts_kwargs = dict(nuts_kwargs or {})
        fallback_nuts_kwargs.update(
            mass_matrix="laplace",
            laplace_hessian_method="finite_difference",
            laplace_fd_relative_step=2.0e-4,
            laplace_map_iterations=200,
            laplace_map_decrement_tolerance=1.0e-4,
            laplace_trust_radius=5.0,
            laplace_warmup=150,
            laplace_target_accept=0.95,
            laplace_max_tree_depth=6,
            laplace_start_at_map=True,
        )
        # Keep the fallback program shape fixed for the parent runner.  The
        # independent-NUTS runner masks padded lanes, so this yields identical
        # active-lane samples while allowing every chunk to reuse one compiled
        # fallback executable even when the number of failed IS lanes varies.
        fallback_width = runner.lane_width
        fallback_runner = runner._fallback_runners.get(fallback_width)
        if fallback_runner is None:
            fallback_runner = build_independent_nuts_runner(
                model,
                nuts_kwargs=fallback_nuts_kwargs,
                mcmc_kwargs=mcmc_kwargs,
                num_warmup=options["num_warmup"],
                num_samples=options["num_samples"],
                lane_width=fallback_width,
                channel_varying_kwargs=channel_varying_kwargs,
            )
            runner._fallback_runners[fallback_width] = fallback_runner
        fallback_samples = get_samples_independent(
            model,
            fallback_key,
            t,
            yerr_array[selected],
            _asarray_f64(y)[selected],
            fallback_init,
            nuts_kwargs=fallback_nuts_kwargs,
            mcmc_kwargs=mcmc_kwargs,
            num_warmup=options["num_warmup"],
            num_samples=options["num_samples"],
            lane_width=fallback_width,
            channel_varying_kwargs=channel_varying_kwargs,
            _runner=fallback_runner,
            **fallback_kwargs,
        )
        jax.tree.leaves(fallback_samples)[0].block_until_ready()
        fallback_seconds = time.perf_counter() - fallback_started
        samples = {
            name: value.at[:, selected].set(fallback_samples[name])
            for name, value in samples.items()
        }
        raw["fell_back"] = raw["fell_back"].at[selected].set(True)
        raw["fallback_seconds"] = jnp.full(
            (num_channels,), fallback_seconds, dtype=jnp.float64
        )

    diagnostics = LaplaceISDiagnostics(**raw)
    _save_diagnostics(
        diagnostics_path,
        diagnostics,
        output=options["output"],
        n_draws=options["n_draws"],
        rounds=options["rounds"],
    )
    if return_diagnostics:
        return samples, diagnostics
    return samples
