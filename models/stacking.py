"""Bayesian stacking utilities for spectroscopic light-curve models.

All public numerical routines consume NumPy arrays and compute in float64.
Pointwise log likelihoods may safely be stored as float32 after evaluation.
"""

from __future__ import annotations

import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from scipy.optimize import minimize
from scipy.special import logsumexp

from models.psis import psis_smooth_log_weights


def _one_draw_loglik(sample, stage_inputs, model):
    conditioned = handlers.substitute(model, data=sample)
    trace = handlers.trace(handlers.seed(conditioned, jax.random.PRNGKey(0))).get_trace(
        stage_inputs.t,
        stage_inputs.yerr,
        y=stage_inputs.y,
        **dict(stage_inputs.model_kwargs),
    )
    observed = [site for site in trace.values()
                if site.get("type") == "sample" and site.get("is_observed", False)]
    if len(observed) != 1:
        raise ValueError(f"Expected exactly one observed likelihood site; found {len(observed)}.")
    site = observed[0]
    value = site["value"]
    logp = site["fn"].log_prob(value)
    # A likelihood_mask is represented by a NumPyro mask handler.
    mask = site.get("mask")
    if mask is not None:
        logp = jnp.where(mask, logp, 0.0)
    return logp


def pointwise_loglik(samples, stage_inputs, model=None):
    """Re-evaluate Normal log densities as ``[draw, channel, time]``.

    ``samples`` is the combined checkpoint posterior mapping. The exact model
    reconstructed by :func:`tools.spectro_stage_inputs.load_stage_inputs` is
    used unless an explicit compatible model is supplied.
    """
    model = stage_inputs.model if model is None else model
    arrays = {name: jnp.asarray(value) for name, value in samples.items()}
    if not arrays:
        raise ValueError("samples must contain at least one posterior site.")
    draws = {value.shape[0] for value in arrays.values()}
    if len(draws) != 1:
        raise ValueError("All posterior arrays must have the same draw dimension.")
    result = jax.vmap(lambda draw: _one_draw_loglik(draw, stage_inputs, model))(arrays)
    result = np.asarray(jax.device_get(result), dtype=np.float64)
    expected = (next(iter(draws)), stage_inputs.num_channels, stage_inputs.num_cadences)
    if result.shape != expected:
        raise ValueError(f"Pointwise likelihood has shape {result.shape}, expected {expected}.")
    return result


def psis_loo(loglik):
    """Return channel-wise point ELPDs, sums, and Pareto k diagnostics."""
    values = np.asarray(loglik, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("loglik must have shape [draw, channel, time].")
    raw = np.moveaxis(-values, 0, -1)  # channel, time, draw
    smooth, khat = psis_smooth_log_weights(jnp.asarray(raw))
    smooth = np.asarray(jax.device_get(smooth), dtype=np.float64)
    elpd_i = logsumexp(np.moveaxis(values, 0, -1) + smooth, axis=-1)
    return elpd_i, np.sum(elpd_i, axis=-1), np.asarray(jax.device_get(khat))


def stacking_weights(elpd_matrix):
    """Maximize the summed log score of a convex predictive mixture."""
    elpd = np.asarray(elpd_matrix, dtype=np.float64)
    if elpd.ndim != 2 or elpd.shape[0] < 1:
        raise ValueError("elpd_matrix must have shape [model, point].")
    n_models = elpd.shape[0]
    offset = np.max(elpd, axis=0)

    def objective(weights):
        return -np.sum(offset + np.log(np.sum(weights[:, None] * np.exp(elpd - offset), axis=0)))

    def gradient(weights):
        scaled = np.exp(elpd - offset)
        denominator = np.sum(weights[:, None] * scaled, axis=0)
        return -np.sum(scaled / denominator, axis=1)

    result = minimize(objective, np.full(n_models, 1.0 / n_models), jac=gradient,
                      method="SLSQP", bounds=[(0.0, 1.0)] * n_models,
                      constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
                                   "jac": lambda w: np.ones_like(w)},
                      options={"ftol": 1e-12, "maxiter": 2000})
    if not result.success:
        raise RuntimeError(f"Stacking optimization failed: {result.message}")
    weights = np.clip(result.x, 0.0, 1.0)
    return weights / weights.sum()


def pseudo_bma_plus_weights(elpd_matrix, n_boot=1000, rng=None):
    """Pseudo-BMA+ weights using a Bayesian bootstrap over LOO points."""
    elpd = np.asarray(elpd_matrix, dtype=np.float64)
    if elpd.ndim != 2:
        raise ValueError("elpd_matrix must have shape [model, point].")
    if n_boot < 1:
        raise ValueError("n_boot must be positive.")
    rng = np.random.default_rng(rng)
    point_weights = rng.dirichlet(np.ones(elpd.shape[1]), size=int(n_boot))
    scores = point_weights @ elpd.T * elpd.shape[1]
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=1, keepdims=True)
    result = weights.mean(axis=0)
    return result / result.sum()


def laplace_log_evidence(map, hessian, logp):
    """Laplace log evidence from a MAP point, negative-log-posterior Hessian and log joint."""
    theta = np.asarray(map, dtype=np.float64).reshape(-1)
    hessian = np.asarray(hessian, dtype=np.float64)
    if hessian.shape != (theta.size, theta.size):
        raise ValueError("hessian shape must match the flattened MAP dimension.")
    sign, logdet = np.linalg.slogdet(hessian)
    if sign <= 0:
        raise ValueError("Laplace Hessian must be positive definite.")
    joint = float(logp(theta) if callable(logp) else logp)
    return joint + 0.5 * theta.size * np.log(2.0 * np.pi) - 0.5 * logdet


def stack_posteriors(depth_samples_per_model, weights, n_out=10000, rng=None):
    """Draw channel-wise posterior mixtures and summarize disagreement."""
    samples = [np.asarray(x, dtype=np.float64) for x in depth_samples_per_model]
    if not samples or any(x.ndim != 2 for x in samples):
        raise ValueError("Each depth posterior must have shape [draw, channel].")
    channels = samples[0].shape[1]
    if any(x.shape[1] != channels for x in samples):
        raise ValueError("All models must contain the same channels.")
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = np.broadcast_to(weights[:, None], (len(samples), channels))
    if weights.shape != (len(samples), channels):
        raise ValueError("weights must have shape [model] or [model, channel].")
    weights = weights / weights.sum(axis=0, keepdims=True)
    rng = np.random.default_rng(rng)
    mixture = np.empty((int(n_out), channels), dtype=np.float64)
    for channel in range(channels):
        choices = rng.choice(len(samples), size=int(n_out), p=weights[:, channel])
        for model_index, model_samples in enumerate(samples):
            selected = np.flatnonzero(choices == model_index)
            if selected.size:
                mixture[selected, channel] = model_samples[
                    rng.integers(model_samples.shape[0], size=selected.size), channel]
    q16, median, q84 = np.percentile(mixture, [16.0, 50.0, 84.0], axis=0)
    single_sigma = np.asarray([(np.percentile(x, 84, axis=0) - np.percentile(x, 16, axis=0)) / 2
                               for x in samples])
    stacked_sigma = (q84 - q16) / 2
    disagreement = stacked_sigma / np.maximum(np.min(single_sigma, axis=0), np.finfo(float).tiny)
    return mixture, {"median": median, "q16": q16, "q84": q84,
                     "depth_err_lo": median - q16, "depth_err_hi": q84 - median,
                     "disagreement": disagreement}


def achromatic_offsets(depth_samples_per_model):
    """Estimate each model's inverse-variance achromatic depth offset.

    The reference in each channel is the unweighted across-model mean median.
    Returned uncertainties are the standard errors of the weighted means.
    """
    samples = np.asarray([np.asarray(x, dtype=np.float64)
                          for x in depth_samples_per_model])
    if samples.ndim != 3:
        raise ValueError("Depth samples must have shape [model, draw, channel].")
    medians = np.median(samples, axis=1)
    variances = np.var(samples, axis=1, ddof=1)
    reference = np.mean(medians, axis=0)
    inverse = 1.0 / np.maximum(variances, np.finfo(float).tiny)
    offsets = np.sum(inverse * (medians - reference), axis=1) / np.sum(inverse, axis=1)
    uncertainties = np.sqrt(1.0 / np.sum(inverse, axis=1))
    return offsets, uncertainties, reference


def aligned_stack_posteriors(depth_samples_per_model, weights, n_out=10000, rng=None):
    """Return shape-only and absolute mixtures plus achromatic offsets.

    Model draws are shifted by their fitted offset. A single weighted-average
    offset is then restored, retaining the model-average absolute level while
    excluding between-model gray shifts from the headline uncertainty.
    """
    offsets, offset_uncertainties, reference = achromatic_offsets(
        depth_samples_per_model
    )
    weights_array = np.asarray(weights, dtype=np.float64)
    model_weights = (weights_array if weights_array.ndim == 1
                     else np.mean(weights_array, axis=1))
    model_weights = model_weights / model_weights.sum()
    average_offset = float(np.sum(model_weights * offsets))
    aligned = [np.asarray(values, dtype=np.float64) - offsets[index] + average_offset
               for index, values in enumerate(depth_samples_per_model)]
    aligned_mixture, aligned_summary = stack_posteriors(
        aligned, weights, n_out=n_out, rng=rng
    )
    absolute_mixture, absolute_summary = stack_posteriors(
        depth_samples_per_model, weights, n_out=n_out,
        rng=None if rng is None else int(rng) + 1
    )
    return {
        "aligned_samples": aligned,
        "aligned_mixture": aligned_mixture,
        "aligned_summary": aligned_summary,
        "absolute_mixture": absolute_mixture,
        "absolute_summary": absolute_summary,
        "offsets": offsets,
        "offset_uncertainties": offset_uncertainties,
        "reference": reference,
        "average_offset": average_offset,
    }


def write_stacked_spectrum(path, wavelength, summary, model_names,
                           depth_samples_per_model, weights,
                           absolute_summary=None, offsets=None,
                           offset_uncertainties=None):
    """Write a self-describing stacked transmission-spectrum CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavelength = np.asarray(wavelength)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = np.broadcast_to(weights[:, None], (len(model_names), wavelength.size))
    model_stats = []
    for values in depth_samples_per_model:
        q16, med, q84 = np.percentile(np.asarray(values), [16, 50, 84], axis=0)
        model_stats.append((med, med - q16, q84 - med))
    fields = ["wavelength", "depth", "depth_err_lo", "depth_err_hi"]
    if absolute_summary is not None:
        fields += ["absolute_depth", "absolute_depth_err_lo", "absolute_depth_err_hi",
                   "absolute_disagreement"]
    for name in model_names:
        fields += [f"{name}_depth", f"{name}_depth_err_lo", f"{name}_depth_err_hi",
                   f"{name}_weight", f"{name}_achromatic_offset",
                   f"{name}_achromatic_offset_err"]
    fields.append("disagreement")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for channel, wave in enumerate(wavelength):
            row = {"wavelength": wave, "depth": summary["median"][channel],
                   "depth_err_lo": summary["depth_err_lo"][channel],
                   "depth_err_hi": summary["depth_err_hi"][channel]}
            if absolute_summary is not None:
                row.update({"absolute_depth": absolute_summary["median"][channel],
                            "absolute_depth_err_lo": absolute_summary["depth_err_lo"][channel],
                            "absolute_depth_err_hi": absolute_summary["depth_err_hi"][channel],
                            "absolute_disagreement": absolute_summary["disagreement"][channel]})
            for m, name in enumerate(model_names):
                med, lo, hi = model_stats[m]
                row.update({f"{name}_depth": med[channel], f"{name}_depth_err_lo": lo[channel],
                            f"{name}_depth_err_hi": hi[channel], f"{name}_weight": weights[m, channel]})
                row[f"{name}_achromatic_offset"] = "" if offsets is None else offsets[m]
                row[f"{name}_achromatic_offset_err"] = ("" if offset_uncertainties is None
                                                         else offset_uncertainties[m])
            row["disagreement"] = summary["disagreement"][channel]
            writer.writerow(row)
    return path


# Deliverable spelling retained for callers that prefer a noun-like writer.
stacked_spectrum = write_stacked_spectrum
