"""Predictive stacking, partially pooled MAP, and Bayesian hierarchical weights."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax


def _scores(elpd):
    values = np.asarray(elpd, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 1 or not np.all(np.isfinite(values)):
        raise ValueError("Scores must be finite [model, channel, point].")
    return values


def stacking_weights(elpd):
    """Simplex optimum with predictive densities computed once, not per iteration."""
    values = np.asarray(elpd, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1 or not np.all(np.isfinite(values)):
        raise ValueError("Scores must be finite [model, point].")
    models, points = values.shape
    if models == 1:
        return np.ones(1)
    scaled = np.exp(values - values.max(axis=0))
    def objective(weights):
        mixture = np.maximum(weights @ scaled, np.finfo(float).tiny)
        return -np.log(mixture).mean(), -(scaled / mixture).mean(axis=1)
    result = minimize(objective, np.full(models, 1/models), jac=True, method="SLSQP",
        bounds=[(0, 1)] * models,
        constraints=dict(type="eq", fun=lambda w: w.sum()-1,
                         jac=lambda w: np.ones_like(w)),
        options=dict(ftol=1e-12, maxiter=2000))
    if not result.success:
        raise RuntimeError(f"Stacking failed: {result.message}")
    weights = np.clip(result.x, 0, 1)
    return weights / weights.sum()


def hierarchical_map_weights(elpd, strength=1.0):
    """Fast exchangeable logistic MAP with fixed pooling precision.

    Maximize summed log predictive density minus strength/2 times squared
    deviations of centered channel logits from their across-channel mean.
    This is a fixed-scale MAP approximation, not sampled hierarchical Bayes.
    There is no smoothing in wavelength and no post-hoc weight convolution.
    """
    values = _scores(elpd)
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("Pooling strength must be finite and nonnegative.")
    models, channels, points = values.shape
    if models == 1:
        return np.ones((1, channels))
    if strength == 0 or channels == 1:
        return np.column_stack([stacking_weights(values[:, c]) for c in range(channels)])
    scaled = np.exp(values - values.max(axis=0, keepdims=True))
    initial = np.maximum(stacking_weights(values.reshape(models, -1)), 1e-5)
    start = np.tile(np.log(initial[1:]/initial[0]), (channels, 1)).T
    def objective(theta):
        logits = np.vstack((np.zeros(channels), theta.reshape(models-1, channels)))
        weights = softmax(logits, axis=0)
        mixture = np.maximum(np.sum(weights[:, :, None]*scaled, axis=0), np.finfo(float).tiny)
        centered = logits - logits.mean(axis=0)
        deviations = centered - centered.mean(axis=1, keepdims=True)
        loss = -np.log(mixture).sum() + .5*strength*np.sum(deviations**2)
        gradient = points*weights - weights*np.sum(scaled/mixture[None], axis=2) + strength*deviations
        return loss / (channels*points), gradient[1:].ravel() / (channels*points)
    result = minimize(objective, start.ravel(), jac=True, method="L-BFGS-B",
                      bounds=[(-25, 25)] * start.size,
                      options=dict(ftol=1e-13, gtol=1e-8, maxiter=2000, maxls=40))
    if not result.success:
        raise RuntimeError(f"Hierarchical MAP failed: {result.message}")
    return softmax(np.vstack((np.zeros(channels), result.x.reshape(models-1, channels))), axis=0)


def hierarchical_weights(elpd, *, seed=0, warmup=500, draws=1000, chains=4,
                         min_ess=100, max_rhat=1.05, target_accept=.95):
    """Sample exchangeable logistic weights and enforce convergence diagnostics.

    Relative to model 0: logits = mu + tau*eta; mu~Normal(0,1),
    tau~HalfNormal(1), eta~Normal(0,1). The score is the log predictive mixture.
    Posterior weight samples are retained, including their cross-channel
    dependence. Model order and reference model are part of this prior.
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    from models.mcmc_diagnostics import rank_diagnostics
    values = _scores(elpd)
    models, channels, _ = values.shape
    if chains < 2 or draws < 4 or warmup < 1:
        raise ValueError("Hierarchical convergence gates require >=2 chains, >=4 draws and warmup.")
    if not np.isfinite(target_accept) or not 0 < target_accept < 1:
        raise ValueError('Hierarchical target_accept must be between zero and one.')
    if models == 1:
        return np.ones((1, channels)), np.ones((chains*draws, 1, channels)), dict(
            divergences=0, max_rhat=1., min_ess=chains*draws)
    values = values - values.max(axis=0, keepdims=True)
    def model(scores):
        mu = numpyro.sample("mu", dist.Normal(0, 1).expand([models-1]).to_event(1))
        tau = numpyro.sample("tau", dist.HalfNormal(1).expand([models-1]).to_event(1))
        eta = numpyro.sample("eta", dist.Normal(0, 1).expand([models-1, channels]).to_event(2))
        logits = jnp.vstack((jnp.zeros(channels), mu[:, None]+tau[:, None]*eta))
        log_weights = jax.nn.log_softmax(logits, axis=0)
        numpyro.deterministic("weights", jnp.exp(log_weights))
        numpyro.factor("predictive_score", jax.nn.logsumexp(log_weights[:, :, None]+scores, axis=0).sum())
    sampler = MCMC(NUTS(model, target_accept_prob=target_accept), num_warmup=warmup,
                   num_samples=draws, num_chains=chains, chain_method="vectorized", progress_bar=False)
    sampler.run(jax.random.PRNGKey(seed), jnp.asarray(values))
    samples = sampler.get_samples(group_by_chain=True)
    diagnostics = rank_diagnostics({key: samples[key] for key in ("mu", "tau", "eta", "weights")})
    info = dict(divergences=int(np.asarray(sampler.get_extra_fields()["diverging"]).sum()),
                max_rhat=float(max(np.max(s) for s in diagnostics['rhat'].values())),
                min_ess=float(min(np.min(s) for s in diagnostics['bulk_ess'].values())),
                min_tail_ess=float(min(np.min(s) for s in diagnostics['tail_ess'].values())),
                diagnostics='Rank-normalized split R-hat and bulk/tail ESS, including latent channel logits.')
    if (info["divergences"] or not np.isfinite(info["max_rhat"]) or info["max_rhat"] > max_rhat
            or not np.isfinite(info["min_ess"]) or info["min_ess"] < min_ess):
        raise RuntimeError(f"Hierarchical weight convergence gate failed: {info}")
    if not np.isfinite(info['min_tail_ess']) or info['min_tail_ess'] < min_ess:
        raise RuntimeError(f"Hierarchical weight convergence gate failed: {info}")
    weight_draws = np.asarray(samples["weights"]).reshape(-1, models, channels)
    return weight_draws.mean(axis=0), weight_draws, info


def model_weights(elpd, *, method="stacking", pooling="channel", seed=0,
                  strength=1., n_boot=1000, **hierarchical_options):
    values = _scores(elpd)
    if method == "hierarchical":
        if pooling != "channel":
            raise ValueError("Hierarchical weights require channel pooling.")
        return hierarchical_weights(values, seed=seed, **hierarchical_options)
    if method == "hierarchical_map":
        if pooling != "channel":
            raise ValueError("Hierarchical MAP requires channel pooling.")
        return hierarchical_map_weights(values, strength), None, {"pooling_precision": strength}
    if method not in {"stacking", "pseudo_bma_plus", "equal"} or pooling not in {"channel", "global"}:
        raise ValueError("Unsupported weight method or pooling.")
    def fit(scores, offset):
        if method == "stacking":
            return stacking_weights(scores)
        if method == "pseudo_bma_plus":
            return pseudo_bma_plus_weights(scores, n_boot=n_boot, rng=seed+offset)
        return np.full(scores.shape[0], 1/scores.shape[0])
    if pooling == "global":
        return fit(values.reshape(values.shape[0], -1), 0), None, {}
    return np.column_stack([fit(values[:, c], c) for c in range(values.shape[1])]), None, {}


def pseudo_bma_plus_weights(elpd, n_boot=1000, rng=None, point_batch_size=1024):
    """Exact Bayesian bootstrap distribution with bounded memory.

    Dirichlet(1,...,1) weights are independent Exp(1) draws divided by their
    sum. Accumulate scores in point blocks rather than allocating an entire
    [bootstrap, channel*time] array, which can exceed device/host memory.
    """
    values = np.asarray(elpd, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1 or not np.all(np.isfinite(values)):
        raise ValueError("Scores must be finite [model, point].")
    if n_boot < 1 or point_batch_size < 1:
        raise ValueError("Bootstrap and point batch counts must be positive.")
    rng = np.random.default_rng(rng)
    models, points = values.shape
    # A per-point constant cancels in every bootstrap's softmax.
    centered = values-values.max(axis=0)
    scores = np.zeros((n_boot, models))
    total = np.zeros(n_boot)
    for start in range(0, points, point_batch_size):
        stop = min(points, start+point_batch_size)
        raw = rng.exponential(size=(n_boot, stop-start))
        scores += raw @ centered[:, start:stop].T
        total += raw.sum(axis=1)
    return softmax(scores/total[:, None]*points, axis=1).mean(axis=0)


def mix_depths(depths, weights, *, n_out=10000, seed=0):
    """Global weights retain whole spectrum draws; channel weights give marginals.

    For global weights, both the candidate identity and posterior draw index
    are shared across the entire output row. With channel weights, this is a
    set of marginal draws and must not be interpreted as a joint spectrum.
    """
    arrays = [np.asarray(x, dtype=np.float64) for x in depths]
    if (not arrays or any(x.ndim != 2 or min(x.shape) < 1 or not np.all(np.isfinite(x)) for x in arrays)
            or len({x.shape[1] for x in arrays}) != 1):
        raise ValueError("Depths must be finite nonempty [draw, channel] arrays.")
    weights = np.asarray(weights, dtype=np.float64)
    channels, models = arrays[0].shape[1], len(arrays)
    if weights.shape not in {(models,), (models, channels)} or np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise ValueError("Invalid model weights.")
    if np.any(weights.sum(axis=0) <= 0) or n_out < 1:
        raise ValueError("Positive total weights and n_out required.")
    weights = weights / weights.sum(axis=0)
    rng = np.random.default_rng(seed)
    mixture = np.empty((n_out, channels))
    if weights.ndim == 1:
        choices = rng.choice(models, size=n_out, p=weights)
        for m, values in enumerate(arrays):
            indices = np.flatnonzero(choices == m)
            mixture[indices] = values[rng.integers(len(values), size=len(indices))]
    else:
        for channel in range(channels):
            choices = rng.choice(models, size=n_out, p=weights[:, channel])
            for m, values in enumerate(arrays):
                indices = np.flatnonzero(choices == m)
                mixture[indices, channel] = values[rng.integers(len(values), size=len(indices)), channel]
    q16, median, q84 = np.percentile(mixture, [16, 50, 84], axis=0)
    widths = [(np.percentile(x, 84, axis=0)-np.percentile(x, 16, axis=0))/2 for x in arrays]
    return mixture, dict(median=median, q16=q16, q84=q84, depth_err_lo=median-q16,
                        depth_err_hi=q84-median,
                        disagreement=(q84-q16)/2/np.maximum(np.min(widths, axis=0), np.finfo(float).tiny))
