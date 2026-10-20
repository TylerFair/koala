"""GPU-batched, independently adapted NUTS for factorized light curves.

The existing spectroscopic models vectorize the forward model over channels,
but a normal NumPyro ``MCMC`` call still treats the whole block as one
high-dimensional NUTS state.  This module instead runs one low-dimensional
NUTS state per channel and vmaps those states over a fixed-width GPU batch.
On CPU with several host devices the batch is split into one block of lanes
per device, so every core advances its own channels.

This backend is only valid when all latent sample
sites factorize by channel; shared *fixed* model arguments are fine, but shared
latent variables are not.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.scipy.linalg import solve_triangular
from numpyro.distributions.transforms import biject_to
from numpyro.infer.hmc import hmc
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model


def _segment_length(total, target):
    """Largest divisor of ``total`` no greater than ``target``.

    Every segment then has the same static length, so the segment program is
    compiled once.  A non-positive ``target`` (or a prime ``total``) yields a
    single segment.
    """
    total = int(total)
    target = int(target)
    if total <= 0:
        return 0
    if target <= 0 or target >= total:
        return total
    for length in range(target, 0, -1):
        if total % length == 0:
            return length
    return total


_LANES = PartitionSpec('lanes')
_DRAW_LANES = PartitionSpec(None, 'lanes')
_REPLICATED = PartitionSpec()
# (init, errors, observations, varying, keys, t, shared) and
# (states, t, errors, observations, varying, shared) program arguments.
_INIT_SPECS = (_LANES,) * 5 + (_REPLICATED,) * 2
_STEP_SPECS = (_LANES, _REPLICATED, _LANES, _LANES, _LANES, _REPLICATED)


def _lane_device_layout(lane_width, requested=None):
    """Host-device mesh and padded lane count for one runner width.

    JAX exposes one CPU device unless ``--xla_force_host_platform_device_count``
    is set, and a vmap on that device keeps only a few cores busy.  With
    several host devices the lane axis is split into equal contiguous blocks,
    each advanced by its own device with its own loops.  ``requested=None``
    shards only when the default backend is CPU, so accelerators keep the
    single-device programs; an integer selects that many CPU devices and 1
    restores the single-device programs.  Returns ``(None, lane_width)`` when
    nothing is sharded.
    """
    if requested is None:
        devices = jax.devices() if jax.default_backend() == 'cpu' else []
    else:
        requested = int(requested)
        if requested < 1:
            raise ValueError('lane_devices must be >= 1.')
        devices = jax.devices('cpu')[:requested]
        if len(devices) < requested:
            raise ValueError(
                f'lane_devices={requested} requested, but JAX exposes {len(devices)} CPU '
                'devices; set XLA_FLAGS=--xla_force_host_platform_device_count before '
                'JAX starts.'
            )
    lane_width = int(lane_width)
    if len(devices) < 2 or lane_width < 2:
        return None, lane_width
    # Use the fewest devices that hold the same per-device load, so 40 lanes
    # on 24 devices become 20 blocks of two rather than 24 padded blocks.
    per_device = math.ceil(lane_width / len(devices))
    count = math.ceil(lane_width / per_device)
    if count < 2:
        return None, lane_width
    return Mesh(np.array(devices[:count]), ('lanes',)), per_device * count


def _shard_lanes(function, mesh, in_specs, out_specs):
    """Jit ``function`` once per device on its block of lanes."""
    if hasattr(jax, 'shard_map'):
        mapped = jax.shard_map(function, mesh=mesh, in_specs=in_specs,
                               out_specs=out_specs, check_vma=False)
    else:  # pragma: no cover - JAX releases before jax.shard_map
        from jax.experimental.shard_map import shard_map
        mapped = shard_map(function, mesh=mesh, in_specs=in_specs,
                           out_specs=out_specs, check_rep=False)
    return jax.jit(mapped)


class _SegmentProgress:
    """A tqdm bar advanced once per compiled segment; silent when disabled."""

    def __init__(self, total, desc, enabled):
        self._bar = None
        if enabled and total > 0:
            try:
                from tqdm.auto import tqdm
            except ImportError:  # pragma: no cover - tqdm is a dependency
                return
            self._bar = tqdm(total=int(total), desc=desc, leave=True,
                             dynamic_ncols=True, mininterval=0.5)

    def update(self, n):
        if self._bar is not None:
            self._bar.update(int(n))

    def close(self):
        if self._bar is not None:
            self._bar.close()


@dataclass(frozen=True)
class IndependentNUTSDiagnostics:
    """Per-draw and final adapted diagnostics for every real channel."""

    num_steps: jax.Array
    accept_prob: jax.Array
    diverging: jax.Array
    step_size: jax.Array
    mean_accept_prob: jax.Array
    num_chains: int = 1
    step_size_by_chain: jax.Array | None = None
    mean_accept_prob_by_chain: jax.Array | None = None
    map_gradient_norm: jax.Array | None = None
    map_newton_decrement: jax.Array | None = None
    map_iterations: jax.Array | None = None
    map_condition_number: jax.Array | None = None
    map_hessian_min_eigenvalue: jax.Array | None = None
    map_hessian_relative_error: jax.Array | None = None


@dataclass(frozen=True)
class LaplaceMetricPreparation:
    """A polished unconstrained start and repaired dense covariance."""

    unconstrained_map: Any
    inverse_mass_matrix: jax.Array
    gradient_norm: jax.Array
    newton_decrement: jax.Array
    iterations: jax.Array
    condition_number: jax.Array
    hessian_min_eigenvalue: jax.Array
    potential_fn: Any = None
    postprocess_fn: Any = None


def dispersed_laplace_starts(preparation, rng_key, num_chains):
    """Independent local starts, backtracked against the actual posterior.

    A covariance describes local curvature. On a curved posterior, a draw
    from its Gaussian approximation can have extremely low density. Shrink
    such offsets along their original direction before independent warmups;
    this changes initialization, not the target or retained draws.
    """
    flat, unravel = ravel_pytree(preparation.unconstrained_map)
    values, vectors = jnp.linalg.eigh(preparation.inverse_mass_matrix)
    noise = jax.random.normal(rng_key, (num_chains, flat.size))
    offsets = .5 * (noise * jnp.sqrt(jnp.maximum(values, 0.))) @ vectors.T
    offsets /= jnp.maximum(1., jnp.max(jnp.abs(offsets), axis=1, keepdims=True))
    potential = lambda point: preparation.potential_fn(unravel(point))
    reference = potential(flat)
    limit = max(10, flat.size)

    def choose(offset):
        initial = potential(flat + offset)
        def condition(state):
            scale, value, count = state
            return ((~jnp.isfinite(value)) | (value > reference + limit)) & (count < 12)
        def shrink(state):
            scale, _, count = state
            scale *= .5
            return scale, potential(flat + scale * offset), count + 1
        scale, value, count = jax.lax.while_loop(
            condition, shrink, (jnp.array(1.), initial, jnp.array(0)))
        acceptable = jnp.isfinite(value) & (value <= reference + limit)
        # The MAP remains a valid final initialization for a pathological
        # Gaussian offset; independent keys and adaptation are retained.
        point = flat + jnp.where(acceptable, scale, 0.) * offset
        return point, initial - reference, jnp.where(acceptable, value-reference, 0.), count

    points, initial, accepted, counts = jax.jit(lambda offsets:jax.lax.map(choose, offsets))(offsets)
    return jax.vmap(unravel)(points), {
        'chain_initial_delta_potential': initial,
        'chain_start_delta_potential': accepted,
        'chain_start_backtracks': counts,
    }


def _asarray_f64(value):
    if value is None:
        return None
    if isinstance(value, (str, bytes, bool)):
        return value
    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError):
        return value
    if jnp.issubdtype(array.dtype, jnp.inexact):
        return array.astype(jnp.float64)
    return array


def _synchronize_cadence_init_values(init_values, model_args):
    """Keep normalized step-width starts consistent with physical repairs.

    The white-light optimizer returns both deterministic physical sites and
    latent sites.  Its support-edge guard repairs ``log_width``; copy that
    repaired starting point into whichever cadence latent is present before
    ``init_to_value`` sees it.  This is an initialization-only coordinate
    conversion and is inactive for the default physical model.
    """
    if not (
        isinstance(init_values, Mapping)
        and "log_width" in init_values
        and (
            "log_width_cadences" in init_values
            or "log_width_unit_logit" in init_values
        )
        and model_args
    ):
        return init_values
    time = jnp.asarray(model_args[0])
    cadence = jnp.median(jnp.diff(jnp.sort(time)))
    synchronized = dict(init_values)
    log_width_cadences = jnp.asarray(init_values["log_width"]) - jnp.log(cadence)
    if "log_width_cadences" in init_values:
        synchronized["log_width_cadences"] = log_width_cadences
    if "log_width_unit_logit" in init_values:
        low = jnp.log(0.5)
        high = jnp.log((30.0 / (24.0 * 60.0)) / cadence)
        fraction = (log_width_cadences - low) / (high - low)
        fraction = jnp.clip(fraction, 1.0e-12, 1.0 - 1.0e-12)
        synchronized["log_width_unit_logit"] = jnp.log(fraction) - jnp.log1p(-fraction)
    return synchronized



def _repair_laplace_hessian(hessian, eigenvalue_floor, scaling="raw"):
    """Regularize a proposal metric, never the posterior potential.

    Correlation scaling makes the floor relative to dimensionless curvature,
    rather than comparing coordinates measured in unrelated units. The raw
    branch intentionally retains the existing operations and default.
    """
    hessian = 0.5 * (hessian + hessian.T)
    hessian = jnp.nan_to_num(hessian, nan=0.0, posinf=1e12, neginf=-1e12)
    if scaling == "raw":
        eigenvalues, eigenvectors = jnp.linalg.eigh(hessian)
        scale = jnp.maximum(jnp.max(jnp.abs(eigenvalues)), 1.0)
        repaired = jnp.maximum(eigenvalues, eigenvalue_floor * scale)
        covariance = (eigenvectors / repaired[None, :]) @ eigenvectors.T
        return covariance, jnp.max(repaired) / jnp.min(repaired), jnp.min(eigenvalues)
    if scaling != "correlation":
        raise ValueError("laplace_metric_scaling must be 'raw' or 'correlation'.")
    diagonal = jnp.abs(jnp.diag(hessian))
    units = jnp.sqrt(jnp.where(diagonal > 0.0, diagonal, 1.0))
    correlation = hessian / units[:, None] / units[None, :]
    values, vectors = jnp.linalg.eigh(correlation)
    scale = jnp.maximum(jnp.max(jnp.abs(values)), 1.0)
    repaired = jnp.maximum(values, eigenvalue_floor * scale)
    covariance = ((vectors / repaired[None, :]) @ vectors.T)
    covariance = covariance / units[:, None] / units[None, :]
    return covariance, jnp.max(repaired) / jnp.min(repaired), jnp.min(jnp.linalg.eigvalsh(hessian))


def _laplace_whitening_factor(covariance):
    """Stable square root even when coordinate variances have different units."""
    units = jnp.sqrt(jnp.diag(covariance))
    correlation = covariance / units[:, None] / units[None, :]
    return units[:, None] * jnp.linalg.cholesky(correlation)


def _affine_laplace_potential(potential, unravel, center, factor):
    """Exact pushforward from flat whitened coordinates to unconstrained sites.

    MAP and factor are fixed for a chain's entire run and continuation. NumPyro
    already includes the support-transform Jacobians; this additional affine
    Jacobian is constant but retained explicitly in the log density.
    """
    _, log_absolute_determinant = jnp.linalg.slogdet(factor)
    def transformed(white):
        return potential(unravel(center + factor @ white)) - log_absolute_determinant
    return transformed


def _stage_potential_generator(potential_generator, static_kwargs):
    """Share the exact lane target graph across MAP derivatives and HMC.

    Non-array model controls stay in the closure. Observations, uncertainty,
    channel arguments and unconstrained positions remain dynamic JAX inputs.
    """
    static_kwargs = dict(static_kwargs)

    @jax.jit
    def staged(position, t, err, dynamic_kwargs):
        kwargs = {**static_kwargs, **dynamic_kwargs}
        return potential_generator(t, err, **kwargs)(position)

    def generate(t, err, **kwargs):
        dynamic_kwargs = {name: value for name, value in kwargs.items()
                          if name not in static_kwargs}
        return lambda position: staged(position, t, err, dynamic_kwargs)

    return generate


def prepare_laplace_metric(
    model, rng_key, init_values, *model_args, model_kwargs=None,
    hessian_method='finite_difference', trust_radius=5.0, max_iterations=200,
    gradient_tolerance=0.0001, decrement_tolerance=0.0001, eigenvalue_floor=1e-08,
    fd_relative_step=0.0002, line_search_steps=8, sequential_evaluations=False,
    metric_scaling="raw", map_execution="fused",
):
    """Prepare a dense Laplace metric for one arbitrary NumPyro posterior.

    ``sequential_evaluations`` bounds temporary memory for expensive surface
    models by evaluating finite-difference and line-search points one at a
    time. It uses the same objective, derivatives, and candidate points.

    ``map_execution='split'`` compiles Hessian, objective/gradient and candidate
    evaluations separately and drives the same MAP iterations from Python.
    This bounds compilation graphs for expensive models. The fused loop
    remains the default; both paths use the same target and candidate points.

    This is the scalar-posterior counterpart of the per-lane preparation used
    by :class:`IndependentNUTSRunner`.  It deliberately returns NumPyro's
    unconstrained pytree so callers can pass it as ``MCMC.run(init_params=...)``.
    """
    method = str(hessian_method).lower()
    if map_execution not in {"fused", "split"}:
        raise ValueError("map_execution must be 'fused' or 'split'.")
    if method not in {"exact", "finite_difference"}:
        raise ValueError("hessian_method must be 'exact' or 'finite_difference'.")
    if trust_radius <= 0 or max_iterations < 1 or line_search_steps < 1:
        raise ValueError("Invalid Laplace MAP iteration controls.")
    model_kwargs = dict(model_kwargs or {})
    init_values = _synchronize_cadence_init_values(init_values, model_args)
    info = initialize_model(
        rng_key, model, init_strategy=init_to_value(values=init_values), model_args=model_args,
        model_kwargs=model_kwargs,
    )
    z0 = info.param_info.z
    flat0, unravel = ravel_pytree(z0)
    potential = lambda flat: info.potential_fn(unravel(flat))
    value_and_grad = jax.value_and_grad(potential)
    gradient_fn = jax.grad(potential)
    if map_execution == "split":
        value_and_grad = jax.jit(value_and_grad)

    def evaluate_points(function, points):
        if sequential_evaluations:
            return jax.lax.map(function, points)
        return jax.vmap(function)(points)

    def central_gradient_hessian(flat, steps):
        offsets = jnp.eye(flat.shape[0], dtype=flat.dtype) * steps[:, None]
        plus = evaluate_points(lambda offset: gradient_fn(flat + offset), offsets)
        minus = evaluate_points(lambda offset: gradient_fn(flat - offset), offsets)
        estimate = (plus - minus) / (2.0 * steps[:, None])
        return 0.5 * (estimate + estimate.T)

    if map_execution == "split":
        central_gradient_hessian = jax.jit(central_gradient_hessian)

    def repair(hessian):
        if metric_scaling != "raw":
            return _repair_laplace_hessian(hessian, eigenvalue_floor, metric_scaling)
        hessian = jnp.nan_to_num(0.5 * (hessian + hessian.T))
        values, vectors = jnp.linalg.eigh(hessian)
        scale = jnp.maximum(jnp.max(jnp.abs(values)), 1.0)
        repaired = jnp.maximum(values, eigenvalue_floor * scale)
        covariance = (vectors / repaired[None, :]) @ vectors.T
        return covariance, repaired[-1] / repaired[0], values[0]

    if method == "exact":
        hessian_fn = jax.hessian(potential)
        if map_execution == "split":
            hessian_fn = jax.jit(hessian_fn)
    else:
        initial_steps = fd_relative_step * jnp.maximum(1.0, jnp.abs(flat0))
        initial_hessian = central_gradient_hessian(flat0, initial_steps)
        coordinate_scale = jnp.clip(
            1.0 / jnp.sqrt(jnp.maximum(jnp.abs(jnp.diag(initial_hessian)), 1e-12)), 0.05, 20.0,
        )
        coordinate_steps = initial_steps * coordinate_scale
        hessian_fn = lambda flat: central_gradient_hessian(flat, coordinate_steps)

    scales = jnp.power(
        jnp.asarray(0.5, flat0.dtype), jnp.arange(line_search_steps, dtype=flat0.dtype)
    )
    value0, gradient0 = value_and_grad(flat0)

    def condition(state):
        index, _, value, gradient, _, decrement = state
        return (
            (index < max_iterations)
            & jnp.isfinite(value)
            & jnp.all(jnp.isfinite(gradient))
            & (jnp.linalg.norm(gradient) > gradient_tolerance)
            & ((decrement_tolerance == 0.0) | (decrement > decrement_tolerance))
        )

    evaluate_candidates = lambda points: evaluate_points(potential, points)
    if map_execution == "split":
        evaluate_candidates = jax.jit(evaluate_candidates)

    def iteration(state):
        index, flat, value, gradient, iterations, _ = state
        covariance, _, _ = repair(hessian_fn(flat))
        direction = -(covariance @ gradient)
        direction *= jnp.minimum(
            1.0, trust_radius / jnp.maximum(jnp.linalg.norm(direction), 1e-12)
        )
        gradient_direction = -gradient * (
            0.1 / jnp.maximum(jnp.linalg.norm(gradient), 1e-12)
        )
        directions = jnp.concatenate(
            (scales[:, None] * direction, scales[:, None] * gradient_direction), axis=0
        )
        candidates = flat[None, :] + directions
        candidate_values = evaluate_candidates(candidates)
        candidate_values = jnp.where(jnp.isfinite(candidate_values), candidate_values, jnp.inf)
        best = jnp.argmin(candidate_values)
        accept = candidate_values[best] < value
        next_flat = jnp.where(accept, candidates[best], flat)
        next_value, next_gradient = value_and_grad(next_flat)
        decrement = jnp.sqrt(jnp.maximum(next_gradient @ covariance @ next_gradient, 0.0))
        return index + 1, next_flat, next_value, next_gradient, iterations + 1, decrement

    initial_state = (
        jnp.asarray(0, jnp.int32), flat0, value0, gradient0,
        jnp.asarray(0, jnp.int32), jnp.asarray(jnp.inf, flat0.dtype),
    )
    if map_execution == "split":
        state = initial_state
        while bool(condition(state)):
            state = iteration(state)
    else:
        state = jax.lax.while_loop(condition, iteration, initial_state)
    _, flat, _, gradient, iterations, _ = state
    covariance, condition_number, minimum_eigenvalue = repair(hessian_fn(flat))
    decrement = jnp.sqrt(jnp.maximum(gradient @ covariance @ gradient, 0.0))
    return LaplaceMetricPreparation(
        unconstrained_map=unravel(flat), inverse_mass_matrix=covariance,
        gradient_norm=jnp.linalg.norm(gradient), newton_decrement=decrement,
        iterations=iterations, condition_number=condition_number,
        hessian_min_eigenvalue=minimum_eigenvalue,
        potential_fn=info.potential_fn,
        postprocess_fn=info.postprocess_fn,
    )


def _pad_first_axis(value, size):
    """Pad an array by repeating its final entry."""
    value = jnp.asarray(value)
    missing = size - value.shape[0]
    if missing <= 0:
        return value
    padding = jnp.repeat(value[-1:], missing, axis=0)
    return jnp.concatenate((value, padding), axis=0)


def _first_tree_lane(value):
    """Take lane zero from every array leaf while preserving PyTree types."""
    return jax.tree_util.tree_map(lambda leaf: leaf[0], value)


def _partition_model_kwargs(
    model_kwargs, channel_varying_kwargs, num_channels, lane_width,
):
    """Split kwargs into channel-varying leaves and shared leaves.

    Channel-varying arguments follow the convention already used by
    ``koala.sampling._slice_by_channel``: their first dimension equals the number of
    light curves.  Each lane receives a length-one channel axis because the
    current vectorized model builders infer ``num_lcs`` from ``yerr``.
    """
    varying = {}
    shared = {}
    channel_varying_kwargs = frozenset(channel_varying_kwargs)
    unknown = channel_varying_kwargs - set(model_kwargs)
    if unknown:
        raise ValueError(
            "channel_varying_kwargs contains unknown model arguments: "
            + ", ".join(sorted(unknown))
        )

    for name, value in model_kwargs.items():
        if value is None:
            shared[name] = None
            continue
        if hasattr(value, "baseline") and (
            hasattr(value, "differences") or hasattr(value, "uniform")
        ):
            if name not in channel_varying_kwargs:
                shared[name] = value
                continue
            if value.baseline.ndim <= 1 or value.baseline.shape[0] != num_channels:
                raise ValueError(
                    f"Channel-varying spot basis {name!r} must have leading "
                    f"dimension {num_channels}."
                )
            varying[name] = type(value)(*(
                None if field is None else
                _pad_first_axis(field, lane_width)[:, None, ...]
                for field in value
            ))
            continue
        array = _asarray_f64(value)
        if name in channel_varying_kwargs:
            if not (
                hasattr(array, "ndim")
                and array.ndim > 0
                and array.shape[0] == num_channels
            ):
                raise ValueError(
                    f"Channel-varying argument {name!r} has shape "
                    f"{getattr(array, 'shape', None)}, expected leading "
                    f"dimension {num_channels}."
                )
            varying[name] = _pad_first_axis(array, lane_width)[:, None, ...]
        else:
            shared[name] = array
    return varying, shared


def _enrich_initial_values(init_params):
    """Add aliases used by the pipeline but not named latent sample sites."""
    enriched = {name: _asarray_f64(value) for name, value in init_params.items()}
    if "u" in enriched:
        u = enriched["u"]
        if hasattr(u, "ndim") and u.ndim >= 2 and u.shape[-1] >= 2:
            enriched.setdefault("c1", u[..., 0])
            enriched.setdefault("c2", u[..., 1])
    if "tau" in enriched:
        enriched.setdefault("log_tau", jnp.log(enriched["tau"]))
    return enriched


def _report_invalid_initial_point(model, init_values, model_args, model_kwargs):
    """Print which sample sites are non-finite or outside support at ``init_values``."""
    try:
        from numpyro import handlers
        with handlers.seed(rng_seed=0):
            tr = handlers.trace(handlers.substitute(model, data=init_values)).get_trace(*model_args, **model_kwargs)
        print("[init diagnostics] per-site status at the initial point:", flush=True)
        for name, site in tr.items():
            if site["type"] != "sample":
                continue
            value = site["value"]
            fn = site["fn"]
            try:
                in_support = bool(jnp.all(fn.support(value)))
            except Exception:
                in_support = None
            try:
                lp = fn.log_prob(value)
                lp_finite = bool(jnp.all(jnp.isfinite(lp)))
                lp_sum = float(jnp.sum(lp))
            except Exception as lp_err:  # pragma: no cover - diagnostics only
                lp_finite, lp_sum = None, f"error: {lp_err}"
            flag = "" if (in_support in (True, None) and lp_finite in (True, None)) else "  <-- PROBLEM"
            print(
                f"[init diagnostics]   {name}: in_support={in_support} "
                f"log_prob_finite={lp_finite} log_prob={lp_sum}{flag}",
                flush=True,
            )
    except Exception as diag_err:  # pragma: no cover - diagnostics only
        print(f"[init diagnostics] could not trace the model: {diag_err}", flush=True)


def _first_lane_initial_values(init_params, num_channels):
    """Values passed to NumPyro's trace-building initialization."""
    first = {}
    for name, value in init_params.items():
        if (
            hasattr(value, "ndim")
            and value.ndim > 0
            and value.shape[0] == num_channels
        ):
            first[name] = value[:1]
    return first


def _prepare_unconstrained_initial_values(
    init_params, model_trace, num_channels, lane_width,
):
    """Build a padded unconstrained z pytree from physical initial values."""
    unconstrained = {}
    for name, site in model_trace.items():
        if (
            site["type"] != "sample"
            or site["is_observed"]
            or site["fn"].support.is_discrete
        ):
            continue

        prototype = jnp.asarray(site["value"])
        if prototype.ndim == 0 or prototype.shape[0] != 1:
            raise ValueError(
                "Independent NUTS only supports latent sites with an explicit "
                f"channel axis. Site {name!r} has traced shape "
                f"{prototype.shape}; a scalar/shared latent would change the "
                "posterior if replicated across lanes."
            )

        if (
            name in {"ld_decorrelated", "ld_smooth"}
            and name not in init_params
            and "c1" in init_params
            and "c2" in init_params
        ):
            # Stage dumps and the production pipeline deliberately retain the
            # public physical LD sites.  A transformed model therefore needs
            # its initial value derived from those sites instead of inheriting
            # the random value used to trace the model scaffold.  The latter
            # can lie outside the Maxted image (h2 <= 0), where the inverse and
            # induced density are undefined.
            from .ld_parameterization import (Power2MaxtedTransform, Power2MaxtedUniformTransform,
                                              Power2SmoothTransform)

            physical = jnp.stack((init_params['c1'], init_params['c2']), axis=-1)
            distribution_transforms = getattr(site["fn"], "transforms", ())
            parameterization = next(
                (
                    transform
                    for transform in distribution_transforms
                    if isinstance(transform, Power2MaxtedTransform)
                    or (name == "ld_smooth" and isinstance(
                        transform.inv, (Power2SmoothTransform, Power2MaxtedUniformTransform)))
                ),
                None,
            )
            if parameterization is None:
                raise ValueError(
                    "Could not identify the limb-darkening transform for "
                    f"the {name} site."
                )
            constrained = _pad_first_axis(parameterization(physical), lane_width)[:, None, ...]
        elif (name == "ld_coefficients" and name not in init_params
              and "c1" in init_params and "c2" in init_params):
            # Direct power-2 coordinates: start from the supplied physical values, never from the random trace
            # prototype, which can lie outside the Maxted-uniform support of the hard-walled box.
            physical = jnp.stack((init_params['c1'], init_params['c2']), axis=-1)
            constrained = _pad_first_axis(physical, lane_width)[:, None, ...]
        elif name in init_params:
            value = init_params[name]
            if not (
                hasattr(value, "ndim")
                and value.ndim > 0
                and value.shape[0] == num_channels
            ):
                raise ValueError(
                    "Independent NUTS requires every matched latent "
                    f"initialization to factorize by channel. Site {name!r} "
                    f"has shape {getattr(value, 'shape', None)}, expected "
                    f"leading dimension {num_channels}."
                )
            constrained = _pad_first_axis(value, lane_width)[:, None, ...]
        else:
            # Match normal NumPyro init_to_value behavior: unspecified sites
            # use the valid value generated while building the model trace.
            constrained = jnp.broadcast_to(prototype, (lane_width,) + prototype.shape)

        transform = biject_to(site["fn"].support)
        unconstrained[name] = transform.inv(constrained)

    if not unconstrained:
        raise ValueError("The model has no continuous latent sample sites.")
    return unconstrained


def _squeeze_internal_channel_axis(samples):
    """Map [draw, lane, 1, ...] back to the established [draw, lane, ...]."""

    def squeeze(value):
        if hasattr(value, "ndim") and value.ndim >= 3 and value.shape[2] == 1:
            return jnp.squeeze(value, axis=2)
        return value

    return jax.tree.map(squeeze, samples)


def _resolve_sampler_options(
    *, num_warmup, num_samples, dense_mass, regularize_mass_matrix, target_accept_prob,
    max_tree_depth, mass_matrix, laplace_warmup, laplace_target_accept, laplace_max_tree_depth,
    laplace_start_at_map, nuts_kwargs, mcmc_kwargs,
):
    nuts = dict(nuts_kwargs or {})
    mcmc = dict(mcmc_kwargs or {})
    # This backend always initializes from the supplied physical values.
    nuts.pop("init_strategy", None)
    # Explicit arguments take precedence over backend dictionaries.
    nuts_defaults = {
        "dense_mass": (bool, True, dense_mass),
        "regularize_mass_matrix": (bool, True, regularize_mass_matrix),
        "target_accept_prob": (float, 0.8, target_accept_prob),
        "max_tree_depth": (int, 10, max_tree_depth),
        "mass_matrix": (str, "adaptive", mass_matrix),
        "laplace_warmup": (int, 150, laplace_warmup),
        "laplace_target_accept": (float, 0.95, laplace_target_accept),
        "laplace_max_tree_depth": (int, 10, laplace_max_tree_depth),
        "laplace_start_at_map": (bool, False, laplace_start_at_map),
        "laplace_map_iterations": (int, 200, None),
        "laplace_map_tolerance": (float, 1.0e-4, None),
        "laplace_map_decrement_tolerance": (float, 1.0e-4, None),
        "laplace_eigenvalue_floor": (float, 1.0e-8, None),
        "laplace_metric_scaling": (str, "raw", None),
        "laplace_coordinates": (str, "unconstrained", None),
        "laplace_hessian_method": (str, "exact", None),
        "laplace_fd_relative_step": (float, 2.0e-4, None),
        # Zero vmaps all finite-difference coordinates; positive sizes bound memory.
        "laplace_fd_batch_size": (int, 1, None),
        "laplace_compare_exact_hessian": (bool, False, None),
        "laplace_map_method": (str, "newton", None),
        "laplace_line_search_steps": (int, 8, None),
        "laplace_trust_radius": (float, 5.0, None),
        "laplace_fuse_program": (bool, False, None),
        # Iterations per compiled segment; zero disables progress splitting.
        "progress_segment": (int, 50, None),
        "synchronize_segments": (bool, True, None),
        "lane_loop": (str, "vmap", None),
        "lane_batch_size": (int, 0, None),
        "forward_mode_differentiation": (bool, False, None),
    }
    mcmc_defaults = {
        "num_warmup": (int, 1000, num_warmup),
        "num_samples": (int, 1000, num_samples),
        "num_chains": (int, 1, None),
        "chain_method": (str, "vectorized", None),
        "progress_bar": (bool, True, None),
    }
    result = {}
    for label, supplied, defaults, ignored in (
        ("NUTS", nuts, nuts_defaults, set()),
        ("MCMC", mcmc, mcmc_defaults, {"jit_model_args"}),
    ):
        unknown = set(supplied) - defaults.keys() - ignored
        if unknown:
            raise ValueError(
                f"Unsupported independent-{label} options: " + ", ".join(sorted(unknown))
            )
        for name, (convert, default, explicit) in defaults.items():
            value = convert(explicit if explicit is not None else supplied.get(name, default))
            result[name] = value.lower() if convert is str else value
    if result["num_chains"] < 1:
        raise ValueError("num_chains must be >= 1.")
    if result["chain_method"] != "vectorized":
        raise ValueError("Independent samplers use chain_method=vectorized; host devices split lanes.")
    if result["mass_matrix"] not in {"adaptive", "laplace"}:
        raise ValueError(
            "mass_matrix must be either 'adaptive' or 'laplace'; got "
            f"{result['mass_matrix']!r}."
        )
    for name in ("laplace_warmup", "laplace_fd_batch_size", "lane_batch_size"):
        if result[name] < 0:
            raise ValueError(f"{name} must be >= 0.")
    for name in ("laplace_map_iterations", "laplace_max_tree_depth", "laplace_line_search_steps"):
        if result[name] < 1:
            raise ValueError(f"{name} must be >= 1.")
    for name in (
        "laplace_map_tolerance", "laplace_eigenvalue_floor",
        "laplace_fd_relative_step", "laplace_trust_radius",
    ):
        if result[name] <= 0.0:
            raise ValueError(f"{name} must be > 0.")
    if result["laplace_map_decrement_tolerance"] < 0.0:
        raise ValueError("laplace_map_decrement_tolerance must be >= 0.")
    if not 0.0 < result["laplace_target_accept"] < 1.0:
        raise ValueError("laplace_target_accept must be between 0 and 1.")
    for name, choices in (
        ("laplace_hessian_method", ("exact", "finite_difference")),
        ("laplace_map_method", ("newton", "diagonal")),
        ("laplace_metric_scaling", ("raw", "correlation")),
        ("laplace_coordinates", ("unconstrained", "whitened")),
        ("lane_loop", ("vmap", "map")),
    ):
        if result[name] not in choices:
            raise ValueError(f"{name} must be {choices[0]!r} or {choices[1]!r}.")
    if result["lane_batch_size"] and result["lane_loop"] != "map":
        raise ValueError("lane_batch_size requires lane_loop='map'.")
    if result["mass_matrix"] != "laplace" and (
        result["laplace_coordinates"] != "unconstrained" or result["laplace_metric_scaling"] != "raw"
    ):
        raise ValueError("Laplace coordinates and metric scaling require mass_matrix='laplace'.")
    return result


def _save_diagnostics(path, diagnostics):
    if path is None:
        return
    steps = jax.device_get(diagnostics.num_steps)
    accept = jax.device_get(diagnostics.accept_prob)
    diverging = jax.device_get(diagnostics.diverging)
    step_size = jax.device_get(diagnostics.step_size)
    payload = {
        "backend": "independent_batched_nuts",
        "num_channels": int(steps.shape[1]),
        "num_draws": int(steps.shape[0]),
        "num_chains": diagnostics.num_chains,
        "draws_per_chain": int(steps.shape[0]) // diagnostics.num_chains,
        "num_divergences": int(diverging.sum()),
        "num_divergences_per_channel": diverging.sum(axis=0).tolist(),
        "mean_num_steps_per_channel": steps.mean(axis=0).tolist(),
        "max_num_steps_per_channel": steps.max(axis=0).tolist(),
        "mean_accept_prob_per_channel": accept.mean(axis=0).tolist(),
        "adapted_step_size_per_channel": step_size.tolist(),
    }
    if diagnostics.num_chains > 1:
        chains = diagnostics.num_chains
        payload['num_divergences_per_chain_per_channel'] = (
            diverging.reshape(-1, chains, steps.shape[1]).sum(axis=0).tolist())
        payload['adapted_step_size_per_chain_per_channel'] = jax.device_get(diagnostics.step_size_by_chain).tolist()
    for name in (
        "map_gradient_norm", "map_newton_decrement", "map_iterations",
        "map_condition_number", "map_hessian_min_eigenvalue",
        "map_hessian_relative_error",
    ):
        value = getattr(diagnostics, name)
        if value is not None:
            payload[f"{name}_per_channel"] = jax.device_get(value).tolist()
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)


def _split_dynamic_and_static_kwargs(shared_kwargs):
    """Separate JAX-compatible runtime values from genuinely static values."""
    dynamic = {}
    static = {}
    for name, value in shared_kwargs.items():
        leaves = jax.tree.leaves(value)
        is_dynamic = True
        for leaf in leaves:
            if leaf is None:
                continue
            if isinstance(leaf, (str, bytes)) or callable(leaf):
                is_dynamic = False
                break
            try:
                array = jnp.asarray(leaf)
            except (TypeError, ValueError):
                is_dynamic = False
                break
            if array.dtype == jnp.dtype("O"):
                is_dynamic = False
                break
        (dynamic if is_dynamic else static)[name] = value
    return dynamic, static


def _static_kwargs_equal(left, right):
    if left.keys() != right.keys():
        return False
    for name in left:
        lhs = left[name]
        rhs = right[name]
        if callable(lhs) or callable(rhs):
            if lhs is not rhs:
                return False
            continue
        try:
            equal = lhs == rhs
            if hasattr(equal, "all"):
                equal = bool(equal.all())
            else:
                equal = bool(equal)
        except (TypeError, ValueError):
            equal = lhs is rhs
        if not equal:
            return False
    return True


class _IndependentSamplerRunner:
    """Reusable shape-specialized vmapped HMC/NUTS transition programs.

    The first call builds the dynamic NumPyro potential and the small set of
    jitted programs used for initialization, adaptation, sampling, and
    postprocessing.  Later calls pass observations, initial states, time,
    channel-varying priors, and all array-valued shared arguments as runtime
    values, so equal-shape wavelength chunks reuse XLA executables without
    retaining values from an earlier chunk.
    """

    def __init__(
        self, model, *, algo, options, lane_width, channel_varying_kwargs=(),
        lane_devices=None,
    ):
        self.model = model
        self.algo = str(algo)
        self.options = dict(options)
        self.lane_width = int(lane_width)
        self.num_chains = int(options.get("num_chains", 1))
        self._execution_width = self.lane_width * self.num_chains
        # Several CPU host devices split the lanes; one device (and every
        # accelerator) keeps ``_device_width == _execution_width``.
        self._mesh, sharded_lanes = _lane_device_layout(self.lane_width, lane_devices)
        self._device_width = sharded_lanes * self.num_chains
        self.channel_varying_kwargs = tuple(channel_varying_kwargs)
        self.program_build_count = 0
        self._initialization_model_info = None
        self._static_shared_kwargs = None
        self._init_program = None
        self._warmup_program = None
        self._warmup_segment = 0
        self._sample_segment = 0
        self._sample_program = None
        self._postprocess_program = None
        self._laplace_sample_program = None
        self._laplace_full_program = None

    def validate_configuration(
        self, model, *, options, lane_width, channel_varying_kwargs,
    ):
        if model is not self.model:
            raise ValueError("Independent sampler runner belongs to another model.")
        if dict(options) != self.options:
            raise ValueError('Independent sampler runner options differ from this call.')
        if int(lane_width) != self.lane_width:
            raise ValueError('Independent sampler runner lane width differs from this call.')
        if tuple(channel_varying_kwargs) != self.channel_varying_kwargs:
            raise ValueError(
                'Independent sampler runner channel-varying arguments differ from this call.',
            )

    def _jit(self, function, in_specs, out_specs):
        """Jit a lane program; with a host-device mesh, one block per device."""
        if self._mesh is None:
            return jax.jit(function)
        return _shard_lanes(function, self._mesh, in_specs, out_specs)

    @staticmethod
    def _merge_lane_kwargs(static_shared, dynamic_shared, varying_lane):
        return {**static_shared, **dynamic_shared, **varying_lane}

    def _postprocess_lanes(self, model_info):
        static_shared = self._static_shared_kwargs
        varying_axes = {name: 0 for name in self.channel_varying_kwargs}
        postprocess_gen = model_info.postprocess_fn

        def postprocess_lane(z, t, err, obs, varying, dynamic_shared):
            kwargs = self._merge_lane_kwargs(static_shared, dynamic_shared, varying)
            return postprocess_gen(t, err, y=obs, **kwargs)(z)

        return jax.vmap(postprocess_lane, in_axes=(0, None, 0, 0, varying_axes, None))


    def _transition_program(self, sample_kernel, *, jitter=False):
        """Vmap a transition while keeping observations dynamic across chunks."""
        static_shared = self._static_shared_kwargs
        varying_axes = {name: 0 for name in self.channel_varying_kwargs}
        options = self.options
        def advance(states, t, errors, observations, varying, dynamic_shared, *transform_args):
            def advance_one(state, err, obs, varying_lane, affine=()):
                kwargs = self._merge_lane_kwargs(static_shared, dynamic_shared, varying_lane)
                if (
                    jitter and self.algo == "HMC"
                    and options.get("trajectory_jitter", 0.0) > 0.0
                ):
                    jitter_key, transition_key = jax.random.split(state.rng_key)
                    spread = max(
                        1, int(round(options['num_steps'] * options['trajectory_jitter'])),
                    )
                    desired_steps = jax.random.randint(
                        jitter_key, (), max(1, options['num_steps'] - spread),
                        options['num_steps'] + spread + 1,
                    )
                    trajectory_length = state.adapt_state.step_size * (
                        desired_steps.astype(jnp.float64) - 1.0e-7
                    )
                    state = state._replace(
                        rng_key=transition_key, trajectory_length=trajectory_length,
                    )
                return sample_kernel(
                    state, model_args=(t, err, *affine), model_kwargs={'y': obs, **kwargs},
                )

            if options.get("lane_loop", "vmap") == "map":
                operands = (states, errors, observations, varying, *transform_args)
                # Bound lockstep NUTS work within each vectorized group.
                # Zero retains one sequential lane; vmap remains the default.
                return jax.lax.map(lambda lane: advance_one(*lane), operands,
                                   batch_size=options.get("lane_batch_size", 0) or None)
            if transform_args:
                return jax.vmap(advance_one, in_axes=(0, 0, 0, varying_axes, 0))(
                    states, errors, observations, varying, transform_args[0])
            return jax.vmap(advance_one, in_axes=(0, 0, 0, varying_axes))(states, errors, observations, varying)

        return advance

    def _build_laplace_programs(
        self, model_info, static_shared_kwargs, initial_prototype,
    ):
        """Build the fused programs used by Laplace-preconditioned HMC/NUTS."""
        self._static_shared_kwargs = dict(static_shared_kwargs)
        options = self.options
        whitened = options.get("laplace_coordinates", "unconstrained") == "whitened"
        _, unravel = ravel_pytree(initial_prototype)
        lane_potential_generator = _stage_potential_generator(
            model_info.potential_fn, self._static_shared_kwargs)
        potential_generator = lane_potential_generator
        if whitened:
            def potential_generator(t, err, center, factor, **kwargs):
                return _affine_laplace_potential(
                    lane_potential_generator(t, err, **kwargs), unravel, center, factor)
        init_kernel, sample_kernel = hmc(
            potential_fn_gen=potential_generator, algo=self.algo,
        )
        static_shared = self._static_shared_kwargs
        varying_axes = {name: 0 for name in self.channel_varying_kwargs}
        _, unravel = ravel_pytree(initial_prototype)
        line_search_scales = jnp.power(
            jnp.asarray(0.5, dtype=jnp.float64),
            jnp.arange(options['laplace_line_search_steps'], dtype=jnp.float64),
        )

        def repaired_inverse_hessian(hessian):
            return _repair_laplace_hessian(
                hessian, options["laplace_eigenvalue_floor"],
                options.get("laplace_metric_scaling", "raw"))

        def map_one(z0, err, obs, varying, t, dynamic_shared):
            kwargs = self._merge_lane_kwargs(static_shared, dynamic_shared, varying)
            potential = lane_potential_generator(t, err, y=obs, **kwargs)
            value_and_grad = jax.value_and_grad(lambda flat: potential(unravel(flat)))
            gradient_fn = jax.grad(lambda flat: potential(unravel(flat)))
            flat0, _ = ravel_pytree(z0)
            value0, gradient0 = value_and_grad(flat0)

            def central_gradient_hessian(flat, coordinate_steps):
                offsets = jnp.eye(flat.shape[0], dtype=flat.dtype)
                offsets = offsets * coordinate_steps[:, None]
                batch_size = options["laplace_fd_batch_size"]
                if batch_size:
                    plus = jax.lax.map(
                        lambda offset: gradient_fn(flat + offset), offsets,
                        batch_size=batch_size,
                    )
                    minus = jax.lax.map(
                        lambda offset: gradient_fn(flat - offset), offsets,
                        batch_size=batch_size,
                    )
                else:
                    plus = jax.vmap(lambda offset: gradient_fn(flat + offset))(offsets)
                    minus = jax.vmap(lambda offset: gradient_fn(flat - offset))(offsets)
                estimate = (plus - minus) / (2.0 * coordinate_steps[:, None])
                return 0.5 * (estimate + estimate.T)

            if options["laplace_hessian_method"] == "exact":
                hessian_fn = jax.hessian(lambda flat: potential(unravel(flat)))
                if options["laplace_map_method"] == "diagonal":
                    initial_hessian = hessian_fn(flat0)
                    fixed_diagonal_covariance = 1.0 / jnp.maximum(
                        jnp.abs(jnp.diag(initial_hessian)), 1.0e-8
                    )
            else:
                # A first diagonal estimate defines a per-coordinate scale.
                # Each subsequent Hessian is 2*d exact gradient evaluations
                # and avoids differentiating the streamed transit kernel.
                relative_step = options["laplace_fd_relative_step"]
                first_steps = relative_step * jnp.maximum(1.0, jnp.abs(flat0))
                first_hessian = central_gradient_hessian(flat0, first_steps)
                diagonal_scale = 1.0 / jnp.sqrt(
                    jnp.maximum(jnp.abs(jnp.diag(first_hessian)), 1.0e-12)
                )
                diagonal_scale = jnp.clip(diagonal_scale, 0.05, 20.0)
                coordinate_steps = first_steps * diagonal_scale
                fixed_diagonal_covariance = 1.0 / jnp.maximum(
                    jnp.abs(jnp.diag(first_hessian)), 1.0e-8
                )

                def hessian_fn(flat):
                    return central_gradient_hessian(flat, coordinate_steps)

            def still_optimizing(state):
                loop_index, _, value, gradient, _, decrement_estimate = state
                gradient_norm = jnp.linalg.norm(gradient)
                decrement_active = (
                    (options["laplace_map_decrement_tolerance"] == 0.0)
                    | (
                        decrement_estimate
                        > options["laplace_map_decrement_tolerance"]
                    )
                )
                return (
                    (loop_index < options["laplace_map_iterations"])
                    & jnp.isfinite(value)
                    & jnp.all(jnp.isfinite(gradient))
                    & (gradient_norm > options["laplace_map_tolerance"])
                    & decrement_active
                )

            def iteration(state):
                loop_index, flat, value, gradient, iterations, _ = state
                gradient_norm = jnp.linalg.norm(gradient)
                active = (
                    jnp.isfinite(value)
                    & jnp.all(jnp.isfinite(gradient))
                    & (gradient_norm > options["laplace_map_tolerance"])
                )
                if options["laplace_map_method"] == "newton":
                    covariance, _, _ = repaired_inverse_hessian(hessian_fn(flat))
                    newton = -(covariance @ gradient)
                else:
                    covariance = jnp.diag(fixed_diagonal_covariance)
                    newton = -(fixed_diagonal_covariance * gradient)
                newton_norm = jnp.linalg.norm(newton)
                # A trust radius prevents a repaired, nearly-flat Hessian
                # direction from proposing a huge unconstrained jump.
                newton = newton * jnp.minimum(
                    1.0, options['laplace_trust_radius'] / jnp.maximum(newton_norm, 1e-12),
                )
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
                candidate_values = jax.vmap(lambda candidate: potential(unravel(candidate)))(candidates)
                candidate_values = jnp.where(
                    jnp.isfinite(candidate_values), candidate_values, jnp.inf,
                )
                # argmin's paired (value, index) reduction can acquire mixed
                # replicated/manual shardings inside shard_map. XLA then
                # propagates a zero tile count and divides by zero in
                # ReshapeSharding. Two scalar reductions preserve first-tie
                # argmin semantics without that mixed-sharding tuple.
                minimum = jnp.min(candidate_values)
                indices = jnp.arange(candidate_values.size)
                best_index = jnp.min(jnp.where(
                    candidate_values == minimum, indices, candidate_values.size))
                best_flat = candidates[best_index]
                best_value = candidate_values[best_index]
                accept = active & (best_value < value)
                next_flat = jnp.where(accept, best_flat, flat)
                next_value, next_gradient = value_and_grad(next_flat)
                decrement_estimate = jnp.sqrt(
                    jnp.maximum(next_gradient @ covariance @ next_gradient, 0.0)
                )
                return (
                    loop_index + jnp.asarray(1, jnp.int32),
                    next_flat,
                    next_value,
                    next_gradient,
                    iterations + active.astype(jnp.int32),
                    decrement_estimate,
                )

            _, flat, _, gradient, iterations, _ = jax.lax.while_loop(
                still_optimizing,
                iteration,
                (
                    jnp.asarray(0, jnp.int32),
                    flat0,
                    value0,
                    gradient0,
                    jnp.asarray(0, jnp.int32),
                    jnp.asarray(jnp.inf, flat0.dtype),
                ),
            )
            hessian = hessian_fn(flat)
            covariance, condition, minimum_eigenvalue = (
                repaired_inverse_hessian(hessian)
            )
            gradient_norm = jnp.linalg.norm(gradient)
            decrement = jnp.sqrt(jnp.maximum(gradient @ covariance @ gradient, 0.0))
            if (
                options["laplace_hessian_method"] == "finite_difference"
                and options["laplace_compare_exact_hessian"]
            ):
                exact_hessian = jax.hessian(lambda value: potential(unravel(value)))(flat)
                relative_error = jnp.linalg.norm(hessian - exact_hessian) / jnp.maximum(jnp.linalg.norm(exact_hessian), 1.0e-30)
            else:
                relative_error = jnp.asarray(jnp.nan, dtype=flat.dtype)
            return (
                flat,
                covariance,
                gradient_norm,
                decrement,
                iterations,
                condition,
                minimum_eigenvalue,
                relative_error,
            )

        map_lanes = jax.vmap(map_one, in_axes=(0, 0, 0, varying_axes, None, None))

        advance = self._transition_program(sample_kernel, jitter=True)

        def prepare_map(
            batched_init, errors, observations, varying, keys, t, dynamic_shared,
        ):
            (
                map_flat,
                covariance,
                gradient_norm,
                decrement,
                iterations,
                condition,
                minimum_eigenvalue,
                relative_error,
            ) = map_lanes(
                jax.tree.map(lambda value:value[::self.num_chains], batched_init),
                errors[::self.num_chains], observations[::self.num_chains],
                jax.tree.map(lambda value:value[::self.num_chains], varying), t, dynamic_shared,
            )
            if self.num_chains > 1:
                prepared = jax.tree.map(lambda value:jnp.repeat(value,self.num_chains,axis=0),
                    (map_flat,covariance,gradient_norm,decrement,iterations,condition,minimum_eigenvalue,relative_error))
                (map_flat,covariance,gradient_norm,decrement,iterations,condition,minimum_eigenvalue,relative_error) = prepared
            map_z = jax.vmap(unravel)(map_flat)
            start_z = map_z if options["laplace_start_at_map"] else batched_init
            if self.num_chains > 1:
                # Share deterministic channel geometry, but use independent,
                # dispersed starts and adaptation for every Markov chain.
                eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
                noise = jax.vmap(lambda key:jax.random.normal(jax.random.fold_in(key,81721), (map_flat.shape[-1],)))(keys)
                offset = jnp.einsum('lij,lj->li',eigenvectors,noise*jnp.sqrt(jnp.maximum(eigenvalues,0.)))
                center = map_flat if options['laplace_start_at_map'] else jax.vmap(lambda z:ravel_pytree(z)[0])(batched_init)
                start_z = jax.vmap(unravel)(center + jnp.clip(.5*offset,-1.,1.))

            def init_one(z, inverse_mass, err, obs, varying_lane, rng, affine=()):
                kwargs = self._merge_lane_kwargs(static_shared, dynamic_shared, varying_lane)
                common = {
                    "num_warmup": options["laplace_warmup"],
                    "step_size": 1.0,
                    "inverse_mass_matrix": inverse_mass,
                    "adapt_step_size": True,
                    "adapt_mass_matrix": False,
                    "dense_mass": not whitened,
                    "target_accept_prob": options["laplace_target_accept"],
                    "find_heuristic_step_size": False,
                    "forward_mode_differentiation": options.get("forward_mode_differentiation", False),
                    "regularize_mass_matrix": False,
                    "model_args": (t, err, *affine),
                    "model_kwargs": {"y": obs, **kwargs},
                    "rng_key": rng,
                }
                if self.algo == "NUTS":
                    common["max_tree_depth"] = options[
                        "laplace_max_tree_depth"
                    ]
                else:
                    if options.get("trajectory_jitter", 0.0) > 0.0:
                        common.update(
                            num_steps=None, trajectory_length=float(options['num_steps']),
                        )
                    else:
                        common.update(num_steps=options['num_steps'], trajectory_length=None)
                return init_kernel(z, **common)

            transforms = ()
            if whitened:
                factor = jax.vmap(_laplace_whitening_factor)(covariance)
                initial_flat = jax.vmap(lambda z: ravel_pytree(z)[0])(start_z)
                white_start = jax.vmap(lambda f, z: solve_triangular(f, z, lower=True))(
                    factor, initial_flat - map_flat)
                identity_metric = jnp.ones_like(map_flat)
                affine = (map_flat, factor)
                states = jax.vmap(init_one, in_axes=(0, 0, 0, 0, varying_axes, 0, 0))(
                    white_start, identity_metric, errors, observations, varying, keys, affine)
                transforms = (affine,)
            else:
                states = jax.vmap(init_one, in_axes=(0, 0, 0, 0, varying_axes, 0))(
                    start_z, covariance, errors, observations, varying, keys,
                )
            diagnostics = (
                gradient_norm,
                decrement,
                iterations,
                condition,
                minimum_eigenvalue,
                relative_error,
            )
            return (states, diagnostics, *transforms)

        def warmup_states(states, t, errors, observations, varying, dynamic_shared, *transforms):
            if options["laplace_warmup"]:
                states = jax.lax.fori_loop(
                    0, options["laplace_warmup"],
                    lambda _, state: advance(
                        state, t, errors, observations, varying, dynamic_shared, *transforms),
                    states,
                )
            return states

        def prepare(batched_init, errors, observations, varying, keys, t, dynamic_shared):
            states, diagnostics, *transforms = prepare_map(
                batched_init, errors, observations, varying, keys, t, dynamic_shared)
            states = warmup_states(states, t, errors, observations, varying, dynamic_shared, *transforms)
            return (states, diagnostics, *transforms)

        # Separate entry points support opt-in phase profiling. Normal inference
        # retains the combined prepare graph; these functions compile only when
        # a profiling caller invokes them.
        extra_specs = (_LANES,) if whitened else ()
        init_output_specs = (_LANES, _LANES, *extra_specs)
        step_specs = (*_STEP_SPECS, *extra_specs)
        self._laplace_map_program = self._jit(prepare_map, _INIT_SPECS, init_output_specs)
        self._laplace_warmup_program = self._jit(warmup_states, step_specs, _LANES)
        self._init_program = self._jit(prepare, _INIT_SPECS, init_output_specs)
        postprocess_lanes = self._postprocess_lanes(model_info)

        def sample_and_postprocess(
            initial, t, errors, observations, varying, dynamic_shared, *transforms,
        ):
            def collect_step(state, _):
                state = advance(state, t, errors, observations, varying, dynamic_shared, *transforms)
                physical_z = state.z
                if whitened:
                    center, factor = transforms[0]
                    physical_z = jax.vmap(lambda z, c, f: unravel(c + f @ z))(
                        state.z, center, factor)
                samples = postprocess_lanes(
                    physical_z, t, errors, observations, varying, dynamic_shared,
                )
                return (state, (samples, state.num_steps, state.accept_prob, state.diverging))

            return jax.lax.scan(
                collect_step,
                initial,
                xs=None,
                length=(
                    options["num_samples"]
                    if options["laplace_fuse_program"]
                    else self._sample_segment
                ),
            )

        self._sample_segment = _segment_length(
            options["num_samples"], options.get("progress_segment", 50)
        )
        self._laplace_sample_program = self._jit(
            sample_and_postprocess, step_specs, (_LANES, _DRAW_LANES))
        if options["laplace_fuse_program"]:
            def prepare_sample_and_postprocess(
                batched_init, errors, observations, varying, keys, t, dynamic_shared,
            ):
                states, map_diagnostics, *transforms = prepare(
                    batched_init, errors, observations, varying, keys, t, dynamic_shared,
                )
                states, outputs = sample_and_postprocess(
                    states, t, errors, observations, varying, dynamic_shared, *transforms,
                )
                return (states, outputs, map_diagnostics, *transforms)

            self._laplace_full_program = self._jit(
                prepare_sample_and_postprocess, _INIT_SPECS, (_LANES, _DRAW_LANES, _LANES, *extra_specs))
        self.program_build_count += 1

    def _build_programs(
        self, model_info, static_shared_kwargs, initial_prototype=None,
    ):
        if self.options.get("mass_matrix") == "laplace":
            if initial_prototype is None:
                raise ValueError("Laplace NUTS requires an initial state prototype.")
            self._build_laplace_programs(model_info, static_shared_kwargs, initial_prototype)
            return
        self._static_shared_kwargs = dict(static_shared_kwargs)
        init_kernel, sample_kernel = hmc(
            potential_fn_gen=model_info.potential_fn, algo=self.algo,
        )
        options = self.options
        static_shared = self._static_shared_kwargs
        varying_axes = {
            name: 0 for name in self.channel_varying_kwargs
        }

        def init_one(z, err, obs, varying, rng, t, dynamic_shared):
            kwargs = self._merge_lane_kwargs(static_shared, dynamic_shared, varying)
            common = {
                "num_warmup": options["num_warmup"],
                "dense_mass": options["dense_mass"],
                "regularize_mass_matrix": options[
                    "regularize_mass_matrix"
                ],
                "target_accept_prob": options["target_accept_prob"],
                "forward_mode_differentiation": options.get("forward_mode_differentiation", False),
                "model_args": (t, err),
                "model_kwargs": {"y": obs, **kwargs},
                "rng_key": rng,
            }
            if self.algo == "NUTS":
                common["max_tree_depth"] = options["max_tree_depth"]
            else:
                common.update(num_steps=options['num_steps'], trajectory_length=None)
            return init_kernel(z, **common)

        self._init_program = self._jit(
            jax.vmap(init_one, in_axes=(0, 0, 0, varying_axes, 0, None, None)),
            _INIT_SPECS, _LANES,
        )

        advance = self._transition_program(sample_kernel)

        if options["num_warmup"]:
            self._warmup_segment = _segment_length(
                options["num_warmup"], options.get("progress_segment", 50)
            )
            self._warmup_program = self._jit(
                lambda initial, t, errors, observations, varying, shared: (
                    jax.lax.fori_loop(
                        0,
                        self._warmup_segment,
                        lambda _, state: advance(
                            state, t, errors, observations, varying, shared
                        ),
                        initial,
                    )
                ),
                _STEP_SPECS, _LANES,
            )

        def collect(initial, t, errors, observations, varying, shared):
            def collect_step(state, _):
                state = advance(state, t, errors, observations, varying, shared)
                return (state, (state.z, state.num_steps, state.accept_prob, state.diverging))

            return jax.lax.scan(collect_step, initial, xs=None, length=self._sample_segment)

        self._sample_segment = _segment_length(
            options["num_samples"], options.get("progress_segment", 50)
        )
        self._sample_program = self._jit(collect, _STEP_SPECS, (_LANES, _DRAW_LANES))
        postprocess_lanes = self._postprocess_lanes(model_info)
        self._postprocess_program = self._jit(
            jax.vmap(postprocess_lanes, in_axes=(0, None, None, None, None, None)),
            (_DRAW_LANES, _REPLICATED, _LANES, _LANES, _LANES, _REPLICATED), _DRAW_LANES,
        )
        self.program_build_count += 1

    @staticmethod
    def _run_segments(program, states, segment, total, desc, show, collect, args,
                      *, synchronize=True):
        """Call a compiled fixed-length ``program`` until ``total`` iterations.

        ``collect=True`` programs return ``(states, outputs)`` from a scan;
        the per-segment outputs are concatenated along the draw axis.
        """
        n_segments = total // segment if segment else 0
        bar = _SegmentProgress(total, desc, show)
        pieces = []
        try:
            for _ in range(n_segments):
                if collect:
                    states, outputs = program(states, *args)
                    pieces.append(outputs)
                else:
                    states = program(states, *args)
                # A visible bar must reflect completed device work. Silent
                # callers can opt out of host waits: array dependencies still
                # serialize transitions, and result consumers synchronize.
                if synchronize:
                    jax.block_until_ready(states)
                if synchronize or not show:
                    bar.update(segment)
            if show and not synchronize and n_segments:
                # Coarse but truthful progress without per-segment device waits.
                jax.block_until_ready(states)
                bar.update(n_segments * segment)
        finally:
            bar.close()
        if not collect:
            return states
        if len(pieces) == 1:
            return states, pieces[0]
        outputs = jax.tree.map(lambda *chunks: jnp.concatenate(chunks, axis=0), *pieces)
        return states, outputs

    def run_raw(self, key, t, yerr, indiv_y, init_params, model_kwargs):
        # A failed fresh run must not leave an older target resumable.
        for name in ('_continuation', '_last_pooled_raw'):
            if hasattr(self, name):
                delattr(self, name)
        if self.num_chains == 1:
            result = self._run_single_lanes(key, t, yerr, indiv_y, init_params, model_kwargs)
            self._last_pooled_raw = result[1]
            return result
        channels = int(indiv_y.shape[0])
        if channels > self.lane_width:
            raise ValueError('The channel count exceeds the runner width.')
        chains = self.num_chains
        def repeat(value):
            def leaf(array):
                if hasattr(array, 'shape') and array.ndim and array.shape[0] == channels:
                    return jnp.repeat(array, chains, axis=0)
                return array
            return jax.tree.map(leaf, value)
        varying = frozenset(self.channel_varying_kwargs)
        expanded_kwargs = {name: repeat(value) if name in varying else value
                           for name, value in model_kwargs.items()}
        samples, raw = self._run_single_lanes(
            key, t, jnp.repeat(yerr, chains, axis=0),
            jnp.repeat(indiv_y, chains, axis=0), repeat(init_params), expanded_kwargs,
        )
        result = self._pool_execution_result(samples, raw, channels)
        self._last_pooled_raw = result[1]
        return result

    def _pool_execution_result(self, samples, raw, channels):
        chains = self.num_chains
        if chains == 1:
            return samples, raw
        def pool(value):
            shaped = value.reshape((value.shape[0], channels, chains) + value.shape[2:])
            return shaped.swapaxes(1, 2).reshape((-1, channels) + value.shape[2:])
        samples = jax.tree.map(pool, samples)
        for name in ('num_steps', 'accept_prob', 'diverging'):
            raw[name] = pool(raw[name])
        for name in ('step_size', 'mean_accept_prob'):
            grouped = raw[name].reshape(channels, chains).T
            raw[name+'_by_chain'] = grouped
            raw[name] = jnp.mean(grouped, axis=0)
        for name in tuple(raw):
            if name.startswith('map_'):
                raw[name] = raw[name].reshape(channels, chains)[:, 0]
        raw['num_chains'] = chains
        return samples, raw

    def continue_raw(self):
        """Advance the last target without tracing, MAP fitting, or warmup.

        Arguments and final chain keys are captured by the preceding run.
        A subsequent fresh run replaces this target and continuation state.
        """
        if not hasattr(self, '_continuation'):
            raise RuntimeError('Continuation requires a completed initial run.')
        states, args, channels, map_diagnostics = self._continuation
        laplace = self.options.get('mass_matrix') == 'laplace'
        program = self._laplace_sample_program if laplace else self._sample_program
        segment = (self.options['num_samples'] if laplace and self.options['laplace_fuse_program']
                   else self._sample_segment)
        states, outputs = self._run_segments(
            program, states, segment, self.options['num_samples'],
            (getattr(self, 'progress_label', '') + ' continuation').strip(),
            bool(self.options.get('progress_bar', True)), collect=True, args=args,
            synchronize=self.options.get('synchronize_segments', True))
        samples, steps, accept_prob, diverging = outputs
        if not laplace:
            samples = self._postprocess_program(samples, *args)
        self._continuation = (states, args, channels, map_diagnostics)
        samples, raw = self._gather_lanes(self._format_execution_result(
            samples, steps, accept_prob, diverging, states, map_diagnostics, channels))
        return self._pool_execution_result(samples, raw, channels // self.num_chains)

    def _run_single_lanes(self, key, t, yerr, indiv_y, init_params, model_kwargs):
        t = _asarray_f64(t)
        yerr = _asarray_f64(yerr)
        indiv_y = _asarray_f64(indiv_y)
        if yerr.ndim != 2 or indiv_y.ndim != 2:
            raise ValueError('yerr and indiv_y must have shape [channel, time].')
        if yerr.shape != indiv_y.shape:
            raise ValueError("yerr and indiv_y must have identical shapes.")
        num_channels = int(yerr.shape[0])
        if num_channels < 1:
            raise ValueError("At least one channel is required.")
        if self._execution_width < num_channels:
            raise ValueError(
                f'lane_width={self._execution_width} cannot hold {num_channels} channels.',
            )

        padded_yerr = _pad_first_axis(yerr, self._execution_width)[:, None, :]
        padded_y = _pad_first_axis(indiv_y, self._execution_width)[:, None, :]
        varying_kwargs, shared_kwargs = _partition_model_kwargs(
            model_kwargs, self.channel_varying_kwargs, num_channels, self._execution_width,
        )
        dynamic_shared, static_shared = _split_dynamic_and_static_kwargs(shared_kwargs)
        if (
            self._static_shared_kwargs is not None
            and not _static_kwargs_equal(static_shared, self._static_shared_kwargs)
        ):
            raise ValueError(
                "A reused independent sampler runner received different "
                "non-array/static model arguments. Build a separate runner "
                "for that static configuration."
            )

        enriched_init = _enrich_initial_values(init_params)
        first_init = _first_lane_initial_values(enriched_init, num_channels)
        first_varying = {
            name: _first_tree_lane(value)
            for name, value in varying_kwargs.items()
        }
        first_kwargs = self._merge_lane_kwargs(static_shared, dynamic_shared, first_varying)
        key_model, key_chains = jax.random.split(key)

        cached_info = self._initialization_model_info
        complete_initialization = cached_info is not None and all(
            name in enriched_init or (
                name in {'ld_smooth','ld_decorrelated','ld_coefficients'}
                and 'c1' in enriched_init and 'c2' in enriched_init
            )
            for name,site in cached_info.model_trace.items()
            if site['type']=='sample' and not site['is_observed']
            and not site['fn'].support.is_discrete
        )
        reuse_supports = bool(
            complete_initialization
            and getattr(self.model,'_koala_fixed_initialization_supports',False)
        )
        if reuse_supports:
            model_info = cached_info
        else:
            # Generic models and omitted initial sites need a fresh trace:
            # their support or randomly supplied start may change by chunk.
            try:
                model_info = initialize_model(
                    key_model, self.model, init_strategy=init_to_value(values=first_init),
                    dynamic_args=True, model_args=(t, padded_yerr[0]),
                    model_kwargs={'y': padded_y[0], **first_kwargs},
                )
            except RuntimeError as err:
                _report_invalid_initial_point(
                    self.model, first_init, (t, padded_yerr[0]),
                    {"y": padded_y[0], **first_kwargs},
                )
                raise RuntimeError(
                    "NumPyro could not initialise the first lane of this chunk "
                    "(see the per-site report above)."
                ) from err
            self._initialization_model_info = model_info
        batched_init = _prepare_unconstrained_initial_values(
            enriched_init, model_info.model_trace, num_channels, self._execution_width,
        )
        if reuse_supports and not all(bool(jnp.all(jnp.isfinite(value)))
                                      for value in jax.tree.leaves(batched_init)):
            raise RuntimeError('Independent sampler initial values are outside their support.')
        if self.num_chains > 1 and self.options.get('mass_matrix') != 'laplace':
            batched_init = {name:value + .1*jax.random.normal(
                jax.random.fold_in(key_model,81721+index), value.shape)
                for index,(name,value) in enumerate(batched_init.items())}
        if self._init_program is None:
            initial_prototype = jax.tree.map(lambda value: value[0], batched_init)
            self._build_programs(model_info, static_shared, initial_prototype=initial_prototype)

        chain_keys = jax.random.split(key_chains, self._execution_width)
        if self._mesh is not None:
            # Keys and starts are drawn at the unsharded width, so a lane's
            # inputs do not depend on the device count.
            batched_init, padded_yerr, padded_y, varying_kwargs, chain_keys = self._place_lanes(
                (batched_init, padded_yerr, padded_y, varying_kwargs, chain_keys))
            t, dynamic_shared = jax.device_put(
                (t, dynamic_shared), NamedSharding(self._mesh, _REPLICATED))
        map_diagnostics = None
        transform_args = ()
        if (
            self.options.get("mass_matrix") == "laplace"
            and self._laplace_full_program is not None
        ):
            states, outputs, map_diagnostics, *transform_args = self._laplace_full_program(
                batched_init, padded_yerr, padded_y, varying_kwargs, chain_keys, t,
                dynamic_shared,
            )
            samples, steps, accept_prob, diverging = outputs
        else:
            prepared = self._init_program(
                batched_init, padded_yerr, padded_y, varying_kwargs, chain_keys, t,
                dynamic_shared,
            )
            if self.options.get("mass_matrix") == "laplace":
                states, map_diagnostics, *transform_args = prepared
            else:
                states = prepared
            label = getattr(self, "progress_label", "")
            show = bool(self.options.get("progress_bar", True))
            if self._warmup_program is not None:
                states = self._run_segments(
                    self._warmup_program, states, self._warmup_segment,
                    self.options['num_warmup'], f'{label} warmup'.strip(), show, collect=False,
                    args=(t, padded_yerr, padded_y, varying_kwargs, dynamic_shared),
                    synchronize=self.options.get('synchronize_segments', True),
                )
            laplace = self.options.get("mass_matrix") == "laplace"
            sample_program = self._laplace_sample_program if laplace else self._sample_program
            states, (samples, steps, accept_prob, diverging) = self._run_segments(
                sample_program, states, self._sample_segment, self.options["num_samples"],
                f"{label} sample".strip(), show, collect=True,
                args=(t, padded_yerr, padded_y, varying_kwargs, dynamic_shared, *transform_args),
                synchronize=self.options.get('synchronize_segments', True),
            )
            if not laplace:
                samples = self._postprocess_program(
                    samples, t, padded_yerr, padded_y, varying_kwargs, dynamic_shared,
                )
        if reuse_supports and not (bool(jnp.all(jnp.isfinite(states.potential_energy)))
                and all(bool(jnp.all(jnp.isfinite(value))) for value in jax.tree.leaves(states.z_grad))):
            raise RuntimeError('Independent sampler received a non-finite posterior target.')
        self._continuation = (states, (t, padded_yerr, padded_y, varying_kwargs, dynamic_shared, *transform_args),
                              num_channels, map_diagnostics)
        return self._gather_lanes(self._format_execution_result(
            samples, steps, accept_prob, diverging, states, map_diagnostics, num_channels))

    def _place_lanes(self, lanes):
        """Pad lane-leading leaves to the device width and split them."""
        sharding = NamedSharding(self._mesh, _LANES)
        return jax.tree.map(
            lambda value: jax.device_put(_pad_first_axis(value, self._device_width), sharding),
            lanes,
        )

    def _gather_lanes(self, result):
        """Return sharded results as uncommitted default-device arrays.

        Unsharded runs return uncommitted arrays; arrays committed to the mesh
        could not be combined with results of other runners or stages.
        """
        if self._mesh is None:
            return result
        return jax.tree.map(lambda value: jnp.asarray(np.asarray(value)), result)

    @staticmethod
    def _format_execution_result(samples, steps, accept_prob, diverging, states,
                                 map_diagnostics, num_channels):
        samples = _squeeze_internal_channel_axis(samples)
        samples = jax.tree.map(lambda value: value[:, :num_channels], samples)
        raw = {
            "num_steps": steps[:, :num_channels],
            "accept_prob": accept_prob[:, :num_channels],
            "diverging": diverging[:, :num_channels],
            "step_size": states.adapt_state.step_size[:num_channels],
            "mean_accept_prob": states.mean_accept_prob[:num_channels],
        }
        if map_diagnostics is not None:
            (
                gradient_norm,
                decrement,
                iterations,
                condition,
                minimum_eigenvalue,
                relative_error,
            ) = map_diagnostics
            raw.update(
                map_gradient_norm=gradient_norm[:num_channels],
                map_newton_decrement=decrement[:num_channels],
                map_iterations=iterations[:num_channels],
                map_condition_number=condition[:num_channels],
                map_hessian_min_eigenvalue=minimum_eigenvalue[:num_channels],
                map_hessian_relative_error=relative_error[:num_channels],
            )
        return samples, raw


class IndependentNUTSRunner(_IndependentSamplerRunner):
    """Reusable independently adapted NUTS runner for one padded width."""

    def __init__(
        self, model, *, options, lane_width, channel_varying_kwargs=(), lane_devices=None,
    ):
        super().__init__(
            model, algo='NUTS', options=options, lane_width=lane_width,
            channel_varying_kwargs=channel_varying_kwargs, lane_devices=lane_devices,
        )


def build_independent_nuts_runner(
    model, *, nuts_kwargs=None, mcmc_kwargs=None, num_warmup=None, num_samples=None,
    lane_width, channel_varying_kwargs=(), dense_mass=None, regularize_mass_matrix=None,
    target_accept_prob=None, max_tree_depth=None, mass_matrix=None, laplace_warmup=None,
    laplace_target_accept=None, laplace_max_tree_depth=None, laplace_start_at_map=None,
    lane_devices=None,
):
    """Build a lazily compiled NUTS runner reusable at one lane width.

    ``lane_devices=None`` splits lanes across every CPU host device when the
    default backend is CPU; ``1`` keeps the single-device programs.
    """
    options = _resolve_sampler_options(
        num_warmup=num_warmup, num_samples=num_samples, dense_mass=dense_mass,
        regularize_mass_matrix=regularize_mass_matrix, target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth, mass_matrix=mass_matrix, laplace_warmup=laplace_warmup,
        laplace_target_accept=laplace_target_accept,
        laplace_max_tree_depth=laplace_max_tree_depth,
        laplace_start_at_map=laplace_start_at_map, nuts_kwargs=nuts_kwargs,
        mcmc_kwargs=mcmc_kwargs,
    )
    if options["num_warmup"] < 0:
        raise ValueError("num_warmup must be >= 0.")
    if options["num_samples"] < 1:
        raise ValueError("num_samples must be >= 1.")
    return IndependentNUTSRunner(
        model, options=options, lane_width=lane_width,
        channel_varying_kwargs=channel_varying_kwargs, lane_devices=lane_devices,
    )


def get_samples_independent(
    model: Callable, key: jax.Array, t, yerr, indiv_y, init_params: Mapping[str, Any], *,
    nuts_kwargs: Mapping[str, Any] | None=None, mcmc_kwargs: Mapping[str, Any] | None=None,
    diagnostics_path: str | None=None, num_warmup: int | None=None,
    num_samples: int | None=None, lane_width: int | None=None,
    channel_varying_kwargs: tuple[str, ...]=(), dense_mass: bool | None=None,
    regularize_mass_matrix: bool | None=None, target_accept_prob: float | None=None,
    max_tree_depth: int | None=None, mass_matrix: str | None=None,
    laplace_warmup: int | None=None, laplace_target_accept: float | None=None,
    laplace_max_tree_depth: int | None=None, laplace_start_at_map: bool | None=None,
    return_diagnostics: bool=False, _runner: IndependentNUTSRunner | None=None, **model_kwargs,
):
    """Sample channel-factorized posteriors with reusable vmapped NUTS.

    Existing callers need not construct a runner. ``koala.pipeline`` supplies the
    private ``_runner`` hook for equal-width chunks so their compiled programs
    are reused while all per-chunk numerical values remain dynamic.
    """
    yerr_array = _asarray_f64(yerr)
    if not hasattr(yerr_array, "ndim") or yerr_array.ndim != 2:
        raise ValueError("yerr and indiv_y must have shape [channel, time].")
    num_channels = int(yerr_array.shape[0])
    resolved_width = num_channels if lane_width is None else int(lane_width)
    options = _resolve_sampler_options(
        num_warmup=num_warmup, num_samples=num_samples, dense_mass=dense_mass,
        regularize_mass_matrix=regularize_mass_matrix, target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth, mass_matrix=mass_matrix, laplace_warmup=laplace_warmup,
        laplace_target_accept=laplace_target_accept,
        laplace_max_tree_depth=laplace_max_tree_depth,
        laplace_start_at_map=laplace_start_at_map, nuts_kwargs=nuts_kwargs,
        mcmc_kwargs=mcmc_kwargs,
    )
    if options["num_warmup"] < 0:
        raise ValueError("num_warmup must be >= 0.")
    if options["num_samples"] < 1:
        raise ValueError("num_samples must be >= 1.")
    if _runner is None:
        runner = IndependentNUTSRunner(
            model, options=options, lane_width=resolved_width,
            channel_varying_kwargs=channel_varying_kwargs,
        )
    else:
        runner = _runner
        runner.validate_configuration(
            model, options=options, lane_width=resolved_width,
            channel_varying_kwargs=channel_varying_kwargs,
        )
    samples, raw = runner.run_raw(key, t, yerr_array, indiv_y, init_params, model_kwargs)
    from .mcmc_diagnostics import PosteriorSamples
    sampled_sites = tuple(name for name, site in runner._initialization_model_info.model_trace.items()
        if site['type'] == 'sample' and not site['is_observed'] and not site['fn'].support.is_discrete)
    samples = PosteriorSamples(samples, num_chains=options["num_chains"], sampled_sites=sampled_sites)
    diagnostics = IndependentNUTSDiagnostics(**raw)
    _save_diagnostics(diagnostics_path, diagnostics)
    if return_diagnostics:
        return samples, diagnostics
    return samples
