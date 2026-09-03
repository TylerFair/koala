"""GPU-batched, independently adapted NUTS for factorized light curves.

The existing spectroscopic models vectorize the forward model over channels,
but a normal NumPyro ``MCMC`` call still treats the whole block as one
high-dimensional NUTS state.  This module instead runs one low-dimensional
NUTS state per channel and vmaps those states over a fixed-width GPU batch.

This backend is deliberately opt-in.  It is only valid when all latent sample
sites factorize by channel; shared *fixed* model arguments are fine, but shared
latent variables are not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from numpyro.distributions.transforms import biject_to
from numpyro.infer.hmc import hmc
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model


@dataclass(frozen=True)
class IndependentNUTSDiagnostics:
    """Per-draw and final adapted diagnostics for every real channel."""

    num_steps: jax.Array
    accept_prob: jax.Array
    diverging: jax.Array
    step_size: jax.Array
    mean_accept_prob: jax.Array
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


def prepare_laplace_metric(
    model,
    rng_key,
    init_values,
    *model_args,
    model_kwargs=None,
    hessian_method="finite_difference",
    trust_radius=5.0,
    max_iterations=200,
    gradient_tolerance=1.0e-4,
    decrement_tolerance=1.0e-4,
    eigenvalue_floor=1.0e-8,
    fd_relative_step=2.0e-4,
    line_search_steps=8,
):
    """Prepare a dense Laplace metric for one arbitrary NumPyro posterior.

    This is the scalar-posterior counterpart of the per-lane preparation used
    by :class:`IndependentNUTSRunner`.  It deliberately returns NumPyro's
    unconstrained pytree so callers can pass it as ``MCMC.run(init_params=...)``.
    """
    method = str(hessian_method).lower()
    if method not in {"exact", "finite_difference"}:
        raise ValueError("hessian_method must be 'exact' or 'finite_difference'.")
    if trust_radius <= 0 or max_iterations < 1 or line_search_steps < 1:
        raise ValueError("Invalid Laplace MAP iteration controls.")
    model_kwargs = dict(model_kwargs or {})
    info = initialize_model(
        rng_key,
        model,
        init_strategy=init_to_value(values=init_values),
        model_args=model_args,
        model_kwargs=model_kwargs,
    )
    z0 = info.param_info.z
    flat0, unravel = ravel_pytree(z0)
    potential = lambda flat: info.potential_fn(unravel(flat))
    value_and_grad = jax.value_and_grad(potential)
    gradient_fn = jax.grad(potential)

    def central_gradient_hessian(flat, steps):
        offsets = jnp.eye(flat.shape[0], dtype=flat.dtype) * steps[:, None]
        plus = jax.vmap(lambda offset: gradient_fn(flat + offset))(offsets)
        minus = jax.vmap(lambda offset: gradient_fn(flat - offset))(offsets)
        estimate = (plus - minus) / (2.0 * steps[:, None])
        return 0.5 * (estimate + estimate.T)

    def repair(hessian):
        hessian = jnp.nan_to_num(0.5 * (hessian + hessian.T))
        values, vectors = jnp.linalg.eigh(hessian)
        scale = jnp.maximum(jnp.max(jnp.abs(values)), 1.0)
        repaired = jnp.maximum(values, eigenvalue_floor * scale)
        covariance = (vectors / repaired[None, :]) @ vectors.T
        return covariance, repaired[-1] / repaired[0], values[0]

    if method == "exact":
        hessian_fn = jax.hessian(potential)
    else:
        initial_steps = fd_relative_step * jnp.maximum(1.0, jnp.abs(flat0))
        initial_hessian = central_gradient_hessian(flat0, initial_steps)
        coordinate_scale = jnp.clip(
            1.0 / jnp.sqrt(jnp.maximum(jnp.abs(jnp.diag(initial_hessian)), 1e-12)),
            0.05,
            20.0,
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
        candidate_values = jax.vmap(potential)(candidates)
        candidate_values = jnp.where(jnp.isfinite(candidate_values), candidate_values, jnp.inf)
        best = jnp.argmin(candidate_values)
        accept = candidate_values[best] < value
        next_flat = jnp.where(accept, candidates[best], flat)
        next_value, next_gradient = value_and_grad(next_flat)
        decrement = jnp.sqrt(jnp.maximum(next_gradient @ covariance @ next_gradient, 0.0))
        return index + 1, next_flat, next_value, next_gradient, iterations + 1, decrement

    _, flat, _, gradient, iterations, _ = jax.lax.while_loop(
        condition,
        iteration,
        (jnp.asarray(0, jnp.int32), flat0, value0, gradient0,
         jnp.asarray(0, jnp.int32), jnp.asarray(jnp.inf, flat0.dtype)),
    )
    covariance, condition_number, minimum_eigenvalue = repair(hessian_fn(flat))
    decrement = jnp.sqrt(jnp.maximum(gradient @ covariance @ gradient, 0.0))
    return LaplaceMetricPreparation(
        unconstrained_map=unravel(flat),
        inverse_mass_matrix=covariance,
        gradient_norm=jnp.linalg.norm(gradient),
        newton_decrement=decrement,
        iterations=iterations,
        condition_number=condition_number,
        hessian_min_eigenvalue=minimum_eigenvalue,
    )


def _pad_first_axis(value, size):
    """Pad an array by repeating its final entry."""
    value = jnp.asarray(value)
    missing = size - value.shape[0]
    if missing <= 0:
        return value
    padding = jnp.repeat(value[-1:], missing, axis=0)
    return jnp.concatenate((value, padding), axis=0)


def _partition_model_kwargs(
    model_kwargs,
    channel_varying_kwargs,
    num_channels,
    lane_width,
):
    """Split kwargs into channel-varying leaves and shared leaves.

    Channel-varying arguments follow the convention already used by
    ``fit_jwst._slice_by_channel``: their first dimension equals the number of
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
    """Add aliases used by fit_jwst but not named latent sample sites."""
    enriched = {name: _asarray_f64(value) for name, value in init_params.items()}
    if "u" in enriched:
        u = enriched["u"]
        if hasattr(u, "ndim") and u.ndim >= 2 and u.shape[-1] >= 2:
            enriched.setdefault("c1", u[..., 0])
            enriched.setdefault("c2", u[..., 1])
    if "tau" in enriched:
        enriched.setdefault("log_tau", jnp.log(enriched["tau"]))
    return enriched


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
    init_params,
    model_trace,
    num_channels,
    lane_width,
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
            name == "ld_decorrelated"
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
            from .ld_parameterization import (
                Power2MaxtedTransform,
                Power2LinearTransform,
                QuadraticKippingTransform,
            )

            physical = jnp.stack(
                (init_params["c1"], init_params["c2"]), axis=-1
            )
            distribution_transforms = getattr(site["fn"], "transforms", ())
            parameterization = next(
                (
                    transform
                    for transform in distribution_transforms
                    if isinstance(
                        transform,
                        (
                            Power2MaxtedTransform,
                            Power2LinearTransform,
                            QuadraticKippingTransform,
                        ),
                    )
                ),
                None,
            )
            if parameterization is None:
                raise ValueError(
                    "Could not identify the limb-darkening transform for "
                    "the ld_decorrelated site."
                )
            constrained = _pad_first_axis(
                parameterization(physical), lane_width
            )[:, None, ...]
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
            constrained = jnp.broadcast_to(
                prototype,
                (lane_width,) + prototype.shape,
            )

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
    *,
    num_warmup,
    num_samples,
    dense_mass,
    regularize_mass_matrix,
    target_accept_prob,
    max_tree_depth,
    mass_matrix,
    laplace_warmup,
    laplace_target_accept,
    laplace_max_tree_depth,
    laplace_start_at_map,
    nuts_kwargs,
    mcmc_kwargs,
):
    nuts = dict(nuts_kwargs or {})
    mcmc = dict(mcmc_kwargs or {})
    # This backend always initializes from the supplied physical values.
    nuts.pop("init_strategy", None)
    supported_nuts = {
        "dense_mass",
        "regularize_mass_matrix",
        "target_accept_prob",
        "max_tree_depth",
        "mass_matrix",
        "laplace_warmup",
        "laplace_target_accept",
        "laplace_max_tree_depth",
        "laplace_start_at_map",
        "laplace_map_iterations",
        "laplace_map_tolerance",
        "laplace_map_decrement_tolerance",
        "laplace_eigenvalue_floor",
        "laplace_hessian_method",
        "laplace_fd_relative_step",
        "laplace_fd_batch_size",
        "laplace_compare_exact_hessian",
        "laplace_map_method",
        "laplace_line_search_steps",
        "laplace_trust_radius",
        "laplace_fuse_program",
    }
    unknown_nuts = set(nuts) - supported_nuts
    if unknown_nuts:
        raise ValueError(
            "Unsupported independent-NUTS options: "
            + ", ".join(sorted(unknown_nuts))
        )
    supported_mcmc = {
        "num_warmup",
        "num_samples",
        "progress_bar",
        "jit_model_args",
    }
    unknown_mcmc = set(mcmc) - supported_mcmc
    if unknown_mcmc:
        raise ValueError(
            "Unsupported independent-MCMC options: "
            + ", ".join(sorted(unknown_mcmc))
        )

    resolved_mass_matrix = str(
        mass_matrix
        if mass_matrix is not None
        else nuts.get("mass_matrix", "adaptive")
    ).lower()
    if resolved_mass_matrix not in {"adaptive", "laplace"}:
        raise ValueError(
            "mass_matrix must be either 'adaptive' or 'laplace'; got "
            f"{resolved_mass_matrix!r}."
        )

    result = {
        "num_warmup": int(
            num_warmup
            if num_warmup is not None
            else mcmc.get("num_warmup", 1000)
        ),
        "num_samples": int(
            num_samples
            if num_samples is not None
            else mcmc.get("num_samples", 1000)
        ),
        "dense_mass": bool(
            dense_mass
            if dense_mass is not None
            else nuts.get("dense_mass", True)
        ),
        "regularize_mass_matrix": bool(
            regularize_mass_matrix
            if regularize_mass_matrix is not None
            else nuts.get("regularize_mass_matrix", True)
        ),
        "target_accept_prob": float(
            target_accept_prob
            if target_accept_prob is not None
            else nuts.get("target_accept_prob", 0.8)
        ),
        "max_tree_depth": int(
            max_tree_depth
            if max_tree_depth is not None
            else nuts.get("max_tree_depth", 10)
        ),
        "mass_matrix": resolved_mass_matrix,
        "laplace_warmup": int(
            laplace_warmup
            if laplace_warmup is not None
            else nuts.get("laplace_warmup", 150)
        ),
        "laplace_target_accept": float(
            laplace_target_accept
            if laplace_target_accept is not None
            else nuts.get("laplace_target_accept", 0.95)
        ),
        "laplace_max_tree_depth": int(
            laplace_max_tree_depth
            if laplace_max_tree_depth is not None
            else nuts.get("laplace_max_tree_depth", 10)
        ),
        "laplace_start_at_map": bool(
            laplace_start_at_map
            if laplace_start_at_map is not None
            else nuts.get("laplace_start_at_map", False)
        ),
        # These expert controls are intentionally not pipeline flags. They
        # keep the MAP solver static and testable without changing its
        # production defaults.
        "laplace_map_iterations": int(nuts.get("laplace_map_iterations", 200)),
        "laplace_map_tolerance": float(
            nuts.get("laplace_map_tolerance", 1.0e-4)
        ),
        "laplace_map_decrement_tolerance": float(
            nuts.get("laplace_map_decrement_tolerance", 1.0e-4)
        ),
        "laplace_eigenvalue_floor": float(
            nuts.get("laplace_eigenvalue_floor", 1.0e-8)
        ),
        "laplace_hessian_method": str(
            nuts.get("laplace_hessian_method", "exact")
        ).lower(),
        "laplace_fd_relative_step": float(
            nuts.get("laplace_fd_relative_step", 2.0e-4)
        ),
        # Zero requests the historical all-coordinate vmap.  A positive
        # value lowers the finite-difference coordinate axis in bounded
        # batches, reducing the simultaneous cadence-sized gradient buffers.
        "laplace_fd_batch_size": int(nuts.get("laplace_fd_batch_size", 1)),
        "laplace_compare_exact_hessian": bool(
            nuts.get("laplace_compare_exact_hessian", False)
        ),
        "laplace_map_method": str(
            nuts.get("laplace_map_method", "newton")
        ).lower(),
        "laplace_line_search_steps": int(
            nuts.get("laplace_line_search_steps", 8)
        ),
        "laplace_trust_radius": float(nuts.get("laplace_trust_radius", 5.0)),
        "laplace_fuse_program": bool(
            nuts.get("laplace_fuse_program", False)
        ),
    }
    if result["laplace_warmup"] < 0:
        raise ValueError("laplace_warmup must be >= 0.")
    if result["laplace_map_iterations"] < 1:
        raise ValueError("laplace_map_iterations must be >= 1.")
    if not 0.0 < result["laplace_target_accept"] < 1.0:
        raise ValueError("laplace_target_accept must be between 0 and 1.")
    if result["laplace_max_tree_depth"] < 1:
        raise ValueError("laplace_max_tree_depth must be >= 1.")
    if result["laplace_map_tolerance"] <= 0.0:
        raise ValueError("laplace_map_tolerance must be > 0.")
    if result["laplace_map_decrement_tolerance"] < 0.0:
        raise ValueError("laplace_map_decrement_tolerance must be >= 0.")
    if result["laplace_eigenvalue_floor"] <= 0.0:
        raise ValueError("laplace_eigenvalue_floor must be > 0.")
    if result["laplace_hessian_method"] not in {
        "exact",
        "finite_difference",
    }:
        raise ValueError(
            "laplace_hessian_method must be 'exact' or "
            "'finite_difference'."
        )
    if result["laplace_fd_relative_step"] <= 0.0:
        raise ValueError("laplace_fd_relative_step must be > 0.")
    if result["laplace_fd_batch_size"] < 0:
        raise ValueError("laplace_fd_batch_size must be >= 0.")
    if result["laplace_map_method"] not in {"newton", "diagonal"}:
        raise ValueError(
            "laplace_map_method must be 'newton' or 'diagonal'."
        )
    if result["laplace_line_search_steps"] < 1:
        raise ValueError("laplace_line_search_steps must be >= 1.")
    if result["laplace_trust_radius"] <= 0.0:
        raise ValueError("laplace_trust_radius must be > 0.")
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
        "num_divergences": int(diverging.sum()),
        "num_divergences_per_channel": diverging.sum(axis=0).tolist(),
        "mean_num_steps_per_channel": steps.mean(axis=0).tolist(),
        "max_num_steps_per_channel": steps.max(axis=0).tolist(),
        "mean_accept_prob_per_channel": accept.mean(axis=0).tolist(),
        "adapted_step_size_per_channel": step_size.tolist(),
    }
    map_fields = {
        "map_gradient_norm_per_channel": diagnostics.map_gradient_norm,
        "map_newton_decrement_per_channel": diagnostics.map_newton_decrement,
        "map_iterations_per_channel": diagnostics.map_iterations,
        "map_condition_number_per_channel": diagnostics.map_condition_number,
        "map_hessian_min_eigenvalue_per_channel": (
            diagnostics.map_hessian_min_eigenvalue
        ),
        "map_hessian_relative_error_per_channel": (
            diagnostics.map_hessian_relative_error
        ),
    }
    for name, value in map_fields.items():
        if value is not None:
            payload[name] = jax.device_get(value).tolist()
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
        self,
        model,
        *,
        algo,
        options,
        lane_width,
        channel_varying_kwargs=(),
    ):
        self.model = model
        self.algo = str(algo)
        self.options = dict(options)
        self.lane_width = int(lane_width)
        self.channel_varying_kwargs = tuple(channel_varying_kwargs)
        self.program_build_count = 0
        self._static_shared_kwargs = None
        self._init_program = None
        self._warmup_program = None
        self._sample_program = None
        self._postprocess_program = None
        self._laplace_sample_program = None
        self._laplace_full_program = None

    def validate_configuration(
        self,
        model,
        *,
        options,
        lane_width,
        channel_varying_kwargs,
    ):
        if model is not self.model:
            raise ValueError("Independent sampler runner belongs to another model.")
        if dict(options) != self.options:
            raise ValueError(
                "Independent sampler runner options differ from this call."
            )
        if int(lane_width) != self.lane_width:
            raise ValueError(
                "Independent sampler runner lane width differs from this call."
            )
        if tuple(channel_varying_kwargs) != self.channel_varying_kwargs:
            raise ValueError(
                "Independent sampler runner channel-varying arguments differ "
                "from this call."
            )

    @staticmethod
    def _merge_lane_kwargs(static_shared, dynamic_shared, varying_lane):
        return {**static_shared, **dynamic_shared, **varying_lane}

    def _build_laplace_programs(
        self,
        model_info,
        static_shared_kwargs,
        initial_prototype,
    ):
        """Build the fused programs used by Laplace-preconditioned HMC/NUTS."""
        self._static_shared_kwargs = dict(static_shared_kwargs)
        init_kernel, sample_kernel = hmc(
            potential_fn_gen=model_info.potential_fn,
            algo=self.algo,
        )
        options = self.options
        static_shared = self._static_shared_kwargs
        varying_axes = {name: 0 for name in self.channel_varying_kwargs}
        _, unravel = ravel_pytree(initial_prototype)
        line_search_scales = jnp.power(
            jnp.asarray(0.5, dtype=jnp.float64),
            jnp.arange(
                options["laplace_line_search_steps"], dtype=jnp.float64
            ),
        )

        def repaired_inverse_hessian(hessian):
            hessian = 0.5 * (hessian + hessian.T)
            hessian = jnp.nan_to_num(
                hessian,
                nan=0.0,
                posinf=1.0e12,
                neginf=-1.0e12,
            )
            eigenvalues, eigenvectors = jnp.linalg.eigh(hessian)
            scale = jnp.maximum(jnp.max(jnp.abs(eigenvalues)), 1.0)
            floor = options["laplace_eigenvalue_floor"] * scale
            repaired = jnp.maximum(eigenvalues, floor)
            covariance = (
                (eigenvectors / repaired[None, :]) @ eigenvectors.T
            )
            condition = jnp.max(repaired) / jnp.min(repaired)
            return covariance, condition, jnp.min(eigenvalues)

        def map_one(z0, err, obs, varying, t, dynamic_shared):
            kwargs = self._merge_lane_kwargs(
                static_shared, dynamic_shared, varying
            )
            potential = model_info.potential_fn(
                t,
                err,
                y=obs,
                **kwargs,
            )
            value_and_grad = jax.value_and_grad(
                lambda flat: potential(unravel(flat))
            )
            gradient_fn = jax.grad(lambda flat: potential(unravel(flat)))
            flat0, _ = ravel_pytree(z0)
            value0, gradient0 = value_and_grad(flat0)

            def central_gradient_hessian(flat, coordinate_steps):
                offsets = jnp.eye(flat.shape[0], dtype=flat.dtype)
                offsets = offsets * coordinate_steps[:, None]
                batch_size = options["laplace_fd_batch_size"]
                if batch_size:
                    plus = jax.lax.map(
                        lambda offset: gradient_fn(flat + offset),
                        offsets,
                        batch_size=batch_size,
                    )
                    minus = jax.lax.map(
                        lambda offset: gradient_fn(flat - offset),
                        offsets,
                        batch_size=batch_size,
                    )
                else:
                    plus = jax.vmap(
                        lambda offset: gradient_fn(flat + offset)
                    )(offsets)
                    minus = jax.vmap(
                        lambda offset: gradient_fn(flat - offset)
                    )(offsets)
                estimate = (plus - minus) / (2.0 * coordinate_steps[:, None])
                return 0.5 * (estimate + estimate.T)

            if options["laplace_hessian_method"] == "exact":
                hessian_fn = jax.hessian(
                    lambda flat: potential(unravel(flat))
                )
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
                first_steps = relative_step * jnp.maximum(
                    1.0, jnp.abs(flat0)
                )
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
                    covariance, _, _ = repaired_inverse_hessian(
                        hessian_fn(flat)
                    )
                    newton = -(covariance @ gradient)
                else:
                    covariance = jnp.diag(fixed_diagonal_covariance)
                    newton = -(fixed_diagonal_covariance * gradient)
                newton_norm = jnp.linalg.norm(newton)
                # A trust radius prevents a repaired, nearly-flat Hessian
                # direction from proposing a huge unconstrained jump.
                newton = newton * jnp.minimum(
                    1.0,
                    options["laplace_trust_radius"]
                    / jnp.maximum(newton_norm, 1.0e-12),
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
                candidate_values = jax.vmap(
                    lambda candidate: potential(unravel(candidate))
                )(candidates)
                candidate_values = jnp.where(
                    jnp.isfinite(candidate_values),
                    candidate_values,
                    jnp.inf,
                )
                best_index = jnp.argmin(candidate_values)
                best_flat = candidates[best_index]
                best_value = candidate_values[best_index]
                accept = active & (best_value < value)
                next_flat = jnp.where(accept, best_flat, flat)
                next_value, next_gradient = value_and_grad(next_flat)
                decrement_estimate = jnp.sqrt(
                    jnp.maximum(
                        next_gradient @ covariance @ next_gradient,
                        0.0,
                    )
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
            decrement = jnp.sqrt(
                jnp.maximum(gradient @ covariance @ gradient, 0.0)
            )
            if (
                options["laplace_hessian_method"] == "finite_difference"
                and options["laplace_compare_exact_hessian"]
            ):
                exact_hessian = jax.hessian(
                    lambda value: potential(unravel(value))
                )(flat)
                relative_error = jnp.linalg.norm(
                    hessian - exact_hessian
                ) / jnp.maximum(jnp.linalg.norm(exact_hessian), 1.0e-30)
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

        map_lanes = jax.vmap(
            map_one,
            in_axes=(0, 0, 0, varying_axes, None, None),
        )

        def advance(states, t, errors, observations, varying, dynamic_shared):
            def advance_one(state, err, obs, varying_lane):
                kwargs = self._merge_lane_kwargs(
                    static_shared, dynamic_shared, varying_lane
                )
                if (
                    self.algo == "HMC"
                    and options.get("trajectory_jitter", 0.0) > 0.0
                ):
                    jitter_key, transition_key = jax.random.split(
                        state.rng_key
                    )
                    spread = max(
                        1,
                        int(
                            round(
                                options["num_steps"]
                                * options["trajectory_jitter"]
                            )
                        ),
                    )
                    desired_steps = jax.random.randint(
                        jitter_key,
                        (),
                        max(1, options["num_steps"] - spread),
                        options["num_steps"] + spread + 1,
                    )
                    trajectory_length = state.adapt_state.step_size * (
                        desired_steps.astype(jnp.float64) - 1.0e-7
                    )
                    state = state._replace(
                        rng_key=transition_key,
                        trajectory_length=trajectory_length,
                    )
                return sample_kernel(
                    state,
                    model_args=(t, err),
                    model_kwargs={"y": obs, **kwargs},
                )

            return jax.vmap(
                advance_one,
                in_axes=(0, 0, 0, varying_axes),
            )(states, errors, observations, varying)

        def prepare(
            batched_init,
            errors,
            observations,
            varying,
            keys,
            t,
            dynamic_shared,
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
                batched_init,
                errors,
                observations,
                varying,
                t,
                dynamic_shared,
            )
            map_z = jax.vmap(unravel)(map_flat)
            start_z = map_z if options["laplace_start_at_map"] else batched_init

            def init_one(z, inverse_mass, err, obs, varying_lane, rng):
                kwargs = self._merge_lane_kwargs(
                    static_shared, dynamic_shared, varying_lane
                )
                common = {
                    "num_warmup": options["laplace_warmup"],
                    "step_size": 1.0,
                    "inverse_mass_matrix": inverse_mass,
                    "adapt_step_size": True,
                    "adapt_mass_matrix": False,
                    "dense_mass": True,
                    "target_accept_prob": options["laplace_target_accept"],
                    "find_heuristic_step_size": False,
                    "regularize_mass_matrix": False,
                    "model_args": (t, err),
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
                            num_steps=None,
                            trajectory_length=float(options["num_steps"]),
                        )
                    else:
                        common.update(
                            num_steps=options["num_steps"],
                            trajectory_length=None,
                        )
                return init_kernel(z, **common)

            states = jax.vmap(
                init_one,
                in_axes=(0, 0, 0, 0, varying_axes, 0),
            )(
                start_z,
                covariance,
                errors,
                observations,
                varying,
                keys,
            )
            if options["laplace_warmup"]:
                states = jax.lax.fori_loop(
                    0,
                    options["laplace_warmup"],
                    lambda _, state: advance(
                        state,
                        t,
                        errors,
                        observations,
                        varying,
                        dynamic_shared,
                    ),
                    states,
                )
            return states, (
                gradient_norm,
                decrement,
                iterations,
                condition,
                minimum_eigenvalue,
                relative_error,
            )

        self._init_program = jax.jit(prepare)
        postprocess_gen = model_info.postprocess_fn

        def postprocess_lane(z, t, err, obs, varying, dynamic_shared):
            kwargs = self._merge_lane_kwargs(
                static_shared, dynamic_shared, varying
            )
            return postprocess_gen(t, err, y=obs, **kwargs)(z)

        postprocess_lanes = jax.vmap(
            postprocess_lane,
            in_axes=(0, None, 0, 0, varying_axes, None),
        )

        def sample_and_postprocess(
            initial,
            t,
            errors,
            observations,
            varying,
            dynamic_shared,
        ):
            def collect_step(state, _):
                state = advance(
                    state,
                    t,
                    errors,
                    observations,
                    varying,
                    dynamic_shared,
                )
                samples = postprocess_lanes(
                    state.z,
                    t,
                    errors,
                    observations,
                    varying,
                    dynamic_shared,
                )
                return state, (
                    samples,
                    state.num_steps,
                    state.accept_prob,
                    state.diverging,
                )

            return jax.lax.scan(
                collect_step,
                initial,
                xs=None,
                length=options["num_samples"],
            )

        self._laplace_sample_program = jax.jit(sample_and_postprocess)
        if options["laplace_fuse_program"]:
            def prepare_sample_and_postprocess(
                batched_init,
                errors,
                observations,
                varying,
                keys,
                t,
                dynamic_shared,
            ):
                states, map_diagnostics = prepare(
                    batched_init,
                    errors,
                    observations,
                    varying,
                    keys,
                    t,
                    dynamic_shared,
                )
                states, outputs = sample_and_postprocess(
                    states,
                    t,
                    errors,
                    observations,
                    varying,
                    dynamic_shared,
                )
                return states, outputs, map_diagnostics

            self._laplace_full_program = jax.jit(
                prepare_sample_and_postprocess
            )
        self.program_build_count += 1

    def _build_programs(
        self,
        model_info,
        static_shared_kwargs,
        initial_prototype=None,
    ):
        if self.options.get("mass_matrix") == "laplace":
            if initial_prototype is None:
                raise ValueError("Laplace NUTS requires an initial state prototype.")
            self._build_laplace_programs(
                model_info,
                static_shared_kwargs,
                initial_prototype,
            )
            return
        self._static_shared_kwargs = dict(static_shared_kwargs)
        init_kernel, sample_kernel = hmc(
            potential_fn_gen=model_info.potential_fn,
            algo=self.algo,
        )
        options = self.options
        static_shared = self._static_shared_kwargs
        varying_axes = {
            name: 0 for name in self.channel_varying_kwargs
        }

        def init_one(z, err, obs, varying, rng, t, dynamic_shared):
            kwargs = self._merge_lane_kwargs(
                static_shared, dynamic_shared, varying
            )
            common = {
                "num_warmup": options["num_warmup"],
                "dense_mass": options["dense_mass"],
                "regularize_mass_matrix": options[
                    "regularize_mass_matrix"
                ],
                "target_accept_prob": options["target_accept_prob"],
                "model_args": (t, err),
                "model_kwargs": {"y": obs, **kwargs},
                "rng_key": rng,
            }
            if self.algo == "NUTS":
                common["max_tree_depth"] = options["max_tree_depth"]
            else:
                common.update(
                    num_steps=options["num_steps"],
                    trajectory_length=None,
                )
            return init_kernel(z, **common)

        self._init_program = jax.jit(
            jax.vmap(
                init_one,
                in_axes=(0, 0, 0, varying_axes, 0, None, None),
            )
        )

        def advance(states, t, errors, observations, varying, dynamic_shared):
            def advance_one(state, err, obs, varying_lane):
                kwargs = self._merge_lane_kwargs(
                    static_shared, dynamic_shared, varying_lane
                )
                return sample_kernel(
                    state,
                    model_args=(t, err),
                    model_kwargs={"y": obs, **kwargs},
                )

            return jax.vmap(
                advance_one,
                in_axes=(0, 0, 0, varying_axes),
            )(states, errors, observations, varying)

        if options["num_warmup"]:
            self._warmup_program = jax.jit(
                lambda initial, t, errors, observations, varying, shared: (
                    jax.lax.fori_loop(
                        0,
                        options["num_warmup"],
                        lambda _, state: advance(
                            state, t, errors, observations, varying, shared
                        ),
                        initial,
                    )
                )
            )

        def collect(initial, t, errors, observations, varying, shared):
            def collect_step(state, _):
                state = advance(
                    state, t, errors, observations, varying, shared
                )
                return state, (
                    state.z,
                    state.num_steps,
                    state.accept_prob,
                    state.diverging,
                )

            return jax.lax.scan(
                collect_step,
                initial,
                xs=None,
                length=options["num_samples"],
            )

        self._sample_program = jax.jit(collect)
        postprocess_gen = model_info.postprocess_fn

        def postprocess_lane(z, t, err, obs, varying, dynamic_shared):
            kwargs = self._merge_lane_kwargs(
                static_shared, dynamic_shared, varying
            )
            return postprocess_gen(t, err, y=obs, **kwargs)(z)

        postprocess_lanes = jax.vmap(
            postprocess_lane,
            in_axes=(0, None, 0, 0, varying_axes, None),
        )
        self._postprocess_program = jax.jit(
            jax.vmap(
                postprocess_lanes,
                in_axes=(0, None, None, None, None, None),
            )
        )
        self.program_build_count += 1

    def run_raw(self, key, t, yerr, indiv_y, init_params, model_kwargs):
        t = _asarray_f64(t)
        yerr = _asarray_f64(yerr)
        indiv_y = _asarray_f64(indiv_y)
        if yerr.ndim != 2 or indiv_y.ndim != 2:
            raise ValueError(
                "yerr and indiv_y must have shape [channel, time]."
            )
        if yerr.shape != indiv_y.shape:
            raise ValueError("yerr and indiv_y must have identical shapes.")
        num_channels = int(yerr.shape[0])
        if num_channels < 1:
            raise ValueError("At least one channel is required.")
        if self.lane_width < num_channels:
            raise ValueError(
                f"lane_width={self.lane_width} cannot hold "
                f"{num_channels} channels."
            )

        padded_yerr = _pad_first_axis(yerr, self.lane_width)[:, None, :]
        padded_y = _pad_first_axis(indiv_y, self.lane_width)[:, None, :]
        varying_kwargs, shared_kwargs = _partition_model_kwargs(
            model_kwargs,
            self.channel_varying_kwargs,
            num_channels,
            self.lane_width,
        )
        dynamic_shared, static_shared = _split_dynamic_and_static_kwargs(
            shared_kwargs
        )
        if (
            self._static_shared_kwargs is not None
            and not _static_kwargs_equal(
                static_shared, self._static_shared_kwargs
            )
        ):
            raise ValueError(
                "A reused independent sampler runner received different "
                "non-array/static model arguments. Build a separate runner "
                "for that static configuration."
            )

        enriched_init = _enrich_initial_values(init_params)
        first_init = _first_lane_initial_values(enriched_init, num_channels)
        first_varying = {
            name: value[0] for name, value in varying_kwargs.items()
        }
        first_kwargs = self._merge_lane_kwargs(
            static_shared, dynamic_shared, first_varying
        )
        key_model, key_chains = jax.random.split(key)

        # Re-trace the lightweight model scaffold for every call so omitted
        # initial sites and parameter-dependent supports also reflect this
        # chunk. The expensive jitted transition programs below retain their
        # identity and executable cache.
        model_info = initialize_model(
            key_model,
            self.model,
            init_strategy=init_to_value(values=first_init),
            dynamic_args=True,
            model_args=(t, padded_yerr[0]),
            model_kwargs={"y": padded_y[0], **first_kwargs},
        )
        batched_init = _prepare_unconstrained_initial_values(
            enriched_init,
            model_info.model_trace,
            num_channels,
            self.lane_width,
        )
        if self._init_program is None:
            initial_prototype = jax.tree.map(
                lambda value: value[0], batched_init
            )
            self._build_programs(
                model_info,
                static_shared,
                initial_prototype=initial_prototype,
            )

        chain_keys = jax.random.split(key_chains, self.lane_width)
        map_diagnostics = None
        if (
            self.options.get("mass_matrix") == "laplace"
            and self._laplace_full_program is not None
        ):
            states, outputs, map_diagnostics = self._laplace_full_program(
                batched_init,
                padded_yerr,
                padded_y,
                varying_kwargs,
                chain_keys,
                t,
                dynamic_shared,
            )
            samples, steps, accept_prob, diverging = outputs
        else:
            prepared = self._init_program(
                batched_init,
                padded_yerr,
                padded_y,
                varying_kwargs,
                chain_keys,
                t,
                dynamic_shared,
            )
            if self.options.get("mass_matrix") == "laplace":
                states, map_diagnostics = prepared
            else:
                states = prepared
            if self._warmup_program is not None:
                states = self._warmup_program(
                    states,
                    t,
                    padded_yerr,
                    padded_y,
                    varying_kwargs,
                    dynamic_shared,
                )
            if self.options.get("mass_matrix") == "laplace":
                states, (samples, steps, accept_prob, diverging) = (
                    self._laplace_sample_program(
                        states,
                        t,
                        padded_yerr,
                        padded_y,
                        varying_kwargs,
                        dynamic_shared,
                    )
                )
            else:
                states, (z_samples, steps, accept_prob, diverging) = (
                    self._sample_program(
                        states,
                        t,
                        padded_yerr,
                        padded_y,
                        varying_kwargs,
                        dynamic_shared,
                    )
                )
                samples = self._postprocess_program(
                    z_samples,
                    t,
                    padded_yerr,
                    padded_y,
                    varying_kwargs,
                    dynamic_shared,
                )
        samples = _squeeze_internal_channel_axis(samples)
        samples = jax.tree.map(
            lambda value: value[:, :num_channels], samples
        )
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
        self,
        model,
        *,
        options,
        lane_width,
        channel_varying_kwargs=(),
    ):
        super().__init__(
            model,
            algo="NUTS",
            options=options,
            lane_width=lane_width,
            channel_varying_kwargs=channel_varying_kwargs,
        )


def build_independent_nuts_runner(
    model,
    *,
    nuts_kwargs=None,
    mcmc_kwargs=None,
    num_warmup=None,
    num_samples=None,
    lane_width,
    channel_varying_kwargs=(),
    dense_mass=None,
    regularize_mass_matrix=None,
    target_accept_prob=None,
    max_tree_depth=None,
    mass_matrix=None,
    laplace_warmup=None,
    laplace_target_accept=None,
    laplace_max_tree_depth=None,
    laplace_start_at_map=None,
):
    """Build a lazily compiled NUTS runner reusable at one lane width."""
    options = _resolve_sampler_options(
        num_warmup=num_warmup,
        num_samples=num_samples,
        dense_mass=dense_mass,
        regularize_mass_matrix=regularize_mass_matrix,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        mass_matrix=mass_matrix,
        laplace_warmup=laplace_warmup,
        laplace_target_accept=laplace_target_accept,
        laplace_max_tree_depth=laplace_max_tree_depth,
        laplace_start_at_map=laplace_start_at_map,
        nuts_kwargs=nuts_kwargs,
        mcmc_kwargs=mcmc_kwargs,
    )
    if options["num_warmup"] < 0:
        raise ValueError("num_warmup must be >= 0.")
    if options["num_samples"] < 1:
        raise ValueError("num_samples must be >= 1.")
    return IndependentNUTSRunner(
        model,
        options=options,
        lane_width=lane_width,
        channel_varying_kwargs=channel_varying_kwargs,
    )


def _get_samples_independent_uncached(
    model: Callable,
    key: jax.Array,
    t,
    yerr,
    indiv_y,
    init_params: Mapping[str, Any],
    *,
    nuts_kwargs: Mapping[str, Any] | None = None,
    mcmc_kwargs: Mapping[str, Any] | None = None,
    diagnostics_path: str | None = None,
    num_warmup: int | None = None,
    num_samples: int | None = None,
    lane_width: int | None = None,
    channel_varying_kwargs: tuple[str, ...] = (),
    dense_mass: bool | None = None,
    regularize_mass_matrix: bool | None = None,
    target_accept_prob: float | None = None,
    max_tree_depth: int | None = None,
    mass_matrix: str | None = None,
    laplace_warmup: int | None = None,
    laplace_target_accept: float | None = None,
    laplace_max_tree_depth: int | None = None,
    laplace_start_at_map: bool | None = None,
    return_diagnostics: bool = False,
    **model_kwargs,
):
    """Sample independent channel posteriors in one vmapped NUTS program.

    The call and returned sample dictionary intentionally mirror
    ``fit_jwst.get_samples``.  Names in ``channel_varying_kwargs`` receive a
    length-one internal channel axis in each lane; all other model arguments
    remain shared.  This explicit list avoids confusing one-planet geometry
    arrays with channel arrays when a block contains one light curve.  The
    optional ``lane_width`` pads a short final block by duplicating its last
    channel, compiles a fixed GPU shape, then discards the dummy lanes.

    Notes
    -----
    Dynamic NUTS trees are vmapped.  XLA masks completed lanes until the
    deepest lane finishes, so similarly difficult channels should be grouped
    together for best throughput.
    """
    options = _resolve_sampler_options(
        num_warmup=num_warmup,
        num_samples=num_samples,
        dense_mass=dense_mass,
        regularize_mass_matrix=regularize_mass_matrix,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        mass_matrix=mass_matrix,
        laplace_warmup=laplace_warmup,
        laplace_target_accept=laplace_target_accept,
        laplace_max_tree_depth=laplace_max_tree_depth,
        laplace_start_at_map=laplace_start_at_map,
        nuts_kwargs=nuts_kwargs,
        mcmc_kwargs=mcmc_kwargs,
    )
    num_warmup = options["num_warmup"]
    num_samples = options["num_samples"]
    dense_mass = options["dense_mass"]
    regularize_mass_matrix = options["regularize_mass_matrix"]
    target_accept_prob = options["target_accept_prob"]
    max_tree_depth = options["max_tree_depth"]

    if num_warmup < 0:
        raise ValueError("num_warmup must be >= 0.")
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1.")

    t = _asarray_f64(t)
    yerr = _asarray_f64(yerr)
    indiv_y = _asarray_f64(indiv_y)
    if yerr.ndim != 2 or indiv_y.ndim != 2:
        raise ValueError("yerr and indiv_y must have shape [channel, time].")
    if yerr.shape != indiv_y.shape:
        raise ValueError("yerr and indiv_y must have identical shapes.")

    num_channels = int(yerr.shape[0])
    if num_channels < 1:
        raise ValueError("At least one channel is required.")
    lane_width = num_channels if lane_width is None else int(lane_width)
    if lane_width < num_channels:
        raise ValueError(
            f"lane_width={lane_width} cannot hold {num_channels} channels."
        )

    padded_yerr = _pad_first_axis(yerr, lane_width)[:, None, :]
    padded_y = _pad_first_axis(indiv_y, lane_width)[:, None, :]
    varying_kwargs, shared_kwargs = _partition_model_kwargs(
        model_kwargs,
        channel_varying_kwargs,
        num_channels,
        lane_width,
    )
    enriched_init = _enrich_initial_values(init_params)
    first_init = _first_lane_initial_values(enriched_init, num_channels)

    def lane_kwargs(varying_lane):
        return {**shared_kwargs, **varying_lane}

    # A dynamic potential generator lets the same compiled NUTS program receive
    # different observations and priors in each vmap lane.
    key_model, key_chains = jax.random.split(key)
    first_kwargs = {
        name: value[0] for name, value in varying_kwargs.items()
    }
    model_info = initialize_model(
        key_model,
        model,
        init_strategy=init_to_value(values=first_init),
        dynamic_args=True,
        model_args=(t, padded_yerr[0]),
        model_kwargs={"y": padded_y[0], **lane_kwargs(first_kwargs)},
    )
    batched_init = _prepare_unconstrained_initial_values(
        enriched_init,
        model_info.model_trace,
        num_channels,
        lane_width,
    )
    init_kernel, sample_kernel = hmc(
        potential_fn_gen=model_info.potential_fn,
        algo="NUTS",
    )

    chain_keys = jax.random.split(key_chains, lane_width)

    def init_one(z, err, obs, varying, rng):
        return init_kernel(
            z,
            num_warmup=num_warmup,
            dense_mass=dense_mass,
            regularize_mass_matrix=regularize_mass_matrix,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
            model_args=(t, err),
            model_kwargs={"y": obs, **lane_kwargs(varying)},
            rng_key=rng,
        )

    states = jax.jit(jax.vmap(init_one))(
        batched_init,
        padded_yerr,
        padded_y,
        varying_kwargs,
        chain_keys,
    )

    def advance(states):
        return jax.vmap(
            lambda state, err, obs, varying: sample_kernel(
                state,
                model_args=(t, err),
                model_kwargs={"y": obs, **lane_kwargs(varying)},
            )
        )(states, padded_yerr, padded_y, varying_kwargs)

    if num_warmup:
        states = jax.jit(
            lambda initial: jax.lax.fori_loop(
                0, num_warmup, lambda _, state: advance(state), initial
            )
        )(states)

    def collect_step(state, _):
        state = advance(state)
        return state, (
            state.z,
            state.num_steps,
            state.accept_prob,
            state.diverging,
        )

    states, (z_samples, num_steps, accept_prob, diverging) = jax.jit(
        lambda initial: jax.lax.scan(
            collect_step, initial, xs=None, length=num_samples
        )
    )(states)

    postprocess_gen = model_info.postprocess_fn

    def postprocess_lane(z, err, obs, varying):
        fn = postprocess_gen(
            t,
            err,
            y=obs,
            **lane_kwargs(varying),
        )
        return fn(z)

    postprocess_draw = jax.vmap(
        postprocess_lane,
        in_axes=(0, 0, 0, 0),
    )
    samples = jax.jit(
        jax.vmap(
            postprocess_draw,
            in_axes=(0, None, None, None),
        )
    )(z_samples, padded_yerr, padded_y, varying_kwargs)
    samples = _squeeze_internal_channel_axis(samples)
    samples = jax.tree.map(lambda value: value[:, :num_channels], samples)

    diagnostics = IndependentNUTSDiagnostics(
        num_steps=num_steps[:, :num_channels],
        accept_prob=accept_prob[:, :num_channels],
        diverging=diverging[:, :num_channels],
        step_size=states.adapt_state.step_size[:num_channels],
        mean_accept_prob=states.mean_accept_prob[:num_channels],
    )
    _save_diagnostics(diagnostics_path, diagnostics)
    if not return_diagnostics:
        return samples
    return samples, diagnostics


def get_samples_independent(
    model: Callable,
    key: jax.Array,
    t,
    yerr,
    indiv_y,
    init_params: Mapping[str, Any],
    *,
    nuts_kwargs: Mapping[str, Any] | None = None,
    mcmc_kwargs: Mapping[str, Any] | None = None,
    diagnostics_path: str | None = None,
    num_warmup: int | None = None,
    num_samples: int | None = None,
    lane_width: int | None = None,
    channel_varying_kwargs: tuple[str, ...] = (),
    dense_mass: bool | None = None,
    regularize_mass_matrix: bool | None = None,
    target_accept_prob: float | None = None,
    max_tree_depth: int | None = None,
    mass_matrix: str | None = None,
    laplace_warmup: int | None = None,
    laplace_target_accept: float | None = None,
    laplace_max_tree_depth: int | None = None,
    laplace_start_at_map: bool | None = None,
    return_diagnostics: bool = False,
    _runner: IndependentNUTSRunner | None = None,
    **model_kwargs,
):
    """Sample channel-factorized posteriors with reusable vmapped NUTS.

    Existing callers need not construct a runner. ``fit_jwst`` supplies the
    private ``_runner`` hook for equal-width chunks so their compiled programs
    are reused while all per-chunk numerical values remain dynamic.
    """
    yerr_array = _asarray_f64(yerr)
    if not hasattr(yerr_array, "ndim") or yerr_array.ndim != 2:
        raise ValueError("yerr and indiv_y must have shape [channel, time].")
    num_channels = int(yerr_array.shape[0])
    resolved_width = num_channels if lane_width is None else int(lane_width)
    options = _resolve_sampler_options(
        num_warmup=num_warmup,
        num_samples=num_samples,
        dense_mass=dense_mass,
        regularize_mass_matrix=regularize_mass_matrix,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        mass_matrix=mass_matrix,
        laplace_warmup=laplace_warmup,
        laplace_target_accept=laplace_target_accept,
        laplace_max_tree_depth=laplace_max_tree_depth,
        laplace_start_at_map=laplace_start_at_map,
        nuts_kwargs=nuts_kwargs,
        mcmc_kwargs=mcmc_kwargs,
    )
    if options["num_warmup"] < 0:
        raise ValueError("num_warmup must be >= 0.")
    if options["num_samples"] < 1:
        raise ValueError("num_samples must be >= 1.")
    if _runner is None:
        runner = IndependentNUTSRunner(
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
    samples, raw = runner.run_raw(
        key,
        t,
        yerr_array,
        indiv_y,
        init_params,
        model_kwargs,
    )
    diagnostics = IndependentNUTSDiagnostics(**raw)
    _save_diagnostics(diagnostics_path, diagnostics)
    if return_diagnostics:
        return samples, diagnostics
    return samples
