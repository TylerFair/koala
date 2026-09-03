"""GPU-batched, independently adapted fixed-step HMC for spectral channels.

This backend has the same factorization requirements as
``models.independent_nuts`` but uses one fixed leapfrog count for every lane.
That removes the lockstep cost of vmapped dynamic NUTS trees: a difficult
channel can adapt a smaller step size without forcing other channels to build
its deeper tree.  The fixed step count is deliberately explicit and must be
chosen by an ESS/second pilot rather than treated as a convergence shortcut.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
from numpyro.infer.hmc import hmc
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model

from .independent_nuts import (
    _IndependentSamplerRunner,
    _asarray_f64,
    _enrich_initial_values,
    _first_lane_initial_values,
    _pad_first_axis,
    _partition_model_kwargs,
    _prepare_unconstrained_initial_values,
    _squeeze_internal_channel_axis,
)


@dataclass(frozen=True)
class IndependentHMCDiagnostics:
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


def _resolve_hmc_options(
    *,
    num_warmup,
    num_samples,
    num_steps,
    dense_mass,
    regularize_mass_matrix,
    target_accept_prob,
    mass_matrix,
    laplace_warmup,
    laplace_target_accept,
    laplace_start_at_map,
    trajectory_jitter,
    nuts_kwargs,
    mcmc_kwargs,
):
    kernel = dict(nuts_kwargs or {})
    mcmc = dict(mcmc_kwargs or {})
    kernel.pop("init_strategy", None)
    supported_kernel = {
        "dense_mass",
        "regularize_mass_matrix",
        "target_accept_prob",
        "num_steps",
        # Dumped NUTS inputs contain this; fixed-step HMC ignores it.
        "max_tree_depth",
        "mass_matrix",
        "laplace_warmup",
        "laplace_target_accept",
        "laplace_start_at_map",
        "laplace_map_iterations",
        "laplace_map_tolerance",
        "laplace_map_decrement_tolerance",
        "laplace_eigenvalue_floor",
        "laplace_hessian_method",
        "laplace_fd_relative_step",
        "laplace_fd_batch_size",
        "laplace_compare_exact_hessian",
        "trajectory_jitter",
        "laplace_map_method",
        "laplace_line_search_steps",
        "laplace_trust_radius",
        "laplace_fuse_program",
    }
    unknown_kernel = set(kernel) - supported_kernel
    if unknown_kernel:
        raise ValueError(
            "Unsupported independent-HMC options: "
            + ", ".join(sorted(unknown_kernel))
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
        else kernel.get("mass_matrix", "adaptive")
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
        "num_steps": int(
            num_steps if num_steps is not None else kernel.get("num_steps", 16)
        ),
        "dense_mass": bool(
            dense_mass
            if dense_mass is not None
            else kernel.get("dense_mass", True)
        ),
        "regularize_mass_matrix": bool(
            regularize_mass_matrix
            if regularize_mass_matrix is not None
            else kernel.get("regularize_mass_matrix", True)
        ),
        "target_accept_prob": float(
            target_accept_prob
            if target_accept_prob is not None
            else kernel.get("target_accept_prob", 0.8)
        ),
        "mass_matrix": resolved_mass_matrix,
        "laplace_warmup": int(
            laplace_warmup
            if laplace_warmup is not None
            else kernel.get("laplace_warmup", 150)
        ),
        "laplace_target_accept": float(
            laplace_target_accept
            if laplace_target_accept is not None
            else kernel.get("laplace_target_accept", 0.85)
        ),
        "laplace_start_at_map": bool(
            laplace_start_at_map
            if laplace_start_at_map is not None
            else kernel.get("laplace_start_at_map", False)
        ),
        "laplace_map_iterations": int(
            kernel.get("laplace_map_iterations", 16)
        ),
        "laplace_map_tolerance": float(
            kernel.get("laplace_map_tolerance", 1.0e-4)
        ),
        "laplace_map_decrement_tolerance": float(
            kernel.get("laplace_map_decrement_tolerance", 0.0)
        ),
        "laplace_eigenvalue_floor": float(
            kernel.get("laplace_eigenvalue_floor", 1.0e-8)
        ),
        "laplace_hessian_method": str(
            kernel.get("laplace_hessian_method", "exact")
        ).lower(),
        "laplace_fd_relative_step": float(
            kernel.get("laplace_fd_relative_step", 2.0e-4)
        ),
        "laplace_fd_batch_size": int(kernel.get("laplace_fd_batch_size", 1)),
        "laplace_compare_exact_hessian": bool(
            kernel.get("laplace_compare_exact_hessian", False)
        ),
        # Kept for the shared preparation implementation; HMC ignores it.
        "laplace_max_tree_depth": 10,
        "laplace_map_method": str(
            kernel.get("laplace_map_method", "newton")
        ).lower(),
        "laplace_line_search_steps": int(
            kernel.get("laplace_line_search_steps", 8)
        ),
        "laplace_trust_radius": float(
            kernel.get("laplace_trust_radius", 1.0)
        ),
        "laplace_fuse_program": bool(
            kernel.get("laplace_fuse_program", False)
        ),
        "trajectory_jitter": float(
            trajectory_jitter
            if trajectory_jitter is not None
            else kernel.get("trajectory_jitter", 0.0)
        ),
    }
    if result["laplace_warmup"] < 0:
        raise ValueError("laplace_warmup must be >= 0.")
    if not 0.0 < result["laplace_target_accept"] < 1.0:
        raise ValueError("laplace_target_accept must be between 0 and 1.")
    if result["laplace_map_iterations"] < 1:
        raise ValueError("laplace_map_iterations must be >= 1.")
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
    if not 0.0 <= result["trajectory_jitter"] < 1.0:
        raise ValueError("trajectory_jitter must be in [0, 1).")
    if result["laplace_map_method"] not in {"newton", "diagonal"}:
        raise ValueError(
            "laplace_map_method must be 'newton' or 'diagonal'."
        )
    if result["laplace_line_search_steps"] < 1:
        raise ValueError("laplace_line_search_steps must be >= 1.")
    if result["laplace_trust_radius"] <= 0.0:
        raise ValueError("laplace_trust_radius must be > 0.")
    return result


def _save_hmc_diagnostics(path, diagnostics):
    if path is None:
        return
    steps = jax.device_get(diagnostics.num_steps)
    accept = jax.device_get(diagnostics.accept_prob)
    diverging = jax.device_get(diagnostics.diverging)
    step_size = jax.device_get(diagnostics.step_size)
    payload = {
        "backend": "independent_batched_hmc",
        "num_channels": int(steps.shape[1]),
        "num_draws": int(steps.shape[0]),
        "fixed_num_steps": (
            int(steps[0, 0]) if (steps == steps[0, 0]).all() else None
        ),
        "num_steps_min": int(steps.min()),
        "num_steps_max": int(steps.max()),
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


class IndependentHMCRunner(_IndependentSamplerRunner):
    """Reusable independently adapted fixed-step HMC runner."""

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
            algo="HMC",
            options=options,
            lane_width=lane_width,
            channel_varying_kwargs=channel_varying_kwargs,
        )


def build_independent_hmc_runner(
    model,
    *,
    nuts_kwargs=None,
    mcmc_kwargs=None,
    num_warmup=None,
    num_samples=None,
    num_steps=None,
    lane_width,
    channel_varying_kwargs=(),
    dense_mass=None,
    regularize_mass_matrix=None,
    target_accept_prob=None,
    mass_matrix=None,
    laplace_warmup=None,
    laplace_target_accept=None,
    laplace_start_at_map=None,
    trajectory_jitter=None,
):
    """Build a lazily compiled fixed-step runner for one padded width."""
    options = _resolve_hmc_options(
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_steps=num_steps,
        dense_mass=dense_mass,
        regularize_mass_matrix=regularize_mass_matrix,
        target_accept_prob=target_accept_prob,
        mass_matrix=mass_matrix,
        laplace_warmup=laplace_warmup,
        laplace_target_accept=laplace_target_accept,
        laplace_start_at_map=laplace_start_at_map,
        trajectory_jitter=trajectory_jitter,
        nuts_kwargs=nuts_kwargs,
        mcmc_kwargs=mcmc_kwargs,
    )
    if options["num_warmup"] < 0:
        raise ValueError("num_warmup must be >= 0.")
    if options["num_samples"] < 1:
        raise ValueError("num_samples must be >= 1.")
    if options["num_steps"] < 1:
        raise ValueError("num_steps must be >= 1.")
    return IndependentHMCRunner(
        model,
        options=options,
        lane_width=lane_width,
        channel_varying_kwargs=channel_varying_kwargs,
    )


def _get_samples_independent_hmc_uncached(
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
    num_steps: int | None = None,
    lane_width: int | None = None,
    channel_varying_kwargs: tuple[str, ...] = (),
    dense_mass: bool | None = None,
    regularize_mass_matrix: bool | None = None,
    target_accept_prob: float | None = None,
    mass_matrix: str | None = None,
    laplace_warmup: int | None = None,
    laplace_target_accept: float | None = None,
    laplace_start_at_map: bool | None = None,
    trajectory_jitter: float | None = None,
    return_diagnostics: bool = False,
    **model_kwargs,
):
    """Sample factorized channel posteriors with fixed-step vmapped HMC.

    Step size and mass matrix adapt independently for every lane.  The number
    of leapfrog steps is shared and static, so all lanes perform exactly the
    same amount of trajectory work per draw.
    """
    options = _resolve_hmc_options(
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_steps=num_steps,
        dense_mass=dense_mass,
        regularize_mass_matrix=regularize_mass_matrix,
        target_accept_prob=target_accept_prob,
        mass_matrix=mass_matrix,
        laplace_warmup=laplace_warmup,
        laplace_target_accept=laplace_target_accept,
        laplace_start_at_map=laplace_start_at_map,
        trajectory_jitter=trajectory_jitter,
        nuts_kwargs=nuts_kwargs,
        mcmc_kwargs=mcmc_kwargs,
    )
    num_warmup = options["num_warmup"]
    num_samples = options["num_samples"]
    num_steps = options["num_steps"]
    if num_warmup < 0:
        raise ValueError("num_warmup must be >= 0.")
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1.")
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1.")

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

    key_model, key_chains = jax.random.split(key)
    first_kwargs = {name: value[0] for name, value in varying_kwargs.items()}
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
        algo="HMC",
    )
    chain_keys = jax.random.split(key_chains, lane_width)

    def init_one(z, err, obs, varying, rng):
        return init_kernel(
            z,
            num_warmup=num_warmup,
            num_steps=num_steps,
            trajectory_length=None,
            dense_mass=options["dense_mass"],
            regularize_mass_matrix=options["regularize_mass_matrix"],
            target_accept_prob=options["target_accept_prob"],
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

    states, (z_samples, sampled_steps, accept_prob, diverging) = jax.jit(
        lambda initial: jax.lax.scan(
            collect_step, initial, xs=None, length=num_samples
        )
    )(states)

    postprocess_gen = model_info.postprocess_fn

    def postprocess_lane(z, err, obs, varying):
        return postprocess_gen(
            t,
            err,
            y=obs,
            **lane_kwargs(varying),
        )(z)

    postprocess_draw = jax.vmap(postprocess_lane, in_axes=(0, 0, 0, 0))
    samples = jax.jit(
        jax.vmap(postprocess_draw, in_axes=(0, None, None, None))
    )(z_samples, padded_yerr, padded_y, varying_kwargs)
    samples = _squeeze_internal_channel_axis(samples)
    samples = jax.tree.map(lambda value: value[:, :num_channels], samples)

    diagnostics = IndependentHMCDiagnostics(
        num_steps=sampled_steps[:, :num_channels],
        accept_prob=accept_prob[:, :num_channels],
        diverging=diverging[:, :num_channels],
        step_size=states.adapt_state.step_size[:num_channels],
        mean_accept_prob=states.mean_accept_prob[:num_channels],
    )
    _save_hmc_diagnostics(diagnostics_path, diagnostics)
    if return_diagnostics:
        return samples, diagnostics
    return samples


def get_samples_independent_hmc(
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
    num_steps: int | None = None,
    lane_width: int | None = None,
    channel_varying_kwargs: tuple[str, ...] = (),
    dense_mass: bool | None = None,
    regularize_mass_matrix: bool | None = None,
    target_accept_prob: float | None = None,
    mass_matrix: str | None = None,
    laplace_warmup: int | None = None,
    laplace_target_accept: float | None = None,
    laplace_start_at_map: bool | None = None,
    trajectory_jitter: float | None = None,
    return_diagnostics: bool = False,
    _runner: IndependentHMCRunner | None = None,
    **model_kwargs,
):
    """Sample factorized posteriors with reusable fixed-step vmapped HMC."""
    yerr_array = _asarray_f64(yerr)
    if not hasattr(yerr_array, "ndim") or yerr_array.ndim != 2:
        raise ValueError("yerr and indiv_y must have shape [channel, time].")
    num_channels = int(yerr_array.shape[0])
    resolved_width = num_channels if lane_width is None else int(lane_width)
    options = _resolve_hmc_options(
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_steps=num_steps,
        dense_mass=dense_mass,
        regularize_mass_matrix=regularize_mass_matrix,
        target_accept_prob=target_accept_prob,
        mass_matrix=mass_matrix,
        laplace_warmup=laplace_warmup,
        laplace_target_accept=laplace_target_accept,
        laplace_start_at_map=laplace_start_at_map,
        trajectory_jitter=trajectory_jitter,
        nuts_kwargs=nuts_kwargs,
        mcmc_kwargs=mcmc_kwargs,
    )
    if options["num_warmup"] < 0:
        raise ValueError("num_warmup must be >= 0.")
    if options["num_samples"] < 1:
        raise ValueError("num_samples must be >= 1.")
    if options["num_steps"] < 1:
        raise ValueError("num_steps must be >= 1.")
    if _runner is None:
        runner = IndependentHMCRunner(
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
    diagnostics = IndependentHMCDiagnostics(**raw)
    _save_hmc_diagnostics(diagnostics_path, diagnostics)
    if return_diagnostics:
        return samples, diagnostics
    return samples
