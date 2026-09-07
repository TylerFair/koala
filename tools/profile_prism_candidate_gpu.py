#!/usr/bin/env python3
"""Profile the PRISM baseline and cadence-reduced candidate by component.

This diagnostic deliberately separates the transit grid evaluation from its
cadence interpolation so GPU transpose/scatter costs are visible.  It also
records compiler memory, HLO operation counts, and cost analysis for the full
candidate potential.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import resource
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model

from jaxoplanet.core.limb_dark import light_curve as stock_light_curve
from models.jaxoplanet.builder import _prepare_power2_poly, get_I_power2
from models.jaxoplanet.core import (
    build_transit_phase_offsets,
    compute_transit_model_window,
    resolve_jaxoplanet_kernel,
)
from models.jaxoplanet.limb_dark_streamed import (
    light_curve as streamed_light_curve,
)
from models.jaxoplanet.transit_grid import (
    _power2_duration_transit_primal,
    build_duration_transit_grid,
    interpolate_duration_transit_grid,
    interpolate_power2_duration_transit,
)
from tools.benchmark_prism_cadence_reduction import _prepare
from tools.profile_prism_cadence import (
    _trace_at_initial,
    _trace_value,
    _transit_function,
    _trend_parameters_and_function,
)


jax.config.update("jax_enable_x64", True)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--grid-nodes", type=int, default=769)
    parser.add_argument(
        "--kernel", choices=("auto", "stock", "streamed"), default="auto"
    )
    parser.add_argument("--output", type=Path)
    return parser


def _block(value):
    return jax.tree.map(
        lambda leaf: leaf.block_until_ready()
        if hasattr(leaf, "block_until_ready")
        else leaf,
        value,
    )


def _memory(executable):
    try:
        report = executable.memory_analysis()
    except Exception as error:  # pragma: no cover - backend dependent
        return {"available": False, "error": str(error)}
    result = {"available": True}
    for name in (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_temp_size_in_bytes",
    ):
        value = getattr(report, name, None)
        if value is not None:
            result[name] = int(value)
    return result


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    try:
        return np.asarray(value).item()
    except (TypeError, ValueError):
        return str(value)


def _hlo_summary(executable):
    try:
        hlo = executable.as_text()
    except Exception as error:  # pragma: no cover - backend dependent
        return {"available": False, "error": str(error)}
    operations = Counter()
    for line in hlo.splitlines():
        match = re.search(r"=\s+(?:[^ ]+\s+)?([a-z][a-z0-9_-]*)\(", line)
        if match is not None:
            operations[match.group(1)] += 1
    watched = (
        "fusion",
        "scatter",
        "gather",
        "reduce",
        "reduce-window",
        "dot",
        "dot-general",
        "exponential",
        "divide",
        "log",
        "sqrt",
        "rsqrt",
        "custom-call",
        "while",
        "sort",
    )
    try:
        cost = _jsonable(executable.cost_analysis())
    except Exception as error:  # pragma: no cover - backend dependent
        cost = {"available": False, "error": str(error)}
    return {
        "available": True,
        "text_bytes": len(hlo.encode("utf-8")),
        "line_count": len(hlo.splitlines()),
        "operation_definitions": int(sum(operations.values())),
        "watched_operation_counts": {
            name: int(operations.get(name, 0)) for name in watched
        },
        "all_operation_counts": dict(operations.most_common()),
        "cost_analysis": cost,
    }


def _profile(name, function, arguments, repeats, *, inspect_hlo=False):
    started = time.perf_counter()
    lowered = jax.jit(function).lower(*arguments)
    lower_seconds = time.perf_counter() - started
    started = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    first_result = _block(executable(*arguments))
    first_seconds = time.perf_counter() - started
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        _block(executable(*arguments))
        samples.append(time.perf_counter() - started)
    result = {
        "name": name,
        "lower_seconds": lower_seconds,
        "compile_seconds": compile_seconds,
        "first_execute_seconds": first_seconds,
        "steady_seconds": samples,
        "steady_median_seconds": float(np.median(samples)),
        "steady_min_seconds": float(np.min(samples)),
        "compiler_memory": _memory(executable),
    }
    if inspect_hlo:
        result["hlo"] = _hlo_summary(executable)
    print(
        f"{name}: compile={compile_seconds:.3f}s "
        f"median={result['steady_median_seconds'] * 1e3:.4f}ms "
        f"min={result['steady_min_seconds'] * 1e3:.4f}ms "
        f"temp={result['compiler_memory'].get('temp_size_in_bytes')}",
        flush=True,
    )
    return first_result, result


def _recover_transit_values(stage, trace):
    builder = dict(stage.model_builder.get("kwargs", {}))
    rors = _trace_value(trace, "rors")
    u = _trace_value(trace, "u")
    if u is None and builder.get("ld_profile") == "power2":
        c1 = _trace_value(trace, "c1")
        c2 = _trace_value(trace, "c2")
        mus, projection = _prepare_power2_poly()
        profiles = get_I_power2(c1[:, None], c2[:, None], mus[None, :])
        u = (projection @ (1.0 - profiles).T).T
    if rors is None or u is None:
        raise ValueError("Could not recover initial radius/limb-darkening values.")
    return rors, u


def _transit_components(stage, trace, grid_nodes, kernel_name):
    model_kwargs = dict(stage.model_kwargs)
    builder = dict(stage.model_builder.get("kwargs", {}))
    indices = jnp.asarray(builder["transit_window_indices"], dtype=jnp.int32)
    t = jnp.asarray(stage.t, dtype=jnp.float64)
    period = jnp.asarray(model_kwargs["PERIOD"], dtype=jnp.float64)
    t0 = jnp.asarray(model_kwargs["mu_t0"], dtype=jnp.float64)
    impact = jnp.asarray(model_kwargs["mu_b"], dtype=jnp.float64)
    duration = jnp.asarray(model_kwargs["mu_duration"], dtype=jnp.float64)
    phase, phase_mask = build_transit_phase_offsets(t, period, t0, duration)
    active_phase = phase[:, indices]
    active_mask = phase_mask[:, indices]
    rors, u = _recover_transit_values(stage, trace)
    c1 = _trace_value(trace, "c1")
    c2 = _trace_value(trace, "c2")
    fixed = {
        "period": period,
        "t0": t0,
        "b": impact,
        "duration": duration,
        "_transit_phase_offsets": phase,
        "_transit_phase_mask": phase_mask,
        "_transit_window_indices": indices,
        "_jaxoplanet_kernel": kernel_name,
        "_ld_profile": builder.get("ld_profile", "quadratic"),
    }

    def active_flux(values, nodes=None):
        lane_rors, lane_u = values

        def one(ror, coefficients):
            params = {**fixed, "rors": ror, "u": coefficients}
            if nodes is not None:
                params["_transit_grid_nodes"] = int(nodes)
            return compute_transit_model_window(params, t)

        return jax.vmap(one)(lane_rors, lane_u)

    active_weight = jnp.asarray(
        0.75 + 0.25 * np.cos(np.arange(indices.shape[0]) * 0.017),
        dtype=jnp.float64,
    )

    def direct_objective(values):
        return jnp.sum(active_flux(values) * active_weight[None, :])

    selected_kernel = resolve_jaxoplanet_kernel(
        kernel_name,
        ld_profile=builder.get("ld_profile", "quadratic"),
        degree=int(u.shape[-1]),
        keplerian=False,
    )
    grid_kernel = (
        streamed_light_curve if selected_kernel == "streamed" else stock_light_curve
    )

    def grid_flux(values):
        lane_rors, lane_u = values

        def one(ror, coefficients):
            radius = ror[0]
            separation = build_duration_transit_grid(
                impact=impact[0], radius_ratio=radius, num_nodes=grid_nodes
            )
            flux = grid_kernel(coefficients, separation, radius, order=10)
            return flux.at[-1].set(0.0)

        return jax.vmap(one)(lane_rors, lane_u)

    grid_weight = jnp.asarray(
        0.75 + 0.25 * np.sin(np.arange(grid_nodes) * 0.031),
        dtype=jnp.float64,
    )

    def grid_objective(values):
        return jnp.sum(grid_flux(values) * grid_weight[None, :])

    initial_grid = jax.lax.stop_gradient(grid_flux((rors, u)))

    def interpolation_forward(fluxes, radii):
        return jax.vmap(
            lambda flux, radius: interpolate_duration_transit_grid(
                flux,
                active_phase[0],
                active_mask[0],
                duration=duration[0],
                impact=impact[0],
                radius_ratio=radius,
            )
        )(fluxes, radii)

    def interpolation_objective(fluxes, radii):
        return jnp.sum(
            interpolation_forward(fluxes, radii) * active_weight[None, :]
        )

    theta = jnp.stack((rors[:, 0], c1, c2), axis=1)

    def reverse_grid_objective(values):
        flux = jax.vmap(
            lambda lane_theta, lane_u: _power2_duration_transit_primal(
                grid_kernel,
                lane_theta,
                lane_u,
                active_phase[0],
                active_mask[0],
                duration=duration[0],
                impact=impact[0],
                num_nodes=grid_nodes,
                order=10,
            )
        )(values, u)
        return jnp.sum(flux * active_weight[None, :])

    def candidate_flux(values):
        return jax.vmap(
            lambda lane_theta, lane_u: interpolate_power2_duration_transit(
                grid_kernel,
                lane_theta[1],
                lane_theta[2],
                lane_u,
                active_phase[0],
                active_mask[0],
                duration=duration[0],
                impact=impact[0],
                radius_ratio=lane_theta[0],
                num_nodes=grid_nodes,
                order=10,
            )
        )(values, u)

    def candidate_transit_objective(values):
        flux = candidate_flux(values)
        return jnp.sum(flux * active_weight[None, :])

    return {
        "initial": (rors, u),
        "radii": rors[:, 0],
        "direct_vg": jax.value_and_grad(direct_objective),
        "grid_vg": jax.value_and_grad(grid_objective),
        "initial_grid": initial_grid,
        "interpolation_forward": interpolation_forward,
        "interpolation_vg": jax.value_and_grad(
            interpolation_objective, argnums=(0, 1)
        ),
        "candidate_transit_vg": jax.value_and_grad(
            candidate_transit_objective
        ),
        "candidate_flux": candidate_flux,
        "theta": theta,
        "reverse_grid_vg": jax.value_and_grad(reverse_grid_objective),
        "active_cadences": int(indices.shape[0]),
        "selected_kernel": selected_kernel,
    }


def _oot_objective(stage, trace):
    coefficients, _, log_jitter, _ = _trend_parameters_and_function(
        stage,
        trace,
        jnp.zeros_like(jnp.asarray(stage.y, dtype=jnp.float64)),
    )
    if len(coefficients) != 3:
        raise ValueError("OOT profiler expects c/v/A explinear coefficients.")
    reference = jnp.asarray(stage.model_kwargs["oot_reference_beta"])
    group_yerr = jnp.asarray(stage.model_kwargs["oot_group_yerr"])
    group_count = jnp.asarray(stage.model_kwargs["oot_group_count"])
    reference_sse = jnp.asarray(stage.model_kwargs["oot_group_reference_sse"])
    x_reference_residual = jnp.asarray(
        stage.model_kwargs["oot_group_x_reference_residual"]
    )
    xx = jnp.asarray(stage.model_kwargs["oot_group_xx"])

    def objective(values, jitter_log):
        beta = jnp.stack(values, axis=1)
        delta = beta - reference
        sse = (
            reference_sse
            - 2.0 * jnp.einsum("lp,lgp->lg", delta, x_reference_residual)
            + jnp.einsum("lp,lgpq,lq->lg", delta, xx, delta)
        )
        variance = group_yerr**2 + jnp.exp(jitter_log[:, None]) ** 2
        log_probability = -0.5 * (
            sse / variance
            + group_count * jnp.log(2.0 * jnp.pi * variance)
        )
        return -jnp.sum(log_probability)

    return coefficients, log_jitter, jax.value_and_grad(
        objective, argnums=(0, 1)
    )


def _potential(stage):
    info = initialize_model(
        stage.rng_key,
        stage.model,
        init_strategy=init_to_value(values=dict(stage.init_params)),
        dynamic_args=False,
        model_args=(stage.t, stage.yerr),
        model_kwargs={"y": stage.y, **dict(stage.model_kwargs)},
        validate_grad=False,
    )
    return jax.value_and_grad(info.potential_fn), info.param_info.z


def _flatten(tree):
    return np.asarray(jax.flatten_util.ravel_pytree(tree)[0])


def main():
    args = _parser().parse_args()
    if args.width < 1 or args.repeats < 1:
        raise ValueError("--width and --repeats must be positive.")
    baseline, candidate, group_summary = _prepare(
        args.dump,
        args.start,
        args.width,
        args.kernel,
        "combined",
        args.grid_nodes,
    )
    baseline_trace = _trace_at_initial(baseline)
    candidate_trace = _trace_at_initial(candidate)
    components = _transit_components(
        baseline, baseline_trace, args.grid_nodes, args.kernel
    )

    # Build the direct full-cadence model once only to provide a fixed transit
    # for the isolated baseline trend+likelihood profile.
    transit_initial, transit, _ = _transit_function(baseline, baseline_trace)
    full_transit_initial = jax.lax.stop_gradient(transit(transit_initial))
    coefficients, _, log_jitter, full_likelihood = (
        _trend_parameters_and_function(
            baseline, baseline_trace, full_transit_initial
        )
    )
    full_likelihood_vg = jax.value_and_grad(
        full_likelihood, argnums=(0, 1)
    )
    active_indices = jnp.asarray(
        candidate.model_builder["kwargs"]["transit_window_indices"],
        dtype=jnp.int32,
    )
    active_time = jnp.asarray(candidate.t, dtype=jnp.float64)[active_indices]
    active_exp = jnp.asarray(
        candidate.model_kwargs["exp_trend"], dtype=jnp.float64
    )[active_indices]
    active_y = jnp.asarray(candidate.y, dtype=jnp.float64)[:, active_indices]
    active_yerr = jnp.asarray(
        candidate.yerr, dtype=jnp.float64
    )[:, active_indices]
    active_mask = candidate.model_kwargs.get("likelihood_mask")
    if active_mask is not None:
        active_mask = jnp.broadcast_to(
            jnp.asarray(active_mask, dtype=bool), jnp.shape(candidate.y)
        )[:, active_indices]

    def active_likelihood(theta, values, jitter_log):
        trend = (
            values[0][:, None]
            + values[1][:, None]
            * (active_time[None, :] - jnp.min(candidate.t))
            + values[2][:, None] * active_exp[None, :]
        )
        prediction = components["candidate_flux"](theta) + trend
        variance = active_yerr**2 + jnp.exp(jitter_log[:, None]) ** 2
        terms = 0.5 * (
            jnp.square(active_y - prediction) / variance
            + jnp.log(2.0 * jnp.pi * variance)
        )
        if active_mask is not None:
            terms = jnp.where(active_mask, terms, 0.0)
        return jnp.sum(terms)

    active_likelihood_vg = jax.value_and_grad(
        active_likelihood, argnums=(0, 1, 2)
    )
    oot_coefficients, oot_log_jitter, oot_vg = _oot_objective(
        candidate, candidate_trace
    )
    baseline_potential, baseline_z = _potential(baseline)
    candidate_potential, candidate_z = _potential(candidate)

    profiles = []
    direct_result, profile = _profile(
        "direct_window_transit_vg",
        components["direct_vg"],
        (components["initial"],),
        args.repeats,
    )
    profiles.append(profile)
    _, profile = _profile(
        "grid_node_kernel_vg",
        components["grid_vg"],
        (components["initial"],),
        args.repeats,
    )
    profiles.append(profile)
    _, profile = _profile(
        "interpolation_forward",
        components["interpolation_forward"],
        (components["initial_grid"], components["radii"]),
        args.repeats,
    )
    profiles.append(profile)
    _, profile = _profile(
        "interpolation_vjp",
        components["interpolation_vg"],
        (components["initial_grid"], components["radii"]),
        args.repeats,
        inspect_hlo=True,
    )
    profiles.append(profile)
    reverse_grid_result, profile = _profile(
        "grid_transit_reverse_scatter_vg",
        components["reverse_grid_vg"],
        (components["theta"],),
        args.repeats,
    )
    profiles.append(profile)
    candidate_transit_result, profile = _profile(
        "candidate_window_transit_vg",
        components["candidate_transit_vg"],
        (components["theta"],),
        args.repeats,
    )
    profiles.append(profile)
    _, profile = _profile(
        "candidate_active_likelihood_vg",
        active_likelihood_vg,
        (components["theta"], coefficients, log_jitter),
        args.repeats,
        inspect_hlo=True,
    )
    profiles.append(profile)
    _, profile = _profile(
        "full_trend_likelihood_vg",
        full_likelihood_vg,
        (coefficients, log_jitter),
        args.repeats,
    )
    profiles.append(profile)
    _, profile = _profile(
        "oot_statistics_vg",
        oot_vg,
        (oot_coefficients, oot_log_jitter),
        args.repeats,
    )
    profiles.append(profile)
    baseline_result, profile = _profile(
        "baseline_full_potential_vg",
        baseline_potential,
        (baseline_z,),
        args.repeats,
    )
    profiles.append(profile)
    candidate_result, profile = _profile(
        "candidate_full_potential_vg",
        candidate_potential,
        (candidate_z,),
        args.repeats,
        inspect_hlo=True,
    )
    profiles.append(profile)

    baseline_value, baseline_gradient = baseline_result
    candidate_value, candidate_gradient = candidate_result
    reverse_grid_value, reverse_grid_gradient = reverse_grid_result
    custom_grid_value, custom_grid_gradient = candidate_transit_result
    custom_grid_gradient_difference = (
        _flatten(custom_grid_gradient) - _flatten(reverse_grid_gradient)
    )
    gradient_difference = _flatten(candidate_gradient) - _flatten(
        baseline_gradient
    )
    report = {
        "dump": str(Path(args.dump).resolve()),
        "platform": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "start": args.start,
        "width": args.width,
        "cadences": baseline.num_cadences,
        "active_cadences": components["active_cadences"],
        "grid_nodes": args.grid_nodes,
        "requested_kernel": args.kernel,
        "selected_kernel": components["selected_kernel"],
        "oot_groups": group_summary,
        "custom_grid_value_relative_difference": float(
            abs(np.asarray(custom_grid_value - reverse_grid_value))
            / max(
                abs(float(np.asarray(reverse_grid_value))),
                np.finfo(float).tiny,
            )
        ),
        "custom_grid_gradient_relative_l2": float(
            np.linalg.norm(custom_grid_gradient_difference)
            / max(
                np.linalg.norm(_flatten(reverse_grid_gradient)),
                np.finfo(float).tiny,
            )
        ),
        "custom_grid_gradient_max_absolute_difference": float(
            np.max(np.abs(custom_grid_gradient_difference), initial=0.0)
        ),
        "potential_relative_difference": float(
            abs(np.asarray(candidate_value - baseline_value))
            / max(abs(float(np.asarray(baseline_value))), np.finfo(float).tiny)
        ),
        "gradient_relative_l2": float(
            np.linalg.norm(gradient_difference)
            / max(
                np.linalg.norm(_flatten(baseline_gradient)),
                np.finfo(float).tiny,
            )
        ),
        "gradient_max_absolute_difference": float(
            np.max(np.abs(gradient_difference), initial=0.0)
        ),
        "profiles": profiles,
        "process_max_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
    }
    rendered = json.dumps(report, indent=2, default=_jsonable)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
