#!/usr/bin/env python3
"""Profile cadence-scaled pieces of a replayed spectroscopic potential.

This tool is intentionally diagnostic: it does not alter the model or sampler.
It reports compile/steady value-and-gradient timings and compiler memory for
the full NumPyro potential, the windowed transit kernel, the trend, and the
Normal likelihood at the dumped initial point.  It also measures the exact
reported-error grouping available for out-of-window sufficient statistics.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import handlers
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model

from models.jaxoplanet.builder import get_I_power2, _prepare_power2_poly
from models.jaxoplanet.core import (
    build_transit_phase_offsets,
    compute_transit_model,
)
from tools.spectro_stage_inputs import load_stage_inputs


jax.config.update("jax_enable_x64", True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser


def _block(value):
    return jax.tree.map(
        lambda leaf: leaf.block_until_ready()
        if hasattr(leaf, "block_until_ready") else leaf,
        value,
    )


def _memory_dict(executable):
    try:
        memory = executable.memory_analysis()
    except Exception as error:  # pragma: no cover - backend dependent
        return {"available": False, "error": str(error)}
    fields = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_temp_size_in_bytes",
    )
    result = {"available": True}
    for field in fields:
        value = getattr(memory, field, None)
        if value is not None:
            result[field] = int(value)
    return result


def _profile(name, function, arguments, repeats):
    started = time.perf_counter()
    lowered = jax.jit(function).lower(*arguments)
    lowered_seconds = time.perf_counter() - started
    started = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    _block(executable(*arguments))
    first_seconds = time.perf_counter() - started
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        _block(executable(*arguments))
        samples.append(time.perf_counter() - started)
    result = {
        "name": name,
        "lower_seconds": lowered_seconds,
        "compile_seconds": compile_seconds,
        "first_execute_seconds": first_seconds,
        "steady_seconds": samples,
        "steady_median_seconds": float(np.median(samples)),
        "steady_min_seconds": float(np.min(samples)),
        "compiler_memory": _memory_dict(executable),
    }
    print(
        f"{name}: compile={compile_seconds:.3f}s, "
        f"steady median={result['steady_median_seconds']:.6f}s, "
        f"temp={result['compiler_memory'].get('temp_size_in_bytes')}",
        flush=True,
    )
    return result


def _trace_at_initial(stage):
    arguments = (
        stage.t,
        stage.yerr,
    )
    keywords = {"y": stage.y, **dict(stage.model_kwargs)}
    scaffold = handlers.trace(
        handlers.seed(stage.model, rng_seed=0)
    ).get_trace(*arguments, **keywords)
    sample_names = {
        name for name, site in scaffold.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    }
    initial_samples = {
        name: value for name, value in stage.init_params.items()
        if name in sample_names
    }
    seeded = handlers.seed(stage.model, rng_seed=0)
    substituted = handlers.substitute(seeded, data=initial_samples)
    return handlers.trace(substituted).get_trace(
        *arguments,
        **keywords,
    )


def _trace_value(trace, name, default=None):
    site = trace.get(name)
    return default if site is None else jnp.asarray(site["value"], dtype=jnp.float64)


def _trend_parameters_and_function(stage, trace, transit_at_initial):
    builder = dict(stage.model_builder.get("kwargs", {}))
    detrend = str(builder.get("detrend_type", "linear"))
    t = jnp.asarray(stage.t, dtype=jnp.float64)
    t_norm = t - jnp.min(t)
    c = _trace_value(trace, "c", jnp.ones((stage.num_channels,)))
    v = _trace_value(trace, "v", jnp.zeros((stage.num_channels,)))
    initial = [c, v]

    def base(coefficients):
        return coefficients[0][:, None] + coefficients[1][:, None] * t_norm

    trend_function = base
    if "quadratic" in detrend or "cubic" in detrend or "quartic" in detrend:
        order = 2
        if "cubic" in detrend:
            order = 3
        if "quartic" in detrend:
            order = 4
        for degree in range(2, order + 1):
            initial.append(_trace_value(trace, f"v{degree}"))

        def polynomial(coefficients):
            value = base(coefficients)
            for degree in range(2, len(coefficients)):
                value = value + coefficients[degree][:, None] * t_norm ** degree
            return value

        trend_function = polynomial

    if "explinear_spectroscopic" in detrend:
        initial.append(_trace_value(trace, "A"))
        exp_trend = jnp.asarray(stage.model_kwargs["exp_trend"], dtype=jnp.float64)
        previous = trend_function

        def fixed_exponential(coefficients):
            return previous(coefficients[:-1]) + coefficients[-1][:, None] * exp_trend

        trend_function = fixed_exponential
    elif "explinear" in detrend:
        initial.extend((_trace_value(trace, "A"), _trace_value(trace, "log_tau")))
        previous = trend_function

        def free_exponential(coefficients):
            return (
                previous(coefficients[:-2])
                + coefficients[-2][:, None]
                * jnp.exp(-t_norm[None, :] / jnp.exp(coefficients[-1][:, None]))
            )

        trend_function = free_exponential

    coefficients = tuple(initial)

    def trend_objective(values):
        trend = trend_function(values)
        return jnp.sum(jnp.square(trend))

    log_jitter = _trace_value(trace, "log_jitter")
    y = jnp.asarray(stage.y, dtype=jnp.float64)
    yerr = jnp.asarray(stage.yerr, dtype=jnp.float64)
    mask = stage.model_kwargs.get("likelihood_mask")
    if mask is not None:
        mask = jnp.broadcast_to(jnp.asarray(mask, dtype=bool), y.shape)

    def likelihood_objective(values, jitter):
        prediction = transit_at_initial + trend_function(values)
        variance = yerr ** 2 + jnp.exp(jitter[:, None]) ** 2
        terms = 0.5 * (
            jnp.square(y - prediction) / variance
            + jnp.log(2.0 * jnp.asarray(math.pi, variance.dtype) * variance)
        )
        if mask is not None:
            terms = jnp.where(mask, terms, 0.0)
        return jnp.sum(terms)

    return coefficients, trend_objective, log_jitter, likelihood_objective


def _transit_function(stage, trace):
    kwargs = dict(stage.model_kwargs)
    builder = dict(stage.model_builder.get("kwargs", {}))
    t = jnp.asarray(stage.t, dtype=jnp.float64)
    period = jnp.asarray(kwargs["PERIOD"], dtype=jnp.float64)
    t0 = jnp.asarray(kwargs["mu_t0"], dtype=jnp.float64)
    b = jnp.asarray(kwargs["mu_b"], dtype=jnp.float64)
    duration = jnp.asarray(kwargs["mu_duration"], dtype=jnp.float64)
    phase_offsets, phase_mask = build_transit_phase_offsets(t, period, t0, duration)
    indices = builder.get("transit_window_indices")
    if str(builder.get("transit_window", "off")) != "auto":
        indices = None
    fixed = {
        "period": period,
        "t0": t0,
        "b": b,
        "duration": duration,
        "_transit_phase_offsets": phase_offsets,
        "_transit_phase_mask": phase_mask,
        "_jaxoplanet_kernel": builder.get("jaxoplanet_kernel", "auto"),
        "_ld_profile": builder.get("ld_profile", "quadratic"),
    }
    if indices is not None:
        fixed["_transit_window_indices"] = jnp.asarray(indices, dtype=jnp.int32)
    rors = _trace_value(trace, "rors")
    u = _trace_value(trace, "u")
    if u is None and builder.get("ld_profile") == "power2":
        c1 = _trace_value(trace, "c1")
        c2 = _trace_value(trace, "c2")
        mus, projection = _prepare_power2_poly()
        profiles = get_I_power2(c1[:, None], c2[:, None], mus[None, :])
        u = (projection @ (1.0 - profiles).T).T
    if u is None:
        raise ValueError("Could not recover the model's limb-darkening coefficients.")

    def transit(values):
        lane_rors, lane_u = values

        def one(ror, coefficients):
            return compute_transit_model(
                {**fixed, "rors": ror, "u": coefficients}, t
            )

        return jax.vmap(one)(lane_rors, lane_u)

    def objective(values):
        flux = transit(values)
        return jnp.sum(jnp.square(flux))

    return (rors, u), transit, objective


def _grouping_summary(stage):
    builder = dict(stage.model_builder.get("kwargs", {}))
    raw_indices = builder.get("transit_window_indices")
    window = np.zeros(stage.num_cadences, dtype=bool)
    if raw_indices is not None and str(builder.get("transit_window", "off")) == "auto":
        window[np.asarray(raw_indices, dtype=np.int64)] = True
    mask = stage.model_kwargs.get("likelihood_mask")
    if mask is None:
        valid = np.ones((stage.num_channels, stage.num_cadences), dtype=bool)
    else:
        valid = np.broadcast_to(np.asarray(mask, dtype=bool), np.shape(stage.y))
    counts = []
    largest = []
    for lane, lane_error in enumerate(np.asarray(stage.yerr)):
        selected = lane_error[valid[lane] & ~window]
        _, frequency = np.unique(selected, return_counts=True)
        counts.append(int(frequency.size))
        largest.append(int(frequency.max(initial=0)))
    return {
        "active_window_cadences": int(np.count_nonzero(window)),
        "out_of_window_cadences_per_lane": [
            int(np.count_nonzero(valid[lane] & ~window))
            for lane in range(stage.num_channels)
        ],
        "exact_yerr_groups_per_lane": counts,
        "largest_exact_group_per_lane": largest,
    }


def main() -> None:
    args = _parser().parse_args()
    if args.width < 1 or args.repeats < 1:
        raise ValueError("--width and --repeats must be positive.")
    stage = load_stage_inputs(args.dump, validate_potential=False).select(
        args.start, args.start + args.width
    )
    trace = _trace_at_initial(stage)
    transit_initial, transit, transit_objective = _transit_function(stage, trace)
    transit_at_initial = transit(transit_initial)
    coefficients, trend_objective, log_jitter, likelihood_objective = (
        _trend_parameters_and_function(stage, trace, transit_at_initial)
    )

    model_info = initialize_model(
        stage.rng_key,
        stage.model,
        init_strategy=init_to_value(values=dict(stage.init_params)),
        dynamic_args=False,
        model_args=(stage.t, stage.yerr),
        model_kwargs={"y": stage.y, **dict(stage.model_kwargs)},
        validate_grad=False,
    )
    potential_vg = jax.value_and_grad(model_info.potential_fn)
    transit_vg = jax.value_and_grad(transit_objective)
    trend_vg = jax.value_and_grad(trend_objective)
    likelihood_vg = jax.value_and_grad(likelihood_objective, argnums=(0, 1))

    profiles = [
        _profile("full_numpyro_potential", potential_vg, (model_info.param_info.z,), args.repeats),
        _profile("windowed_transit", transit_vg, (transit_initial,), args.repeats),
        _profile("trend", trend_vg, (coefficients,), args.repeats),
        _profile(
            "normal_likelihood_with_fixed_transit",
            likelihood_vg,
            (coefficients, log_jitter),
            args.repeats,
        ),
    ]
    builder_summary = dict(stage.model_builder)
    builder_kwargs = dict(builder_summary.get("kwargs", {}))
    builder_indices = builder_kwargs.pop("transit_window_indices", None)
    if builder_indices is not None:
        builder_kwargs["transit_window_index_count"] = int(
            np.size(builder_indices)
        )
    builder_summary["kwargs"] = builder_kwargs
    report = {
        "dump": str(Path(args.dump).resolve()),
        "platform": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "start": args.start,
        "width": args.width,
        "num_cadences": stage.num_cadences,
        "model_builder": builder_summary,
        "grouping": _grouping_summary(stage),
        "profiles": profiles,
        "process_max_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
    }
    rendered = json.dumps(report, indent=2, default=lambda value: np.asarray(value).tolist())
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
