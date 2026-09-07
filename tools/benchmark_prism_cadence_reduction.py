#!/usr/bin/env python3
"""Compare exact long-cadence PRISM potential paths on a stage-input dump."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model

from tools.spectro_stage_inputs import load_stage_inputs
from models.cadence_reduction import (
    STATISTIC_KEYS,
    build_explinear_oot_statistics,
)


jax.config.update("jax_enable_x64", True)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--widths", default="4,8,16,40")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--candidate", choices=("cadence", "grid", "combined"),
        default="combined",
    )
    parser.add_argument("--grid-nodes", type=int, default=769)
    parser.add_argument(
        "--kernel", choices=("auto", "stock", "streamed"),
        default="auto",
    )
    parser.add_argument("--output", type=Path)
    return parser


def _block(value):
    return jax.tree.map(
        lambda leaf: leaf.block_until_ready()
        if hasattr(leaf, "block_until_ready") else leaf,
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


def _prepare(path, start, width, kernel, candidate_kind, grid_nodes):
    use_cadence = candidate_kind in {"cadence", "combined"}
    use_grid = candidate_kind in {"grid", "combined"}
    baseline = load_stage_inputs(
        path,
        validate_potential=False,
        builder_overrides={
            "cadence_reduction": "off",
            "transit_grid": "off",
            "transit_grid_nodes": grid_nodes,
            "transit_grid_non_grazing": True,
            "jaxoplanet_kernel": kernel,
        },
    ).select(start, start + width)
    candidate = load_stage_inputs(
        path,
        validate_potential=False,
        builder_overrides={
            "cadence_reduction": "auto" if use_cadence else "off",
            "transit_grid": "auto" if use_grid else "off",
            "transit_grid_nodes": grid_nodes,
            "transit_grid_non_grazing": True,
            "jaxoplanet_kernel": kernel,
        },
    ).select(start, start + width)
    if use_cadence:
        reference_beta = np.column_stack((
            np.asarray(candidate.init_params['c']),
            np.asarray(candidate.init_params['v']),
            np.asarray(candidate.init_params['A']),
        ))
        statistics = build_explinear_oot_statistics(
            candidate.t,
            candidate.y,
            candidate.yerr,
            candidate.model_builder['kwargs']['transit_window_indices'],
            candidate.model_kwargs['exp_trend'],
            reference_beta,
            candidate.model_kwargs.get('likelihood_mask'),
        )
        candidate = replace(
            candidate,
            model_kwargs={**dict(candidate.model_kwargs), **statistics},
            channel_varying_kwargs=tuple(dict.fromkeys(
                (*candidate.channel_varying_kwargs, *STATISTIC_KEYS)
            )),
        )
    builder = dict(candidate.model_builder.get("kwargs", {}))
    window = builder.get("transit_window_indices")
    if window is None or builder.get("transit_window") != "auto":
        raise ValueError("The dump must contain static transit-window indices.")
    in_window = np.zeros(candidate.num_cadences, dtype=bool)
    in_window[np.asarray(window, dtype=np.int64)] = True
    likelihood_mask = candidate.model_kwargs.get("likelihood_mask")
    valid = (
        np.ones(np.shape(candidate.y), dtype=bool)
        if likelihood_mask is None
        else np.broadcast_to(
            np.asarray(likelihood_mask, dtype=bool), np.shape(candidate.y)
        )
    )
    group_counts = [
        int(np.unique(np.asarray(candidate.yerr)[lane, valid[lane] & ~in_window]).size)
        for lane in range(candidate.num_channels)
    ]
    return baseline, candidate, {
        "minimum": min(group_counts), "maximum": max(group_counts)
    }


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


def _profile(function, argument, repeats):
    started = time.perf_counter()
    lowered = jax.jit(function).lower(argument)
    lower_seconds = time.perf_counter() - started
    started = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    first = _block(executable(argument))
    first_seconds = time.perf_counter() - started
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        _block(executable(argument))
        timings.append(time.perf_counter() - started)
    return first, {
        "lower_seconds": lower_seconds,
        "compile_seconds": compile_seconds,
        "first_execute_seconds": first_seconds,
        "steady_seconds": timings,
        "steady_median_seconds": float(np.median(timings)),
        "steady_min_seconds": float(np.min(timings)),
        "compiler_memory": _memory(executable),
    }


def _flat(tree):
    return np.asarray(jax.flatten_util.ravel_pytree(tree)[0])


def _run_width(path, start, width, repeats, kernel, candidate_kind, grid_nodes):
    baseline, candidate, group_summary = _prepare(
        path, start, width, kernel, candidate_kind, grid_nodes
    )
    baseline_fn, baseline_z = _potential(baseline)
    candidate_fn, candidate_z = _potential(candidate)
    baseline_result, baseline_profile = _profile(
        baseline_fn, baseline_z, repeats
    )
    candidate_result, candidate_profile = _profile(
        candidate_fn, candidate_z, repeats
    )
    baseline_value, baseline_gradient = baseline_result
    candidate_value, candidate_gradient = candidate_result
    gradient_difference = _flat(candidate_gradient) - _flat(baseline_gradient)
    baseline_gradient_flat = _flat(baseline_gradient)
    value_difference = float(np.asarray(candidate_value - baseline_value))
    baseline_value_float = float(np.asarray(baseline_value))
    gradient_relative = float(
        np.linalg.norm(gradient_difference)
        / max(np.linalg.norm(baseline_gradient_flat), np.finfo(float).tiny)
    )
    speedup = (
        baseline_profile["steady_median_seconds"]
        / candidate_profile["steady_median_seconds"]
    )
    print(
        f"width {width}: speedup={speedup:.3f}x, "
        f"potential_rel={abs(value_difference) / max(abs(baseline_value_float), np.finfo(float).tiny):.3e}, "
        f"gradient_rel={gradient_relative:.3e}",
        flush=True,
    )
    return {
        "width": width,
        "candidate_kind": candidate_kind,
        "transit_grid_nodes": grid_nodes,
        "cadences": candidate.num_cadences,
        "active_cadences": int(np.size(
            candidate.model_builder["kwargs"]["transit_window_indices"]
        )),
        "out_of_window_groups_min": group_summary["minimum"],
        "out_of_window_groups_max": group_summary["maximum"],
        "potential_absolute_difference": abs(value_difference),
        "potential_relative_difference": (
            abs(value_difference)
            / max(abs(baseline_value_float), np.finfo(float).tiny)
        ),
        "gradient_max_absolute_difference": float(
            np.max(np.abs(gradient_difference), initial=0.0)
        ),
        "gradient_relative_l2": gradient_relative,
        "baseline": baseline_profile,
        "candidate": candidate_profile,
        "candidate_speedup": speedup,
    }


def main():
    args = _parser().parse_args()
    widths = [int(value) for value in args.widths.split(",")]
    if any(width < 1 for width in widths) or args.repeats < 1:
        raise ValueError("Widths and repeats must be positive.")
    if args.grid_nodes < 13:
        raise ValueError("--grid-nodes must be at least 13.")
    report = {
        "dump": str(Path(args.dump).resolve()),
        "platform": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "start": args.start,
        "results": [
            _run_width(
                args.dump,
                args.start,
                width,
                args.repeats,
                args.kernel,
                args.candidate,
                args.grid_nodes,
            )
            for width in widths
        ],
    }
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
