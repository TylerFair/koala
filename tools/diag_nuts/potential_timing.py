#!/usr/bin/env python3
"""CPU forward and value-plus-gradient timings at three production shapes."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diag_nuts.common import block, json_dump, jsonable, make_spectro_problem


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--cadences", default="230,2058,40787")
    p.add_argument("--channels", default="1,40")
    p.add_argument("--seed", type=int, default=20260903)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument(
        "--output",
        default="acceleration_reports/diag_nuts/potential_timing.json",
    )
    return p


def time_one(fn, arg, repeats):
    compile_start = time.perf_counter()
    block(fn(arg))
    compile_seconds = time.perf_counter() - compile_start
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        block(fn(arg))
        timings.append(time.perf_counter() - start)
    values = np.asarray(timings)
    return {
        "compile_and_first_seconds": compile_seconds,
        "repeats": repeats,
        "seconds": timings,
        "median_seconds": float(np.median(values)),
        "min_seconds": float(np.min(values)),
        "max_seconds": float(np.max(values)),
    }


def main() -> None:
    args = parser().parse_args()
    cadences = [int(value) for value in args.cadences.split(",")]
    channels = [int(value) for value in args.channels.split(",")]
    rows = []
    for cadence_count in cadences:
        for channel_count in channels:
            problem = make_spectro_problem(
                channel_count, cadence_count, args.seed + cadence_count + channel_count
            )
            info = initialize_model(
                jax.random.PRNGKey(args.seed),
                problem.model,
                init_strategy=init_to_value(values=problem.init_params),
                model_args=(problem.t, problem.yerr),
                model_kwargs={"y": problem.y, **problem.model_kwargs},
            )
            potential = jax.jit(info.potential_fn)
            value_grad = jax.jit(jax.value_and_grad(info.potential_fn))
            repeats = min(args.repeats, 3) if cadence_count >= 40000 else args.repeats
            forward = time_one(potential, info.param_info.z, repeats)
            vg = time_one(value_grad, info.param_info.z, repeats)
            row = {
                "cadences": cadence_count,
                "channels": channel_count,
                "active_window_cadences": problem.metadata["window_cadences"],
                "active_window_fraction": problem.metadata["window_fraction"],
                "forward": forward,
                "value_plus_gradient": vg,
                "value_plus_gradient_over_forward": (
                    vg["median_seconds"] / forward["median_seconds"]
                ),
                "per_channel_forward_seconds": forward["median_seconds"] / channel_count,
                "per_channel_value_plus_gradient_seconds": (
                    vg["median_seconds"] / channel_count
                ),
            }
            rows.append(row)
            print(
                f"cadences={cadence_count} channels={channel_count} "
                f"forward={forward['median_seconds']:.6g}s "
                f"value+grad={vg['median_seconds']:.6g}s",
                flush=True,
            )
    json_dump(
        args.output,
        jsonable(
            {
                "platform": str(jax.devices()[0]),
                "float64": bool(jax.config.jax_enable_x64),
                "timing_scope": (
                    "jitted complete NumPyro potential: transforms, priors, "
                    "streamed power-2 transit, linear trend, jitter, likelihood"
                ),
                "rows": rows,
            }
        ),
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
