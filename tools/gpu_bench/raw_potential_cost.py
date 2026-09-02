#!/usr/bin/env python3
"""Time a dumped production NumPyro potential and its value-plus-gradient."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import numpy as np
import numpyro

from tools.spectro_stage_inputs import load_stage_inputs


def _block(value):
    leaves = jax.tree.leaves(value)
    if leaves:
        leaves[0].block_until_ready()
    return value


def _measure(function, argument, repeats):
    started = time.perf_counter()
    cold_value = _block(function(argument))
    cold_seconds = time.perf_counter() - started
    del cold_value

    # Two additional synchronized calls keep compilation out of the steady
    # distribution and allow GPU clocks/allocations to settle.
    for _ in range(2):
        _block(function(argument))

    seconds = []
    for _ in range(repeats):
        started = time.perf_counter()
        _block(function(argument))
        seconds.append(time.perf_counter() - started)
    values = np.asarray(seconds, dtype=np.float64)
    return {
        "cold_seconds": cold_seconds,
        "steady_repeats": repeats,
        "steady_median_seconds": float(np.median(values)),
        "steady_min_seconds": float(np.min(values)),
        "steady_max_seconds": float(np.max(values)),
        "steady_all_seconds": values.tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    # Dumps are produced on a different accelerator. The scalar potential is
    # O(1e5), and cross-GPU reassociation can move its final reduction by
    # O(1e-5). Keep validation enabled with a cross-device absolute tolerance.
    stage = load_stage_inputs(
        args.dump, validate_potential=True, potential_atol=1.0e-4
    )
    end = stage.num_channels if args.end is None else args.end
    selected = stage.select(args.start, end)

    nuts_kwargs = dict(selected.nuts_kwargs)
    nuts_kwargs.pop("init_strategy", None)
    model_info = numpyro.infer.util.initialize_model(
        selected.rng_key,
        selected.model,
        init_strategy=numpyro.infer.init_to_value(values=selected.init_params),
        dynamic_args=False,
        model_args=(selected.t, selected.yerr),
        model_kwargs={"y": selected.y, **dict(selected.model_kwargs)},
        validate_grad=False,
    )
    initial = model_info.param_info.z
    potential = jax.jit(model_info.potential_fn)
    value_and_grad = jax.jit(jax.value_and_grad(model_info.potential_fn))

    payload = {
        "dump": str(Path(args.dump).resolve()),
        "channels": [args.start, end],
        "num_channels": selected.num_channels,
        "num_cadences": selected.num_cadences,
        "device": str(jax.devices()[0]),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "potential": _measure(potential, initial, args.repeats),
        "value_and_gradient": _measure(value_and_grad, initial, args.repeats),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{output}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
