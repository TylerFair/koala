#!/usr/bin/env python3
"""GPU replay CLI with cross-accelerator initial-potential tolerance."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_sampler_on_stage_inputs as replay
from tools.spectro_stage_inputs import load_stage_inputs as _load_stage_inputs

_diagnostic_arrays = replay._diagnostic_arrays


def _load_with_cross_accelerator_tolerance(path, **kwargs):
    kwargs["potential_atol"] = max(
        float(kwargs.get("potential_atol", 0.0)), 1.0e-4
    )
    stage = _load_stage_inputs(path, **kwargs)
    backend = None
    for index, value in enumerate(sys.argv[1:]):
        if value == "--backend" and index + 2 <= len(sys.argv[1:]):
            backend = sys.argv[1:][index + 1]
        elif value.startswith("--backend="):
            backend = value.split("=", 1)[1]
    if backend == "independent_nuts":
        # The dump stores the joint sampler's diagonal-mass setting. Remove
        # only that inherited override so independent NUTS uses its documented
        # small dense mass matrix per lane; retain every other stored option.
        nuts_kwargs = dict(stage.nuts_kwargs)
        nuts_kwargs.pop("dense_mass", None)
        stage = replace(stage, nuts_kwargs=nuts_kwargs)
    return stage


replay.load_stage_inputs = _load_with_cross_accelerator_tolerance


def _all_dataclass_diagnostics(value):
    arrays = _diagnostic_arrays(value)
    if value is not None and hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            arrays[name] = replay.np.asarray(
                replay.jax.device_get(getattr(value, name))
            )
    return arrays


replay._diagnostic_arrays = _all_dataclass_diagnostics


if __name__ == "__main__":
    raise SystemExit(replay.main())
