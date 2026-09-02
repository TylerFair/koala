#!/usr/bin/env python3
"""Compare exact real-stage potential/gradient before and after cadence padding."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np
import numpyro

from fit_jwst import _pad_spectro_cadences_exact
from tools.spectro_stage_inputs import load_stage_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump")
    parser.add_argument("--channels", type=int, default=1)
    args = parser.parse_args()
    stage = load_stage_inputs(args.dump).select(0, args.channels)
    init = numpyro.infer.init_to_value(values=stage.init_params)
    base = numpyro.infer.util.initialize_model(
        stage.rng_key, stage.model, init_strategy=init,
        model_args=(stage.t, stage.yerr),
        model_kwargs={"y": stage.y, **stage.model_kwargs},
        dynamic_args=False, validate_grad=False,
    )
    t, y, yerr, mask = _pad_spectro_cadences_exact(stage.t, stage.y, stage.yerr)
    padded_kwargs = dict(stage.model_kwargs)
    pad = int(t.shape[0]) - stage.num_cadences
    for name, value in stage.model_kwargs.items():
        try:
            array = jnp.asarray(value)
        except (TypeError, ValueError):
            continue
        if array.ndim and array.shape[-1] == stage.num_cadences:
            padded_kwargs[name] = jnp.pad(
                array,
                [(0, 0)] * (array.ndim - 1) + [(0, pad)],
                constant_values=0.0,
            )
    padded_kwargs["likelihood_mask"] = mask
    padded = numpyro.infer.util.initialize_model(
        stage.rng_key, stage.model, init_strategy=init,
        model_args=(t, yerr), model_kwargs={"y": y, **padded_kwargs},
        dynamic_args=False, validate_grad=False,
    )
    z = base.param_info.z
    value0, grad0 = jax.value_and_grad(base.potential_fn)(z)
    value1, grad1 = jax.value_and_grad(padded.potential_fn)(z)
    leaves0 = jax.tree.leaves(grad0)
    leaves1 = jax.tree.leaves(grad1)
    grad_abs = max(
        float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        for a, b in zip(leaves0, leaves1)
    )
    result = {
        "dump": os.path.abspath(args.dump),
        "channels": args.channels,
        "cadences_original": int(stage.num_cadences),
        "cadences_padded": int(t.shape[0]),
        "potential_original": float(value0),
        "potential_padded": float(value1),
        "potential_abs_difference": float(abs(value1 - value0)),
        "gradient_max_abs_difference": grad_abs,
        "passes_1e_10": bool(abs(value1 - value0) <= 1e-10 and grad_abs <= 1e-10),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
