#!/usr/bin/env python3
"""Trace the production Laplace MAP iterations for one quadratic-LD lane."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model

from models.independent_nuts import (
    _enrich_initial_values,
    _first_lane_initial_values,
    _partition_model_kwargs,
    _prepare_unconstrained_initial_values,
    _split_dynamic_and_static_kwargs,
)
from models.ld_parameterization import QuadraticKippingTransform
from tools.spectro_stage_inputs import load_stage_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump")
    parser.add_argument("--channel", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    stage = load_stage_inputs(
        args.dump, potential_atol=1e-8,
        builder_overrides={"ld_parameterization": "decorrelated"},
    ).select(args.channel, args.channel + 1)
    varying, shared = _partition_model_kwargs(
        stage.model_kwargs,
        tuple(k for k in stage.channel_varying_kwargs if k in stage.model_kwargs),
        1, 1
    )
    dynamic, static = _split_dynamic_and_static_kwargs(shared)
    kwargs = {**static, **dynamic, **{k: v[0] for k, v in varying.items()}}
    enriched = _enrich_initial_values(stage.init_params)
    key_model, _ = jax.random.split(stage.rng_key)
    yerr = stage.yerr[:, None, :]
    obs = stage.y[:, None, :]
    info = initialize_model(
        key_model, stage.model,
        init_strategy=init_to_value(values=_first_lane_initial_values(enriched, 1)),
        dynamic_args=True, model_args=(stage.t, yerr[0]),
        model_kwargs={"y": obs[0], **kwargs},
    )
    batched = _prepare_unconstrained_initial_values(
        enriched, info.model_trace, 1, 1
    )
    flat, unravel = ravel_pytree(jax.tree.map(lambda x: x[0], batched))
    potential = info.potential_fn(stage.t, yerr[0], y=obs[0], **kwargs)
    vg = jax.jit(jax.value_and_grad(lambda x: potential(unravel(x))))
    grad = jax.jit(jax.grad(lambda x: potential(unravel(x))))
    relative_step = 2.0e-4
    first_steps = relative_step * jnp.maximum(1.0, jnp.abs(flat))

    def fd_hessian(x, steps):
        eye = jnp.eye(x.size, dtype=x.dtype) * steps[:, None]
        plus = jax.vmap(lambda off: grad(x + off))(eye)
        minus = jax.vmap(lambda off: grad(x - off))(eye)
        result = (plus - minus) / (2.0 * steps[:, None])
        return 0.5 * (result + result.T)

    first_h = fd_hessian(flat, first_steps)
    scale = jnp.clip(
        1.0 / jnp.sqrt(jnp.maximum(jnp.abs(jnp.diag(first_h)), 1e-12)),
        0.05, 20.0,
    )
    steps = first_steps * scale
    line_scales = 0.5 ** jnp.arange(12, dtype=jnp.float64)
    inverse = QuadraticKippingTransform().inv
    rows = []
    for iteration in range(args.iterations + 1):
        value, gradient = vg(flat)
        state = unravel(flat)
        q = np.asarray(state["ld_decorrelated"]).reshape(-1, 2)[0]
        coeff = np.asarray(inverse(jnp.asarray(q)))
        hessian = fd_hessian(flat, steps)
        eig, vec = jnp.linalg.eigh(jnp.nan_to_num(hessian, nan=0., posinf=1e12, neginf=-1e12))
        largest = jnp.maximum(jnp.max(jnp.abs(eig)), 1.0)
        repaired = jnp.maximum(eig, 1e-8 * largest)
        covariance = (vec / repaired[None, :]) @ vec.T
        condition = float(jnp.max(repaired) / jnp.min(repaired))
        rows.append({
            "iteration": iteration, "potential": float(value),
            "gradient_norm": float(jnp.linalg.norm(gradient)),
            "condition_number": condition,
            "raw_min_eigenvalue": float(jnp.min(eig)),
            "q1": float(q[0]), "q2": float(q[1]),
            "u1": float(coeff[0]), "u2": float(coeff[1]),
            "physical_edge_distance": float(np.min(np.r_[coeff, 1.0 - coeff])),
        })
        if iteration == args.iterations:
            break
        newton = -(covariance @ gradient)
        newton *= jnp.minimum(1.0, 5.0 / jnp.maximum(jnp.linalg.norm(newton), 1e-12))
        gradient_step = -gradient * (0.1 / jnp.maximum(jnp.linalg.norm(gradient), 1e-12))
        directions = jnp.concatenate(
            (line_scales[:, None] * newton, line_scales[:, None] * gradient_step), axis=0
        )
        candidates = flat[None, :] + directions
        values = jax.vmap(lambda x: potential(unravel(x)))(candidates)
        values = jnp.where(jnp.isfinite(values), values, jnp.inf)
        best = int(jnp.argmin(values))
        if float(values[best]) < float(value):
            flat = candidates[best]
            rows[-1]["accepted_direction"] = "newton" if best < 12 else "gradient"
            rows[-1]["accepted_scale"] = float(line_scales[best % 12])
        else:
            rows[-1]["accepted_direction"] = "none"
            rows[-1]["accepted_scale"] = 0.0
    payload = {
        "dump": str(Path(args.dump).resolve()), "channel": args.channel,
        "iterations": rows,
        "summary": {
            "initial": rows[0], "final": rows[-1],
            "minimum_physical_edge_distance": min(r["physical_edge_distance"] for r in rows),
            "maximum_condition_number": max(r["condition_number"] for r in rows),
        },
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
