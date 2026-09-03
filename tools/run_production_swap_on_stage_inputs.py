#!/usr/bin/env python3
"""Run the production Laplace-NUTS gate and sampler-swap chain on a dump."""

import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import numpy as np

import fit_jwst
from tools.run_sampler_on_stage_inputs import _arviz_diagnostics, _base_key_for_seed
from tools.spectro_stage_inputs import load_stage_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--seed", type=int, default=343)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    prefix = Path(args.output_prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    stage = load_stage_inputs(
        args.dump, potential_atol=1e-4,
        builder_overrides={"ld_parameterization": "coefficients"},
    )
    nuts = dict(stage.nuts_kwargs)
    nuts.pop("init_strategy", None)
    nuts.update(
        mass_matrix="laplace", target_accept_prob=0.95,
        max_tree_depth=5, laplace_warmup=150,
        laplace_target_accept=0.95, laplace_start_at_map=True,
        laplace_trust_radius=5.0, laplace_hessian_method="finite_difference",
        laplace_fd_batch_size=1,
    )
    mcmc = dict(stage.mcmc_kwargs)
    mcmc.update(num_warmup=args.warmup, num_samples=args.samples, progress_bar=False)
    key, source = _base_key_for_seed(stage, args.seed)
    started = time.perf_counter()
    samples = fit_jwst.get_samples_chunked(
        stage.model, key, stage.t, stage.yerr, stage.y, stage.init_params,
        chunk_size=args.chunk_size, nuts_kwargs=nuts, mcmc_kwargs=mcmc,
        output_dir=str(prefix.parent), checkpoint_prefix=prefix.name,
        sampler_backend="independent_nuts",
        channel_varying_kwargs=tuple(
            k for k in stage.channel_varying_kwargs if k in stage.model_kwargs
        ),
        checkpoint_signature={
            "stage_dump": str(Path(args.dump).resolve()),
            "ld_parameterization": "coefficients", "seed_index": args.seed,
        },
        spectro_min_depth_ess=400.0, spectro_max_divergences=0,
        **stage.model_kwargs,
    )
    wall = time.perf_counter() - started
    arrays = {k: np.asarray(jax.device_get(v)) for k, v in samples.items()}
    with open(f"{prefix}.pkl", "wb") as stream:
        pickle.dump(arrays, stream, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(f"{prefix}.npz", **arrays)
    provenance = list(samples.sampler_used or [])
    diagnostics = []
    for path in sorted((prefix.parent / "chunks").glob(f"{prefix.name}_*chunk_*.diagnostics.json")):
        diagnostics.append(json.loads(path.read_text()))
    attempts = Counter()
    attempt_divergences = Counter()
    for item in diagnostics:
        for attempt in item.get("sampler_gate_attempts", []):
            lane_count = len(attempt.get("input_local_lanes", []))
            if not lane_count:
                lane_count = len(attempt.get("depth_ess_per_channel") or [])
            attempts[attempt["sampler"]] += lane_count
            attempt_divergences[attempt["sampler"]] += int(
                np.sum(attempt.get("num_divergences_per_channel", 0))
            )
    summary = {
        "dump": str(Path(args.dump).resolve()), "seed_index": args.seed,
        "rng_source": source, "wall_seconds": wall,
        "num_channels": stage.num_channels, "sampler_used": provenance,
        "sampler_counts": dict(Counter(provenance)),
        "attempted_lane_counts": dict(attempts),
        "attempt_divergences": dict(attempt_divergences),
        "nuts_kwargs": nuts, "mcmc_kwargs": mcmc,
        "chunk_diagnostics": diagnostics,
    }
    Path(f"{prefix}.production.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    Path(f"{prefix}.arviz.json").write_text(
        json.dumps(_arviz_diagnostics(arrays, stage.num_channels), indent=2) + "\n"
    )
    print(json.dumps({k: summary[k] for k in ("wall_seconds", "sampler_counts", "attempt_divergences")}, indent=2))


if __name__ == "__main__":
    main()
