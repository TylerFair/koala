#!/usr/bin/env python
"""Run an isolated real white-light fit from an existing pipeline config."""

import argparse
import runpy
import sys
import json
import time
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mass-matrix", choices=("adaptive", "laplace"), required=True)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument(
        "--hessian-method",
        choices=("exact", "finite_difference"),
        default="finite_difference",
    )
    parser.add_argument("--map-iterations", type=int, default=200)
    parser.add_argument("--laplace-target-accept", type=float, default=0.9)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    flags = config.setdefault("flags", {})
    flags.update(
        analysis_stage="whitelight",
        random_seed=args.seed,
        whitelight_mass_matrix=args.mass_matrix,
        whitelight_num_samples=args.samples,
        save_whitelight_trace=True,
        whitelight_laplace_hessian_method=args.hessian_method,
        whitelight_laplace_map_iterations=args.map_iterations,
        whitelight_laplace_target_accept=args.laplace_target_accept,
        pre_nuts_gradient_diagnostic=False,
    )
    if args.mass_matrix == "laplace":
        flags["whitelight_laplace_warmup"] = 200 if args.warmup is None else args.warmup
    else:
        flags["whitelight_num_warmup"] = 1000 if args.warmup is None else args.warmup
    config["host_device"] = "gpu"
    config["output_dir"] = args.output_dir

    config_path = Path(args.output_dir) / "resolved_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    sys.argv = ["fit_jwst.py", "--config", str(config_path)]
    started = time.perf_counter()
    runpy.run_path("fit_jwst.py", run_name="__main__")
    elapsed = time.perf_counter() - started
    diagnostics_path = config_path.parent / "whitelight_mcmc_diagnostics.json"
    if diagnostics_path.exists():
        diagnostics = json.loads(diagnostics_path.read_text())
        diagnostics["pipeline_whitelight_wall_seconds"] = elapsed
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
