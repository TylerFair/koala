#!/usr/bin/env python3
"""Run one immutable timed pipeline variant from an existing YAML config."""
import argparse
import os
import subprocess
import time
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--minimal-plots", action="store_true")
    parser.add_argument("--adaptive-white", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.base).read_text())
    cfg["output_dir"] = args.output_dir
    flags = cfg.setdefault("flags", {})
    flags["phase_timers"] = True
    if args.minimal_plots:
        flags["plots"] = "minimal"
    if args.fast:
        flags.update(
            compile_box=True,
            jax_compilation_cache_dir=(
                "/scratch/midway3/tfairnington/jax_cache/efficiency_p3_shared"
            ),
            whitelight_mass_matrix="laplace",
            whitelight_laplace_warmup=200,
            whitelight_num_samples=1000,
        )
        cfg.setdefault("stellar", {})["ld_prior_cache_dir"] = (
            "/scratch/midway3/tfairnington/ld_prior_cache/efficiency_p3_shared"
        )
    if args.adaptive_white:
        flags["whitelight_mass_matrix"] = "adaptive"
    root = Path("/scratch/midway3/tfairnington") / args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    config_path = root / "effective_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    env = dict(os.environ)
    env.update(JAX_ENABLE_X64="1", FIT_JWST_PHASE_TIMERS="1")
    started = time.perf_counter()
    with (root / "pipeline.log").open("w") as stream:
        result = subprocess.run(
            ["python", "fit_jwst.py", "--config", str(config_path)],
            env=env, stdout=stream, stderr=subprocess.STDOUT,
        )
    (root / "process_wall.txt").write_text(
        f"{time.perf_counter() - started:.9f}\n"
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
