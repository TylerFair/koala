#!/usr/bin/env python3
"""Materialize a model-stacking YAML matrix and V100 queue scripts."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import yaml


REPO = Path("/project/ekempton/tfairnington/JWST")
SCRATCH_ROOT = Path("/scratch/midway3/tfairnington/accel_stacking")


def _merge(target, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def materialize(base_config, matrix_spec, queue_start=300):
    with Path(base_config).open() as stream:
        base = yaml.safe_load(stream)
    with Path(matrix_spec).open() as stream:
        spec = yaml.safe_load(stream)
    dataset = str(spec["dataset"])
    variants = list(spec["variants"])
    if not variants or queue_start < 300 or queue_start + len(variants) - 1 > 419:
        raise ValueError("The matrix must use one or more queue numbers in 300--419.")
    config_dir = REPO / "configs_stacking"
    queue_dir = REPO / "acceleration_reports/gpu_queue/pending"
    config_dir.mkdir(parents=True, exist_ok=True)
    queue_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for offset, variant in enumerate(variants):
        name = str(variant["name"])
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Unsafe variant name {name!r}.")
        config = copy.deepcopy(base)
        _merge(config, dict(variant.get("overrides", {})))
        config["path"] = str(SCRATCH_ROOT / dataset)
        config["output_dir"] = name
        flags = config.setdefault("flags", {})
        flags.update({
            "analysis_stage": spec.get("analysis_stage", "prep"),
            "spectro_sampler": "independent_nuts",
            "spectro_mass_matrix": "laplace",
            "spectro_jitter_prior": "lognormal",
            "spectro_min_depth_ess": 400,
        })
        config_path = config_dir / f"{dataset}_{name}.yaml"
        if config_path.exists():
            raise FileExistsError(config_path)
        with config_path.open("x") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        number = queue_start + offset
        result_dir = Path("/scratch/midway3/tfairnington/accel_gpu_results") / f"{number}_stacking_{name}"
        script_path = queue_dir / f"{number}_stacking_{name}.sh"
        if script_path.exists():
            raise FileExistsError(script_path)
        script = f"""#!/usr/bin/env bash
set -euo pipefail
cd /project/ekempton/tfairnington/JWST
export JAX_ENABLE_X64=1
mkdir -p {result_dir}
export FIT_JWST_DUMP_SAMPLER_INPUTS={result_dir}/stage_inputs
start=$(date +%s)
python fit_jwst.py --config {config_path}
end=$(date +%s)
printf '%s\\n' \"$((end-start))\" > {result_dir}/wall_seconds.txt
"""
        with script_path.open("x") as stream:
            stream.write(script)
        os.chmod(script_path, 0o755)
        outputs.append((config_path, script_path))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_config")
    parser.add_argument("matrix_spec")
    parser.add_argument("--queue-start", type=int, default=300)
    args = parser.parse_args()
    for config, script in materialize(args.base_config, args.matrix_spec, args.queue_start):
        print(f"{config}\t{script}")


if __name__ == "__main__":
    main()
