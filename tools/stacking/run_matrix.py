#!/usr/bin/env python3
"""Materialize a portable set of model-stacking fits."""

from __future__ import annotations

import argparse
import copy
import os
import shlex
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
LEGACY_FIT_ROOT = Path("/scratch/midway3/tfairnington/accel_stacking")
LEGACY_RESULT_ROOT = Path("/scratch/midway3/tfairnington/accel_gpu_results")


def _merge(target, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _safe_component(value, label):
    value = str(value)
    if not value or not value.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Unsafe {label} {value!r}.")
    return value


def _validated_variants(spec):
    variants = list(spec.get("variants", ()))
    if not variants:
        raise ValueError("The matrix must define at least one variant.")
    names = []
    for variant in variants:
        names.append(_safe_component(variant["name"], "variant name"))
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate variant names: {duplicates}.")
    return variants


def materialize(base_config, matrix_spec, workspace="stacking_workspace"):
    """Write local fit configs, a run script, and a stacking manifest.

    Paths stored in the manifest are relative to the manifest itself, so the
    workspace can be moved as one directory before running the stacker.
    """
    base_config = Path(base_config).resolve()
    matrix_spec = Path(matrix_spec).resolve()
    workspace = Path(workspace).resolve()
    with base_config.open() as stream:
        base = yaml.safe_load(stream)
    with matrix_spec.open() as stream:
        spec = yaml.safe_load(stream)

    dataset = _safe_component(spec["dataset"], "dataset name")
    variants = _validated_variants(spec)
    config_dir = workspace / "configs"
    fit_root = workspace / "fits"
    stage_root = workspace / "stage_inputs"
    config_dir.mkdir(parents=True, exist_ok=True)
    fit_root.mkdir(parents=True, exist_ok=True)
    stage_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": dataset,
        "variants": [],
    }
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'PYTHON_BIN="${PYTHON:-python}"',
        'export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"',
        f"cd {shlex.quote(str(Path.cwd().resolve()))}",
        "",
    ]
    outputs = []
    for variant in variants:
        name = str(variant["name"])
        config = copy.deepcopy(base)
        _merge(config, dict(variant.get("overrides", {})))
        fit_dir = fit_root / name
        stage_dir = stage_root / name
        config["output_dir"] = str(fit_dir)
        flags = config.setdefault("flags", {})
        flags["analysis_stage"] = spec.get(
            "analysis_stage", flags.get("analysis_stage", "all")
        )
        flags.setdefault("spectro_sampler", "independent_nuts")
        flags.setdefault("spectro_min_depth_ess", 400)

        config_path = config_dir / f"{dataset}_{name}.yaml"
        if config_path.exists():
            raise FileExistsError(config_path)
        with config_path.open("x") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)

        manifest["variants"].append({
            "name": name,
            "config": str(config_path.relative_to(workspace)),
            "fit_dir": str(fit_dir.relative_to(workspace)),
            "stage_inputs": str(stage_dir.relative_to(workspace)),
        })
        commands.extend([
            f"mkdir -p {shlex.quote(str(stage_dir))}",
            (
                f"FIT_JWST_DUMP_SAMPLER_INPUTS={shlex.quote(str(stage_dir))} "
                f'"$PYTHON_BIN" {shlex.quote(str(REPO / "fit_jwst.py"))} '
                f"--config {shlex.quote(str(config_path))}"
            ),
            "",
        ])
        outputs.append(config_path)

    manifest_path = workspace / "analysis.yaml"
    run_script = workspace / "run_all.sh"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    if run_script.exists():
        raise FileExistsError(run_script)
    with manifest_path.open("x") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False)
    with run_script.open("x") as stream:
        stream.write("\n".join(commands))
    os.chmod(run_script, 0o755)
    return outputs, manifest_path, run_script


def materialize_legacy_midway(base_config, matrix_spec, queue_start=300):
    """Retain the former site-specific queue materializer for old campaigns."""
    with Path(base_config).open() as stream:
        base = yaml.safe_load(stream)
    with Path(matrix_spec).open() as stream:
        spec = yaml.safe_load(stream)
    dataset = _safe_component(spec["dataset"], "dataset name")
    variants = _validated_variants(spec)
    if queue_start < 300 or queue_start + len(variants) - 1 > 429:
        raise ValueError("Legacy queue numbers must be in 300--429.")
    config_dir = REPO / "configs_stacking"
    queue_dir = REPO / "acceleration_reports/gpu_queue/pending"
    config_dir.mkdir(parents=True, exist_ok=True)
    queue_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for offset, variant in enumerate(variants):
        name = str(variant["name"])
        config = copy.deepcopy(base)
        _merge(config, dict(variant.get("overrides", {})))
        config["path"] = str(LEGACY_FIT_ROOT / dataset)
        config["output_dir"] = name
        flags = config.setdefault("flags", {})
        flags["analysis_stage"] = spec.get(
            "analysis_stage", flags.get("analysis_stage", "prep")
        )
        flags.setdefault("spectro_sampler", "independent_nuts")
        flags.setdefault("spectro_min_depth_ess", 400)
        config_path = config_dir / f"{dataset}_{name}.yaml"
        if config_path.exists():
            raise FileExistsError(config_path)
        with config_path.open("x") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        number = queue_start + offset
        result_dir = LEGACY_RESULT_ROOT / f"{number}_stacking_{name}"
        script_path = queue_dir / f"{number}_stacking_{name}.sh"
        if script_path.exists():
            raise FileExistsError(script_path)
        script = f"""#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(str(REPO))}
export JAX_ENABLE_X64=1
mkdir -p {shlex.quote(str(result_dir))}
export FIT_JWST_DUMP_SAMPLER_INPUTS={shlex.quote(str(result_dir / "stage_inputs"))}
python fit_jwst.py --config {shlex.quote(str(config_path))}
"""
        with script_path.open("x") as stream:
            stream.write(script)
        os.chmod(script_path, 0o755)
        outputs.append((config_path, script_path))
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Create isolated configurations for a model-stacking matrix."
    )
    parser.add_argument("base_config")
    parser.add_argument("matrix_spec")
    parser.add_argument(
        "--workspace",
        default="stacking_workspace",
        help="Local directory for configs, fits, stage inputs, and manifest.",
    )
    parser.add_argument(
        "--legacy-midway",
        action="store_true",
        help="Generate the former site-specific Midway queue files.",
    )
    parser.add_argument("--queue-start", type=int, default=300)
    args = parser.parse_args()

    if args.legacy_midway:
        for config, script in materialize_legacy_midway(
            args.base_config, args.matrix_spec, args.queue_start
        ):
            print(f"{config}\t{script}")
        return

    configs, manifest, script = materialize(
        args.base_config, args.matrix_spec, args.workspace
    )
    for config in configs:
        print(f"Wrote {config}")
    print(f"Analysis manifest: {manifest}")
    print(f"Run fits: bash {script}")


if __name__ == "__main__":
    main()
