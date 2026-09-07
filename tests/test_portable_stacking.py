import json
import os
import pickle
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from tools.stacking.run_matrix import materialize
from tools.stacking.stack_spectra import _load_chunks, _variant_paths


def _write_yaml(path, value):
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_materialized_workspace_is_portable_and_preserves_user_flags(tmp_path):
    source = tmp_path / "source files"
    source.mkdir()
    base_path = source / "base config.yaml"
    matrix_path = source / "matrix spec.yaml"
    workspace = tmp_path / "workspace with spaces"
    _write_yaml(
        base_path,
        {
            "path": "/data/input",
            "output_dir": "unused",
            "flags": {
                "spectro_sampler": "independent_hmc",
                "spectro_min_depth_ess": 321,
                "spectro_max_divergences": 2,
            },
        },
    )
    _write_yaml(
        matrix_path,
        {
            "dataset": "demo_dataset",
            "analysis_stage": "highres",
            "variants": [
                {"name": "wide_prior", "overrides": {"flags": {"ld_prior": "uniform"}}}
            ],
        },
    )

    configs, manifest_path, run_script = materialize(
        base_path, matrix_path, workspace
    )
    config = yaml.safe_load(configs[0].read_text(encoding="utf-8"))
    assert config["flags"]["spectro_sampler"] == "independent_hmc"
    assert config["flags"]["spectro_min_depth_ess"] == 321
    assert config["flags"]["spectro_max_divergences"] == 2
    assert config["flags"]["ld_prior"] == "uniform"
    assert config["flags"]["analysis_stage"] == "highres"

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    variant = manifest["variants"][0]
    assert not Path(variant["config"]).is_absolute()
    assert not Path(variant["fit_dir"]).is_absolute()
    assert not Path(variant["stage_inputs"]).is_absolute()
    combined = run_script.read_text(encoding="utf-8") + manifest_path.read_text(
        encoding="utf-8"
    )
    assert "/scratch/" not in combined

    capture = tmp_path / "captured argv.txt"
    python_stub = tmp_path / "python stub"
    python_stub.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$FIT_JWST_DUMP_SAMPLER_INPUTS\" \"$@\" > \"$CAPTURE\"\n",
        encoding="utf-8",
    )
    python_stub.chmod(0o755)
    elsewhere = tmp_path / "different cwd"
    elsewhere.mkdir()
    environment = os.environ.copy()
    environment.update(PYTHON=str(python_stub), CAPTURE=str(capture))
    subprocess.run(["bash", str(run_script)], cwd=elsewhere, env=environment, check=True)
    captured = capture.read_text(encoding="utf-8").splitlines()
    assert captured[0] == str(workspace / "stage_inputs" / "wide_prior")
    assert Path(captured[1]).name == "fit_jwst.py"
    assert captured[2] == "--config"
    assert captured[3] == str(configs[0])


@pytest.mark.parametrize("name", ["../escape", "space name", "", "name/child"])
def test_materialize_rejects_unsafe_variant_names(tmp_path, name):
    base = tmp_path / "base.yaml"
    matrix = tmp_path / "matrix.yaml"
    _write_yaml(base, {"flags": {}})
    _write_yaml(matrix, {"dataset": "demo", "variants": [{"name": name}]})
    with pytest.raises(ValueError, match="Unsafe variant name"):
        materialize(base, matrix, tmp_path / "workspace")


def test_materialize_rejects_duplicate_names(tmp_path):
    base = tmp_path / "base.yaml"
    matrix = tmp_path / "matrix.yaml"
    _write_yaml(base, {"flags": {}})
    _write_yaml(
        matrix,
        {"dataset": "demo", "variants": [{"name": "same"}, {"name": "same"}]},
    )
    with pytest.raises(ValueError, match="Duplicate variant names"):
        materialize(base, matrix, tmp_path / "workspace")


def test_variant_paths_resolve_relative_to_manifest(tmp_path):
    manifest = tmp_path / "moved workspace" / "analysis.yaml"
    variant = {
        "name": "model",
        "fit_dir": "fits/model",
        "stage_inputs": "stage_inputs/model",
    }
    fit_dir, stage_dir = _variant_paths(manifest, {}, variant, 0)
    assert fit_dir == (manifest.parent / "fits/model").resolve()
    assert stage_dir == (manifest.parent / "stage_inputs/model").resolve()


def _write_family(output, prefix, ranges, *, stage_kind, value):
    chunks = output / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    (chunks / f"{prefix}.manifest.json").write_text(
        json.dumps({"checkpoint_signature": {"stage": stage_kind}}),
        encoding="utf-8",
    )
    for start, stop in ranges:
        payload = {"samples": {"depths": np.full((3, stop - start, 1), value)}}
        with (chunks / f"{prefix}_chunk_{start}_{stop}.pkl").open("wb") as stream:
            pickle.dump(payload, stream)


def _stage(label, kind, count):
    return SimpleNamespace(
        meta={
            "stage_label": label,
            "stage_kind": kind,
            "num_channels": count,
            "wavelength": np.arange(count),
        }
    )


def test_load_chunks_selects_requested_stage_when_lowres_is_longer(tmp_path):
    _write_family(
        tmp_path, "target_R20_low", [(0, 3), (3, 6)],
        stage_kind="low_resolution", value=20,
    )
    _write_family(
        tmp_path, "target_R100_high", [(0, 2)],
        stage_kind="high_resolution", value=100,
    )
    samples = _load_chunks(
        tmp_path, _stage("target_R100_high_resolution", "high_resolution", 2)
    )
    assert samples["depths"].shape == (3, 2, 1)
    assert np.all(samples["depths"] == 100)


def test_load_chunks_uses_manifest_stage_when_prefix_and_counts_match(tmp_path):
    _write_family(
        tmp_path, "target_low", [(0, 2)], stage_kind="low_resolution", value=20
    )
    _write_family(
        tmp_path, "target_high", [(0, 2)], stage_kind="high_resolution", value=100
    )
    samples = _load_chunks(
        tmp_path, _stage("target_high_resolution", "high_resolution", 2)
    )
    assert np.all(samples["depths"] == 100)


def test_load_chunks_rejects_stale_incomplete_or_ambiguous_family(tmp_path):
    stage = _stage("target_high_resolution", "high_resolution", 3)
    _write_family(
        tmp_path, "target_good", [(0, 3)], stage_kind="high_resolution", value=1
    )
    _write_family(
        tmp_path, "target_stale", [(0, 2)], stage_kind="high_resolution", value=2
    )
    with pytest.raises(ValueError, match="Incomplete checkpoint families"):
        _load_chunks(tmp_path, stage)

    for path in (tmp_path / "chunks").glob("target_stale*"):
        path.unlink()
    _write_family(
        tmp_path, "target_other", [(0, 3)], stage_kind="high_resolution", value=3
    )
    with pytest.raises(ValueError, match="Ambiguous checkpoint families"):
        _load_chunks(tmp_path, stage)
