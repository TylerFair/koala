#!/usr/bin/env python3
"""Immutable end-to-end parity run for one stellar-informed dataset.

Every candidate is executed through fit_jwst.py in its own output directory.
Existing paths are rejected rather than resumed or overwritten.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import arviz as az
import numpy as np
import yaml

PROJECT = Path("/project/ekempton/tfairnington/JWST")
SAVED_ROOT = Path("/cds2/ekempton/tfairnington/STELLARINFORMED")
SCRATCH_ROOT = Path("/scratch/midway3/tfairnington/accel_parity")
PYTHON = Path("/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python")
sys.path.insert(0, str(PROJECT))

from tools.campaign.run_dataset import (  # noqa: E402
    GATES,
    apply_calibrated_gates,
    compare_samples,
    compare_whitelight,
    concatenate_reference_chunks,
    match_wavelength_product,
)

CHUNK_RE = re.compile(r"_chunk_(\d+)_(\d+)\.pkl$")


def exclusive_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def load_chunks(directory: Path, channels: int) -> dict:
    groups = {}
    for path in directory.glob("chunks/*.pkl"):
        match = CHUNK_RE.search(path.name)
        if match:
            prefix = path.name[: match.start()]
            groups.setdefault(prefix, []).append((int(match[1]), int(match[2]), path))
    complete = []
    for prefix, parts in groups.items():
        parts.sort()
        spans = [(a, b) for a, b, _ in parts]
        if not spans or spans[0][0] != 0 or spans[-1][1] != channels or any(
            left[1] != right[0] for left, right in zip(spans, spans[1:])
        ):
            continue
        payloads = []
        for _, _, path in parts:
            with path.open("rb") as stream:
                payloads.append(pickle.load(stream))
        complete.append((prefix, {k: np.concatenate([p[k] for p in payloads], axis=1)
                                  for k in payloads[0]}))
    if len(complete) != 1:
        raise ValueError(f"Expected one {channels}-channel checkpoint group; found {[x[0] for x in complete]}")
    return complete[0][1]


def depth_ess(samples: dict) -> dict:
    values = np.asarray(samples.get("depths", np.asarray(samples["rors"]) ** 2))
    if values.ndim > 2:
        values = values[..., 0]
    ess = np.asarray(az.ess(az.from_dict(posterior={"depth": values[None]}))["depth"])
    return {"min": float(np.nanmin(ess)), "median": float(np.nanmedian(ess)),
            "p05": float(np.nanpercentile(ess, 5))}


def spectrum_metrics(candidate: dict, reference: dict, wavelengths: np.ndarray) -> dict:
    cand = np.asarray(candidate.get("depths", np.asarray(candidate["rors"]) ** 2))
    ref = np.asarray(reference.get("depths", np.asarray(reference["rors"]) ** 2))
    if cand.ndim > 2: cand = cand[..., 0]
    if ref.ndim > 2: ref = ref[..., 0]
    cm, rm = np.median(cand, axis=0), np.median(ref, axis=0)
    cs, rs = np.std(cand, axis=0, ddof=1), np.std(ref, axis=0, ddof=1)
    delta_ppm = (cm - rm) * 1e6
    weight = 1.0 / np.maximum(rs * 1e6, 1e-30) ** 2
    design = np.column_stack([np.ones(wavelengths.size), wavelengths - np.average(wavelengths, weights=weight)])
    beta = np.linalg.solve(design.T @ (weight[:, None] * design), design.T @ (weight * delta_ppm))
    ratio = cs / rs
    return {
        "weighted_mean_offset_ppm": float(beta[0]),
        "slope_ppm_per_um": float(beta[1]),
        "rms_channel_median_difference_ppm": float(np.sqrt(np.mean(delta_ppm ** 2))),
        "sigma_ratio_median": float(np.median(ratio)),
        "sigma_ratio_p05": float(np.percentile(ratio, 5)),
        "sigma_ratio_p95": float(np.percentile(ratio, 95)),
    }


def candidate_flags(label: str, prism: bool) -> dict:
    common = {
        "spectro_laplace_hessian_method": "finite_difference",
        "spectro_laplace_fd_relative_step": 2e-4,
        "spectro_laplace_trust_radius": 5.0,
        "spectro_laplace_warmup": 200 if label == "A99" else 150,
        "spectro_laplace_start_at_map": True,
        "spectro_laplace_fuse_program": True,
        "spectro_jitter_prior": "lognormal",
        "lowres_num_samples": 2000 if label in {"A", "A99", "B"} else 1000,
        "highres_num_samples": 2000 if label in {"A", "A99", "B"} else 1000,
        "vmap_chunk": 4 if prism else 40,
    }
    if label == "A":
        common.update(spectro_sampler="independent_nuts", spectro_mass_matrix="laplace",
                      spectro_laplace_target_accept=.95, spectro_laplace_max_tree_depth=6)
    elif label == "A99":
        common.update(spectro_sampler="independent_nuts", spectro_mass_matrix="laplace",
                      spectro_laplace_target_accept=.99, spectro_laplace_max_tree_depth=10)
    elif label == "B":
        common.update(spectro_sampler="independent_hmc", spectro_mass_matrix="laplace",
                      spectro_laplace_target_accept=.85, spectro_hmc_num_steps=8,
                      spectro_hmc_trajectory_jitter=.25)
    elif label == "C":
        common = {"spectro_sampler": "laplace_is", "spectro_jitter_prior": "lognormal",
                  "vmap_chunk": 4 if prism else 40}
    elif label == "D":
        # Production-control sampler on the regenerated campaign inputs.
        common = {"spectro_sampler": "joint_nuts", "spectro_jitter_prior": "log_uniform",
                  "lowres_num_warmup": 1000, "lowres_num_samples": 1000,
                  "highres_num_warmup": 1000, "highres_num_samples": 1000,
                  "vmap_chunk": 4 if prism else 40}
    else:
        raise ValueError(label)
    return common


def run_fit(config: Path, log: Path, dump_dir: Path) -> dict:
    env = os.environ.copy()
    env.update(JAX_ENABLE_X64="1", FIT_JWST_DUMP_SAMPLER_INPUTS=str(dump_dir))
    started = time.perf_counter()
    with log.open("x", encoding="utf-8") as stream:
        proc = subprocess.run([str(PYTHON), "fit_jwst.py", "-c", str(config)], cwd=PROJECT,
                              env=env, stdout=stream, stderr=subprocess.STDOUT)
    return {"returncode": proc.returncode, "wall_seconds": time.perf_counter() - started,
            "log": str(log)}


def run_a99_from_dumps(result_dir: Path, reference: dict, wavelengths: np.ndarray) -> dict:
    """Replay the immutable A dumps without rerunning white light or data setup."""
    dump_dir = result_dir / "dumps_A"
    dumps = {}
    for path in sorted(dump_dir.glob("*_inputs.pkl")):
        with path.open("rb") as stream:
            kind = str(pickle.load(stream)["meta"]["stage_kind"])
        dumps[kind] = path
    if set(dumps) != {"low_resolution", "high_resolution"}:
        raise ValueError(f"Expected low/high A dumps in {dump_dir}; found {sorted(dumps)}")

    replay_dir = result_dir / "a99_replay"
    replay_dir.mkdir(exist_ok=False)
    runs = {}
    divergence_total = 0
    high_samples = None
    for kind in ("low_resolution", "high_resolution"):
        prefix = replay_dir / kind
        with dumps[kind].open("rb") as stream:
            channels = int(pickle.load(stream)["meta"]["num_channels"])
        command = [
            str(PYTHON), "tools/run_sampler_on_stage_inputs.py", str(dumps[kind]),
            "--backend", "independent_nuts", "--start", "0", "--end", str(channels),
            "--warmup", "200", "--samples", "2000", "--chunk-size", "40",
            "--platform", "gpu", "--seed", "0", "--potential-atol", "1e-4",
            "--nuts-override", "mass_matrix=laplace",
            "--nuts-override", "laplace_warmup=200",
            "--nuts-override", "laplace_target_accept=0.99",
            "--nuts-override", "laplace_max_tree_depth=10",
            "--nuts-override", "laplace_start_at_map=True",
            "--nuts-override", "laplace_fuse_program=True",
            "--output-prefix", str(prefix),
        ]
        started = time.perf_counter()
        log = result_dir / f"fit_A99_{kind}.log"
        with log.open("x", encoding="utf-8") as stream:
            proc = subprocess.run(command, cwd=PROJECT, stdout=stream,
                                  stderr=subprocess.STDOUT)
        runs[kind] = {"returncode": proc.returncode,
                      "wall_seconds": time.perf_counter() - started,
                      "log": str(log), "command": command}
        if proc.returncode:
            return {"label": "A99", "status": "fit_failed", "stages": runs}
        diagnostics = json.loads(Path(f"{prefix}.diagnostics.json").read_text())
        divergence_total += int(diagnostics["num_divergences_total"])
        if kind == "high_resolution":
            with Path(f"{prefix}.pkl").open("rb") as stream:
                high_samples = pickle.load(stream)

    raw = compare_samples(high_samples, reference, 0, wavelengths.size)["rows"]
    return {
        "label": "A99", "status": "completed", "source_dumps": str(dump_dir),
        "output": str(replay_dir), "stages": runs,
        "num_divergences": divergence_total,
        "fidelity": apply_calibrated_gates(raw),
        "depth_ess": depth_ess(high_samples),
        "spectrum": spectrum_metrics(high_samples, reference, wavelengths),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", default="A,B,C")
    parser.add_argument("--staged-reference", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--resume-existing", action="store_true",
                        help="Add A99 from existing A stage dumps; never reruns fit_jwst.")
    args = parser.parse_args()
    source = Path(args.config).resolve()
    base = yaml.safe_load(source.read_text())
    run_name = str(base["output_dir"])
    result_dir = SCRATCH_ROOT / args.run_tag / run_name
    if args.resume_existing:
        if not result_dir.is_dir():
            raise FileNotFoundError(result_dir)
    else:
        result_dir.mkdir(parents=True, exist_ok=False)
    with Path(args.staged_reference).open("rb") as stream:
        staged = pickle.load(stream)
    reference, wavelengths = staged["samples"], np.asarray(staged["wavelengths"])
    channels = wavelengths.size
    labels = [x.strip().upper() for x in args.candidates.split(",") if x.strip()]
    if args.resume_existing:
        if labels != ["A99"]:
            raise ValueError("--resume-existing requires --candidates A99")
        previous = json.loads((result_dir / "result_final.json").read_text())
        if (result_dir / "result_after_A99.json").exists():
            raise FileExistsError(result_dir / "result_after_A99.json")
        row = run_a99_from_dumps(result_dir, reference, wavelengths)
        result = dict(previous)
        result["status"] = "completed"
        result["candidates"] = [*previous.get("candidates", []), row]
        exclusive_text(result_dir / "result_after_A99.json",
                       json.dumps(result, indent=2, default=str) + "\n")
        print(f"PARITY_RESULT {result_dir / 'result_after_A99.json'}")
        return 0 if row["status"] == "completed" else 2
    informed = str(base.get("flags", {}).get("ld_prior", "")).lower() in {"stellarprior", "informed"}
    if not informed and "C" in labels:
        labels.remove("C")
    result = {"dataset": run_name, "source_config": str(source), "run_tag": args.run_tag,
              "gates": GATES, "candidates": [], "status": "running"}
    for label in labels:
        cfg = yaml.safe_load(source.read_text())
        cfg["output_dir"] = f"{run_name}_PARITY_{args.run_tag}_{label}"
        cfg["host_device"] = "gpu"
        # The LD grid is independent of the sampler.  Keep one immutable cache
        # per dataset/run tag so B and C do not repeat the expensive 125-point
        # stellar-atmosphere interpolation performed by A.
        cfg.setdefault("stellar", {})["ld_prior_cache_dir"] = str(
            result_dir / "shared_ld_prior_cache"
        )
        cfg.setdefault("flags", {}).update(candidate_flags(label, "PRISM" in str(cfg.get("instrument", "")).upper()))
        cfg["flags"]["whitelight_num_warmup"] = 1000
        cfg["flags"]["whitelight_num_samples"] = 1000
        cfg_path = PROJECT / "configs_parity" / f"{args.run_tag}_{source.stem}_{label}.yaml"
        exclusive_text(cfg_path, yaml.safe_dump(cfg, sort_keys=False))
        output = Path(cfg.get("path", ".")) / cfg["output_dir"]
        if output.exists():
            raise FileExistsError(output)
        row = {"label": label, "config": str(cfg_path), "output": str(output)}
        row["fit"] = run_fit(cfg_path, result_dir / f"fit_{label}.log", result_dir / f"dumps_{label}")
        if row["fit"]["returncode"] == 0:
            try:
                samples = load_chunks(output, channels)
                raw = compare_samples(samples, reference, 0, channels)["rows"]
                row["fidelity"] = apply_calibrated_gates(raw)
                row["depth_ess"] = depth_ess(samples)
                row["spectrum"] = spectrum_metrics(samples, reference, wavelengths)
                row["status"] = "completed"
            except Exception as exc:
                row.update(status="analysis_failed", error=f"{type(exc).__name__}: {exc}")
        else:
            row["status"] = "fit_failed"
        result["candidates"].append(row)
        exclusive_text(result_dir / f"result_after_{label}.json", json.dumps(result, indent=2, default=str) + "\n")
    result["status"] = "completed"
    exclusive_text(result_dir / "result_final.json", json.dumps(result, indent=2, default=str) + "\n")
    stamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    manifest = PROJECT / "acceleration_reports" / "OVERNIGHT_MANIFEST.md"
    with manifest.open("a", encoding="utf-8") as stream:
        stream.write(
            f"- {stamp}: parity {run_name} completed; "
            f"candidates={','.join(labels)}; result={result_dir / 'result_final.json'}\n"
        )
    print(f"PARITY_RESULT {result_dir / 'result_final.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
