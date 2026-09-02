#!/usr/bin/env python3
"""Run one saved-population dataset through the acceleration campaign."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import pickle
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

PROJECT = Path("/project/ekempton/tfairnington/JWST")
SAVED_ROOT = Path("/cds2/ekempton/tfairnington/STELLARINFORMED")
DEFAULT_RESULT_ROOT = Path("/scratch/midway3/tfairnington/accel_campaign")
PYTHON = Path("/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python")

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools.run_sampler_on_stage_inputs import compare_samples

GATES = {
    "depth/rors": {"median": 0.133, "ratio": (0.816, 1.226)},
    "trend c,v": {"median": 0.164, "ratio": (0.826, 1.210)},
    "LD c1,c2": {"median": 0.202, "ratio": (0.754, 1.327)},
    "other": {"median": 0.150, "ratio": (0.829, 1.207)},
}
EXCLUDED_GATE_SITES = {"log_jitter", "total_error"}


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _site_base(name: str) -> str:
    return name.split("[", 1)[0]


def _site_class(name: str) -> str | None:
    base = _site_base(name)
    if base in EXCLUDED_GATE_SITES:
        return None
    if base in {"depth", "depths", "rors"}:
        return "depth/rors"
    if base == "c" or re.fullmatch(r"v\d*", base):
        return "trend c,v"
    if base in {"c1", "c2"}:
        return "LD c1,c2"
    return "other"


def apply_calibrated_gates(rows: list[dict]) -> dict:
    by_site = defaultdict(list)
    gated = []
    for source in rows:
        row = dict(source)
        site_class = _site_class(row["site"])
        row["site_class"] = site_class
        if site_class is None:
            row["gate_included"] = False
            row["gate_pass"] = None
        else:
            gate = GATES[site_class]
            row["gate_included"] = True
            row["gate_pass"] = bool(
                row["abs_median_shift_ref_sigma"] <= gate["median"]
                and gate["ratio"][0]
                <= row["sigma_ratio"]
                <= gate["ratio"][1]
            )
            gated.append(row)
        by_site[row["site"]].append(row)

    summaries = []
    for site in sorted(by_site):
        group = by_site[site]
        shifts = np.asarray(
            [row["abs_median_shift_ref_sigma"] for row in group],
            dtype=np.float64,
        )
        ratios = np.asarray(
            [row["sigma_ratio"] for row in group], dtype=np.float64
        )
        included = [row for row in group if row["gate_included"]]
        summaries.append(
            {
                "site": site,
                "site_class": _site_class(site),
                "num_channels": len(group),
                "median_shift_p95": float(np.percentile(shifts, 95)),
                "median_shift_max": float(np.max(shifts)),
                "sigma_ratio_min": float(np.min(ratios)),
                "sigma_ratio_max": float(np.max(ratios)),
                "gate_pass_fraction": (
                    None
                    if not included
                    else float(
                        np.mean([bool(row["gate_pass"]) for row in included])
                    )
                ),
            }
        )
    return {
        "gate_definition": GATES,
        "excluded_sites": sorted(EXCLUDED_GATE_SITES),
        "rows": list(rows),
        "per_site": summaries,
        "gate_num_rows": len(gated),
        "gate_num_passed": sum(bool(row["gate_pass"]) for row in gated),
        "gate_pass_fraction": (
            None
            if not gated
            else float(np.mean([bool(row["gate_pass"]) for row in gated]))
        ),
        "gate_pass": bool(gated) and all(bool(row["gate_pass"]) for row in gated),
    }


def clone_campaign_config(config_path: Path, campaign_dir: Path) -> tuple[Path, dict]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config {config_path} is not a mapping.")
    original_output = str(config["output_dir"])
    config["output_dir"] = f"{original_output}_CAMPAIGN"
    config["host_device"] = "gpu"
    flags = config.setdefault("flags", {})
    flags["whitelight_num_warmup"] = 1000
    flags["whitelight_num_samples"] = 1000
    flags["spectro_jitter_prior"] = "lognormal"
    flags["save_whitelight_trace"] = False
    campaign_dir.mkdir(parents=True, exist_ok=True)
    output = campaign_dir / f"campaign_{config_path.name}"
    _atomic_text(output, yaml.safe_dump(config, sort_keys=False))
    return output, {
        "saved_run_name": original_output,
        "campaign_output_name": config["output_dir"],
        "fits_path": str(
            Path(config.get("path", "."))
            / str(config.get("input_dir", ""))
            / str(config["fits_file"])
        ),
        "instrument": config.get("instrument"),
        "order": config.get("order"),
        "nrs": config.get("nrs"),
    }


def _run_logged(command: list[str], log_path: Path, env=None) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("COMMAND " + " ".join(command) + "\n")
        stream.flush()
        process = subprocess.run(
            command,
            cwd=PROJECT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - started
    return {
        "command": command,
        "log": str(log_path),
        "returncode": int(process.returncode),
        "wall_seconds": float(elapsed),
    }


def _dump_payload(path: Path) -> dict:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or "meta" not in payload:
        raise ValueError(f"Not a stage-input dump: {path}")
    return payload


def discover_dumps(dump_dir: Path) -> dict[str, Path]:
    result = {}
    for path in sorted(dump_dir.glob("*_inputs.pkl")):
        kind = str(_dump_payload(path)["meta"].get("stage_kind", ""))
        if kind in result:
            raise ValueError(f"Multiple {kind} dumps in {dump_dir}.")
        result[kind] = path
    required = {"low_resolution", "high_resolution"}
    if set(result) != required:
        raise ValueError(
            f"Expected low/high dumps in {dump_dir}; found {sorted(result)}."
        )
    return result


_CHUNK_RE = re.compile(r"^(?P<prefix>.+)_chunk_(?P<start>\d+)_(?P<end>\d+)\.pkl$")


def concatenate_reference_chunks(saved_run: Path, expected_channels: int) -> tuple[dict, dict]:
    groups = defaultdict(list)
    for path in sorted((saved_run / "chunks").glob("*.pkl")):
        match = _CHUNK_RE.match(path.name)
        if match:
            groups[match.group("prefix")].append(
                (int(match.group("start")), int(match.group("end")), path)
            )
    candidates = []
    failures = []
    for prefix, chunks in groups.items():
        chunks.sort()
        if not chunks or chunks[0][0] != 0:
            failures.append(f"{prefix}: does not begin at channel zero")
            continue
        cursor = 0
        mappings = []
        valid = True
        for start, end, path in chunks:
            if start != cursor or end <= start:
                failures.append(f"{prefix}: non-contiguous at {start}:{end}")
                valid = False
                break
            with path.open("rb") as stream:
                mapping = pickle.load(stream)
            if not isinstance(mapping, dict):
                failures.append(f"{prefix}: {path.name} is not a dict")
                valid = False
                break
            widths = {
                np.asarray(value).shape[1]
                for value in mapping.values()
                if np.asarray(value).ndim >= 2
            }
            if widths != {end - start}:
                failures.append(f"{prefix}: filename/data width mismatch")
                valid = False
                break
            mappings.append(mapping)
            cursor = end
        if valid and cursor == expected_channels:
            sites = set(mappings[0])
            if any(set(mapping) != sites for mapping in mappings[1:]):
                failures.append(f"{prefix}: sites differ across chunks")
                continue
            combined = {
                site: np.concatenate(
                    [np.asarray(mapping[site]) for mapping in mappings], axis=1
                )
                for site in sorted(sites)
            }
            candidates.append((prefix, combined, chunks))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one contiguous {expected_channels}-channel checkpoint "
            f"group in {saved_run}; found {[item[0] for item in candidates]}; "
            f"failures={failures}."
        )
    prefix, combined, chunks = candidates[0]
    return combined, {
        "prefix": prefix,
        "chunks": [str(path) for _, _, path in chunks],
        "channels": expected_channels,
    }


def _read_wavelength_csv(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return np.asarray([], dtype=np.float64)
    keys = rows[0].keys()
    key = next(
        (name for name in ("wavelength_center", "wavelength") if name in keys),
        None,
    )
    if key is None:
        raise ValueError(f"No wavelength column in {path}.")
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def match_wavelength_product(saved_run: Path, dump_payload: dict) -> dict:
    expected = np.asarray(dump_payload["meta"].get("wavelength"), dtype=np.float64)
    matches = []
    examined = []
    for path in sorted(saved_run.glob("*_wavelengths*.csv")):
        measured = _read_wavelength_csv(path)
        examined.append({"path": str(path), "channels": int(measured.size)})
        if measured.shape == expected.shape and np.allclose(
            measured, expected, rtol=1.0e-7, atol=1.0e-10
        ):
            matches.append(path)
    if not matches:
        raise ValueError(
            f"No saved wavelength grid matches dump shape {expected.shape}; "
            f"examined={examined}."
        )
    # Duplicate visit-labelled products can be byte-equivalent; prefer the
    # un-suffixed canonical product and record all matches.
    chosen = min(matches, key=lambda path: (path.stem.count("_V"), len(path.name)))
    return {"chosen": str(chosen), "matches": [str(path) for path in matches]}


def _literal_float(value) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        parsed = value
    array = np.asarray(parsed, dtype=np.float64).reshape(-1)
    return float(array[0])


def _summary_sigma(row: dict, base: str) -> tuple[float, float, float]:
    direct = _literal_float(row.get(f"{base}_err"))
    low = _literal_float(row.get(f"{base}_err_low"))
    high = _literal_float(row.get(f"{base}_err_high"))
    if not np.isfinite(low):
        low = direct
    if not np.isfinite(high):
        high = direct
    sigma = float(np.nanmean([abs(low), abs(high)]))
    return sigma, abs(low), abs(high)


def _component_series(array: np.ndarray, channels: int):
    array = np.asarray(array)
    if array.ndim < 2 or array.shape[1] != channels:
        return []
    if array.ndim == 2:
        return [("", array)]
    flattened = array.reshape(array.shape[0], channels, -1)
    result = []
    for index in range(flattened.shape[-1]):
        suffix = "".join(
            f"[{item}]"
            for item in np.unravel_index(index, array.shape[2:])
        )
        result.append((suffix, flattened[:, :, index]))
    return result


def compare_lowres_summary(candidate: dict, csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reference_rows = list(csv.DictReader(stream))
    channels = len(reference_rows)
    aliases = {"depths": "depth", "A_spot": "A_spot"}
    rows = []
    for site in sorted(candidate):
        reference_name = aliases.get(site, site)
        if reference_name not in reference_rows[0]:
            continue
        for suffix, values in _component_series(candidate[site], channels):
            for channel, reference_row in enumerate(reference_rows):
                sample = np.asarray(values[:, channel], dtype=np.float64)
                q16, median, q84 = np.percentile(sample, [16, 50, 84])
                reference_median = _literal_float(reference_row[reference_name])
                sigma, sigma_low, sigma_high = _summary_sigma(
                    reference_row, reference_name
                )
                if not np.isfinite(sigma) or sigma <= 0.0:
                    continue
                rows.append(
                    {
                        "site": f"{site}{suffix}",
                        "channel": channel,
                        "abs_median_shift_ref_sigma": abs(
                            median - reference_median
                        )
                        / sigma,
                        "sigma_ratio": float(np.std(sample, ddof=1) / sigma),
                        "p16_shift_ref_sigma": float(
                            (q16 - (reference_median - sigma_low)) / sigma
                        ),
                        "p84_shift_ref_sigma": float(
                            (q84 - (reference_median + sigma_high)) / sigma
                        ),
                    }
                )
    if not rows:
        raise ValueError(f"No comparable low-resolution sites in {csv_path}.")
    return rows


def match_lowres_summary(saved_run: Path, dump_payload: dict) -> Path:
    channels = int(dump_payload["meta"]["num_channels"])
    matches = []
    for path in sorted(saved_run.glob("*_bestfit_params*.csv")):
        if "whitelight" in path.name:
            continue
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                count = sum(1 for _ in csv.DictReader(stream))
        except (OSError, csv.Error):
            continue
        if count == channels:
            matches.append(path)
    if not matches:
        raise ValueError(
            f"No {channels}-row low-resolution bestfit CSV in {saved_run}."
        )
    return min(matches, key=lambda path: (path.stem.count("_V"), len(path.name)))


def compare_whitelight(old_path: Path, new_path: Path) -> dict:
    with old_path.open(newline="", encoding="utf-8") as stream:
        old = next(csv.DictReader(stream))
    with new_path.open(newline="", encoding="utf-8") as stream:
        new = next(csv.DictReader(stream))
    parameters = []
    for name in ("duration", "t0", "b", "rors", "depths"):
        if name not in old or name not in new:
            continue
        old_value = _literal_float(old[name])
        new_value = _literal_float(new[name])
        sigma, _, _ = _summary_sigma(old, name)
        parameters.append(
            {
                "parameter": name,
                "saved": old_value,
                "campaign": new_value,
                "difference": new_value - old_value,
                "saved_sigma": sigma,
                "abs_shift_saved_sigma": (
                    None
                    if not np.isfinite(sigma) or sigma <= 0.0
                    else abs(new_value - old_value) / sigma
                ),
            }
        )
    finite = [
        row["abs_shift_saved_sigma"]
        for row in parameters
        if row["abs_shift_saved_sigma"] is not None
    ]
    return {
        "saved_csv": str(old_path),
        "campaign_csv": str(new_path),
        "parameters": parameters,
        "max_abs_shift_saved_sigma": max(finite) if finite else None,
    }


def _candidate_options(label: str, prism: bool) -> tuple[str, list[str]]:
    common = [
        "--builder-override",
        "jitter_prior=lognormal",
    ]
    metric = [
        "--nuts-override",
        "mass_matrix=laplace",
        "--nuts-override",
        "laplace_hessian_method=finite_difference",
        "--nuts-override",
        "laplace_fd_relative_step=0.0002",
        "--nuts-override",
        "laplace_fuse_program=True",
        "--nuts-override",
        f"laplace_map_iterations={200 if prism else 16}",
        "--nuts-override",
        "laplace_warmup=150",
        "--nuts-override",
        f"laplace_target_accept={0.99 if prism else 0.95}",
        "--nuts-override",
        f"laplace_start_at_map={str(prism)}",
    ]
    if label == "A":
        return "independent_nuts", common + metric + [
            "--nuts-override",
            "laplace_max_tree_depth=5",
        ]
    if label == "B":
        return "independent_hmc", common + metric + [
            "--nuts-override",
            "num_steps=8",
            "--nuts-override",
            "trajectory_jitter=0.25",
        ]
    if label == "C":
        return "laplace_is", common + [
            "--backend-override",
            "laplace_is_num_draws=4096",
            "--backend-override",
            "laplace_is_rounds=2",
            "--backend-override",
            "laplace_is_draw_chunk_size=256",
            "--backend-override",
            f"laplace_is_map_maxiter={200 if prism else 200}",
            "--backend-override",
            "laplace_is_map_tol=1e-8",
            "--backend-override",
            "laplace_is_student_df=3.0",
            "--backend-override",
            "laplace_is_scale_inflation=1.5",
            "--backend-override",
            "laplace_is_fallback=True",
        ]
    raise ValueError(f"Unknown candidate label {label!r}.")


def _candidate_diagnostics(prefix: Path) -> dict:
    diagnostics = json.loads(
        Path(f"{prefix}.diagnostics.json").read_text(encoding="utf-8")
    )
    timing = json.loads(
        Path(f"{prefix}.timing.json").read_text(encoding="utf-8")
    )
    arviz = json.loads(
        Path(f"{prefix}.arviz.json").read_text(encoding="utf-8")
    )
    khat = [
        row.get("pareto_k_max")
        for row in diagnostics.get("chunks", [])
        if row.get("pareto_k_max") is not None
    ]
    return {
        "wall_seconds": timing.get("total_wall_seconds"),
        "recorded_compile_seconds": timing.get("recorded_compile_seconds"),
        "divergences": diagnostics.get("num_divergences_total", 0),
        "min_bulk_ess": arviz.get("ess_bulk_min"),
        "max_pareto_k": max(khat) if khat else None,
        "chunks": diagnostics.get("chunks", []),
    }


def _find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern} in {directory}.")
    return min(matches, key=lambda path: (path.stem.count("_V"), len(path.name)))


def _recover_reference_wall(saved_run: Path) -> dict:
    candidates = []
    patterns = [
        re.compile(r"(?:total|wall)[ _-]*(?:time)?[^0-9]*(\d+(?:\.\d+)?)\s*s", re.I),
        re.compile(r"^real\s+(\d+(?:\.\d+)?)$", re.M),
    ]
    for path in sorted(saved_run.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".log", ".out"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in patterns:
            for match in pattern.finditer(text):
                candidates.append(
                    {"seconds": float(match.group(1)), "source": str(path)}
                )
    return {
        "wall_seconds": max((row["seconds"] for row in candidates), default=None),
        "candidates": candidates,
        "note": (
            None
            if candidates
            else "No recoverable .log/.out timing in the saved run directory."
        ),
    }


def _write_dataset_markdown(result: dict, path: Path) -> None:
    lines = [
        f"# Campaign result: {result['dataset']}\n",
        f"- Instrument: `{result['instrument']}`",
        f"- Config: `{result['config']}`",
        f"- Saved run: `{result['saved_run']}`",
        f"- FITS: `{result['fits_path']}`",
        f"- Dump bridge: `{result['dump_bridge']}`",
        f"- Total wall: `{result.get('total_wall_seconds')}` s",
        "",
        "| Candidate | Stage | Status | Wall (s) | Gate pass fraction | Divergences | Min ESS | max k-hat |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result.get("candidate_rows", []):
        fidelity = row.get("fidelity") or {}
        diagnostics = row.get("diagnostics") or {}
        lines.append(
            "| {candidate} | {stage} | {status} | {wall} | {gate} | {div} | {ess} | {khat} |".format(
                candidate=row["candidate"],
                stage=row["stage"],
                status=row["status"],
                wall=diagnostics.get("wall_seconds", ""),
                gate=fidelity.get("gate_pass_fraction", ""),
                div=diagnostics.get("divergences", ""),
                ess=diagnostics.get("min_bulk_ess", ""),
                khat=diagnostics.get("max_pareto_k", ""),
            )
        )
    geometry = result.get("white_light_geometry", {})
    lines.extend(
        [
            "",
            "## White-light geometry",
            "",
            f"Maximum saved-posterior-normalized shift: `{geometry.get('max_abs_shift_saved_sigma')}` sigma.",
            "",
        ]
    )
    _atomic_text(path, "\n".join(lines))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument(
        "--candidates",
        default="A",
        help="Comma-separated subset of A,B,C; A is always required.",
    )
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument(
        "--saved-root",
        default=str(SAVED_ROOT),
        help="Root containing saved run directories (stage to /scratch on GPU nodes).",
    )
    parser.add_argument("--skip-fit", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    candidates = [item.strip().upper() for item in args.candidates.split(",") if item.strip()]
    if "A" not in candidates or any(item not in {"A", "B", "C"} for item in candidates):
        raise ValueError("--candidates must include A and contain only A,B,C.")

    campaign_config, metadata = clone_campaign_config(
        config_path, PROJECT / "configs_campaign"
    )
    saved_run = Path(args.saved_root).resolve() / metadata["saved_run_name"]
    fits_path = Path(metadata["fits_path"])
    if not saved_run.is_dir():
        raise FileNotFoundError(f"Saved run does not exist: {saved_run}")
    if not fits_path.is_file():
        raise FileNotFoundError(f"FITS input does not exist: {fits_path}")

    result_dir = Path(args.result_root).resolve() / metadata["saved_run_name"]
    dump_dir = result_dir / "dumps"
    candidate_dir = result_dir / "candidates"
    reference_dir = result_dir / "reference"
    for directory in (dump_dir, candidate_dir, reference_dir):
        directory.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "dataset": metadata["saved_run_name"],
        "instrument": metadata["instrument"],
        "order": metadata["order"],
        "nrs": metadata["nrs"],
        "config": str(config_path),
        "campaign_config": str(campaign_config),
        "saved_run": str(saved_run),
        "fits_path": str(fits_path),
        "dump_bridge": {
            "enabled": True,
            "lowres_warmup": 2,
            "lowres_samples": 2,
            "lowres_max_tree_depth": 2,
            "white_light_warmup": 1000,
            "white_light_samples": 1000,
        },
        "candidate_rows": [],
        "status": "running",
    }
    previous_result_path = result_dir / "result.json"
    if args.skip_fit and previous_result_path.is_file():
        previous = json.loads(previous_result_path.read_text(encoding="utf-8"))
        if previous.get("fit"):
            result["fit"] = previous["fit"]
        result["resumed_from_existing_dumps"] = True
    campaign_started = time.perf_counter()
    _atomic_json(result_dir / "result.json", result)

    if not args.skip_fit:
        env = os.environ.copy()
        env.update(
            {
                "JAX_ENABLE_X64": "1",
                "FIT_JWST_DUMP_SAMPLER_INPUTS": str(dump_dir),
                "FIT_JWST_DUMP_SAMPLER_INPUTS_EXIT": "1",
                "FIT_JWST_DUMP_BRIDGE_LOWRES": "1",
                "FIT_JWST_DUMP_BRIDGE_LOWRES_WARMUP": "2",
                "FIT_JWST_DUMP_BRIDGE_LOWRES_SAMPLES": "2",
                "FIT_JWST_DUMP_BRIDGE_LOWRES_MAX_TREE_DEPTH": "2",
            }
        )
        result["fit"] = _run_logged(
            [str(PYTHON), "fit_jwst.py", "-c", str(campaign_config)],
            result_dir / "fit.log",
            env=env,
        )
        _atomic_json(result_dir / "result.json", result)
        if result["fit"]["returncode"] != 0:
            result["status"] = "fit_failed"
            result["total_wall_seconds"] = time.perf_counter() - campaign_started
            _atomic_json(result_dir / "result.json", result)
            _write_dataset_markdown(result, result_dir / "result.md")
            return 2

    try:
        dumps = discover_dumps(dump_dir)
        dump_payloads = {name: _dump_payload(path) for name, path in dumps.items()}
        result["dumps"] = {name: str(path) for name, path in dumps.items()}
        for kind, payload in dump_payloads.items():
            result.setdefault("alignment", {})[kind] = {
                "wavelength": match_wavelength_product(saved_run, payload)
            }
        high_channels = int(dump_payloads["high_resolution"]["meta"]["num_channels"])
        high_reference, high_manifest = concatenate_reference_chunks(
            saved_run, high_channels
        )
        high_reference_path = reference_dir / "high_resolution_samples.pkl"
        with high_reference_path.open("wb") as stream:
            pickle.dump(high_reference, stream, protocol=pickle.HIGHEST_PROTOCOL)
        result["alignment"]["high_resolution"]["reference"] = high_manifest
        low_summary_path = match_lowres_summary(
            saved_run, dump_payloads["low_resolution"]
        )
        result["alignment"]["low_resolution"]["reference_summary"] = str(
            low_summary_path
        )
    except Exception as error:
        result["status"] = "alignment_failed"
        result["alignment_error"] = f"{type(error).__name__}: {error}"
        result["total_wall_seconds"] = time.perf_counter() - campaign_started
        _atomic_json(result_dir / "result.json", result)
        _write_dataset_markdown(result, result_dir / "result.md")
        return 3

    prism = "PRISM" in str(metadata["instrument"]).upper()
    candidate_chunk_size = 4 if prism else 40
    for label in candidates:
        backend, options = _candidate_options(label, prism)
        for stage_kind in ("low_resolution", "high_resolution"):
            payload = dump_payloads[stage_kind]
            channels = int(payload["meta"]["num_channels"])
            prefix = candidate_dir / f"{label}_{backend}_{stage_kind}"
            command = [
                str(PYTHON),
                "tools/run_sampler_on_stage_inputs.py",
                str(dumps[stage_kind]),
                "--backend",
                backend,
                "--start",
                "0",
                "--end",
                str(channels),
                "--warmup",
                "150",
                "--samples",
                "1000",
                "--chunk-size",
                str(candidate_chunk_size),
                "--platform",
                "gpu",
                "--seed",
                "0",
                "--potential-atol",
                "1e-4",
                *options,
                "--output-prefix",
                str(prefix),
            ]
            if stage_kind == "high_resolution":
                command.extend(["--compare", str(high_reference_path)])
            run = _run_logged(
                command,
                result_dir / f"candidate_{label}_{stage_kind}.log",
            )
            row = {
                "candidate": label,
                "backend": backend,
                "stage": stage_kind,
                "status": "failed" if run["returncode"] else "completed",
                "run": run,
            }
            if run["returncode"] == 0:
                try:
                    with Path(f"{prefix}.pkl").open("rb") as stream:
                        candidate_samples = pickle.load(stream)
                    if stage_kind == "high_resolution":
                        raw = compare_samples(
                            candidate_samples, high_reference, 0, channels
                        )["rows"]
                    else:
                        raw = compare_lowres_summary(
                            candidate_samples, low_summary_path
                        )
                    row["fidelity"] = apply_calibrated_gates(raw)
                    row["diagnostics"] = _candidate_diagnostics(prefix)
                except Exception as error:
                    row["status"] = "analysis_failed"
                    row["analysis_error"] = f"{type(error).__name__}: {error}"
            result["candidate_rows"].append(row)
            _atomic_json(result_dir / "result.json", result)

    campaign_output = (
        Path(yaml.safe_load(campaign_config.read_text())["path"])
        / yaml.safe_load(campaign_config.read_text())["output_dir"]
    )
    try:
        old_white = _find_one(saved_run, "*_whitelight_bestfit_params*.csv")
        new_white = _find_one(campaign_output, "*_whitelight_bestfit_params*.csv")
        result["white_light_geometry"] = compare_whitelight(old_white, new_white)
    except Exception as error:
        result["white_light_geometry"] = {
            "error": f"{type(error).__name__}: {error}"
        }
    result["reference_timing"] = _recover_reference_wall(saved_run)
    result["status"] = "completed"
    result["total_wall_seconds"] = time.perf_counter() - campaign_started
    _atomic_json(result_dir / "result.json", result)
    _write_dataset_markdown(result, result_dir / "result.md")
    print(f"CAMPAIGN_RESULT {result_dir / 'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
