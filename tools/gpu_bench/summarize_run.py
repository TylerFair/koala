#!/usr/bin/env python3
"""Summarize one replay prefix into compact quantitative JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _distribution(values):
    array = np.asarray(values)
    if not array.size:
        return None
    return {
        "mean": float(np.mean(array)),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix")
    parser.add_argument("--output")
    args = parser.parse_args()
    prefix = Path(args.prefix).resolve()

    timing = _load_json(Path(f"{prefix}.timing.json"))
    arviz = _load_json(Path(f"{prefix}.arviz.json"))
    compile_seconds = float(timing["recorded_compile_seconds"])
    total_seconds = float(timing["total_wall_seconds"])
    payload = {
        "prefix": str(prefix),
        "backend": timing["backend"],
        "channels": timing["channels"],
        "warmup": timing["warmup"],
        "samples": timing["samples"],
        "total_wall_seconds": total_seconds,
        "recorded_compile_seconds": compile_seconds,
        "wall_minus_recorded_compile_seconds": max(
            total_seconds - compile_seconds, 0.0
        ),
    }

    with np.load(f"{prefix}.diagnostics.npz", allow_pickle=False) as diagnostics:
        for field in (
            "num_steps",
            "accept_prob",
            "pareto_k",
            "is_ess",
            "imh_acceptance",
            "map_gradient_norm",
            "hessian_condition_number",
            "map_iterations",
        ):
            arrays = [
                diagnostics[name].ravel()
                for name in diagnostics.files
                if name.endswith("_" + field)
            ]
            if arrays:
                payload[field] = _distribution(np.concatenate(arrays))
        for field in ("diverging", "gate_passed", "fell_back", "converged"):
            arrays = [
                diagnostics[name].ravel()
                for name in diagnostics.files
                if name.endswith("_" + field)
            ]
            if arrays:
                values = np.concatenate(arrays)
                payload[field + "_count"] = int(np.count_nonzero(values))
                payload[field + "_total"] = int(values.size)

    site_ess = {}
    for row in arviz["rows"]:
        value = row.get("ess_bulk")
        if value is not None and np.isfinite(value):
            site_ess.setdefault(row["site"], []).append(float(value))
    payload["ess_bulk"] = {
        site: {
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
            "worst_channel": int(
                min(
                    (
                        row
                        for row in arviz["rows"]
                        if row["site"] == site and row.get("ess_bulk") is not None
                    ),
                    key=lambda row: row["ess_bulk"],
                )["channel"]
            ),
        }
        for site, values in sorted(site_ess.items())
    }
    all_ess = [value for values in site_ess.values() for value in values]
    payload["ess_bulk_all"] = (
        {
            "min": float(np.min(all_ess)),
            "median": float(np.median(all_ess)),
            "max": float(np.max(all_ess)),
        }
        if all_ess
        else None
    )

    comparison_path = Path(f"{prefix}.comparison.json")
    if comparison_path.exists():
        comparison = _load_json(comparison_path)
        requested_sites = {
            "rors[0]",
            "depths[0]",
            "c",
            "v",
            "log_jitter",
            "c1",
            "c2",
        }
        requested_rows = [
            row for row in comparison["rows"] if row["site"] in requested_sites
        ]
        requested_worst_median = (
            max(
                requested_rows,
                key=lambda row: row["abs_median_shift_ref_sigma"],
            )
            if requested_rows
            else None
        )
        requested_worst_ratio = (
            max(
                requested_rows,
                key=lambda row: abs(np.log(row["sigma_ratio"]))
                if row["sigma_ratio"] > 0
                else np.inf,
            )
            if requested_rows
            else None
        )
        payload["fidelity"] = {
            "pass": comparison["pass"],
            "num_failed": comparison["num_failed"],
            "num_rows": comparison["num_rows"],
            "worst_median_shift": comparison["worst_median_shift"],
            "worst_sigma_ratio": comparison["worst_sigma_ratio"],
            "requested_sites_pass": all(
                row["pass"] for row in requested_rows
            ),
            "requested_sites_num_failed": sum(
                not row["pass"] for row in requested_rows
            ),
            "requested_sites_num_rows": len(requested_rows),
            "requested_sites_worst_median_shift": requested_worst_median,
            "requested_sites_worst_sigma_ratio": requested_worst_ratio,
            "requested_sites_by_site": {
                site: {
                    "num_failed": sum(
                        not row["pass"]
                        for row in requested_rows
                        if row["site"] == site
                    ),
                    "num_rows": sum(
                        row["site"] == site for row in requested_rows
                    ),
                    "max_abs_median_shift_ref_sigma": max(
                        row["abs_median_shift_ref_sigma"]
                        for row in requested_rows
                        if row["site"] == site
                    ),
                    "sigma_ratio_min": min(
                        row["sigma_ratio"]
                        for row in requested_rows
                        if row["site"] == site
                    ),
                    "sigma_ratio_max": max(
                        row["sigma_ratio"]
                        for row in requested_rows
                        if row["site"] == site
                    ),
                }
                for site in sorted({row["site"] for row in requested_rows})
            },
        }

    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
