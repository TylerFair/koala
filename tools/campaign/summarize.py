#!/usr/bin/env python3
"""Aggregate dataset campaign result.json files into JSON and Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


POPULATION_COUNTS = {
    "G395H": 32,
    "SOSS order 1": 18,
    "SOSS order 2": 13,
    "PRISM": 4,
    "G395M": 3,
}


def _mode(instrument, order):
    instrument = str(instrument or "")
    if "SOSS" in instrument:
        return f"SOSS order {int(order)}"
    for name in ("G395H", "PRISM", "G395M"):
        if name in instrument:
            return name
    return instrument or "unknown"


def summarize(result_paths):
    datasets = []
    rows = []
    for path in sorted(Path(item) for item in result_paths):
        result = json.loads(path.read_text(encoding="utf-8"))
        fit_wall = (result.get("fit") or {}).get("wall_seconds")
        candidate_runs = [
            (row.get("run") or {}).get("wall_seconds")
            for row in result.get("candidate_rows", [])
            if (row.get("run") or {}).get("wall_seconds") is not None
        ]
        candidate_a_runs = [
            (row.get("run") or {}).get("wall_seconds")
            for row in result.get("candidate_rows", [])
            if row.get("candidate") == "A"
            and (row.get("run") or {}).get("wall_seconds") is not None
        ]
        total_wall = result.get("total_wall_seconds")
        measured_parts = (fit_wall or 0.0) + sum(candidate_runs)
        overhead = (
            max(0.0, total_wall - measured_parts)
            if total_wall is not None
            else 0.0
        )
        candidate_a_stages = {
            row.get("stage")
            for row in result.get("candidate_rows", [])
            if row.get("candidate") == "A" and row.get("status") == "completed"
        }
        a_only_wall = (
            None
            if fit_wall is None
            or candidate_a_stages != {"low_resolution", "high_resolution"}
            else fit_wall + sum(candidate_a_runs) + overhead
        )
        datasets.append(
            {
                "dataset": result["dataset"],
                "instrument": result.get("instrument"),
                "mode": _mode(result.get("instrument"), result.get("order")),
                "status": result.get("status"),
                "total_wall_seconds": total_wall,
                "fit_wall_seconds": fit_wall,
                "a_only_wall_seconds": a_only_wall,
                "white_geometry_max_shift_sigma": (
                    result.get("white_light_geometry") or {}
                ).get("max_abs_shift_saved_sigma"),
            }
        )
        for row in result.get("candidate_rows", []):
            fidelity = row.get("fidelity") or {}
            diagnostics = row.get("diagnostics") or {}
            rows.append(
                {
                    "dataset": result["dataset"],
                    "instrument": result.get("instrument"),
                    "candidate": row["candidate"],
                    "backend": row.get("backend"),
                    "stage": row["stage"],
                    "status": row["status"],
                    "wall_seconds": diagnostics.get("wall_seconds"),
                    "gate_pass_fraction": fidelity.get("gate_pass_fraction"),
                    "gate_pass": fidelity.get("gate_pass"),
                    "divergences": diagnostics.get("divergences"),
                    "min_bulk_ess": diagnostics.get("min_bulk_ess"),
                    "max_pareto_k": diagnostics.get("max_pareto_k"),
                    "per_site": fidelity.get("per_site", []),
                }
            )
    walls = [
        row["total_wall_seconds"]
        for row in datasets
        if row["total_wall_seconds"] is not None
    ]
    mode_walls = {}
    for row in datasets:
        if row["a_only_wall_seconds"] is not None:
            mode_walls.setdefault(row["mode"], []).append(
                row["a_only_wall_seconds"]
            )
    projection_rows = []
    proxy = {
        "SOSS order 2": "SOSS order 1",
        "G395M": "G395H",
    }
    projected_total = 0.0
    projection_complete = True
    for mode, count in POPULATION_COUNTS.items():
        source = mode if mode in mode_walls else proxy.get(mode)
        values = mode_walls.get(source or "", [])
        estimate = float(np.mean(values)) if values else None
        if estimate is None:
            projection_complete = False
        else:
            projected_total += count * estimate
        projection_rows.append(
            {
                "mode": mode,
                "count": count,
                "pilot_source": source,
                "a_only_seconds_per_dataset": estimate,
                "projected_seconds": None if estimate is None else count * estimate,
            }
        )
    return {
        "datasets": datasets,
        "rows": rows,
        "num_datasets": len(datasets),
        "pilot_total_wall_seconds": float(np.sum(walls)) if walls else None,
        "pilot_mean_wall_seconds": float(np.mean(walls)) if walls else None,
        "projected_70_serial_seconds": (
            projected_total if projection_complete else None
        ),
        "projected_70_two_gpu_ideal_seconds": (
            projected_total / 2.0 if projection_complete else None
        ),
        "population_projection": projection_rows,
        "projection_note": (
            "A-only projection; SOSS order 1 pilots proxy order 2 and G395H "
            "pilots proxy G395M. Two-GPU value is an ideal load-balance lower bound."
        ),
    }


def markdown(summary):
    lines = [
        "# Accelerated sampler validation campaign summary\n",
        "| Dataset | Instrument | Candidate | Stage | Status | Wall (s) | Gate fraction | Divergences | Min ESS | max k-hat |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {dataset} | {instrument} | {candidate} | {stage} | {status} | {wall} | {gate} | {div} | {ess} | {khat} |".format(
                dataset=row["dataset"],
                instrument=row["instrument"],
                candidate=row["candidate"],
                stage=row["stage"],
                status=row["status"],
                wall=row["wall_seconds"],
                gate=row["gate_pass_fraction"],
                div=row["divergences"],
                ess=row["min_bulk_ess"],
                khat=row["max_pareto_k"],
            )
        )
    lines.extend(
        [
            "",
            "## Per-site fidelity",
            "",
            "| Dataset | Candidate | Stage | Site | Median shift p95/max (sigma) | Sigma-ratio min/max | Gate fraction |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        for site in row.get("per_site", []):
            lines.append(
                "| {dataset} | {candidate} | {stage} | {site} | {p95}/{maximum} | {rmin}/{rmax} | {gate} |".format(
                    dataset=row["dataset"],
                    candidate=row["candidate"],
                    stage=row["stage"],
                    site=site.get("site"),
                    p95=site.get("median_shift_p95"),
                    maximum=site.get("median_shift_max"),
                    rmin=site.get("sigma_ratio_min"),
                    rmax=site.get("sigma_ratio_max"),
                    gate=site.get("gate_pass_fraction"),
                )
            )
    lines.extend(
        [
            "",
            "## Population cost projection",
            "",
            "| Mode | Runs | Pilot proxy | A-only wall/run (s) | Projected wall (s) |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in summary["population_projection"]:
        lines.append(
            "| {mode} | {count} | {source} | {wall} | {projected} |".format(
                mode=row["mode"],
                count=row["count"],
                source=row["pilot_source"],
                wall=row["a_only_seconds_per_dataset"],
                projected=row["projected_seconds"],
            )
        )
    lines.extend(
        [
            "",
            f"Pilot datasets: {summary['num_datasets']}",
            f"Pilot total wall: {summary['pilot_total_wall_seconds']} s",
            f"Mean-based 70-run serial projection: {summary['projected_70_serial_seconds']} s",
            f"Ideal two-GPU projection: {summary['projected_70_two_gpu_ideal_seconds']} s",
            f"Projection assumptions: {summary['projection_note']}",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = summarize(args.results)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prefix.with_suffix(".md").write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
