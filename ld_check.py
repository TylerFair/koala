#!/usr/bin/env python3
"""Audit stellar-informed LD prior cache widths for near-delta-function sigmas.

Example:
    python tyler_scripts/audit_ld_prior_sigma_floor.py \
        --glob "/scratch/midway3/tfairnington/*STELLARINFORMEDLD*" \
        --threshold 0.01
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd


STOP_TOKENS = {
    "SOSS",
    "ORDER1",
    "ORDER2",
    "G395M",
    "G395H",
    "NRS1",
    "NRS2",
    "NIRISS",
    "NIRSPEC",
    "MIRI",
    "LRS",
    "PRISM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan ld_prior_cache CSVs for c1/c2 stellar sigma values below a floor."
    )
    parser.add_argument(
        "--glob",
        required=True,
        help="Quoted glob for run directories, e.g. '/scratch/.../*STELLARINFORMEDLD*'",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Flag any cache with c1_sigma_star or c2_sigma_star below this value.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional path to write the per-file audit table as CSV.",
    )
    return parser.parse_args()


def infer_planet(run_dir_name: str) -> str:
    parts = run_dir_name.split("_")
    keep: list[str] = []
    for part in parts:
        if part in STOP_TOKENS or part.startswith("ORDER"):
            break
        keep.append(part)
    return "_".join(keep) if keep else run_dir_name


def load_flagged_rows(run_glob: str, threshold: float) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    scanned = 0
    for run_path_str in sorted(glob.glob(run_glob)):
        run_path = Path(run_path_str)
        cache_dir = run_path / "ld_prior_cache"
        if not cache_dir.is_dir():
            continue
        for csv_path in sorted(cache_dir.glob("*.csv")):
            scanned += 1
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                rows.append(
                    {
                        "planet": infer_planet(run_path.name),
                        "run_dir": run_path.name,
                        "cache_file": csv_path.name,
                        "status": f"read_failed: {exc}",
                    }
                )
                continue

            required = {"c1_sigma_star", "c2_sigma_star"}
            if not required.issubset(df.columns):
                rows.append(
                    {
                        "planet": infer_planet(run_path.name),
                        "run_dir": run_path.name,
                        "cache_file": csv_path.name,
                        "status": "missing_sigma_columns",
                    }
                )
                continue

            c1_min = float(df["c1_sigma_star"].min())
            c2_min = float(df["c2_sigma_star"].min())
            low_c1 = c1_min < threshold
            low_c2 = c2_min < threshold
            rows.append(
                {
                    "planet": infer_planet(run_path.name),
                    "run_dir": run_path.name,
                    "cache_file": csv_path.name,
                    "status": "flagged" if (low_c1 or low_c2) else "ok",
                    "low_c1": low_c1,
                    "low_c2": low_c2,
                    "c1_min": c1_min,
                    "c2_min": c2_min,
                    "c1_median": float(df["c1_sigma_star"].median()),
                    "c2_median": float(df["c2_sigma_star"].median()),
                }
            )
    return rows, scanned


def print_summary(rows: list[dict[str, object]], threshold: float, scanned: int) -> None:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    flagged_rows = [r for r in rows if r.get("status") == "flagged"]
    other_rows = [r for r in rows if r.get("status") not in {"ok", "flagged"}]

    flagged_planets = sorted({str(r["planet"]) for r in flagged_rows})
    flagged_runs = sorted({str(r["run_dir"]) for r in flagged_rows})

    print(f"Scanned cache files: {scanned}")
    print(f"Threshold: {threshold:g}")
    print(f"OK cache files: {len(ok_rows)}")
    print(f"Flagged cache files: {len(flagged_rows)}")
    print(f"Other/problem cache files: {len(other_rows)}")
    print(f"Flagged unique runs: {len(flagged_runs)}")
    print(f"Flagged unique planets: {len(flagged_planets)}")
    print()

    if flagged_planets:
        print("Flagged planets:")
        for planet in flagged_planets:
            print(f"  - {planet}")
        print()

    if flagged_rows:
        print("Flagged cache files:")
        for row in flagged_rows:
            print(
                "  - "
                f"{row['run_dir']}/{row['cache_file']}: "
                f"low_c1={row['low_c1']}, low_c2={row['low_c2']}, "
                f"c1_min={row['c1_min']:.6g}, c2_min={row['c2_min']:.6g}"
            )
        print()

    if other_rows:
        print("Problem cache files:")
        for row in other_rows:
            print(f"  - {row['run_dir']}/{row['cache_file']}: {row['status']}")


def main() -> None:
    args = parse_args()
    rows, scanned = load_flagged_rows(args.glob, args.threshold)
    print_summary(rows, args.threshold, scanned)
    if args.output_csv:
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        print()
        print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()

