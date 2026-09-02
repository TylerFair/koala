#!/usr/bin/env python3
"""Combine precond2 timing, fidelity, and ESS summaries into one table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCIENCE_SITES = {"A_spot", "c", "c1", "c2", "depths", "rors", "v"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = Path(args.results)
    summaries = Path(args.summaries)
    rows = []
    for timing_path in sorted(results.glob("*_fd_*.timing.json")):
        stem = timing_path.name.removesuffix(".timing.json")
        summary_path = summaries / f"{stem}.json"
        if not summary_path.exists() or "_full" in stem:
            continue
        timing = json.loads(timing_path.read_text())
        summary = json.loads(summary_path.read_text())
        comparison = summary["comparison"]
        ess = summary["ess_bulk_by_site"]
        site_medians = [
            value["median"]
            for site, value in ess.items()
            if site in SCIENCE_SITES
        ]
        site_minima = [
            value["min"]
            for site, value in ess.items()
            if site in SCIENCE_SITES
        ]
        wall = float(timing["chunks"][0]["wall_seconds"])
        compile_seconds = float(
            timing["chunks"][0]["total_recorded_compile_seconds"]
        )
        residual = wall - compile_seconds
        science_ess_median = float(np.median(site_medians))
        rows.append(
            {
                "configuration": stem,
                "wall_seconds": wall,
                "compile_seconds": compile_seconds,
                "residual_seconds": residual,
                "science_passed": comparison["science_passed"],
                "science_values": comparison["science_values"],
                "science_ess_min": float(np.min(site_minima)),
                "science_site_median_ess": science_ess_median,
                "science_site_median_ess_per_wall_second": (
                    science_ess_median / wall
                ),
                "science_site_median_ess_per_residual_second": (
                    science_ess_median / residual
                ),
                "wall_seconds_adjusted_to_1000_median_ess": (
                    wall * 1000.0 / science_ess_median
                ),
                "residual_seconds_adjusted_to_1000_median_ess": (
                    residual * 1000.0 / science_ess_median
                ),
                "sampler_diagnostics": summary["sampler_diagnostics"][0],
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n")
    for row in rows:
        print(
            row["configuration"],
            f"wall={row['wall_seconds']:.2f}",
            f"compile={row['compile_seconds']:.2f}",
            f"gate={row['science_passed']}/{row['science_values']}",
            f"ESSmed/s={row['science_site_median_ess_per_wall_second']:.2f}",
            f"equal1000={row['wall_seconds_adjusted_to_1000_median_ess']:.2f}",
        )


if __name__ == "__main__":
    main()
