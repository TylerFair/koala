#!/usr/bin/env python3
"""Aggregate immutable parity outputs, including geometry and diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.campaign.run_dataset import compare_whitelight


def _diagnostics(output: Path) -> dict:
    high, low = [], []
    for path in output.glob("chunks/*.diagnostics.json"):
        payload = json.loads(path.read_text())
        (low if "_R20_" in path.name else high).append(payload)
    def summarize(items):
        return {
            "chunks": len(items),
            "divergences": int(sum(x.get("num_divergences", 0) for x in items)),
            "map_not_converged": int(sum(
                sum(not bool(v) for v in x.get("converged_per_channel", [])) for x in items
            )),
            "fallback_channels": int(sum(x.get("num_fallback_channels", 0) for x in items)),
            "max_pareto_k": float(max(
                [v for x in items for v in x.get("pareto_k_per_channel", [])] or [float("nan")]
            )),
        }
    return {"low": summarize(low), "high": summarize(high)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = {}
    for path in args.root.glob("**/result_after_*.json"):
        result = json.loads(path.read_text())
        row = result.get("candidates", [])[-1]
        if row.get("status") != "completed":
            continue
        key = (result["dataset"], row["label"])
        if key in candidates and path.stat().st_mtime <= candidates[key][0]:
            continue
        output = Path(row["output"])
        saved = Path("/cds2/ekempton/tfairnington/STELLARINFORMED") / result["dataset"]
        geometry = None
        try:
            new_csv = next(output.glob("*_whitelight_bestfit_params.csv"))
            old_csv = next(saved.glob("*_whitelight_bestfit_params.csv"))
            geometry = compare_whitelight(old_csv, new_csv)
        except Exception as exc:
            geometry = {"error": f"{type(exc).__name__}: {exc}"}
        enriched = dict(row)
        enriched["dataset"] = result["dataset"]
        enriched["run_tag"] = result["run_tag"]
        enriched["geometry"] = geometry
        enriched["diagnostics"] = _diagnostics(output)
        candidates[key] = (path.stat().st_mtime, enriched)
    rows = [x[1] for x in sorted(candidates.values(), key=lambda item: (item[1]["dataset"], item[1]["label"]))]
    payload = {"rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=True)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
