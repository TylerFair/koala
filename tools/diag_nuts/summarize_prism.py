#!/usr/bin/env python3
"""Compare a PRISM candidate with a finite-draw joint-NUTS reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from summarize_precond import _components, _load_samples, ess_by_site


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    candidate = _load_samples(args.candidate)
    reference = _load_samples(args.reference)
    rows = []
    for site in sorted(set(candidate) & set(reference)):
        cand = np.asarray(candidate[site])[:, : args.end]
        ref = np.asarray(reference[site])[:, : args.end]
        if cand.ndim < 2 or ref.ndim < 2:
            continue
        for suffix, cand_component in _components(cand):
            ref_component = dict(_components(ref)).get(suffix)
            if ref_component is None:
                continue
            for channel in range(args.end):
                cand_values = cand_component[:, channel].astype(np.float64)
                ref_values = ref_component[:, channel].astype(np.float64)
                ref_sigma = np.std(ref_values, ddof=1)
                # Fixed deterministic sites (for example PRISM's fixed LD)
                # have no reference sampling scale and are not comparable in
                # sigma units.
                if not np.isfinite(ref_sigma) or ref_sigma <= 0.0:
                    continue
                rows.append(
                    {
                        "site": f"{site}{suffix}",
                        "base_site": site,
                        "channel": channel,
                        "median_shift_ref_sigma": float(
                            (np.median(cand_values) - np.median(ref_values))
                            / ref_sigma
                        ),
                        "sigma_ratio": float(
                            np.std(cand_values, ddof=1) / ref_sigma
                        ),
                    }
                )

    by_site = {}
    for site in sorted({row["base_site"] for row in rows}):
        selected = [row for row in rows if row["base_site"] == site]
        shifts = np.abs([row["median_shift_ref_sigma"] for row in selected])
        ratios = np.asarray([row["sigma_ratio"] for row in selected])
        by_site[site] = {
            "median_abs_median_shift_ref_sigma": float(np.median(shifts)),
            "max_abs_median_shift_ref_sigma": float(np.max(shifts)),
            "sigma_ratio_min": float(np.min(ratios)),
            "sigma_ratio_median": float(np.median(ratios)),
            "sigma_ratio_max": float(np.max(ratios)),
        }

    with np.load(args.diagnostics, allow_pickle=False) as diagnostics:
        prefix = "chunk_000_"
        gradients = np.asarray(diagnostics[prefix + "map_gradient_norm"])
        decrements = np.asarray(diagnostics[prefix + "map_newton_decrement"])
        iterations = np.asarray(diagnostics[prefix + "map_iterations"])
        conditions = np.asarray(diagnostics[prefix + "map_condition_number"])
        map_by_channel = [
            {
                "channel": channel,
                "gradient_norm": float(gradients[channel]),
                "newton_decrement": float(decrements[channel]),
                "iterations": int(iterations[channel]),
                "condition_number": float(conditions[channel]),
            }
            for channel in range(args.end)
        ]

    payload = {
        "candidate": str(Path(args.candidate).resolve()),
        "reference": str(Path(args.reference).resolve()),
        "candidate_draws": int(next(iter(candidate.values())).shape[0]),
        "reference_draws": int(next(iter(reference.values())).shape[0]),
        "by_site": by_site,
        "rows": rows,
        "ess_bulk_by_site": ess_by_site(candidate, 0, args.end),
        "map_by_channel": map_by_channel,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("by_site", "ess_bulk_by_site", "map_by_channel")}, indent=2))


if __name__ == "__main__":
    main()
