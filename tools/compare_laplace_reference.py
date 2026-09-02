#!/usr/bin/env python3
"""Compare benchmark summaries to a pooled joint-NUTS reference."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def _summary(reference):
    output = {}
    for site, value in reference.items():
        array = np.asarray(value, dtype=np.float64)
        if array.ndim < 2:
            continue
        array = array.reshape((array.shape[0], array.shape[1], -1))
        output[site] = {
            "median": np.median(array, axis=0),
            "sigma": np.std(array, axis=0, ddof=1),
            "q16": np.quantile(array, 0.16, axis=0),
            "q84": np.quantile(array, 0.84, axis=0),
        }
    return output


def _gate_map(noise_floor):
    result = {}
    if noise_floor is None:
        return result
    for gate in noise_floor["calibrated_gates"]:
        for site_component in gate["seed_median_shift_p95_by_site"]:
            result[site_component.split("[")[0]] = gate
    return result


def _aggregate(values):
    flat = np.asarray(values, dtype=np.float64).ravel()
    return {
        "min": float(np.min(flat)),
        "median": float(np.median(flat)),
        "p95": float(np.quantile(flat, 0.95)),
        "max": float(np.max(flat)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--method", default="laplace_is")
    reference_group = parser.add_mutually_exclusive_group(required=True)
    reference_group.add_argument("--reference-pkl")
    reference_group.add_argument("--reference-json")
    parser.add_argument("--reference-method", default="independent_nuts")
    parser.add_argument("--noise-floor-json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    candidate_payload = json.loads(Path(args.candidate_json).read_text())
    candidate = candidate_payload["results"][args.method]["sites"]
    if args.reference_pkl is not None:
        with open(args.reference_pkl, "rb") as stream:
            reference = _summary(pickle.load(stream))
    else:
        reference_payload = json.loads(Path(args.reference_json).read_text())
        reference_sites = reference_payload["results"][args.reference_method]["sites"]
        reference = {
            site: {
                name: np.asarray(values, dtype=np.float64)
                for name, values in summary.items()
                if name in {"median", "sigma", "q16", "q84"}
            }
            for site, summary in reference_sites.items()
        }
    noise_floor = (
        None
        if args.noise_floor_json is None
        else json.loads(Path(args.noise_floor_json).read_text())
    )
    gates = _gate_map(noise_floor)
    comparisons = {}
    total_gate_values = 0
    total_gate_passes = 0
    for site in sorted(set(candidate) & set(reference)):
        cand = {name: np.asarray(value) for name, value in candidate[site].items()}
        ref = reference[site]
        scale = np.maximum(ref["sigma"], np.finfo(np.float64).tiny)
        median_shift = np.abs(cand["median"] - ref["median"]) / scale
        sigma_ratio = cand["sigma"] / scale
        q16_shift = (cand["q16"] - ref["q16"]) / scale
        q84_shift = (cand["q84"] - ref["q84"]) / scale
        site_result = {
            "abs_median_shift_over_sigma_ref": median_shift.tolist(),
            "sigma_ratio": sigma_ratio.tolist(),
            "q16_shift_over_sigma_ref": q16_shift.tolist(),
            "q84_shift_over_sigma_ref": q84_shift.tolist(),
            "aggregate": {
                "abs_median_shift_over_sigma_ref": _aggregate(median_shift),
                "sigma_ratio": _aggregate(sigma_ratio),
                "abs_q16_shift_over_sigma_ref": _aggregate(np.abs(q16_shift)),
                "abs_q84_shift_over_sigma_ref": _aggregate(np.abs(q84_shift)),
            },
        }
        if site in gates:
            gate = gates[site]
            passes = (
                (median_shift <= gate["median_shift_limit_sigma"])
                & (sigma_ratio >= gate["sigma_ratio_lower"])
                & (sigma_ratio <= gate["sigma_ratio_upper"])
            )
            total_gate_values += passes.size
            total_gate_passes += int(passes.sum())
            site_result["calibrated_gate"] = {
                "site_class": gate["site_class"],
                "median_shift_limit_sigma": gate["median_shift_limit_sigma"],
                "sigma_ratio_lower": gate["sigma_ratio_lower"],
                "sigma_ratio_upper": gate["sigma_ratio_upper"],
                "passed": passes.tolist(),
                "num_passed": int(passes.sum()),
                "num_values": int(passes.size),
            }
        comparisons[site] = site_result

    output = {
        "candidate_json": str(Path(args.candidate_json).resolve()),
        "method": args.method,
        "reference_pkl": (
            None if args.reference_pkl is None
            else str(Path(args.reference_pkl).resolve())
        ),
        "reference_json": (
            None if args.reference_json is None
            else str(Path(args.reference_json).resolve())
        ),
        "reference_method": (
            None if args.reference_json is None else args.reference_method
        ),
        "noise_floor_json": (
            None if args.noise_floor_json is None
            else str(Path(args.noise_floor_json).resolve())
        ),
        "calibrated_gate_passed": total_gate_passes,
        "calibrated_gate_values": total_gate_values,
        "sites": comparisons,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "calibrated_gate_passed": total_gate_passes,
        "calibrated_gate_values": total_gate_values,
        "sites": {
            site: result["aggregate"] for site, result in comparisons.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
