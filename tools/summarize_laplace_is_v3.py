#!/usr/bin/env python3
"""Calibrated science fidelity, ESS, and spectrum offsets for v3 runs."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools.campaign.run_dataset import apply_calibrated_gates
from tools.run_sampler_on_stage_inputs import compare_samples
from tools.spectro_stage_inputs import load_stage_inputs


def load_samples(path):
    with open(path, "rb") as stream:
        return {k: np.asarray(v) for k, v in pickle.load(stream).items()}


def reference_slice(samples, start, count):
    return {
        k: (v if v.ndim < 2 or v.shape[1] == count else v[:, start:start + count])
        for k, v in samples.items()
    }


def ess_summary(path):
    data = json.load(open(path, encoding="utf-8"))
    grouped = defaultdict(list)
    for row in data.get("rows", []):
        value = row.get("ess_bulk")
        if value is not None and np.isfinite(value):
            grouped[row["site"]].append(float(value))
    return {
        site: {"min": min(values), "median": float(np.median(values))}
        for site, values in sorted(grouped.items())
    }


def spectrum(candidate, reference, wavelength):
    site = "depths" if "depths" in candidate and "depths" in reference else "rors"
    cand = np.asarray(candidate[site])
    ref = np.asarray(reference[site])
    if site == "rors":
        cand, ref = cand ** 2, ref ** 2
    while cand.ndim > 2:
        cand = cand[..., 0]
        ref = ref[..., 0]
    delta = (np.median(cand, axis=0) - np.median(ref, axis=0)) * 1.0e6
    wavelength = np.asarray(wavelength, dtype=float)[:delta.size]
    good = np.isfinite(delta) & np.isfinite(wavelength)
    x, y = wavelength[good], delta[good]
    slope, intercept = np.polyfit(x - np.mean(x), y, 1)
    return {
        "site": site,
        "channels": int(y.size),
        "mean_depth_offset_ppm": float(np.mean(y)),
        "slope_ppm_per_micron": float(slope),
        "rms_channel_median_difference_ppm": float(np.sqrt(np.mean(y ** 2))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("reference")
    parser.add_argument("dump")
    parser.add_argument("--arviz", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidate = load_samples(args.candidate)
    count = np.asarray(next(iter(candidate.values()))).shape[1]
    reference = reference_slice(load_samples(args.reference), args.start, count)
    raw = compare_samples(candidate, reference, 0, count)
    calibrated = apply_calibrated_gates(raw["rows"])
    stage = load_stage_inputs(args.dump, validate_potential=False).select(
        args.start, args.start + count
    )
    result = {
        "candidate": args.candidate,
        "reference": args.reference,
        "calibrated": calibrated,
        "ess_bulk_by_site": ess_summary(args.arviz),
        "spectrum": spectrum(candidate, reference, stage.meta["wavelength"]),
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        f"science pass {calibrated['gate_num_passed']}/"
        f"{calibrated['gate_num_rows']}"
    )
    print(json.dumps(result["spectrum"], indent=2))


if __name__ == "__main__":
    main()
