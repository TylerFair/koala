#!/usr/bin/env python
"""Compare pooled white-light traces and quantify the seed null."""

import argparse
import glob
import json

import arviz as az
import numpy as np
import pandas as pd


def load(paths):
    datasets = [az.from_netcdf(path).posterior for path in paths]
    common = sorted(set.intersection(*(set(data.data_vars) for data in datasets)))
    pooled = {}
    per_seed = {}
    for name in common:
        arrays = [np.asarray(data[name]).reshape(-1, *data[name].shape[2:]) for data in datasets]
        pooled[name] = np.concatenate(arrays, axis=0)
        per_seed[name] = arrays
    return pooled, per_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ref_paths = sorted(
        path for pattern in args.reference.split(",") for path in glob.glob(pattern)
    )
    cand_paths = sorted(
        path for pattern in args.candidate.split(",") for path in glob.glob(pattern)
    )
    reference, reference_seeds = load(ref_paths)
    candidate, candidate_seeds = load(cand_paths)
    rows = []
    for name in sorted(set(reference) & set(candidate)):
        ref = reference[name]
        cand = candidate[name]
        for index in np.ndindex(ref.shape[1:]) if ref.ndim > 1 else [()]:
            ref_values = ref[(slice(None),) + index]
            cand_values = cand[(slice(None),) + index]
            ref_sigma = float(np.std(ref_values, ddof=1))
            ref_median = float(np.median(ref_values))
            cand_median = float(np.median(cand_values))
            candidate_seed_medians = [
                float(np.median(values[(slice(None),) + index]))
                for values in candidate_seeds[name]
            ]
            reference_seed_medians = [
                float(np.median(values[(slice(None),) + index]))
                for values in reference_seeds[name]
            ]
            if not np.isfinite(ref_sigma) or ref_sigma == 0.0:
                continue
            rows.append({
                "site": name + ("[" + ",".join(map(str, index)) + "]" if index else ""),
                "reference_median": ref_median,
                "candidate_median": cand_median,
                "reference_sigma": ref_sigma,
                "candidate_sigma": float(np.std(cand_values, ddof=1)),
                "median_shift_sigma": (cand_median - ref_median) / ref_sigma,
                "sigma_ratio": float(np.std(cand_values, ddof=1)) / ref_sigma,
                "candidate_seed_null_sigma": (
                    max(candidate_seed_medians) - min(candidate_seed_medians)
                ) / ref_sigma,
                "reference_seed_null_sigma": (
                    max(reference_seed_medians) - min(reference_seed_medians)
                ) / ref_sigma,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, index=False)
    print(json.dumps({
        "reference_files": ref_paths,
        "candidate_files": cand_paths,
        "max_abs_shift_sigma": float(frame.median_shift_sigma.abs().max()),
        "sigma_ratio_range": [float(frame.sigma_ratio.min()), float(frame.sigma_ratio.max())],
        "output": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
