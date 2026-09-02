#!/usr/bin/env python3
"""Summarize preconditioned-NUTS accuracy, ESS, and lockstep diagnostics."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import arviz as az
import numpy as np


def _load_samples(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {name: np.asarray(data[name]) for name in data.files}
    with path.open("rb") as stream:
        return {name: np.asarray(value) for name, value in pickle.load(stream).items()}


def _components(array):
    trailing = array.shape[2:]
    if not trailing:
        return [("", array)]
    return [
        (
            "[" + ",".join(str(index) for index in component) + "]",
            array[(slice(None), slice(None), *component)],
        )
        for component in np.ndindex(trailing)
    ]


def _gate_by_site(noise_floor):
    result = {}
    for gate in noise_floor["calibrated_gates"]:
        for component in gate["seed_median_shift_p95_by_site"]:
            result[component.split("[")[0]] = gate
    return result


def _slice_channels(array, start, end):
    width = end - start
    if array.ndim < 2:
        return array
    if array.shape[1] == width:
        return array
    if array.shape[1] >= end:
        return array[:, start:end]
    raise ValueError(
        f"Array with {array.shape[1]} channels cannot supply {start}:{end}."
    )


def compare(candidate, reference, noise_floor, start, end):
    gates = _gate_by_site(noise_floor)
    rows = []
    for site in sorted(set(candidate) & set(reference)):
        if site not in gates:
            continue
        cand_components = dict(
            _components(_slice_channels(np.asarray(candidate[site]), start, end))
        )
        ref_components = dict(
            _components(_slice_channels(np.asarray(reference[site]), start, end))
        )
        gate = gates[site]
        for suffix in sorted(set(cand_components) & set(ref_components)):
            cand_values = cand_components[suffix]
            ref_values = ref_components[suffix]
            for local_channel in range(end - start):
                cand = np.asarray(cand_values[:, local_channel], dtype=np.float64)
                ref = np.asarray(ref_values[:, local_channel], dtype=np.float64)
                ref_sigma = float(np.std(ref, ddof=1))
                median_shift = float(
                    abs(np.median(cand) - np.median(ref)) / ref_sigma
                )
                sigma_ratio = float(np.std(cand, ddof=1) / ref_sigma)
                passed = bool(
                    median_shift <= gate["median_shift_limit_sigma"]
                    and gate["sigma_ratio_lower"]
                    <= sigma_ratio
                    <= gate["sigma_ratio_upper"]
                )
                rows.append(
                    {
                        "site": f"{site}{suffix}",
                        "base_site": site,
                        "site_class": gate["site_class"],
                        "channel": start + local_channel,
                        "median_shift_sigma": median_shift,
                        "sigma_ratio": sigma_ratio,
                        "pass": passed,
                    }
                )
    by_site = {}
    for site in sorted({row["base_site"] for row in rows}):
        selected = [row for row in rows if row["base_site"] == site]
        by_site[site] = {
            "passed": sum(row["pass"] for row in selected),
            "values": len(selected),
            "median_shift_sigma_max": max(
                row["median_shift_sigma"] for row in selected
            ),
            "sigma_ratio_min": min(row["sigma_ratio"] for row in selected),
            "sigma_ratio_median": float(
                np.median([row["sigma_ratio"] for row in selected])
            ),
            "sigma_ratio_max": max(row["sigma_ratio"] for row in selected),
        }
    science_sites = {"A_spot", "c", "c1", "c2", "depths", "rors", "v"}
    science = [row for row in rows if row["base_site"] in science_sites]
    return {
        "passed": sum(row["pass"] for row in rows),
        "values": len(rows),
        "science_passed": sum(row["pass"] for row in science),
        "science_values": len(science),
        "by_site": by_site,
        "failures": [row for row in rows if not row["pass"]],
        "rows": rows,
    }


def ess_by_site(samples, start, end):
    output = {}
    for site, array in sorted(samples.items()):
        array = np.asarray(array)
        if array.ndim < 2:
            continue
        try:
            array = _slice_channels(array, start, end)
        except ValueError:
            continue
        values = []
        for _, component in _components(array):
            for channel in range(component.shape[1]):
                values.append(
                    float(
                        np.asarray(
                            az.ess(
                                np.asarray(component[:, channel])[None, :],
                                method="bulk",
                            )
                        )
                    )
                )
        finite = np.asarray([value for value in values if np.isfinite(value)])
        if finite.size:
            output[site] = {
                "min": float(np.min(finite)),
                "median": float(np.median(finite)),
                "max": float(np.max(finite)),
                "values": int(finite.size),
            }
    return output


def sampler_diagnostics(path):
    if path is None:
        return None
    chunks = []
    with np.load(path, allow_pickle=False) as data:
        prefixes = sorted(
            {name.rsplit("_", 2)[0] for name in data.files if name.endswith("num_steps")}
        )
        for prefix in prefixes:
            steps = np.asarray(data[f"{prefix}_num_steps"], dtype=np.float64)
            row = {
                "chunk": prefix,
                "num_steps_mean": float(np.mean(steps)),
                "num_steps_median": float(np.median(steps)),
            }
            for name, operation in (
                ("diverging", lambda value: int(np.count_nonzero(value))),
                ("accept_prob", lambda value: float(np.mean(value))),
                ("step_size", lambda value: float(np.median(value))),
            ):
                key = f"{prefix}_{name}"
                if key in data:
                    output_name = {
                        "diverging": "divergences",
                        "accept_prob": "accept_prob_mean",
                        "step_size": "step_size_median",
                    }[name]
                    row[output_name] = operation(data[key])
            if steps.ndim >= 2:
                lane_max = np.max(steps, axis=1)
                row.update(
                    lane_max_steps_mean=float(np.mean(lane_max)),
                    lane_max_steps_median=float(np.median(lane_max)),
                    lane_max_steps_max=float(np.max(lane_max)),
                    lockstep_ratio_mean=float(
                        np.mean(lane_max) / np.mean(steps)
                    ),
                )
            for name in (
                "map_gradient_norm",
                "map_newton_decrement",
                "map_iterations",
                "map_condition_number",
                "map_hessian_min_eigenvalue",
                "map_hessian_relative_error",
            ):
                key = f"{prefix}_{name}"
                if key in data:
                    row[f"{name}_median"] = float(np.median(data[key]))
                    row[f"{name}_max"] = float(np.max(data[key]))
            chunks.append(row)
    return chunks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--noise-floor", required=True)
    parser.add_argument("--diagnostics")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    candidate = _load_samples(args.candidate)
    reference = _load_samples(args.reference)
    noise_floor = json.loads(Path(args.noise_floor).read_text())
    payload = {
        "candidate": str(Path(args.candidate).resolve()),
        "reference": str(Path(args.reference).resolve()),
        "noise_floor": str(Path(args.noise_floor).resolve()),
        "channels": [args.start, args.end],
        "comparison": compare(
            candidate, reference, noise_floor, args.start, args.end
        ),
        "ess_bulk_by_site": ess_by_site(candidate, args.start, args.end),
        "sampler_diagnostics": sampler_diagnostics(args.diagnostics),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary = {
        "comparison": {
            name: payload["comparison"][name]
            for name in ("passed", "values", "science_passed", "science_values")
        },
        "by_site": payload["comparison"]["by_site"],
        "ess_bulk_by_site": payload["ess_bulk_by_site"],
        "sampler_diagnostics": payload["sampler_diagnostics"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
