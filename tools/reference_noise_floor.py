#!/usr/bin/env python3
"""Measure seed-to-seed NUTS noise and build a pooled stage reference."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


GATE_CLASS_ORDER = (
    "depth/rors",
    "trend c,v",
    "LD c1,c2",
    "noise log_jitter/total_error",
    "other",
)


def _load_samples(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Sample file must contain a dict; received {type(value)}")
    return {str(name): np.asarray(array) for name, array in value.items()}


def _seed_label(path, fallback):
    match = re.search(r"(?:^|[_-])seed(\d+)(?:[_-]|\.|$)", Path(path).name)
    return f"seed{match.group(1)}" if match else f"sample{fallback}"


def _site_components(name, array, num_channels):
    if array.ndim < 2 or array.shape[1] != num_channels:
        return []
    if array.ndim == 2:
        return [(name, array)]
    flattened = array.reshape(array.shape[0], num_channels, -1)
    trailing_shape = array.shape[2:]
    result = []
    for flat_index in range(flattened.shape[2]):
        component = np.unravel_index(flat_index, trailing_shape)
        suffix = "".join(f"[{index}]" for index in component)
        result.append((f"{name}{suffix}", flattened[:, :, flat_index]))
    return result


def _site_base(component_name):
    return component_name.split("[", 1)[0]


def _site_class(component_name):
    base = _site_base(component_name)
    if base in {"depth", "depths", "rors"}:
        return "depth/rors"
    if base == "c" or re.fullmatch(r"v\d*", base):
        return "trend c,v"
    if base in {"c1", "c2"}:
        return "LD c1,c2"
    if base in {"log_jitter", "total_error"}:
        return "noise log_jitter/total_error"
    return "other"


def _scaled_shift(value_a, value_b, sigma):
    difference = float(value_a - value_b)
    if sigma > 0.0:
        return difference / sigma
    if difference == 0.0:
        return 0.0
    return float(np.copysign(np.inf, difference))


def _finite_percentile(values, percentile):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def _finite_min(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return None if values.size == 0 else float(np.min(values))


def _finite_max(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return None if values.size == 0 else float(np.max(values))


def _summarize_rows(rows):
    by_site = defaultdict(list)
    for row in rows:
        by_site[row["site"]].append(row)
    summaries = []
    for site in sorted(by_site):
        group = by_site[site]
        median_values = [row["abs_median_shift_pooled_sigma"] for row in group]
        ratio_values = [row["sigma_ratio"] for row in group]
        p16_values = [abs(row["p16_shift_pooled_sigma"]) for row in group]
        p84_values = [abs(row["p84_shift_pooled_sigma"]) for row in group]
        summaries.append(
            {
                "site": site,
                "site_class": _site_class(site),
                "num_channel_seed_pairs": len(group),
                "median_shift_max": _finite_max(median_values),
                "median_shift_p95": _finite_percentile(median_values, 95),
                "sigma_ratio_min": _finite_min(ratio_values),
                "sigma_ratio_max": _finite_max(ratio_values),
                "sigma_ratio_p05": _finite_percentile(ratio_values, 5),
                "sigma_ratio_p95": _finite_percentile(ratio_values, 95),
                "abs_p16_shift_max": _finite_max(p16_values),
                "abs_p16_shift_p95": _finite_percentile(p16_values, 95),
                "abs_p84_shift_max": _finite_max(p84_values),
                "abs_p84_shift_p95": _finite_percentile(p84_values, 95),
            }
        )
    return summaries


def _calibrated_gates(rows):
    by_class = defaultdict(list)
    for row in rows:
        by_class[_site_class(row["site"])].append(row)
    result = []
    for site_class in GATE_CLASS_ORDER:
        group = by_class.get(site_class, [])
        if not group:
            continue
        site_groups = defaultdict(list)
        for row in group:
            site_groups[row["site"]].append(
                row["abs_median_shift_pooled_sigma"]
            )
        per_site_p95 = {
            site: _finite_percentile(values, 95)
            for site, values in sorted(site_groups.items())
        }
        median_p95 = _finite_max(
            [value for value in per_site_p95.values() if value is not None]
        )
        ratios = []
        for row in group:
            ratio = float(row["sigma_ratio"])
            if np.isfinite(ratio) and ratio > 0.0:
                ratios.extend((ratio, 1.0 / ratio))
        ratio_min = _finite_min(ratios)
        ratio_max = _finite_max(ratios)
        result.append(
            {
                "site_class": site_class,
                "num_channel_seed_pairs": len(group),
                "seed_median_shift_p95": median_p95,
                "seed_median_shift_p95_by_site": per_site_p95,
                "median_shift_limit_sigma": (
                    None if median_p95 is None else max(0.1, 1.5 * median_p95)
                ),
                "seed_sigma_ratio_symmetric_min": ratio_min,
                "seed_sigma_ratio_symmetric_max": ratio_max,
                "sigma_ratio_lower": (
                    None if ratio_min is None else ratio_min / 1.05
                ),
                "sigma_ratio_upper": (
                    None if ratio_max is None else ratio_max * 1.05
                ),
            }
        )
    return result


def compute_noise_floor(sample_sets, labels, start):
    if not sample_sets:
        raise ValueError("At least one sample set is required.")
    site_names = set(sample_sets[0])
    for samples in sample_sets[1:]:
        if set(samples) != site_names:
            raise ValueError("Seed sample sets contain different site names.")
    first = next(iter(sample_sets[0].values()))
    if first.ndim < 2:
        raise ValueError("Samples must use [draw, channel, ...] layout.")
    num_channels = int(first.shape[1])
    for name in site_names:
        shape_tail = sample_sets[0][name].shape[1:]
        for samples in sample_sets:
            if samples[name].shape[1:] != shape_tail:
                raise ValueError(f"Seed sample shapes differ for site {name!r}.")

    rows = []
    for index_a, index_b in itertools.combinations(range(len(sample_sets)), 2):
        samples_a = sample_sets[index_a]
        samples_b = sample_sets[index_b]
        for name in sorted(site_names):
            components_a = dict(
                _site_components(name, samples_a[name], num_channels)
            )
            components_b = dict(
                _site_components(name, samples_b[name], num_channels)
            )
            if set(components_a) != set(components_b):
                raise ValueError(f"Seed components differ for site {name!r}.")
            for component_name in sorted(components_a):
                values_a = np.asarray(components_a[component_name], dtype=np.float64)
                values_b = np.asarray(components_b[component_name], dtype=np.float64)
                for channel in range(num_channels):
                    channel_a = values_a[:, channel]
                    channel_b = values_b[:, channel]
                    q16_a, median_a, q84_a = np.percentile(
                        channel_a, [16, 50, 84]
                    )
                    q16_b, median_b, q84_b = np.percentile(
                        channel_b, [16, 50, 84]
                    )
                    sigma_a = float(np.std(channel_a, ddof=1))
                    sigma_b = float(np.std(channel_b, ddof=1))
                    pooled_sigma = float(
                        np.sqrt((sigma_a**2 + sigma_b**2) / 2.0)
                    )
                    ratio = (
                        sigma_a / sigma_b
                        if sigma_b > 0.0
                        else (1.0 if sigma_a == 0.0 else float("inf"))
                    )
                    rows.append(
                        {
                            "site": component_name,
                            "site_class": _site_class(component_name),
                            "channel": int(start + channel),
                            "seed_a": labels[index_a],
                            "seed_b": labels[index_b],
                            "sigma_a": sigma_a,
                            "sigma_b": sigma_b,
                            "pooled_sigma": pooled_sigma,
                            "abs_median_shift_pooled_sigma": abs(
                                _scaled_shift(median_a, median_b, pooled_sigma)
                            ),
                            "sigma_ratio": float(ratio),
                            "p16_shift_pooled_sigma": _scaled_shift(
                                q16_a, q16_b, pooled_sigma
                            ),
                            "p84_shift_pooled_sigma": _scaled_shift(
                                q84_a, q84_b, pooled_sigma
                            ),
                        }
                    )
    return {
        "pairwise_rows": rows,
        "per_site_summary": _summarize_rows(rows),
        "calibrated_gates": _calibrated_gates(rows),
        "num_channels": num_channels,
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_pickle(path, value):
    temporary = Path(f"{path}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _atomic_json(path, value):
    temporary = Path(f"{path}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _print_tables(result):
    print(
        "site                              n    median p95    median max  "
        "ratio min  ratio max   |p16| p95   |p84| p95"
    )
    for row in result["per_site_summary"]:
        print(
            f"{row['site'][:32]:32s} {row['num_channel_seed_pairs']:5d} "
            f"{row['median_shift_p95']:13.4g} {row['median_shift_max']:13.4g} "
            f"{row['sigma_ratio_min']:10.4g} {row['sigma_ratio_max']:10.4g} "
            f"{row['abs_p16_shift_p95']:11.4g} {row['abs_p84_shift_p95']:11.4g}"
        )
    print("\nCalibrated gates")
    print("site class                         median limit   sigma-ratio interval")
    for row in result["calibrated_gates"]:
        print(
            f"{row['site_class'][:34]:34s} "
            f"{row['median_shift_limit_sigma']:12.4g}   "
            f"[{row['sigma_ratio_lower']:.4g}, {row['sigma_ratio_upper']:.4g}]"
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", help="Source *_inputs.pkl dump.")
    parser.add_argument(
        "samples",
        nargs="+",
        help="One or more per-seed sample .pkl/.npz files.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument(
        "--output-dir",
        default="/scratch/midway3/tfairnington/accel_stage_inputs/references",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sample_sets = [_load_samples(path) for path in args.samples]
    labels = [_seed_label(path, index) for index, path in enumerate(args.samples)]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Seed labels are not unique: {labels}")
    result = compute_noise_floor(sample_sets, labels, args.start)
    end = args.end if args.end is not None else args.start + result["num_channels"]
    if end - args.start != result["num_channels"]:
        raise ValueError(
            f"Channel range {args.start}:{end} does not match sample width "
            f"{result['num_channels']}."
        )

    pooled = {}
    for name in sorted(sample_sets[0]):
        pooled[name] = np.concatenate(
            [samples[name] for samples in sample_sets], axis=0
        )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_stem = Path(args.dump).stem
    output_stem = (
        f"{dump_stem}_ch{args.start}_{end}_pooled_joint_nuts"
    )
    pooled_path = output_dir / f"{output_stem}.pkl"
    json_path = output_dir / f"{output_stem}_noise_floor.json"
    _atomic_pickle(pooled_path, pooled)

    result.update(
        {
            "schema_version": 1,
            "dump": str(Path(args.dump).resolve()),
            "sample_files": [str(Path(path).resolve()) for path in args.samples],
            "seed_labels": labels,
            "channels": [args.start, end],
            "num_seed_sets": len(sample_sets),
            "draws_per_seed": [
                int(next(iter(samples.values())).shape[0])
                for samples in sample_sets
            ],
            "pooled_draws": int(next(iter(pooled.values())).shape[0]),
            "pooled_reference": str(pooled_path),
            "normalization": (
                "sqrt((sample_sigma_a^2 + sample_sigma_b^2) / 2)"
            ),
            "gate_definition": {
                "median_shift": (
                    "max(0.1 sigma, 1.5 * maximum per-site p95 absolute "
                    "seed-to-seed median shift within the class, in "
                    "pooled-sigma units)"
                ),
                "sigma_ratio": (
                    "full observed class spread including reciprocal ratios, "
                    "widened multiplicatively by 5%"
                ),
            },
        }
    )
    _atomic_json(json_path, result)
    _print_tables(result)
    print(f"\nSaved pooled reference: {pooled_path}")
    print(f"Saved noise floor: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
