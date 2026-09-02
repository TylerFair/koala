"""Combine corrected PRISM parallel checkpoints and compare with serial."""

import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fit_jwst import _concatenate_chunk_samples


ROOT = Path("/scratch/midway3/tfairnington")
PARALLEL = ROOT / "HAT-P-65_PRISM_P2_SPEED_PARALLEL_HIGH" / "chunks"
SERIAL = ROOT / "HAT-P-65_PRISM_P2_SPEED_SERIAL" / "chunks"
REPORT = Path("acceleration_reports/diag_compile_box/prism_parallel_high_compare.json")
COMBINED = PARALLEL / "prism_high_parallel_combined.pkl"
PATTERN = re.compile(r"_chunk_(\d+)_(\d+)\.pkl$")


def indexed(directory):
    result = {}
    for path in directory.glob("*R50*chunk_*_*.pkl"):
        match = PATTERN.search(path.name)
        if match:
            result[(int(match.group(1)), int(match.group(2)))] = path
    return result


started = time.perf_counter()
parallel_paths = indexed(PARALLEL)
serial_paths = indexed(SERIAL)
ranges = sorted(serial_paths)
if set(parallel_paths) != set(ranges):
    raise RuntimeError(
        f"range mismatch: parallel={sorted(parallel_paths)}, serial={ranges}"
    )

parallel_chunks = []
serial_chunks = []
for bounds in ranges:
    with parallel_paths[bounds].open("rb") as stream:
        parallel_chunks.append(pickle.load(stream))
    with serial_paths[bounds].open("rb") as stream:
        serial_chunks.append(pickle.load(stream))

parallel = _concatenate_chunk_samples(parallel_chunks)
serial = _concatenate_chunk_samples(serial_chunks)
with COMBINED.open("xb") as stream:
    pickle.dump(parallel, stream)

sites = {}
all_equal = True
overall_max = 0.0
for name in sorted(set(parallel) | set(serial)):
    if name not in parallel or name not in serial:
        sites[name] = {"missing": "parallel" if name not in parallel else "serial"}
        all_equal = False
        continue
    left = np.asarray(parallel[name])
    right = np.asarray(serial[name])
    shape_equal = left.shape == right.shape
    exact = bool(shape_equal and np.array_equal(left, right, equal_nan=True))
    if shape_equal:
        finite = np.isfinite(left) & np.isfinite(right)
        max_abs = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
    else:
        max_abs = float("inf")
    sites[name] = {
        "parallel_shape": list(left.shape),
        "serial_shape": list(right.shape),
        "bitwise_equal": exact,
        "max_abs_difference": max_abs,
    }
    all_equal &= exact
    overall_max = max(overall_max, max_abs)

payload = {
    "ranges": [list(item) for item in ranges],
    "parallel_checkpoint_files": {str(k): str(v) for k, v in parallel_paths.items()},
    "serial_checkpoint_files": {str(k): str(v) for k, v in serial_paths.items()},
    "combined_output": str(COMBINED),
    "combine_and_compare_wall_seconds": time.perf_counter() - started,
    "all_sites_bitwise_equal": all_equal,
    "all_sites_within_1e-12": all(v.get("max_abs_difference", float("inf")) <= 1e-12 for v in sites.values()),
    "overall_max_abs_difference": overall_max,
    "sites": sites,
}
REPORT.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
