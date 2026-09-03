#!/usr/bin/env python3
"""Compare independently produced PRISM chunk replays with one serial replay."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def _load(path):
    with open(path, "rb") as stream:
        return pickle.load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--part", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    serial = _load(args.serial)
    parts = [_load(path) for path in args.part]
    combined = {
        name: np.concatenate([np.asarray(part[name]) for part in parts], axis=1)
        for name in serial
    }
    if set(combined) != set(serial):
        raise ValueError("Serial and parallel sample sites differ")
    sites = {}
    exact = True
    for name in sorted(serial):
        reference = np.asarray(serial[name])
        candidate = combined[name]
        equal = bool(np.array_equal(reference, candidate, equal_nan=True))
        exact &= equal
        sites[name] = {
            "shape": list(reference.shape),
            "draw_for_draw_equal": equal,
            "max_abs_difference": float(
                np.nanmax(np.abs(reference - candidate))
            ),
        }
    payload = {"draw_for_draw_equal": exact, "sites": sites}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if exact else 1)


if __name__ == "__main__":
    main()
