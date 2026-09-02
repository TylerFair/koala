#!/usr/bin/env python3
"""Benchmark the generic fused contraction as a direct-quadratic candidate.

This reuses the direct-u1/u2 quadratic benchmark harness but substitutes the
degree-agnostic fused Green-basis contraction for the specialized evaluator.
Inputs remain literal ``u1,u2`` coefficients; no Kipping transform is used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import jax

jax.config.update("jax_enable_x64", True)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.jaxoplanet.limb_dark_fused import light_curve as fused_light_curve
from tools import benchmark_jaxoplanet_quadratic_gpu as benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(benchmark.PRESETS), required=True)
    parser.add_argument("--batch", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()
    if args.platform != "auto":
        jax.config.update("jax_platform_name", args.platform)

    preset = benchmark.PRESETS[args.preset]
    times, metadata = benchmark.load_time_grid(
        preset["path"], attribute="time", transit_t0=preset["t0"]
    )
    original_loader = benchmark.specialized_light_curve
    benchmark.specialized_light_curve = lambda: fused_light_curve
    try:
        result = benchmark.run_case(
            times,
            args.batch,
            period=preset["period"],
            duration=preset["duration"],
            impact=0.45,
            u1=0.30,
            u2=0.20,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    finally:
        benchmark.specialized_light_curve = original_loader

    result["candidate"] = "fused_degree2_direct_u1_u2_no_kipping"
    benchmark._print_result(result)
    payload = {
        "environment": {
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
            "time_grid": metadata,
        },
        "result": result,
    }
    if args.json_path:
        destination = Path(args.json_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
        print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
