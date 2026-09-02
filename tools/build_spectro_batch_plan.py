#!/usr/bin/env python3
"""Build a fit-consumable difficulty batch plan from pilot diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import uuid


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.spectro_manifest_tools import build_batch_plan_from_chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-chunk",
        action="append",
        required=True,
        metavar="START:STOP=PATH",
        help=(
            "Pilot diagnostic with explicit global channel indices. Repeat for "
            "every chunk; comma-separated exact indices are also accepted."
        ),
    )
    parser.add_argument("--nominal-width", required=True, type=int)
    parser.add_argument("--quarantine-width", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.nominal_width < 1 or args.quarantine_width < 1:
        raise ValueError("nominal-width and quarantine-width must be positive.")
    plan, sources = build_batch_plan_from_chunks(
        args.pilot_chunk,
        nominal_width=args.nominal_width,
        quarantine_width=args.quarantine_width,
        require_workload_provenance=True,
    )
    _atomic_write_json(args.output, plan.to_manifest())
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "fingerprint_sha256": plan.fingerprint_sha256,
                "num_channels": len(plan.original_channel_indices),
                "num_batches": len(plan.batches),
                "sources": sources,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
