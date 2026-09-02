#!/usr/bin/env python3
"""Stage the small saved-reference subset needed on GPU compute nodes."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def stage_run(source: Path, destination_root: Path) -> Path:
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination = destination_root / source.name
    destination.mkdir(parents=True, exist_ok=True)
    chunks = source / "chunks"
    if not chunks.is_dir():
        raise FileNotFoundError(chunks)
    destination_chunks = destination / "chunks"
    destination_chunks.mkdir(exist_ok=True)
    for path in sorted(chunks.glob("*.pkl")):
        shutil.copy2(path, destination_chunks / path.name)
    patterns = (
        "*_wavelengths*.csv",
        "*_bestfit_params*.csv",
        "*.log",
        "*.out",
    )
    copied = []
    for pattern in patterns:
        for path in sorted(source.glob(pattern)):
            target = destination / path.name
            shutil.copy2(path, target)
            copied.append(target)
    for name in ("logs",):
        child = source / name
        if child.is_dir():
            shutil.copytree(child, destination / name, dirs_exist_ok=True)
    if not list(destination_chunks.glob("*.pkl")):
        raise ValueError(f"No checkpoint chunks staged from {source}.")
    if not copied:
        raise ValueError(f"No CSV/log metadata staged from {source}.")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination_root")
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args(argv)
    destination_root = Path(args.destination_root).resolve()
    for item in args.runs:
        destination = stage_run(Path(item).resolve(), destination_root)
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
