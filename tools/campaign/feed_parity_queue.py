#!/usr/bin/env python3
"""Conservatively feed immutable parity templates to the file GPU queue."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=Path("acceleration_reports/gpu_queue"))
    parser.add_argument("--max-pending", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--stop", default="09:00")
    parser.add_argument("--first", type=int, default=213)
    args = parser.parse_args()
    hour, minute = map(int, args.stop.split(":"))
    stop = dt.datetime.now().astimezone().replace(hour=hour, minute=minute, second=0, microsecond=0)
    if stop <= dt.datetime.now().astimezone():
        raise ValueError(f"stop time {stop} is not in the future")
    templates = sorted(
        (p for p in (args.queue / "templates").glob("*_parity_*.sh")
         if int(p.name.split("_", 1)[0]) >= args.first),
        key=lambda p: int(p.name.split("_", 1)[0]),
    )
    submitted = {
        p.name for state in ("pending", "running", "done")
        for p in (args.queue / state).glob("*_parity_*.sh")
    }
    observed_exits = {p.name for p in (args.queue / "done").glob("*_parity_*.exit")}
    while dt.datetime.now().astimezone() < stop:
        exits = {p.name for p in (args.queue / "done").glob("*_parity_*.exit")}
        for name in sorted(exits - observed_exits):
            path = args.queue / "done" / name
            print(f"{dt.datetime.now().astimezone().isoformat()} completed {name}: {path.read_text().strip()}", flush=True)
        observed_exits = exits
        pending = list((args.queue / "pending").glob("*_parity_*.sh"))
        while len(pending) < args.max_pending:
            source = next((p for p in templates if p.name not in submitted), None)
            if source is None:
                print("No unsubmitted parity templates remain.", flush=True)
                return
            target = args.queue / "pending" / source.name
            with source.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            os.chmod(target, source.stat().st_mode)
            submitted.add(source.name)
            pending.append(target)
            print(f"{dt.datetime.now().astimezone().isoformat()} queued {target.name}", flush=True)
        time.sleep(args.poll_seconds)
    print(f"Reached stop time {stop.isoformat()}; no further submissions.", flush=True)


if __name__ == "__main__":
    main()
