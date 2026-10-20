"""Run balanced CI shards, releasing JAX compilation memory after each file."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def discover_files(root):
    return sorted(str(p.relative_to(root)) for p in (root / "tests").rglob("test_*.py"))


def make_shards(paths, durations, count):
    """Assign every file exactly once with deterministic longest-first packing."""
    if count < 1:
        raise ValueError("shard count must be positive")
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate test file")
    groups = [[] for _ in range(count)]
    totals = [0.0] * count
    for path in sorted(paths, key=lambda p: (-float(durations.get(p, 30.0)), p)):
        index = min(range(count), key=lambda i: (totals[i], i))
        groups[index].append(path)
        totals[index] += float(durations.get(path, 30.0))
    return [sorted(group) for group in groups], totals


def junit_counts(path):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    return {
        "passed": sum(not any(c.tag in {"failure", "error", "skipped"} for c in t) for t in cases),
        "failed": sum(t.find("failure") is not None for t in cases),
        "errors": sum(t.find("error") is not None for t in cases),
        "skipped": sum(t.find("skipped") is not None for t in cases),
    }


def acceptable_exit(code, counts):
    # pytest returns 5 for a file containing only deselected slow tests or a
    # module-level optional-dependency skip. Errors must never become success.
    if counts["failed"] or counts["errors"]:
        return False
    return code == 0 or (code == 5 and counts["passed"] == 0)


def group_rss_kib(group):
    """Conservative aggregate RSS including child Python/compiler processes."""
    total = 0
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            stat = (directory / "stat").read_text().rpartition(") ")[2].split()
            if int(stat[2]) != group:
                continue
            match = re.search(r"^VmRSS:\s+(\d+)", (directory / "status").read_text(), re.M)
            if match:
                total += int(match.group(1))
        except (OSError, ValueError, IndexError):
            # Processes can exit between reading stat and status.
            continue
    return total


def run_file(path, output, memory_bytes, timeout_seconds):
    label = Path(path).stem + "-" + hashlib.sha256(path.encode()).hexdigest()[:10]
    logfile = output / f"{label}.log"
    xmlfile = output / f"{label}.xml"
    timefile = output / f"{label}.time"
    command = [
        "/usr/bin/time", "-v", "-o", str(timefile), sys.executable, "-m", "pytest",
        path, "-m", "not slow and not gpu", "-v", "-rfE", "--tb=short",
        "--durations=0", f"--junitxml={xmlfile}", "-p", "no:cacheprovider",
    ]

    def limits():
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

    started = time.monotonic()
    peak = 0
    stopped = None
    print(f"START {path}", flush=True)
    with logfile.open("w") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, preexec_fn=limits)
        while process.poll() is None:
            peak = max(peak, group_rss_kib(process.pid))
            if peak * 1024 > memory_bytes:
                stopped = "aggregate RSS exceeded memory budget"
            elif time.monotonic() - started > timeout_seconds:
                stopped = "file timeout"
            if stopped:
                os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(0.25)
        code = process.wait()
    elapsed = time.monotonic() - started
    try:
        counts = junit_counts(xmlfile)
    except (OSError, ET.ParseError):
        counts = {"passed": 0, "failed": 0, "errors": 1, "skipped": 0}
    timed_peak = 0
    if timefile.exists():
        match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", timefile.read_text())
        if match:
            timed_peak = int(match.group(1))
    result = {"file": path, "returncode": code, "seconds": elapsed,
              "aggregate_peak_rss_kib": peak, "time_peak_rss_kib": timed_peak,
              "limit_reason": stopped, **counts}
    result["ok"] = stopped is None and acceptable_exit(code, counts)
    print(json.dumps(result, sort_keys=True), flush=True)
    if not result["ok"]:
        print(logfile.read_text()[-16000:], flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--durations", type=Path, default=ROOT / "tests/ci_durations.json")
    parser.add_argument("--memory-bytes", type=int, default=7_000_000_000)
    parser.add_argument("--file-timeout", type=int, default=600)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.shard < args.shards:
        parser.error("shard index must be in [0, shards)")
    paths = discover_files(ROOT)
    args.output = args.output.resolve()
    durations = json.loads(args.durations.read_text())["files"]
    groups, totals = make_shards(paths, durations, args.shards)
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {"shard": args.shard, "shards": args.shards, "files": groups[args.shard],
            "estimated_seconds": totals[args.shard], "all_estimated_seconds": totals,
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "memory_limit_bytes": args.memory_bytes}
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, sort_keys=True), flush=True)
    if args.plan_only:
        return 0
    started = time.monotonic()
    results = []
    for path in groups[args.shard]:
        results.append(run_file(path, args.output, args.memory_bytes, args.file_timeout))
        # Keep partial evidence even if the enclosing allocation ends early.
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    summary = {"shard": args.shard, "seconds": time.monotonic() - started,
               "ok": all(r["ok"] for r in results), "files": len(results),
               **{key: sum(r[key] for r in results) for key in ["passed", "failed", "errors", "skipped"]},
               "peak_rss_kib": max((r["aggregate_peak_rss_kib"] for r in results), default=0)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
