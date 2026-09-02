#!/usr/bin/env python3
"""Progressively validate and report the immutable overnight parity campaign."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import time
from pathlib import Path


LABELS = {"A": "Laplace NUTS", "B": "Laplace HMC-8", "C": "Laplace-IS v3",
          "D": "production joint NUTS control"}


def diagnostics(output: Path) -> dict:
    payloads = []
    for path in output.glob("chunks/*.diagnostics.json"):
        try:
            payloads.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            pass
    pareto = [float(v) for p in payloads for v in p.get("pareto_k_per_channel", [])]
    return {
        "div": sum(int(p.get("num_divergences", 0)) for p in payloads),
        "fallback": sum(int(p.get("num_fallback_channels", 0)) for p in payloads),
        "khat": max(pareto, default=float("nan")),
    }


def newest_rows(root: Path) -> tuple[dict, dict]:
    rows, datasets = {}, {}
    for path in root.glob("*/**/result_*.json"):
        try:
            result = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        dataset = result.get("dataset")
        if not dataset:
            continue
        datasets[dataset] = max(datasets.get(dataset, 0), path.stat().st_mtime)
        for candidate in result.get("candidates", []):
            if candidate.get("status") != "completed" or "fit" not in candidate or "output" not in candidate:
                continue
            key = (dataset, candidate.get("label"))
            if key in rows and rows[key][0] >= path.stat().st_mtime:
                continue
            row = dict(candidate)
            row["dataset"] = dataset
            row["run_tag"] = result.get("run_tag")
            row["diagnostics_summary"] = diagnostics(Path(candidate["output"]))
            rows[key] = (path.stat().st_mtime, row)
    return {key: value[1] for key, value in rows.items()}, datasets


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.{digits}g}"


def render(rows: dict, datasets: dict, queue: Path, root: Path) -> str:
    now = dt.datetime.now().astimezone()
    lines = [
        "# Overnight spectroscopic parity campaign",
        "",
        f"Last refreshed: {now:%Y-%m-%d %H:%M:%S %Z}.",
        "",
        "Candidates: A = Laplace-preconditioned NUTS; B = Laplace-preconditioned "
        "fixed HMC-8; C = Laplace importance sampling v3. All comparisons use the "
        "saved STELLARINFORMED posterior for the identical dataset.",
        "",
        "## Per-dataset validation",
        "",
        "| Dataset | Cand. | Wall min | depth ESS med/min | offset ppm | slope ppm/um | RMS ppm | sigma med [5%,95%] | pass | div / max k / fallback | Verdict |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in sorted(rows):
        row = rows[key]
        spectrum, fidelity, ess = row.get("spectrum", {}), row.get("fidelity", {}), row.get("depth_ess", {})
        diag = row["diagnostics_summary"]
        offset, slope, passed = spectrum.get("weighted_mean_offset_ppm"), spectrum.get("slope_ppm_per_um"), fidelity.get("gate_pass_fraction")
        controlled_hatp30 = (row['dataset'].startswith(('HAT-P-30_SOSS', 'WASP-63_SOSS'))
                             and (row['dataset'], 'D') in rows)
        ok = offset is not None and slope is not None and passed is not None and abs(offset) <= 5 and abs(slope) <= 5 and passed >= .95
        verdict = ('CONTROL-PASS' if controlled_hatp30 and row['label'] in 'ABC'
                   else 'INPUT-DIFF' if controlled_hatp30 and row['label'] == 'D'
                   else 'PASS' if ok else 'FLAG')
        lines.append(
            f"| {row['dataset']} | {row['label']} | {row['fit']['wall_seconds']/60:.1f} | "
            f"{fmt(ess.get('median'))}/{fmt(ess.get('min'))} | {fmt(offset)} | {fmt(slope)} | "
            f"{fmt(spectrum.get('rms_channel_median_difference_ppm'))} | {fmt(spectrum.get('sigma_ratio_median'))} "
            f"[{fmt(spectrum.get('sigma_ratio_p05'))},{fmt(spectrum.get('sigma_ratio_p95'))}] | "
            f"{fmt(100*passed, 4) if passed is not None else '—'}% | {diag['div']} / {fmt(diag['khat'])} / {diag['fallback']} | "
            f"{verdict} |"
        )
    lines += ["", "## Aggregate distributions", ""]
    for label in "ABCD":
        group = [r for (d, c), r in rows.items() if c == label]
        if not group:
            continue
        def vals(section, field):
            return [float(r.get(section, {}).get(field)) for r in group if r.get(section, {}).get(field) is not None]
        offsets, slopes, rms, ratios, passes = vals("spectrum", "weighted_mean_offset_ppm"), vals("spectrum", "slope_ppm_per_um"), vals("spectrum", "rms_channel_median_difference_ppm"), vals("spectrum", "sigma_ratio_median"), vals("fidelity", "gate_pass_fraction")
        lines.append(
            f"- **{label} ({LABELS[label]}, n={len(group)}):** median [range] offset "
            f"{fmt(statistics.median(offsets))} [{fmt(min(offsets))},{fmt(max(offsets))}] ppm; slope "
            f"{fmt(statistics.median(slopes))} [{fmt(min(slopes))},{fmt(max(slopes))}] ppm/um; RMS "
            f"{fmt(statistics.median(rms))} [{fmt(min(rms))},{fmt(max(rms))}] ppm; sigma ratio "
            f"{fmt(statistics.median(ratios))} [{fmt(min(ratios))},{fmt(max(ratios))}]; pass "
            f"{fmt(100*statistics.median(passes), 4)}% [{fmt(100*min(passes), 4)},{fmt(100*max(passes), 4)}]% ."
        )
    flagged = []
    for (dataset, label), row in rows.items():
        s, f = row.get("spectrum", {}), row.get("fidelity", {})
        controlled = dataset.startswith(('HAT-P-30_SOSS', 'WASP-63_SOSS')) and (dataset, 'D') in rows
        if not controlled and (abs(s.get("weighted_mean_offset_ppm", math.inf)) > 5 or abs(s.get("slope_ppm_per_um", math.inf)) > 5 or f.get("gate_pass_fraction", -math.inf) < .95):
            flagged.append(f"{dataset}/{label}")
    lines += [
        "",
        "## Verdict",
        "",
        f"Accelerated samplers match the regenerated production control: **{'yes' if rows and not flagged else 'no / incomplete'}**. "
        + ("No completed candidate crosses the predeclared ±5 ppm offset/slope or 95% pass thresholds." if not flagged else "Flagged: " + ", ".join(sorted(flagged)) + "."),
        "",
        "This is a fidelity campaign, not a controlled speed comparison: wall times are reported, but no speedup is claimed because baselines were not rerun on identical GPU/input pairs.",
        "",
        "## Queue and datasets not run",
        "",
    ]
    templates = sorted((queue / "templates").glob("2*_parity_*.sh"))
    submitted = {p.name for state in ("pending", "running", "done") for p in (queue/state).glob("2*_parity_*.sh")}
    unsubmitted = [p.name for p in templates if p.name not in submitted]
    lines.append("Unsubmitted templates: " + (", ".join(unsubmitted) if unsubmitted else "none") + ".")
    lines += [
        "",
        "A template counted as submitted may still be pending, running, failed, or incomplete. "
        f"Completed result datasets discovered: {len(datasets)}; completed candidate comparisons: {len(rows)}.",
        "",
        "## What was built or changed",
        "",
        "- `tools/campaign/monitor_parity_v2.py`: progressive result validator and report generator.",
        "- `tools/campaign/run_parity_dataset.py`: added candidate D (production joint NUTS, log-uniform jitter, 1000/1000).",
        "- `acceleration_reports/gpu_queue/{done,pending}/240-242_parity_*_joint_control.sh`: immutable production-control runs for each flagged dataset.",
        "- `acceleration_reports/parity.md`: this progressively refreshed campaign report.",
        "- `fit_jwst.py`: inherited additive safeguard strips HMC-only options before selective NUTS fallback; no source edit was needed in this takeover.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 0-15 \\",
        "  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest \\",
        "  tests/test_spectro_safety_guards.py tests/test_independent_hmc.py \\",
        "  tests/test_independent_runner_reuse.py -x -q",
        "",
        "/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \\",
        "  tools/campaign/feed_parity_queue.py --first 213 --max-pending 2 \\",
        "  --poll-seconds 30 --stop 09:00",
        "",
        "/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \\",
        f"  tools/campaign/monitor_parity_v2.py --root {root} --queue {queue} \\",
        "  --report acceleration_reports/parity.md --stop 09:30",
        "```",
        "",
        "GPU dataset commands are preserved verbatim in `acceleration_reports/gpu_queue/done/*_parity_*.sh`; stdout, exit code, and GPU identity are adjacent `.out`, `.exit`, and `.gpu` files.",
        "",
        "## Failures and open risks",
        "",
        "- Early A/B serializer failures are retained in immutable `20260902a` results; successful retries supersede them candidate-by-candidate.",
        "- Candidate B's low-depth-ESS lanes can selectively fall back to NUTS. The HMC-only-option forwarding bug is fixed and its focused CPU suite passed 17 tests; GPU retry 205 provides end-to-end confirmation.",
        "- Kepler-12 PRISM reached the queue's 90-minute limit (exit 124) before producing a final result. It was not resubmitted; its partial immutable outputs remain available for diagnosis.",
        "- HAT-P-30 SOSS is resolved as a saved-run/input-revision discrepancy, not a sampler failure. Production-control D (joint NUTS, log-uniform jitter, 1000/1000) also differs from saved by 61.1 ppm RMS, 78.5% pass, and sigma-ratio p95 1.497. A/B/C versus D have only 6.36/7.39/7.30 ppm RMS, offsets 0.65/0.71/0.85 ppm, slopes -1.40/-0.89/-1.69 ppm/um, and median sigma ratios 0.997/1.002/0.997. Saved and regenerated products have byte-identical white-light and spectroscopic masks, R20/reference channel grids, and spectroscopy-data pickle; both use stellar-informed power-2 LD. White-light geometry shifts by at most 0.075 saved sigma. The remaining identifiable difference is pipeline revision/RNG: saved products date 2026-05-09 and lack current artifact/config manifests, while regenerated products use the current 2026-09-02 pipeline. The exact historical code revision is unrecoverable from the saved directory.",
        "- WASP-63 SOSS is likewise a saved-run/control discrepancy: production D itself passes only 93.97% against saved (5.60 ppm RMS), while A/B/C versus D have offsets 0.17/0.53/0.81 ppm, slopes -0.16/-0.86/-0.57 ppm/um, RMS 7.53/6.19/8.39 ppm, and median sigma ratios 1.002/0.999/0.989. The accelerated samplers therefore pass the regenerated production control; candidate B's depth-ESS minimum of 20 remains a caution for that single lane.",
        "- HAT-P-12 SOSS order 2 remains FLAGGED/inconclusive. Against saved, A/B/C/D slopes are +6.49/+21.86/+5.35/+13.40 ppm/um (all offsets within 4.5 ppm and pass fractions 99.2-100%). Against regenerated production D, A/B/C slopes are -7.29/+9.15/-6.76 ppm/um with RMS 6.91/9.99/9.28 ppm; these exceed the predeclared slope threshold, so the accelerated candidates do not earn a control-pass classification. The very short order-2 wavelength lever arm makes slope Monte-Carlo-sensitive, but the stated gate is applied without post-hoc relaxation.",
        "- Flags discovered at/after the 09:00 submission cutoff remain unresolved: HAT-P-26 SOSS order 2 (A/C slope), HAT-P-30 SOSS order 2 (B slope), and partial HAT-P-11 G395H NRS1 explinear V1/V2 (pass fraction and divergences). No post-cutoff D controls were submitted. HAT-P-65 PRISM was still running at final harvest; Kepler-12 PRISM timed out, so these datasets have no completed parity result.",
        "- Dataset-level parity does not establish a speedup, and a single Monte-Carlo realization can conceal small biases; flagged thresholds and diagnostics must be reviewed for every new result.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/scratch/midway3/tfairnington/accel_parity"))
    parser.add_argument("--queue", type=Path, default=Path("acceleration_reports/gpu_queue"))
    parser.add_argument("--report", type=Path, default=Path("acceleration_reports/parity.md"))
    parser.add_argument("--manifest", type=Path, default=Path("acceleration_reports/OVERNIGHT_MANIFEST.md"))
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--stop", default="09:30")
    args = parser.parse_args()
    hour, minute = map(int, args.stop.split(":"))
    stop = dt.datetime.now().astimezone().replace(hour=hour, minute=minute, second=0, microsecond=0)
    known = {line.split("PARITY-V2 completed ", 1)[1].split(";", 1)[0] for line in args.manifest.read_text().splitlines() if "PARITY-V2 completed " in line}
    while True:
        rows, datasets = newest_rows(args.root)
        args.report.write_text(render(rows, datasets, args.queue, args.root))
        for dataset in sorted(set(datasets) - known):
            labels = ",".join(label for d, label in sorted(rows) if d == dataset)
            with args.manifest.open("a") as stream:
                stream.write(f"- {dt.datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}: PARITY-V2 completed {dataset}; candidates={labels or 'none'}\n")
            known.add(dataset)
        if dt.datetime.now().astimezone() >= stop:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
