"""Pure helpers for the spectroscopic sampler benchmark suite.

This module intentionally has no JAX dependency.  The command-line harness
can therefore plan subprocess-isolated GPU cases without initializing CUDA in
the parent process, and the selection/reporting logic remains cheap to test.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
VALID_BACKENDS = ("joint_nuts", "independent_nuts", "independent_hmc")
VALID_TREND_MODES = ("sampled_uniform", "gaussian_marginalized")


def parse_choice_list(
    value: str | Sequence[str],
    *,
    valid: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    """Parse a comma-separated choice list, preserving order and uniqueness."""
    raw = value.split(",") if isinstance(value, str) else list(value)
    choices = tuple(item.strip() for item in raw if item.strip())
    if not choices:
        raise ValueError(f"{label} must contain at least one value.")
    unknown = [choice for choice in choices if choice not in valid]
    if unknown:
        raise ValueError(
            f"Unknown {label}: {', '.join(unknown)}; expected one of "
            + ", ".join(valid)
        )
    if len(set(choices)) != len(choices):
        raise ValueError(f"{label} must not contain duplicates.")
    return choices


def parse_positive_int_list(value: str | Sequence[int]) -> tuple[int, ...]:
    raw = value.split(",") if isinstance(value, str) else list(value)
    try:
        widths = tuple(int(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("resident widths must be comma-separated integers.") from exc
    if not widths or any(width < 1 for width in widths):
        raise ValueError("resident widths must contain positive integers.")
    if len(set(widths)) != len(widths):
        raise ValueError("resident widths must not contain duplicates.")
    return widths


def summarize_timing(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one cold run followed by zero or more repeat runs.

    Each input item needs ``wall_seconds`` and ``compile_seconds``.  Compilation
    is measured by JAX monitoring events in the executable harness; subtracting
    it gives a useful, explicitly labelled execution-time estimate.
    """
    if not runs:
        raise ValueError("At least one timed run is required.")
    wall = [float(run["wall_seconds"]) for run in runs]
    compile_time = [float(run.get("compile_seconds", 0.0)) for run in runs]
    if any(not math.isfinite(value) or value < 0 for value in wall + compile_time):
        raise ValueError("Timing values must be finite and non-negative.")
    repeat_indices = list(range(1, len(runs)))
    repeat_wall = [wall[index] for index in repeat_indices]
    repeat_execution = [
        max(0.0, wall[index] - compile_time[index]) for index in repeat_indices
    ]
    steady_wall = statistics.median(repeat_wall) if repeat_wall else wall[0]
    steady_execution = (
        statistics.median(repeat_execution)
        if repeat_execution
        else max(0.0, wall[0] - compile_time[0])
    )
    return {
        "cold_wall_seconds": wall[0],
        "cold_compile_seconds": compile_time[0],
        "cold_execution_estimate_seconds": max(0.0, wall[0] - compile_time[0]),
        "repeat_wall_seconds": repeat_wall,
        "repeat_compile_seconds": [compile_time[index] for index in repeat_indices],
        "steady_state_wall_seconds": steady_wall,
        "steady_state_execution_estimate_seconds": steady_execution,
        "cold_minus_steady_wall_seconds": max(0.0, wall[0] - steady_wall),
        "num_timed_runs": len(runs),
        "compile_measurement": (
            "sum of JAX jaxpr-trace, jaxpr-to-MLIR, and backend-compile "
            "duration events emitted during the run"
        ),
        "steady_state_definition": (
            "median full warmup+sampling wall time after the first process-local run; "
            "with no repeats, the cold run is used"
        ),
    }


def candidate_id(backend: str, trend_mode: str, channels: int) -> str:
    return f"{backend}__{trend_mode}__width{int(channels)}"


def quality_reasons(
    result: Mapping[str, Any],
    *,
    max_divergence_fraction: float,
    min_channel_ess: float,
    max_device_memory_bytes: int | None = None,
    max_tree_depth_saturation_fraction: float = 0.05,
) -> list[str]:
    """Return reasons a measured candidate is unsafe for automatic selection."""
    if result.get("status") != "ok":
        return ["case did not complete"]
    diagnostics = result.get("diagnostics", {})
    posterior = result.get("posterior", {})
    reasons: list[str] = []
    divergence_fraction = diagnostics.get(
        "max_divergence_fraction_per_channel",
        diagnostics.get("divergence_fraction"),
    )
    if divergence_fraction is None or not math.isfinite(float(divergence_fraction)):
        reasons.append("divergence fraction is unavailable")
    elif float(divergence_fraction) > max_divergence_fraction:
        reasons.append(
            f"divergence fraction {float(divergence_fraction):.6g} exceeds "
            f"{max_divergence_fraction:.6g}"
        )
    saturation = diagnostics.get("tree_depth_saturation_fraction_max")
    if saturation is not None:
        if not math.isfinite(float(saturation)):
            reasons.append("tree-depth saturation fraction is non-finite")
        elif float(saturation) > max_tree_depth_saturation_fraction:
            reasons.append(
                f"tree-depth saturation fraction {float(saturation):.6g} "
                f"exceeds {max_tree_depth_saturation_fraction:.6g}"
            )
    ess_min = posterior.get("rors_ess_min")
    if ess_min is None or not math.isfinite(float(ess_min)):
        reasons.append("minimum channel ESS is unavailable")
    elif float(ess_min) < min_channel_ess:
        reasons.append(
            f"minimum channel ESS {float(ess_min):.6g} is below "
            f"{min_channel_ess:.6g}"
        )
    ess_rate = posterior.get("rors_ess_sum_per_second")
    if ess_rate is None or not math.isfinite(float(ess_rate)) or float(ess_rate) <= 0:
        reasons.append("ESS/second is unavailable or non-positive")
    if not posterior.get("all_finite", False):
        reasons.append("posterior samples are not all finite")
    if max_device_memory_bytes is not None:
        peak = result.get("memory", {}).get("peak_device_bytes")
        if peak is None:
            reasons.append(
                "peak device memory is unavailable under an active memory constraint"
            )
        elif int(peak) > int(max_device_memory_bytes):
            reasons.append(
                f"peak device memory {int(peak)} exceeds {int(max_device_memory_bytes)}"
            )
    return reasons


def select_candidates(
    results: Iterable[Mapping[str, Any]],
    *,
    max_divergence_fraction: float,
    min_channel_ess: float,
    max_device_memory_bytes: int | None = None,
    max_tree_depth_saturation_fraction: float = 0.05,
) -> dict[str, Any]:
    """Select measured ESS/s winners independently for each trend prior."""
    measured = list(results)
    selections: dict[str, Any] = {}
    for trend_mode in VALID_TREND_MODES:
        candidates = [item for item in measured if item.get("trend_mode") == trend_mode]
        evaluated = []
        for item in candidates:
            reasons = quality_reasons(
                item,
                max_divergence_fraction=max_divergence_fraction,
                min_channel_ess=min_channel_ess,
                max_device_memory_bytes=max_device_memory_bytes,
                max_tree_depth_saturation_fraction=(
                    max_tree_depth_saturation_fraction
                ),
            )
            rate = item.get("posterior", {}).get("rors_ess_sum_per_second")
            evaluated.append(
                {
                    "case_id": item.get("case_id"),
                    "eligible": not reasons,
                    "rejection_reasons": reasons,
                    "rors_ess_sum_per_second": rate,
                }
            )
        eligible = [entry for entry in evaluated if entry["eligible"]]
        winner = (
            max(eligible, key=lambda entry: float(entry["rors_ess_sum_per_second"]))
            if eligible
            else None
        )
        selections[trend_mode] = {
            "selected_case_id": None if winner is None else winner["case_id"],
            "objective": "maximum measured summed channel rors ESS per steady-state second",
            "candidates": evaluated,
        }
    return {
        "selection_scope": "separate within each trend-prior family",
        "cross_trend_selection_performed": False,
        "quality_constraints": {
            "max_divergence_fraction": max_divergence_fraction,
            "min_channel_ess": min_channel_ess,
            "max_device_memory_bytes": max_device_memory_bytes,
            "max_tree_depth_saturation_fraction": (
                max_tree_depth_saturation_fraction
            ),
        },
        "by_trend_mode": selections,
    }


def build_measured_comparisons(results: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build only apples-to-apples ratios (same trend prior and channel count)."""
    successful = [
        item for item in results
        if item.get("status") == "ok"
        and item.get("posterior", {}).get("all_finite", False)
    ]
    comparisons: list[dict[str, Any]] = []
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for item in successful:
        groups.setdefault((str(item["trend_mode"]), int(item["channels"])), []).append(item)
    for (trend_mode, channels), group in sorted(groups.items()):
        baseline = next((item for item in group if item["backend"] == "joint_nuts"), None)
        if baseline is None:
            continue
        base_wall = float(baseline["timing"]["steady_state_wall_seconds"])
        base_rate_raw = baseline["posterior"].get("rors_ess_sum_per_second")
        if base_rate_raw is None or base_wall <= 0:
            continue
        base_rate = float(base_rate_raw)
        if not math.isfinite(base_rate) or base_rate <= 0:
            continue
        for item in group:
            if item is baseline:
                continue
            wall = float(item["timing"]["steady_state_wall_seconds"])
            rate_raw = item["posterior"].get("rors_ess_sum_per_second")
            if rate_raw is None or wall <= 0:
                continue
            rate = float(rate_raw)
            if not math.isfinite(rate) or rate <= 0:
                continue
            comparisons.append(
                {
                    "baseline_case_id": baseline["case_id"],
                    "candidate_case_id": item["case_id"],
                    "trend_mode": trend_mode,
                    "channels": channels,
                    "wall_speedup": base_wall / wall,
                    "ess_per_second_ratio": rate / base_rate,
                    "prior_family_identical": True,
                    "measured_not_projected": True,
                }
            )
    return comparisons


def build_trend_comparisons(results: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Report measured trend-mode ratios while explicitly marking prior mismatch."""
    successful = [
        item for item in results
        if item.get("status") == "ok"
        and item.get("posterior", {}).get("all_finite", False)
    ]
    comparisons: list[dict[str, Any]] = []
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for item in successful:
        groups.setdefault((str(item["backend"]), int(item["channels"])), []).append(item)
    for (backend, channels), group in sorted(groups.items()):
        sampled = next(
            (item for item in group if item["trend_mode"] == "sampled_uniform"),
            None,
        )
        marginalized = next(
            (item for item in group if item["trend_mode"] == "gaussian_marginalized"),
            None,
        )
        if sampled is None or marginalized is None:
            continue
        sampled_wall = float(sampled["timing"]["steady_state_wall_seconds"])
        marginalized_wall = float(
            marginalized["timing"]["steady_state_wall_seconds"]
        )
        sampled_rate_raw = sampled["posterior"].get("rors_ess_sum_per_second")
        marginalized_rate_raw = marginalized["posterior"].get(
            "rors_ess_sum_per_second"
        )
        sampled_rate = (
            None if sampled_rate_raw is None else float(sampled_rate_raw)
        )
        marginalized_rate = (
            None if marginalized_rate_raw is None else float(marginalized_rate_raw)
        )
        ess_ratio = (
            None
            if sampled_rate is None
            or marginalized_rate is None
            or not math.isfinite(sampled_rate)
            or not math.isfinite(marginalized_rate)
            or sampled_rate <= 0
            else marginalized_rate / sampled_rate
        )
        sampled_median = sampled["posterior"].get("rors_median_per_channel", [])
        marginalized_median = marginalized["posterior"].get(
            "rors_median_per_channel", []
        )
        sampled_sd = sampled["posterior"].get("rors_sd_per_channel", [])
        marginalized_sd = marginalized["posterior"].get(
            "rors_sd_per_channel", []
        )
        standardized = []
        if (
            len(sampled_median)
            == len(marginalized_median)
            == len(sampled_sd)
            == len(marginalized_sd)
        ):
            for lhs, rhs, lhs_sd, rhs_sd in zip(
                sampled_median, marginalized_median, sampled_sd, marginalized_sd
            ):
                pooled = math.sqrt(0.5 * (float(lhs_sd) ** 2 + float(rhs_sd) ** 2))
                standardized.append(abs(float(rhs) - float(lhs)) / max(pooled, 1.0e-300))
        comparisons.append(
            {
                "sampled_case_id": sampled["case_id"],
                "marginalized_case_id": marginalized["case_id"],
                "backend": backend,
                "channels": channels,
                "measured_wall_speedup": sampled_wall / marginalized_wall,
                "measured_ess_per_second_ratio": ess_ratio,
                "rors_median_max_pooled_sd": max(standardized) if standardized else None,
                "prior_family_identical": False,
                "posterior_equivalence_claimed": False,
                "interpretation": (
                    "performance pilot only: sampled Uniform and marginalized "
                    "Gaussian trend priors differ"
                ),
            }
        )
    return comparisons


def atomic_write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    """Atomically write strict JSON, rejecting NaN/Infinity."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
