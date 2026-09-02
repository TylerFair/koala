"""Build fit-consumable batching manifests from measured benchmark artifacts."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def parse_pilot_chunk_spec(spec: str) -> tuple[tuple[int, ...], Path]:
    """Parse ``START:STOP=PATH`` or ``i,j,k=PATH`` with explicit global indices."""
    if "=" not in spec:
        raise ValueError(
            "pilot chunk must be START:STOP=PATH or comma-separated-indices=PATH"
        )
    channel_text, path_text = spec.split("=", 1)
    channel_text = channel_text.strip()
    path_text = path_text.strip()
    if not channel_text or not path_text:
        raise ValueError("pilot chunk channel specification and path cannot be empty.")
    if ":" in channel_text:
        parts = channel_text.split(":")
        if len(parts) != 2:
            raise ValueError("pilot range must contain exactly one colon.")
        try:
            start, stop = (int(value) for value in parts)
        except ValueError as exc:
            raise ValueError("pilot range bounds must be integers.") from exc
        if start < 0 or stop <= start:
            raise ValueError("pilot range must satisfy 0 <= START < STOP.")
        indices = tuple(range(start, stop))
    else:
        try:
            indices = tuple(int(value.strip()) for value in channel_text.split(","))
        except ValueError as exc:
            raise ValueError("pilot channel indices must be integers.") from exc
        if not indices or any(index < 0 for index in indices):
            raise ValueError("pilot channel indices must be non-negative.")
        if len(set(indices)) != len(indices):
            raise ValueError("pilot chunk contains duplicate channel indices.")
    return indices, Path(path_text)


def aggregate_pilot_diagnostics(
    chunk_specs: Sequence[str],
    *,
    require_zero_based_contiguous: bool = True,
):
    """Aggregate independent-sampler diagnostics with exact global indices.

    Explicit index specifications avoid guessing from checkpoint filenames.
    Chunks may be supplied in any order, but overlapping indices are rejected.
    Fit-consumable plans require coverage exactly ``range(num_channels)``.
    """
    from models.channel_batching import PilotDiagnostics

    if not chunk_specs:
        raise ValueError("At least one pilot diagnostic chunk is required.")
    records: dict[int, tuple[float, float, int, float]] = {}
    draw_counts = set()
    sources = []
    for raw_spec in chunk_specs:
        indices, path = parse_pilot_chunk_spec(raw_spec)
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        diagnostic_sha256 = sha256(path.read_bytes()).hexdigest()
        pilot = PilotDiagnostics.from_mapping(payload, channel_indices=indices)
        if len(pilot.channel_indices) != len(indices):
            raise ValueError(
                f"Pilot {path} has {len(pilot.channel_indices)} channels, "
                f"but {raw_spec!r} specifies {len(indices)}."
            )
        if pilot.num_draws is None:
            raise ValueError(f"Pilot {path} does not record num_draws.")
        draw_counts.add(pilot.num_draws)
        for index, mean_steps, max_steps, divergences, step_size in zip(
            pilot.channel_indices,
            pilot.mean_num_steps,
            pilot.max_num_steps,
            pilot.num_divergences,
            pilot.step_size,
        ):
            if index in records:
                raise ValueError(f"Global channel {index} occurs in multiple pilot chunks.")
            records[index] = (mean_steps, max_steps, divergences, step_size)
        sources.append(
            {
                "path": str(path.resolve()),
                "channel_indices": list(indices),
                "num_draws": pilot.num_draws,
                "diagnostic_sha256": diagnostic_sha256,
                "sampling_workload_fingerprint_sha256": payload.get(
                    "sampling_workload_fingerprint_sha256"
                ),
                "sampler_backend": payload.get("sampler_backend"),
            }
        )
    if len(draw_counts) != 1:
        raise ValueError(
            "All pilot chunks must use the same num_draws so divergence counts "
            "have one well-defined rate."
        )
    ordered = tuple(sorted(records))
    if require_zero_based_contiguous and ordered != tuple(range(len(ordered))):
        raise ValueError(
            "Fit-consumable pilot chunks must cover every global channel exactly "
            f"once from 0 through N-1; received {ordered}."
        )
    values = [records[index] for index in ordered]
    pilot = PilotDiagnostics(
        channel_indices=ordered,
        mean_num_steps=tuple(value[0] for value in values),
        max_num_steps=tuple(value[1] for value in values),
        num_divergences=tuple(value[2] for value in values),
        step_size=tuple(value[3] for value in values),
        num_draws=draw_counts.pop(),
    )
    return pilot, sources


def build_batch_plan_from_chunks(
    chunk_specs: Sequence[str],
    *,
    nominal_width: int,
    quarantine_width: int = 1,
    require_workload_provenance: bool = False,
):
    from models.channel_batching import build_difficulty_batch_plan

    pilot, sources = aggregate_pilot_diagnostics(chunk_specs)
    recorded_fingerprints = [
        source["sampling_workload_fingerprint_sha256"] for source in sources
    ]
    present_fingerprints = {
        value for value in recorded_fingerprints if value not in {None, ""}
    }
    if present_fingerprints and any(
        value in {None, ""} for value in recorded_fingerprints
    ):
        raise ValueError(
            "Pilot diagnostics mix provenance-aware and legacy files. Rerun "
            "every pilot chunk for the same current sampling workload."
        )
    if len(present_fingerprints) > 1:
        raise ValueError(
            "Pilot diagnostics came from different sampling workloads; their "
            "workload fingerprints do not match."
        )
    if require_workload_provenance and not present_fingerprints:
        raise ValueError(
            "Pilot diagnostics have no sampling workload fingerprint. Rerun "
            "the independent-sampler pilot with the current fit code before "
            "building a fit-consumable plan."
        )

    provenance = None
    if present_fingerprints:
        backends = {
            source["sampler_backend"]
            for source in sources
            if source["sampler_backend"] not in {None, ""}
        }
        if len(backends) > 1:
            raise ValueError(
                "Pilot diagnostics mix different sampler backends."
            )
        provenance = {
            "artifact_kind": "jwst_spectro_channel_batch_plan",
            "sampling_workload_fingerprint_sha256": (
                present_fingerprints.pop()
            ),
            "pilot_sampler_backend": (
                next(iter(backends)) if backends else None
            ),
            "pilot_num_draws": int(pilot.num_draws),
            "pilot_diagnostics": [
                {
                    "channel_indices": source["channel_indices"],
                    "diagnostic_sha256": source["diagnostic_sha256"],
                }
                for source in sorted(
                    sources, key=lambda item: item["channel_indices"][0]
                )
            ],
        }

    plan = build_difficulty_batch_plan(
        pilot,
        nominal_width=nominal_width,
        quarantine_width=quarantine_width,
        provenance=provenance,
    )
    return plan, sources


def build_width_selection_from_results(
    results: Iterable[Mapping[str, Any]],
    *,
    backend: str,
    trend_mode: str,
    total_memory_bytes: int,
    memory_fraction: float,
    eligible_case_ids: set[str] | None = None,
    provenance: Mapping[str, Any] | None = None,
):
    """Convert isolated width measurements into a ``WidthSelection``."""
    from models.channel_batching import WidthMeasurement, select_autotuned_width

    measurements = []
    skipped = []
    for result in results:
        if result.get("backend") != backend or result.get("trend_mode") != trend_mode:
            continue
        case_id = str(result.get("case_id"))
        if result.get("status") != "ok":
            skipped.append({"case_id": case_id, "reason": "case failed"})
            continue
        if eligible_case_ids is not None and case_id not in eligible_case_ids:
            skipped.append({"case_id": case_id, "reason": "quality gate failed"})
            continue
        peak = result.get("memory", {}).get("peak_device_bytes")
        throughput = result.get("posterior", {}).get("rors_ess_sum_per_second")
        if peak is None:
            skipped.append({"case_id": case_id, "reason": "peak device memory unavailable"})
            continue
        if throughput is None:
            skipped.append({"case_id": case_id, "reason": "ESS/second unavailable"})
            continue
        measurements.append(
            WidthMeasurement(
                width=int(result["channels"]),
                peak_memory_bytes=int(peak),
                throughput=float(throughput),
            )
        )
    if not measurements:
        raise ValueError(
            f"No usable width measurements for backend={backend!r}, "
            f"trend_mode={trend_mode!r}. Skipped: {skipped}"
        )
    selection = select_autotuned_width(
        measurements,
        total_memory_bytes=total_memory_bytes,
        memory_fraction=memory_fraction,
        provenance=provenance,
    )
    return selection, skipped
