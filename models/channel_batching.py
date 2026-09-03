"""Deterministic channel batching and resident-width autotuning.

This module is intentionally independent of the sampling implementation.  A
short pilot run can be summarized with :class:`PilotDiagnostics`, converted
into a difficulty-aware :class:`ChannelBatchPlan`, and then consumed by any
runner capable of slicing arbitrary channel indices.  The plan carries enough
information to checkpoint it, fingerprint it, and restore samples to their
original wavelength order.

The width autotuner likewise operates only on measured peak memory and
throughput.  It never extrapolates an unmeasured width, which makes its choice
safe to cache per device and workload shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from os import PathLike
from typing import Any, Mapping, Sequence

import numpy as np


BATCH_PLAN_SCHEMA_VERSION = 2
WIDTH_SELECTION_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SpectroMemoryModel:
    """Affine peak-memory model for a fixed sampler/model family.

    The draw term represents compact retained posterior/diagnostic arrays;
    cadence-sized temporaries are represented separately so the rule remains
    useful across instrument modes.
    """

    intercept_bytes: float
    bytes_per_lane: float
    bytes_per_lane_cadence: float
    bytes_per_draw_lane: float = 0.0

    def predict(self, width: int, cadences: int, draws: int = 0) -> int:
        width, cadences, draws = int(width), int(cadences), int(draws)
        if width < 1 or cadences < 1 or draws < 0:
            raise ValueError("width/cadences must be positive and draws non-negative.")
        value = (
            self.intercept_bytes
            + self.bytes_per_lane * width
            + self.bytes_per_lane_cadence * width * cadences
            + self.bytes_per_draw_lane * width * draws
        )
        if not math.isfinite(value) or value < 0:
            raise ValueError("Memory model predicted an invalid byte count.")
        return int(math.ceil(value))


def fit_spectro_memory_model(measurements):
    """Least-squares fit of the documented affine peak-memory model."""
    rows, peaks = [], []
    for item in measurements:
        width = int(item["width"])
        cadences = int(item["cadences"])
        draws = int(item.get("draws", 0))
        peak = float(item["peak_memory_bytes"])
        if min(width, cadences) < 1 or draws < 0 or peak < 0:
            raise ValueError("Invalid memory-model measurement.")
        rows.append((1.0, width, width * cadences, width * draws))
        peaks.append(peak)
    if len(rows) < 4:
        raise ValueError("At least four measurements are required to fit the model.")
    coefficients, _, rank, _ = np.linalg.lstsq(
        np.asarray(rows, dtype=np.float64),
        np.asarray(peaks, dtype=np.float64),
        rcond=None,
    )
    if rank < 3:
        raise ValueError("Memory-model measurements do not constrain cadence scaling.")
    coefficients = np.maximum(coefficients, 0.0)
    return SpectroMemoryModel(*coefficients)


def resolve_spectro_auto_width(
    model,
    *,
    bytes_limit,
    cadences,
    draws,
    speed_cap,
    max_channels,
    headroom_fraction=0.25,
):
    """Largest predicted-safe width, capped at the measured speed sweet spot."""
    if not isinstance(model, SpectroMemoryModel):
        model = SpectroMemoryModel(**dict(model))
    bytes_limit = int(bytes_limit)
    headroom_fraction = float(headroom_fraction)
    if bytes_limit <= 0 or not 0.0 <= headroom_fraction < 1.0:
        raise ValueError("Invalid device limit or headroom fraction.")
    cap = min(int(speed_cap), int(max_channels))
    if cap < 1:
        raise ValueError("speed_cap and max_channels must be positive.")
    budget = int(math.floor(bytes_limit * (1.0 - headroom_fraction)))
    if model.predict(1, cadences, draws) > budget:
        raise ValueError("Even one spectroscopic lane exceeds the memory budget.")
    low, high = 1, cap
    while low < high:
        middle = (low + high + 1) // 2
        if model.predict(middle, cadences, draws) <= budget:
            low = middle
        else:
            high = middle - 1
    return low


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _one_dimensional_tuple(
    name: str,
    values: Sequence[Any],
    *,
    dtype,
) -> tuple:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got {array.shape}.")
    return tuple(array.tolist())


def _first_present(mapping: Mapping[str, Any], names: Sequence[str]):
    for name in names:
        if name in mapping:
            return mapping[name]
    raise KeyError(
        "Missing pilot diagnostic; expected one of " + ", ".join(names)
    )


def _normalize_axis_index(axis: int, ndim: int) -> int:
    axis = int(axis)
    if not -ndim <= axis < ndim:
        raise ValueError(f"channel_axis={axis} is invalid for an {ndim}-D array.")
    return axis % ndim


@dataclass(frozen=True)
class PilotDiagnostics:
    """Per-channel statistics produced by a short independently adapted run.

    ``channel_indices`` records the wavelength order of the input arrays.  It
    need not be contiguous, which permits plans for a selected detector/order
    while retaining global channel identifiers.
    """

    channel_indices: tuple[int, ...]
    mean_num_steps: tuple[float, ...]
    max_num_steps: tuple[float, ...]
    num_divergences: tuple[int, ...]
    step_size: tuple[float, ...]
    num_draws: int | None = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "channel_indices",
            _one_dimensional_tuple(
                "channel_indices", self.channel_indices, dtype=np.int64
            ),
        )
        object.__setattr__(
            self,
            "mean_num_steps",
            _one_dimensional_tuple(
                "mean_num_steps", self.mean_num_steps, dtype=np.float64
            ),
        )
        object.__setattr__(
            self,
            "max_num_steps",
            _one_dimensional_tuple(
                "max_num_steps", self.max_num_steps, dtype=np.float64
            ),
        )
        object.__setattr__(
            self,
            "num_divergences",
            _one_dimensional_tuple(
                "num_divergences", self.num_divergences, dtype=np.int64
            ),
        )
        object.__setattr__(
            self,
            "step_size",
            _one_dimensional_tuple("step_size", self.step_size, dtype=np.float64),
        )

        lengths = {
            len(self.channel_indices),
            len(self.mean_num_steps),
            len(self.max_num_steps),
            len(self.num_divergences),
            len(self.step_size),
        }
        if len(lengths) != 1:
            raise ValueError("All pilot diagnostics must have the same length.")
        if not self.channel_indices:
            raise ValueError("Pilot diagnostics must contain at least one channel.")
        if len(set(self.channel_indices)) != len(self.channel_indices):
            raise ValueError("channel_indices must be unique.")

        mean_steps = np.asarray(self.mean_num_steps)
        max_steps = np.asarray(self.max_num_steps)
        divergences = np.asarray(self.num_divergences)
        step_size = np.asarray(self.step_size)
        if not np.all(np.isfinite(mean_steps)) or np.any(mean_steps < 0):
            raise ValueError("mean_num_steps must be finite and non-negative.")
        if not np.all(np.isfinite(max_steps)) or np.any(max_steps < mean_steps):
            raise ValueError(
                "max_num_steps must be finite and at least mean_num_steps."
            )
        if np.any(divergences < 0):
            raise ValueError("num_divergences must be non-negative.")
        if not np.all(np.isfinite(step_size)) or np.any(step_size <= 0):
            raise ValueError("step_size must be finite and strictly positive.")

        if self.num_draws is not None:
            object.__setattr__(self, "num_draws", int(self.num_draws))
            if self.num_draws < 1:
                raise ValueError("num_draws must be positive when provided.")
            if np.any(divergences > self.num_draws):
                raise ValueError(
                    "Per-channel divergences cannot exceed num_draws."
                )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        channel_indices: Sequence[int] | None = None,
    ) -> "PilotDiagnostics":
        """Load either the public schema or independent-NUTS JSON schema."""

        mean_steps = _first_present(
            payload,
            ("mean_num_steps", "mean_num_steps_per_channel"),
        )
        max_steps = _first_present(
            payload,
            ("max_num_steps", "max_num_steps_per_channel"),
        )
        divergences = _first_present(
            payload,
            # The independent-NUTS JSON contains both a scalar total under
            # ``num_divergences`` and this per-channel vector.  Prefer the
            # vector whenever both are present.
            ("num_divergences_per_channel", "num_divergences"),
        )
        step_size = _first_present(
            payload,
            ("step_size", "adapted_step_size_per_channel"),
        )
        if channel_indices is None:
            channel_indices = payload.get("channel_indices")
        if channel_indices is None:
            channel_indices = range(len(mean_steps))

        return cls(
            channel_indices=tuple(channel_indices),
            mean_num_steps=tuple(mean_steps),
            max_num_steps=tuple(max_steps),
            num_divergences=tuple(divergences),
            step_size=tuple(step_size),
            num_draws=payload.get("num_draws"),
        )

    @classmethod
    def from_json(
        cls,
        path: str | PathLike[str],
        *,
        channel_indices: Sequence[int] | None = None,
    ) -> "PilotDiagnostics":
        with open(path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
        return cls.from_mapping(payload, channel_indices=channel_indices)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "channel_indices": list(self.channel_indices),
            "mean_num_steps": list(self.mean_num_steps),
            "max_num_steps": list(self.max_num_steps),
            "num_divergences": list(self.num_divergences),
            "step_size": list(self.step_size),
            "num_draws": self.num_draws,
        }


@dataclass(frozen=True)
class DifficultyConfig:
    """Weights and robust-tail gates used to identify difficult channels."""

    mean_steps_weight: float = 1.0
    max_steps_weight: float = 0.5
    step_size_weight: float = 0.5
    divergence_weight: float = 8.0
    pathology_step_ratio: float = 4.0
    pathology_step_size_ratio: float = 4.0
    pathology_score_z: float = 3.5
    pathology_tail_fraction: float = 0.10
    quarantine_divergences: bool = True

    def __post_init__(self):
        weights = (
            self.mean_steps_weight,
            self.max_steps_weight,
            self.step_size_weight,
            self.divergence_weight,
        )
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("Difficulty weights must be finite and non-negative.")
        if self.pathology_step_ratio <= 1.0:
            raise ValueError("pathology_step_ratio must be greater than one.")
        if self.pathology_step_size_ratio <= 1.0:
            raise ValueError(
                "pathology_step_size_ratio must be greater than one."
            )
        if not math.isfinite(self.pathology_score_z) or self.pathology_score_z <= 0:
            raise ValueError("pathology_score_z must be finite and positive.")
        if not (0.0 < self.pathology_tail_fraction <= 1.0):
            raise ValueError("pathology_tail_fraction must be in (0, 1].")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "mean_steps_weight": self.mean_steps_weight,
            "max_steps_weight": self.max_steps_weight,
            "step_size_weight": self.step_size_weight,
            "divergence_weight": self.divergence_weight,
            "pathology_step_ratio": self.pathology_step_ratio,
            "pathology_step_size_ratio": self.pathology_step_size_ratio,
            "pathology_score_z": self.pathology_score_z,
            "pathology_tail_fraction": self.pathology_tail_fraction,
            "quarantine_divergences": self.quarantine_divergences,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DifficultyConfig":
        return cls(**dict(payload))


@dataclass(frozen=True)
class ChannelDifficulty:
    channel_index: int
    score: float
    pathological: bool
    reasons: tuple[str, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "channel_index": self.channel_index,
            "score": self.score,
            "pathological": self.pathological,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ChannelDifficulty":
        return cls(
            channel_index=int(payload["channel_index"]),
            score=float(payload["score"]),
            pathological=bool(payload["pathological"]),
            reasons=tuple(payload.get("reasons", ())),
        )


@dataclass(frozen=True)
class ChannelBatch:
    """One execution batch and its fixed-width padding recipe."""

    batch_id: str
    channel_indices: tuple[int, ...]
    padded_channel_indices: tuple[int, ...]
    lane_width: int
    quarantined: bool
    min_difficulty: float
    max_difficulty: float

    def __post_init__(self):
        object.__setattr__(
            self,
            "channel_indices",
            tuple(int(value) for value in self.channel_indices),
        )
        object.__setattr__(
            self,
            "padded_channel_indices",
            tuple(int(value) for value in self.padded_channel_indices),
        )
        if not self.channel_indices:
            raise ValueError("A channel batch cannot be empty.")
        if self.lane_width < len(self.channel_indices):
            raise ValueError("lane_width cannot be smaller than the active batch.")
        if len(self.padded_channel_indices) != self.lane_width:
            raise ValueError(
                "padded_channel_indices must contain exactly lane_width entries."
            )
        if self.padded_channel_indices[: len(self.channel_indices)] != self.channel_indices:
            raise ValueError("Padded indices must begin with the active channels.")
        if any(
            value != self.channel_indices[-1]
            for value in self.padded_channel_indices[len(self.channel_indices) :]
        ):
            raise ValueError("Padding must repeat the final active channel.")

    @property
    def num_active(self) -> int:
        return len(self.channel_indices)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "channel_indices": list(self.channel_indices),
            "padded_channel_indices": list(self.padded_channel_indices),
            "lane_width": self.lane_width,
            "quarantined": self.quarantined,
            "min_difficulty": self.min_difficulty,
            "max_difficulty": self.max_difficulty,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ChannelBatch":
        return cls(
            batch_id=str(payload["batch_id"]),
            channel_indices=tuple(payload["channel_indices"]),
            padded_channel_indices=tuple(payload["padded_channel_indices"]),
            lane_width=int(payload["lane_width"]),
            quarantined=bool(payload["quarantined"]),
            min_difficulty=float(payload["min_difficulty"]),
            max_difficulty=float(payload["max_difficulty"]),
        )


@dataclass(frozen=True)
class ChannelBatchPlan:
    """A serializable, fingerprinted difficulty-aware execution plan."""

    pilot: PilotDiagnostics
    config: DifficultyConfig
    nominal_width: int
    quarantine_width: int
    difficulties: tuple[ChannelDifficulty, ...]
    batches: tuple[ChannelBatch, ...]
    strategy: str = "difficulty_bucketed"
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self):
        object.__setattr__(self, "nominal_width", int(self.nominal_width))
        object.__setattr__(self, "quarantine_width", int(self.quarantine_width))
        object.__setattr__(self, "difficulties", tuple(self.difficulties))
        object.__setattr__(self, "batches", tuple(self.batches))
        if self.provenance is not None:
            if not isinstance(self.provenance, Mapping):
                raise ValueError("Channel-batch provenance must be a mapping.")
            normalized = json.loads(
                json.dumps(
                    dict(self.provenance),
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            object.__setattr__(self, "provenance", normalized)
        if self.nominal_width < 1:
            raise ValueError("nominal_width must be positive.")
        if not (1 <= self.quarantine_width <= self.nominal_width):
            raise ValueError(
                "quarantine_width must be between one and nominal_width."
            )

        planned = [
            channel
            for batch in self.batches
            for channel in batch.channel_indices
        ]
        if len(planned) != len(set(planned)):
            raise ValueError("A channel appears in more than one batch.")
        if set(planned) != set(self.pilot.channel_indices):
            raise ValueError("Batches must cover every pilot channel exactly once.")
        difficulty_indices = [item.channel_index for item in self.difficulties]
        if difficulty_indices != list(self.pilot.channel_indices):
            raise ValueError(
                "Difficulty records must follow the original wavelength order."
            )

    @property
    def original_channel_indices(self) -> tuple[int, ...]:
        return self.pilot.channel_indices

    @property
    def execution_channel_indices(self) -> tuple[int, ...]:
        return tuple(
            channel for batch in self.batches for channel in batch.channel_indices
        )

    def fingerprint_inputs(self) -> dict[str, Any]:
        """Return JSON-compatible inputs suitable for checkpoint hashing."""

        return {
            "schema_version": BATCH_PLAN_SCHEMA_VERSION,
            "strategy": self.strategy,
            "nominal_width": self.nominal_width,
            "quarantine_width": self.quarantine_width,
            "pilot": self.pilot.to_manifest(),
            "difficulty_config": self.config.to_manifest(),
            "difficulties": [item.to_manifest() for item in self.difficulties],
            "batches": [batch.to_manifest() for batch in self.batches],
            "provenance": self.provenance,
        }

    @property
    def fingerprint_sha256(self) -> str:
        return _canonical_sha256(self.fingerprint_inputs())

    def to_manifest(self) -> dict[str, Any]:
        payload = self.fingerprint_inputs()
        payload["fingerprint_sha256"] = self.fingerprint_sha256
        return payload

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "ChannelBatchPlan":
        if int(payload.get("schema_version", -1)) != BATCH_PLAN_SCHEMA_VERSION:
            raise ValueError("Unsupported channel-batch manifest schema.")
        plan = cls(
            pilot=PilotDiagnostics.from_mapping(payload["pilot"]),
            config=DifficultyConfig.from_mapping(payload["difficulty_config"]),
            nominal_width=int(payload["nominal_width"]),
            quarantine_width=int(payload["quarantine_width"]),
            difficulties=tuple(
                ChannelDifficulty.from_mapping(item)
                for item in payload["difficulties"]
            ),
            batches=tuple(
                ChannelBatch.from_mapping(item) for item in payload["batches"]
            ),
            strategy=str(payload["strategy"]),
            provenance=payload.get("provenance"),
        )
        recorded = payload.get("fingerprint_sha256")
        if recorded is not None and recorded != plan.fingerprint_sha256:
            raise ValueError("Channel-batch manifest fingerprint does not match.")
        return plan


def _difficulty_records(
    pilot: PilotDiagnostics,
    config: DifficultyConfig,
) -> tuple[ChannelDifficulty, ...]:
    mean_steps = np.asarray(pilot.mean_num_steps, dtype=np.float64)
    max_steps = np.asarray(pilot.max_num_steps, dtype=np.float64)
    step_size = np.asarray(pilot.step_size, dtype=np.float64)
    divergences = np.asarray(pilot.num_divergences, dtype=np.float64)

    # Medians make a single pathological channel unable to distort the
    # reference scale.  The tiny floor matters only for synthetic zero-step
    # diagnostics (a real NUTS draw always takes at least one step).
    tiny = np.finfo(np.float64).tiny
    mean_reference = max(float(np.median(mean_steps)), tiny)
    max_reference = max(float(np.median(max_steps)), tiny)
    step_reference = max(float(np.median(step_size)), tiny)
    divergence_rate = (
        divergences / pilot.num_draws
        if pilot.num_draws is not None
        else (divergences > 0).astype(np.float64)
    )

    scores = (
        config.mean_steps_weight
        * np.log2(np.maximum(mean_steps, tiny) / mean_reference)
        + config.max_steps_weight
        * np.log2(np.maximum(max_steps, tiny) / max_reference)
        + config.step_size_weight
        * np.log2(step_reference / np.maximum(step_size, tiny))
        + config.divergence_weight * divergence_rate
    )
    score_median = float(np.median(scores))
    median_absolute_deviation = float(np.median(np.abs(scores - score_median)))
    robust_scale = max(1.4826 * median_absolute_deviation, 0.5)
    robust_z = (scores - score_median) / robust_scale

    tail_count = max(
        1,
        int(math.ceil(len(scores) * config.pathology_tail_fraction)),
    )
    original_positions = range(len(scores))
    hardest_positions = sorted(
        original_positions,
        key=lambda position: (-float(scores[position]), position),
    )[:tail_count]
    robust_tail = set(hardest_positions)

    records = []
    for position, channel_index in enumerate(pilot.channel_indices):
        reasons = []
        if config.quarantine_divergences and divergences[position] > 0:
            reasons.append("divergences")
        if mean_steps[position] > config.pathology_step_ratio * mean_reference:
            reasons.append("mean_steps_ratio")
        if max_steps[position] > config.pathology_step_ratio * max_reference:
            reasons.append("max_steps_ratio")
        if step_size[position] * config.pathology_step_size_ratio < step_reference:
            reasons.append("step_size_ratio")
        if position in robust_tail and robust_z[position] >= config.pathology_score_z:
            reasons.append("robust_score_tail")
        records.append(
            ChannelDifficulty(
                channel_index=channel_index,
                score=float(scores[position]),
                pathological=bool(reasons),
                reasons=tuple(reasons),
            )
        )
    return tuple(records)


def _make_batches(
    records: Sequence[ChannelDifficulty],
    *,
    width: int,
    prefix: str,
    quarantined: bool,
) -> list[ChannelBatch]:
    batches = []
    for start in range(0, len(records), width):
        group = records[start : start + width]
        channel_indices = tuple(item.channel_index for item in group)
        padded = channel_indices + (channel_indices[-1],) * (
            width - len(channel_indices)
        )
        scores = [item.score for item in group]
        batches.append(
            ChannelBatch(
                batch_id=f"{prefix}{len(batches):04d}",
                channel_indices=channel_indices,
                padded_channel_indices=padded,
                lane_width=width,
                quarantined=quarantined,
                min_difficulty=min(scores),
                max_difficulty=max(scores),
            )
        )
    return batches


def build_difficulty_batch_plan(
    pilot: PilotDiagnostics | Mapping[str, Any],
    *,
    nominal_width: int,
    quarantine_width: int = 1,
    config: DifficultyConfig | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> ChannelBatchPlan:
    """Group channels by pilot cost and isolate the pathological tail.

    Within each population channels are sorted from hardest to easiest, using
    original wavelength position as the stable tie-breaker.  The final batch
    at each resident width is padded by repeating its last active channel.
    """

    if not isinstance(pilot, PilotDiagnostics):
        pilot = PilotDiagnostics.from_mapping(pilot)
    nominal_width = int(nominal_width)
    quarantine_width = int(quarantine_width)
    if nominal_width < 1:
        raise ValueError("nominal_width must be positive.")
    if not (1 <= quarantine_width <= nominal_width):
        raise ValueError(
            "quarantine_width must be between one and nominal_width."
        )
    config = DifficultyConfig() if config is None else config
    records = _difficulty_records(pilot, config)
    positions = {
        channel_index: position
        for position, channel_index in enumerate(pilot.channel_indices)
    }

    def stable_hardest_first(item: ChannelDifficulty):
        return (-item.score, positions[item.channel_index])

    pathological = sorted(
        (item for item in records if item.pathological),
        key=stable_hardest_first,
    )
    ordinary = sorted(
        (item for item in records if not item.pathological),
        key=stable_hardest_first,
    )
    # Quarantined batches run first so a numerically broken channel fails fast
    # and cannot hold an otherwise productive full-width batch hostage.
    batches = _make_batches(
        pathological,
        width=quarantine_width,
        prefix="quarantine-",
        quarantined=True,
    )
    batches.extend(
        _make_batches(
            ordinary,
            width=nominal_width,
            prefix="regular-",
            quarantined=False,
        )
    )
    return ChannelBatchPlan(
        pilot=pilot,
        config=config,
        nominal_width=nominal_width,
        quarantine_width=quarantine_width,
        difficulties=records,
        batches=tuple(batches),
        provenance=provenance,
    )


def restore_channel_order(
    values_by_batch: Sequence[Any],
    plan: ChannelBatchPlan,
    *,
    channel_axis: int = 1,
) -> np.ndarray:
    """Concatenate batch outputs and restore the pilot's wavelength order.

    Each input may contain only active channels (the normal sampler contract)
    or all padded lanes.  Padded lanes are trimmed before restoration.
    """

    if len(values_by_batch) != len(plan.batches):
        raise ValueError(
            "values_by_batch must contain one array for every planned batch."
        )
    active_values = []
    for value, batch in zip(values_by_batch, plan.batches):
        array = np.asarray(value)
        axis = _normalize_axis_index(channel_axis, array.ndim)
        axis_size = array.shape[axis]
        if axis_size not in {batch.num_active, batch.lane_width}:
            raise ValueError(
                f"Batch {batch.batch_id!r} has channel-axis size {axis_size}; "
                f"expected {batch.num_active} active or {batch.lane_width} padded."
            )
        if axis_size != batch.num_active:
            array = np.take(array, range(batch.num_active), axis=axis)
        active_values.append(array)

    axis = _normalize_axis_index(channel_axis, active_values[0].ndim)
    concatenated = np.concatenate(active_values, axis=axis)
    execution_positions = {
        channel: position
        for position, channel in enumerate(plan.execution_channel_indices)
    }
    restoration = [
        execution_positions[channel]
        for channel in plan.original_channel_indices
    ]
    return np.take(concatenated, restoration, axis=axis)


def restore_sample_mapping(
    samples_by_batch: Sequence[Mapping[str, Any]],
    plan: ChannelBatchPlan,
    *,
    channel_axis: int = 1,
) -> dict[str, np.ndarray]:
    """Restore every sample site in a list of per-batch dictionaries."""

    if len(samples_by_batch) != len(plan.batches):
        raise ValueError(
            "samples_by_batch must contain one mapping for every planned batch."
        )
    expected = set(samples_by_batch[0])
    for batch_samples in samples_by_batch[1:]:
        if set(batch_samples) != expected:
            raise ValueError("All sample batches must contain identical sites.")
    return {
        name: restore_channel_order(
            [batch_samples[name] for batch_samples in samples_by_batch],
            plan,
            channel_axis=channel_axis,
        )
        for name in sorted(expected)
    }


@dataclass(frozen=True)
class WidthMeasurement:
    """Measured performance for one resident channel width."""

    width: int
    peak_memory_bytes: int
    throughput: float

    def __post_init__(self):
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "peak_memory_bytes", int(self.peak_memory_bytes))
        object.__setattr__(self, "throughput", float(self.throughput))
        if self.width < 1:
            raise ValueError("Measured width must be positive.")
        if self.peak_memory_bytes < 0:
            raise ValueError("peak_memory_bytes must be non-negative.")
        if not math.isfinite(self.throughput) or self.throughput <= 0:
            raise ValueError("throughput must be finite and positive.")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "WidthMeasurement":
        return cls(
            width=int(payload.get("width", payload.get("lane_width"))),
            peak_memory_bytes=int(
                payload.get(
                    "peak_memory_bytes",
                    payload.get("peak_hbm_bytes"),
                )
            ),
            throughput=float(
                payload.get(
                    "throughput",
                    payload.get("ess_per_second"),
                )
            ),
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "peak_memory_bytes": self.peak_memory_bytes,
            "throughput": self.throughput,
        }


@dataclass(frozen=True)
class WidthSelection:
    """The cacheable result of a measured resident-width sweep."""

    selected: WidthMeasurement
    measurements: tuple[WidthMeasurement, ...]
    total_memory_bytes: int
    memory_fraction: float
    memory_budget_bytes: int
    eligible_widths: tuple[int, ...]
    rejected_widths: tuple[int, ...]
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self):
        if self.provenance is None:
            return
        if not isinstance(self.provenance, Mapping):
            raise ValueError("Width-selection provenance must be a mapping.")
        normalized = json.loads(
            json.dumps(
                dict(self.provenance),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        object.__setattr__(self, "provenance", normalized)

    def fingerprint_inputs(self) -> dict[str, Any]:
        return {
            "schema_version": WIDTH_SELECTION_SCHEMA_VERSION,
            "total_memory_bytes": self.total_memory_bytes,
            "memory_fraction": self.memory_fraction,
            "memory_budget_bytes": self.memory_budget_bytes,
            "measurements": [item.to_manifest() for item in self.measurements],
            "selected_width": self.selected.width,
            "eligible_widths": list(self.eligible_widths),
            "rejected_widths": list(self.rejected_widths),
            "provenance": self.provenance,
        }

    @property
    def fingerprint_sha256(self) -> str:
        return _canonical_sha256(self.fingerprint_inputs())

    def to_manifest(self) -> dict[str, Any]:
        payload = self.fingerprint_inputs()
        payload["fingerprint_sha256"] = self.fingerprint_sha256
        return payload

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "WidthSelection":
        if int(payload.get("schema_version", -1)) != WIDTH_SELECTION_SCHEMA_VERSION:
            raise ValueError("Unsupported width-selection manifest schema.")
        selection = select_autotuned_width(
            tuple(
                WidthMeasurement.from_mapping(item)
                for item in payload["measurements"]
            ),
            total_memory_bytes=int(payload["total_memory_bytes"]),
            memory_fraction=float(payload["memory_fraction"]),
            provenance=payload.get("provenance"),
        )
        if int(payload["selected_width"]) != selection.selected.width:
            raise ValueError(
                "Width-selection manifest does not contain the optimal "
                "measured width."
            )
        recorded = payload.get("fingerprint_sha256")
        if recorded is not None and recorded != selection.fingerprint_sha256:
            raise ValueError("Width-selection manifest fingerprint does not match.")
        return selection


def select_autotuned_width(
    measurements: Sequence[WidthMeasurement | Mapping[str, Any]],
    *,
    total_memory_bytes: int,
    memory_fraction: float = 0.70,
    provenance: Mapping[str, Any] | None = None,
) -> WidthSelection:
    """Select maximum measured throughput under the allowed HBM fraction.

    Ties prefer lower peak memory and then the smaller resident width.  This
    conservative tie-breaker avoids spending memory when it buys no measured
    throughput.  A width that was not measured is never selected.
    """

    normalized = tuple(
        item
        if isinstance(item, WidthMeasurement)
        else WidthMeasurement.from_mapping(item)
        for item in measurements
    )
    if not normalized:
        raise ValueError("At least one width measurement is required.")
    widths = [item.width for item in normalized]
    if len(widths) != len(set(widths)):
        raise ValueError("Each resident width may be measured only once.")
    total_memory_bytes = int(total_memory_bytes)
    memory_fraction = float(memory_fraction)
    if total_memory_bytes <= 0:
        raise ValueError("total_memory_bytes must be positive.")
    if not math.isfinite(memory_fraction) or not (0.0 < memory_fraction <= 1.0):
        raise ValueError("memory_fraction must be in (0, 1].")

    memory_budget_bytes = int(math.floor(total_memory_bytes * memory_fraction))
    eligible = tuple(
        item for item in normalized if item.peak_memory_bytes <= memory_budget_bytes
    )
    if not eligible:
        smallest = min(item.peak_memory_bytes for item in normalized)
        raise ValueError(
            "No measured width fits the memory budget: "
            f"budget={memory_budget_bytes} bytes, smallest peak={smallest} bytes."
        )
    selected = max(
        eligible,
        key=lambda item: (
            item.throughput,
            -item.peak_memory_bytes,
            -item.width,
        ),
    )
    ordered = tuple(sorted(normalized, key=lambda item: item.width))
    eligible_widths = tuple(sorted(item.width for item in eligible))
    rejected_widths = tuple(
        item.width
        for item in ordered
        if item.peak_memory_bytes > memory_budget_bytes
    )
    return WidthSelection(
        selected=selected,
        measurements=ordered,
        total_memory_bytes=total_memory_bytes,
        memory_fraction=memory_fraction,
        memory_budget_bytes=memory_budget_bytes,
        eligible_widths=eligible_widths,
        rejected_widths=rejected_widths,
        provenance=provenance,
    )


__all__ = [
    "BATCH_PLAN_SCHEMA_VERSION",
    "WIDTH_SELECTION_SCHEMA_VERSION",
    "ChannelBatch",
    "ChannelBatchPlan",
    "ChannelDifficulty",
    "DifficultyConfig",
    "PilotDiagnostics",
    "WidthMeasurement",
    "WidthSelection",
    "SpectroMemoryModel",
    "build_difficulty_batch_plan",
    "restore_channel_order",
    "restore_sample_mapping",
    "select_autotuned_width",
    "fit_spectro_memory_model",
    "resolve_spectro_auto_width",
]
