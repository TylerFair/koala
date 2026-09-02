import json

import numpy as np
import pytest

from models.channel_batching import (
    ChannelBatchPlan,
    DifficultyConfig,
    PilotDiagnostics,
    WidthMeasurement,
    WidthSelection,
    build_difficulty_batch_plan,
    restore_channel_order,
    restore_sample_mapping,
    select_autotuned_width,
)


def _pilot(
    mean_steps,
    *,
    max_steps=None,
    divergences=None,
    step_size=None,
    channel_indices=None,
    num_draws=100,
):
    count = len(mean_steps)
    if max_steps is None:
        max_steps = np.asarray(mean_steps) * 2.0
    if divergences is None:
        divergences = np.zeros(count, dtype=int)
    if step_size is None:
        step_size = np.full(count, 0.1)
    if channel_indices is None:
        channel_indices = np.arange(count)
    return PilotDiagnostics(
        channel_indices=tuple(channel_indices),
        mean_num_steps=tuple(mean_steps),
        max_num_steps=tuple(max_steps),
        num_divergences=tuple(divergences),
        step_size=tuple(step_size),
        num_draws=num_draws,
    )


def _no_quarantine_config():
    return DifficultyConfig(
        pathology_step_ratio=1.0e12,
        pathology_step_size_ratio=1.0e12,
        pathology_score_z=1.0e12,
        quarantine_divergences=False,
    )


def test_existing_independent_nuts_diagnostic_schema_is_consumed():
    payload = {
        "num_channels": 3,
        "num_draws": 50,
        "num_divergences": 1,
        "mean_num_steps_per_channel": [3.0, 7.0, 12.0],
        "max_num_steps_per_channel": [7, 15, 31],
        "num_divergences_per_channel": [0, 1, 0],
        "adapted_step_size_per_channel": [0.2, 0.1, 0.05],
    }

    pilot = PilotDiagnostics.from_mapping(
        payload,
        channel_indices=[101, 103, 109],
    )

    assert pilot.channel_indices == (101, 103, 109)
    assert pilot.mean_num_steps == (3.0, 7.0, 12.0)
    assert pilot.num_divergences == (0, 1, 0)
    assert pilot.num_draws == 50


def test_difficulty_plan_is_deterministic_stable_and_json_roundtrips():
    # Equal metrics exercise the original-position tie-breaker.  The channel
    # identifiers are deliberately non-monotonic to distinguish stable input
    # order from an accidental numeric sort.
    pilot = _pilot(
        [8.0] * 5,
        max_steps=[15.0] * 5,
        channel_indices=[7, 2, 9, 4, 1],
    )

    first = build_difficulty_batch_plan(
        pilot,
        nominal_width=3,
        config=_no_quarantine_config(),
    )
    second = build_difficulty_batch_plan(
        pilot,
        nominal_width=3,
        config=_no_quarantine_config(),
    )

    assert first.execution_channel_indices == (7, 2, 9, 4, 1)
    assert first.to_manifest() == second.to_manifest()
    assert first.fingerprint_sha256 == second.fingerprint_sha256
    encoded = json.dumps(first.to_manifest(), allow_nan=False, sort_keys=True)
    restored = ChannelBatchPlan.from_manifest(json.loads(encoded))
    assert restored == first
    assert restored.fingerprint_sha256 == first.fingerprint_sha256


def test_batch_plan_provenance_is_normalized_and_fingerprinted():
    provenance = {
        "sampling_workload_fingerprint_sha256": "abc123",
        "artifact_kind": "jwst_spectro_channel_batch_plan",
        "nested": {"value": 2},
    }
    plan = build_difficulty_batch_plan(
        _pilot([4.0, 8.0]),
        nominal_width=2,
        config=_no_quarantine_config(),
        provenance=provenance,
    )
    restored = ChannelBatchPlan.from_manifest(
        json.loads(json.dumps(plan.to_manifest(), allow_nan=False))
    )
    assert restored.provenance == provenance
    assert restored.fingerprint_sha256 == plan.fingerprint_sha256

    changed = build_difficulty_batch_plan(
        _pilot([4.0, 8.0]),
        nominal_width=2,
        config=_no_quarantine_config(),
        provenance={**provenance, "sampling_workload_fingerprint_sha256": "different"},
    )
    assert changed.fingerprint_sha256 != plan.fingerprint_sha256


def test_batch_padding_and_original_wavelength_order_recovery():
    pilot = _pilot(
        [2.0, 32.0, 4.0, 16.0, 8.0],
        max_steps=[3.0, 63.0, 7.0, 31.0, 15.0],
        channel_indices=[10, 11, 12, 13, 14],
    )
    plan = build_difficulty_batch_plan(
        pilot,
        nominal_width=3,
        config=_no_quarantine_config(),
    )

    assert plan.execution_channel_indices != plan.original_channel_indices
    assert len(plan.batches) == 2
    assert plan.batches[-1].num_active == 2
    assert plan.batches[-1].padded_channel_indices[-1] == (
        plan.batches[-1].channel_indices[-1]
    )

    # Supply padded arrays to ensure the duplicate dummy lane is discarded.
    batch_values = [
        np.broadcast_to(
            np.asarray(batch.padded_channel_indices)[None, :, None],
            (2, batch.lane_width, 1),
        )
        for batch in plan.batches
    ]
    restored = restore_channel_order(batch_values, plan, channel_axis=1)
    expected = np.broadcast_to(
        np.asarray(plan.original_channel_indices)[None, :, None],
        (2, len(plan.original_channel_indices), 1),
    )
    np.testing.assert_array_equal(restored, expected)

    mapped = restore_sample_mapping(
        [{"rors": value, "trend": value + 100} for value in batch_values],
        plan,
    )
    np.testing.assert_array_equal(mapped["rors"], expected)
    np.testing.assert_array_equal(mapped["trend"], expected + 100)


def test_pathological_tail_is_quarantined_into_singleton_batches():
    pilot = _pilot(
        [8, 9, 7, 8, 512, 9, 8, 7],
        max_steps=[15, 15, 15, 15, 1023, 15, 15, 15],
        divergences=[0, 0, 0, 0, 2, 0, 0, 0],
        step_size=[0.1, 0.09, 0.11, 0.1, 0.001, 0.09, 0.1, 0.11],
    )

    plan = build_difficulty_batch_plan(
        pilot,
        nominal_width=4,
        quarantine_width=1,
    )

    quarantined = [batch for batch in plan.batches if batch.quarantined]
    regular = [batch for batch in plan.batches if not batch.quarantined]
    assert len(quarantined) == 1
    assert quarantined[0].channel_indices == (4,)
    assert quarantined[0].lane_width == 1
    assert all(4 not in batch.channel_indices for batch in regular)
    record = plan.difficulties[4]
    assert record.pathological
    assert "divergences" in record.reasons
    assert "mean_steps_ratio" in record.reasons
    assert "step_size_ratio" in record.reasons


def test_quarantine_width_can_group_only_the_pathological_population():
    pilot = _pilot(
        [8, 256, 8, 512, 8, 8],
        max_steps=[15, 511, 15, 1023, 15, 15],
        divergences=[0, 1, 0, 1, 0, 0],
        step_size=[0.1, 0.005, 0.1, 0.001, 0.1, 0.1],
    )
    plan = build_difficulty_batch_plan(
        pilot,
        nominal_width=4,
        quarantine_width=2,
    )

    assert plan.batches[0].quarantined
    assert plan.batches[0].lane_width == 2
    assert set(plan.batches[0].channel_indices) == {1, 3}
    assert not plan.batches[1].quarantined
    assert plan.batches[1].lane_width == 4


def test_autotuner_selects_fastest_measured_width_under_memory_fraction():
    measurements = [
        {"lane_width": 8, "peak_hbm_bytes": 200, "ess_per_second": 10.0},
        {"lane_width": 16, "peak_hbm_bytes": 500, "ess_per_second": 14.0},
        {"lane_width": 32, "peak_hbm_bytes": 800, "ess_per_second": 100.0},
    ]

    selection = select_autotuned_width(
        measurements,
        total_memory_bytes=1000,
        memory_fraction=0.70,
    )

    assert selection.memory_budget_bytes == 700
    assert selection.selected.width == 16
    assert selection.eligible_widths == (8, 16)
    assert selection.rejected_widths == (32,)
    encoded = json.dumps(selection.to_manifest(), allow_nan=False)
    assert WidthSelection.from_manifest(json.loads(encoded)) == selection


def test_autotuner_tie_prefers_lower_memory_and_rejects_no_fit():
    selection = select_autotuned_width(
        [
            WidthMeasurement(width=8, peak_memory_bytes=200, throughput=20.0),
            WidthMeasurement(width=16, peak_memory_bytes=400, throughput=20.0),
        ],
        total_memory_bytes=1000,
    )
    assert selection.selected.width == 8

    with pytest.raises(ValueError, match="No measured width fits"):
        select_autotuned_width(
            [WidthMeasurement(width=1, peak_memory_bytes=800, throughput=1.0)],
            total_memory_bytes=1000,
            memory_fraction=0.70,
        )
