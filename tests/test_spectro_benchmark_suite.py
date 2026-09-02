import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest

from models.channel_batching import ChannelBatchPlan, WidthSelection
from tools.benchmark_spectro_samplers import (
    _CompilationEvents,
    _case_seeds,
    _source_fingerprints,
    _summarize_posterior,
    _validate_args,
    build_parser,
)
from tools.benchmark_independent_nuts_gpu import make_problem
from tools.spectro_benchmark_core import (
    atomic_write_json,
    build_measured_comparisons,
    build_trend_comparisons,
    parse_choice_list,
    parse_positive_int_list,
    quality_reasons,
    select_candidates,
    summarize_timing,
)
from tools.spectro_manifest_tools import (
    aggregate_pilot_diagnostics,
    build_batch_plan_from_chunks,
    build_width_selection_from_results,
    parse_pilot_chunk_spec,
)


def _result(
    case_id,
    backend,
    trend,
    channels,
    *,
    wall=10.0,
    ess_rate=20.0,
    ess_min=100.0,
    divergences=0.0,
    peak=100,
    median=None,
):
    if median is None:
        median = [0.1] * channels
    return {
        "status": "ok",
        "case_id": case_id,
        "backend": backend,
        "trend_mode": trend,
        "channels": channels,
        "timing": {"steady_state_wall_seconds": wall},
        "diagnostics": {"divergence_fraction": divergences},
        "memory": {"peak_device_bytes": peak},
        "posterior": {
            "all_finite": True,
            "rors_ess_min": ess_min,
            "rors_ess_sum_per_second": ess_rate,
            "rors_median_per_channel": median,
            "rors_sd_per_channel": [0.01] * channels,
        },
    }


def test_choice_width_and_cli_validation_are_strict():
    assert parse_choice_list(
        "joint_nuts,independent_hmc",
        valid=("joint_nuts", "independent_hmc"),
        label="backends",
    ) == ("joint_nuts", "independent_hmc")
    assert parse_positive_int_list("8,16,40") == (8, 16, 40)
    with pytest.raises(ValueError, match="Unknown backends"):
        parse_choice_list("magic", valid=("joint_nuts",), label="backends")
    with pytest.raises(ValueError, match="duplicates"):
        parse_positive_int_list("8,8")

    args = build_parser().parse_args(
        [
            "--backends",
            "joint_nuts",
            "--emit-width-selection",
            "width.json",
        ]
    )
    with pytest.raises(ValueError, match="manifest-backend"):
        _validate_args(args)


def test_jax_compile_events_and_timing_summary_are_separate():
    events = _CompilationEvents()
    mark = events.mark()
    events.listener(
        "/jax/core/compile/jaxpr_trace_duration", 1.25, fun_name="a"
    )
    events.listener(
        "/jax/core/compile/backend_compile_duration", 2.0, fun_name="a"
    )
    measured = events.since(mark)
    assert measured["compile_seconds"] == pytest.approx(3.25)
    assert measured["jaxpr_trace_count"] == 1
    assert measured["backend_compile_count"] == 1

    summary = summarize_timing(
        [
            {"wall_seconds": 12.0, "compile_seconds": 4.0},
            {"wall_seconds": 7.0, "compile_seconds": 1.0},
            {"wall_seconds": 9.0, "compile_seconds": 1.0},
        ]
    )
    assert summary["cold_compile_seconds"] == 4.0
    assert summary["steady_state_wall_seconds"] == 8.0
    assert summary["steady_state_execution_estimate_seconds"] == 7.0


def test_autotuning_rejects_bad_diagnostics_and_stays_within_prior_family():
    results = [
        _result("safe", "joint_nuts", "sampled_uniform", 8, ess_rate=10.0),
        _result(
            "fast_but_bad",
            "independent_nuts",
            "sampled_uniform",
            8,
            ess_rate=1000.0,
            divergences=0.1,
        ),
        _result(
            "marginal",
            "independent_hmc",
            "gaussian_marginalized",
            8,
            ess_rate=40.0,
        ),
    ]
    selection = select_candidates(
        results,
        max_divergence_fraction=0.01,
        min_channel_ess=50,
    )
    assert selection["cross_trend_selection_performed"] is False
    assert selection["by_trend_mode"]["sampled_uniform"]["selected_case_id"] == "safe"
    assert (
        selection["by_trend_mode"]["gaussian_marginalized"]["selected_case_id"]
        == "marginal"
    )


def test_quality_gate_uses_worst_lane_and_tree_depth_saturation():
    result = _result(
        "bad_lane", "independent_nuts", "sampled_uniform", 80,
        divergences=0.00625,
    )
    result["diagnostics"].update(
        {
            "max_divergence_fraction_per_channel": 0.5,
            "tree_depth_saturation_fraction_max": 0.2,
        }
    )
    reasons = quality_reasons(
        result,
        max_divergence_fraction=0.01,
        min_channel_ess=50,
        max_tree_depth_saturation_fraction=0.05,
    )
    assert any("divergence fraction 0.5" in reason for reason in reasons)
    assert any("tree-depth saturation fraction 0.2" in reason for reason in reasons)


def test_memory_constraint_rejects_unavailable_peak():
    result = _result(
        "missing_memory", "independent_nuts", "sampled_uniform", 8,
        peak=None,
    )
    reasons = quality_reasons(
        result,
        max_divergence_fraction=0.01,
        min_channel_ess=50,
        max_device_memory_bytes=10_000,
    )
    assert any("peak device memory is unavailable" in reason for reason in reasons)


def test_posterior_finiteness_checks_every_site_and_leaf():
    draws = {
        "rors": np.stack(
            (
                np.linspace(0.14, 0.15, 16),
                np.linspace(0.145, 0.155, 16),
            ),
            axis=1,
        )[..., None],
        "trend": {
            "c": np.ones((16, 2)),
            "v": np.concatenate(
                (np.zeros((15, 2)), np.asarray([[0.0, np.inf]])), axis=0
            ),
        },
    }
    summary = _summarize_posterior(draws, steady_seconds=1.0)
    assert summary["all_finite"] is False
    assert summary["posterior_leaf_count"] == 3
    assert summary["nonfinite_sites"] == ["trend"]
    assert summary["nonfinite_leaf_paths"] == ["trend.v"]
    assert summary["nonfinite_value_count"] == 1


def test_width_sweep_problem_is_one_nested_realization_and_seeds_ignore_width():
    small = make_problem(2, 9, 7123, reference_channels=4)
    large = make_problem(4, 9, 7123, reference_channels=4)
    np.testing.assert_array_equal(np.asarray(small[1]), np.asarray(large[1]))
    np.testing.assert_array_equal(np.asarray(small[2]), np.asarray(large[2])[:2])
    np.testing.assert_array_equal(np.asarray(small[3]), np.asarray(large[3])[:2])
    for name in small[5]:
        small_value = np.asarray(small[5][name])
        large_value = np.asarray(large[5][name])
        if small_value.ndim and large_value.shape[0] == 4:
            np.testing.assert_array_equal(small_value, large_value[:2])
        else:
            np.testing.assert_array_equal(small_value, large_value)
    assert small[7]["synthetic_reference_channels"] == 4
    assert small[7]["synthetic_channel_subset"] == "prefix[0:2]"
    assert small[7]["synthetic_channel_order"] == (
        "base2_van_der_corput_prefix"
    )
    np.testing.assert_array_equal(
        small[7]["synthetic_channel_phase"],
        large[7]["synthetic_channel_phase"][:2],
    )
    # The first four nested lanes already cover both halves of the domain.
    assert min(large[7]["synthetic_channel_phase"]) <= -0.5
    assert max(large[7]["synthetic_channel_phase"]) >= 0.5
    assert _case_seeds(99, "independent_nuts", "sampled_uniform") == (
        99,
        1_001_111,
    )


def test_sampler_benchmark_fingerprints_full_measured_workload():
    fingerprints = _source_fingerprints()
    required = {
        "tools/benchmark_independent_nuts_gpu.py",
        "models/jaxoplanet/core.py",
        "models/jaxoplanet/limb_dark_streamed.py",
        "models/channel_batching.py",
        "models/independent_nuts.py",
        "models/independent_hmc.py",
    }
    assert required <= fingerprints.keys()


def test_comparisons_require_matching_width_and_mark_trend_prior_difference():
    joint = _result("joint", "joint_nuts", "sampled_uniform", 8, wall=20, ess_rate=10)
    independent = _result(
        "ind", "independent_nuts", "sampled_uniform", 8, wall=5, ess_rate=30
    )
    wrong_width = _result(
        "ind16", "independent_nuts", "sampled_uniform", 16, wall=2, ess_rate=50
    )
    marginalized = _result(
        "marg", "joint_nuts", "gaussian_marginalized", 8, wall=4, ess_rate=50
    )
    sampler = build_measured_comparisons([joint, independent, wrong_width, marginalized])
    assert len(sampler) == 1
    assert sampler[0]["wall_speedup"] == 4.0
    assert sampler[0]["prior_family_identical"] is True

    trend = build_trend_comparisons([joint, marginalized])
    assert len(trend) == 1
    assert trend[0]["measured_wall_speedup"] == 5.0
    assert trend[0]["prior_family_identical"] is False
    assert trend[0]["posterior_equivalence_claimed"] is False

    missing_ess = _result(
        "missing", "joint_nuts", "sampled_uniform", 4, ess_rate=1.0
    )
    missing_ess["posterior"]["rors_ess_sum_per_second"] = None
    assert build_measured_comparisons([missing_ess]) == []


def _write_diagnostics(
    path: Path,
    means,
    divergences=None,
    *,
    workload_fingerprint=None,
    sampler_backend="independent_nuts",
):
    count = len(means)
    payload = {
        "num_draws": 50,
        "mean_num_steps_per_channel": list(means),
        "max_num_steps_per_channel": [2 * value for value in means],
        "num_divergences_per_channel": (
            [0] * count if divergences is None else list(divergences)
        ),
        "adapted_step_size_per_channel": [1.0 / value for value in means],
    }
    if workload_fingerprint is not None:
        payload.update(
            {
                "sampling_workload_fingerprint_sha256": workload_fingerprint,
                "sampler_backend": sampler_backend,
            }
        )
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_pilot_chunks_use_explicit_global_indices_and_emit_fit_plan(tmp_path):
    first = tmp_path / "chunk_a.json"
    second = tmp_path / "chunk_b.json"
    _write_diagnostics(first, [4.0, 8.0])
    _write_diagnostics(second, [128.0, 16.0], divergences=[1, 0])

    indices, path = parse_pilot_chunk_spec(f"2:4={second}")
    assert indices == (2, 3)
    assert path == second
    # Deliberately reverse input order; aggregation returns wavelength order.
    specs = [f"2:4={second}", f"0:2={first}"]
    pilot, sources = aggregate_pilot_diagnostics(specs)
    assert pilot.channel_indices == (0, 1, 2, 3)
    assert pilot.mean_num_steps == (4.0, 8.0, 128.0, 16.0)
    assert len(sources) == 2

    plan, _ = build_batch_plan_from_chunks(
        specs, nominal_width=2, quarantine_width=1
    )
    manifest_path = tmp_path / "plan.json"
    atomic_write_json(manifest_path, plan.to_manifest())
    loaded = ChannelBatchPlan.from_manifest(json.loads(manifest_path.read_text()))
    assert loaded.original_channel_indices == (0, 1, 2, 3)
    assert any(batch.quarantined for batch in loaded.batches)


def test_pilot_chunk_overlap_or_gap_is_rejected(tmp_path):
    one = tmp_path / "one.json"
    two = tmp_path / "two.json"
    _write_diagnostics(one, [2.0, 3.0])
    _write_diagnostics(two, [4.0, 5.0])
    with pytest.raises(ValueError, match="multiple"):
        aggregate_pilot_diagnostics([f"0:2={one}", f"1:3={two}"])
    with pytest.raises(ValueError, match="cover every global"):
        aggregate_pilot_diagnostics([f"0:2={one}", f"3:5={two}"])


def test_fit_consumable_plan_requires_one_exact_pilot_workload(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    fingerprint = "a" * 64
    _write_diagnostics(
        first, [4.0, 8.0], workload_fingerprint=fingerprint
    )
    _write_diagnostics(
        second, [16.0, 32.0], workload_fingerprint=fingerprint
    )
    specs = [f"0:2={first}", f"2:4={second}"]
    plan, sources = build_batch_plan_from_chunks(
        specs,
        nominal_width=2,
        require_workload_provenance=True,
    )
    assert plan.provenance["artifact_kind"] == (
        "jwst_spectro_channel_batch_plan"
    )
    assert plan.provenance["sampling_workload_fingerprint_sha256"] == (
        fingerprint
    )
    assert len(plan.provenance["pilot_diagnostics"]) == 2
    assert all(len(source["diagnostic_sha256"]) == 64 for source in sources)

    _write_diagnostics(
        second, [16.0, 32.0], workload_fingerprint="b" * 64
    )
    with pytest.raises(ValueError, match="different sampling workloads"):
        build_batch_plan_from_chunks(
            specs,
            nominal_width=2,
            require_workload_provenance=True,
        )

    legacy = tmp_path / "legacy.json"
    _write_diagnostics(legacy, [4.0, 8.0])
    with pytest.raises(ValueError, match="no sampling workload fingerprint"):
        build_batch_plan_from_chunks(
            [f"0:2={legacy}"],
            nominal_width=2,
            require_workload_provenance=True,
        )


def test_measured_width_results_emit_roundtrippable_fit_manifest(tmp_path):
    results = [
        _result(
            "w8", "independent_nuts", "gaussian_marginalized", 8,
            ess_rate=20.0, peak=200,
        ),
        _result(
            "w16", "independent_nuts", "gaussian_marginalized", 16,
            ess_rate=30.0, peak=600,
        ),
        _result(
            "other", "independent_hmc", "gaussian_marginalized", 8,
            ess_rate=100.0, peak=100,
        ),
    ]
    provenance = {
        "sampler_backend": "independent_nuts",
        "trend_inference": "gaussian_marginalized",
        "num_cadences": 513,
    }
    selection, skipped = build_width_selection_from_results(
        results,
        backend="independent_nuts",
        trend_mode="gaussian_marginalized",
        total_memory_bytes=1000,
        memory_fraction=0.70,
        eligible_case_ids={"w8", "w16"},
        provenance=provenance,
    )
    assert skipped == []
    assert selection.selected.width == 16
    path = tmp_path / "width.json"
    atomic_write_json(path, selection.to_manifest())
    loaded = WidthSelection.from_manifest(json.loads(path.read_text()))
    assert loaded.selected.width == 16
    assert loaded.provenance == provenance


def test_atomic_json_rejects_nan(tmp_path):
    with pytest.raises(ValueError):
        atomic_write_json(tmp_path / "bad.json", {"value": float("nan")})
    assert not (tmp_path / "bad.json").exists()
