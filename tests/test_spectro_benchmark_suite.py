import os

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest

from tools.benchmark_spectro_samplers import (
    _CompilationEvents,
    _summarize_posterior,
    _validate_args,
    build_parser,
)
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


def test_atomic_json_rejects_nan(tmp_path):
    with pytest.raises(ValueError):
        atomic_write_json(tmp_path / "bad.json", {"value": float("nan")})
    assert not (tmp_path / "bad.json").exists()
