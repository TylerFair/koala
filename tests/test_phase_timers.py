import json
import time

import fit_jwst


def test_phase_timer_is_opt_in_and_writes_atomic_summary(tmp_path):
    timer = fit_jwst._PipelinePhaseTimers()
    with timer.phase("disabled"):
        pass
    assert timer.events == []
    timer.enable(str(tmp_path))
    with timer.phase("data_load", cache="hit"):
        time.sleep(0.001)
    timer.write()
    payload = json.loads((tmp_path / "phase_timings.json").read_text())
    assert payload["schema_version"] == 1
    assert payload["phase_totals_seconds"]["data_load"] > 0
    assert payload["events"][0]["cache"] == "hit"
    assert not list(tmp_path.glob("*.tmp.*"))


def test_minimal_plot_wrapper_skips_only_diagnostic_plot(monkeypatch, tmp_path):
    timer = fit_jwst._PipelinePhaseTimers()
    monkeypatch.setattr(fit_jwst, "_PHASE_TIMERS", timer)
    timer.enable(str(tmp_path))
    called = []
    monkeypatch.setattr(fit_jwst, "plot_noise_binning_robust", lambda *a, **k: called.append(True))
    monkeypatch.setattr(fit_jwst, "save_results", lambda *a, **k: called.append("csv"))
    fit_jwst._install_timed_output_helpers(minimal_plots=True)
    fit_jwst.plot_noise_binning_robust(None, None, None)
    fit_jwst.save_results(None, None, None, None)
    assert called == ["csv"]
    assert any(event.get("skipped_by_minimal_plots") for event in timer.events)
