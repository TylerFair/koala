import json

import fit_jwst


def test_sampler_diagnostic_annotation_sanitizes_optional_nan(tmp_path):
    path = tmp_path / "diagnostics.json"
    path.write_text(json.dumps({"map_hessian_relative_error_per_channel": [float("nan"), 0.1]}))
    fit_jwst._annotate_sampler_diagnostics(
        path,
        sampling_workload_fingerprint="abc",
        sampler_backend="independent_nuts",
    )
    payload = json.loads(path.read_text())
    assert payload["map_hessian_relative_error_per_channel"] == [None, 0.1]
    assert payload["sampling_workload_fingerprint_sha256"] == "abc"
