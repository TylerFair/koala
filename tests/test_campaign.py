import csv
import pickle

import numpy as np

from tools.campaign.run_dataset import (
    apply_calibrated_gates,
    compare_lowres_summary,
    concatenate_reference_chunks,
)
from tools.campaign.summarize import summarize


def test_reference_chunks_are_sorted_and_concatenated(tmp_path):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    for start, end in ((2, 4), (0, 2)):
        with (chunks / f"target_Rx_chunk_{start}_{end}.pkl").open("wb") as stream:
            pickle.dump(
                {"c": np.full((3, end - start), start, dtype=np.float64)},
                stream,
            )
    samples, manifest = concatenate_reference_chunks(tmp_path, 4)
    assert samples["c"].shape == (3, 4)
    assert np.all(samples["c"][:, :2] == 0)
    assert np.all(samples["c"][:, 2:] == 2)
    assert manifest["channels"] == 4


def test_lowres_summary_and_noise_exclusion(tmp_path):
    path = tmp_path / "R20_bestfit_params.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["wavelength", "rors", "rors_err_low", "rors_err_high"],
        )
        writer.writeheader()
        writer.writerow(
            {"wavelength": 1.0, "rors": "[0.1]", "rors_err_low": "[0.01]", "rors_err_high": "[0.01]"}
        )
        writer.writerow(
            {"wavelength": 2.0, "rors": "[0.2]", "rors_err_low": "[0.02]", "rors_err_high": "[0.02]"}
        )
    standardized = np.linspace(-1.0, 1.0, 21)
    standardized -= np.median(standardized)
    standardized /= np.std(standardized, ddof=1)
    candidate = {
        "rors": np.stack(
            [0.1 + 0.01 * standardized, 0.2 + 0.02 * standardized], axis=1
        )[..., None],
        "log_jitter": np.zeros((20, 2)),
    }
    rows = compare_lowres_summary(candidate, path)
    rows.append(
        {
            "site": "log_jitter",
            "channel": 0,
            "abs_median_shift_ref_sigma": 99.0,
            "sigma_ratio": 99.0,
            "p16_shift_ref_sigma": 99.0,
            "p84_shift_ref_sigma": 99.0,
        }
    )
    result = apply_calibrated_gates(rows)
    assert result["gate_num_rows"] == 2
    assert result["gate_pass_fraction"] == 1.0


def test_summarizer(tmp_path):
    result = {
        "dataset": "tiny",
        "instrument": "NIRISS/SOSS",
        "order": 1,
        "status": "completed",
        "total_wall_seconds": 10.0,
        "fit": {"wall_seconds": 7.0},
        "white_light_geometry": {"max_abs_shift_saved_sigma": 0.2},
        "candidate_rows": [],
    }
    path = tmp_path / "result.json"
    import json

    path.write_text(json.dumps(result))
    summary = summarize([path])
    # Only one of the three required pilot proxy classes is present, so the
    # stratified 70-run projection is intentionally incomplete.
    assert summary["projected_70_serial_seconds"] is None
