import json
import pickle

import numpy as np

from tools.reference_noise_floor import compute_noise_floor, main


def _sample_set(offset):
    draws = np.arange(12, dtype=np.float64).reshape(6, 2)
    return {
        "rors": (draws + offset)[..., None],
        "depths": ((draws + offset) / 10.0)[..., None],
        "c": draws + 2.0 * offset,
        "log_jitter": -draws + offset,
    }


def test_noise_floor_pairwise_rows_and_gates():
    result = compute_noise_floor(
        [_sample_set(0.0), _sample_set(0.1), _sample_set(0.2)],
        ["seed0", "seed1", "seed2"],
        start=4,
    )
    # Four scalar components x two channels x three seed pairs.
    assert len(result["pairwise_rows"]) == 24
    assert {row["channel"] for row in result["pairwise_rows"]} == {4, 5}
    assert {row["site_class"] for row in result["calibrated_gates"]} == {
        "depth/rors",
        "trend c,v",
        "noise log_jitter/total_error",
    }
    assert all(
        row["median_shift_limit_sigma"] >= 0.1
        for row in result["calibrated_gates"]
    )


def test_main_writes_standard_pooled_layout(tmp_path):
    sample_paths = []
    for seed, offset in enumerate((0.0, 0.1, 0.2)):
        path = tmp_path / f"stage_seed{seed}_samples.pkl"
        with path.open("wb") as stream:
            pickle.dump(_sample_set(offset), stream)
        sample_paths.append(path)
    dump = tmp_path / "tiny_inputs.pkl"
    dump.touch()
    output_dir = tmp_path / "references"

    assert main(
        [
            str(dump),
            *(str(path) for path in sample_paths),
            "--start",
            "4",
            "--end",
            "6",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0
    pooled_path = output_dir / (
        "tiny_inputs_ch4_6_pooled_joint_nuts.pkl"
    )
    noise_path = output_dir / (
        "tiny_inputs_ch4_6_pooled_joint_nuts_noise_floor.json"
    )
    with pooled_path.open("rb") as stream:
        pooled = pickle.load(stream)
    assert pooled["rors"].shape == (18, 2, 1)
    noise = json.loads(noise_path.read_text())
    assert noise["pooled_draws"] == 18
    assert noise["seed_labels"] == ["seed0", "seed1", "seed2"]
