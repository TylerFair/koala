import json
import os
from pathlib import Path
import subprocess
import sys


def test_benchmark_gp_smoke(tmp_path):
    output = tmp_path / "gp.json"
    environment = os.environ.copy()
    environment.update(
        {
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "OMP_NUM_THREADS": "8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    command = [
        sys.executable,
        "tools/benchmark_gp.py",
        "--sizes", "64",
        "--cadences", "33",
        "--repeats", "2",
        "--json", str(output),
    ]
    subprocess.run(command, cwd=Path(__file__).parents[1], env=environment, check=True)
    result = json.loads(output.read_text(encoding="utf-8"))

    assert result["schema_version"] == 1
    assert result["metadata"]["platform"] == "cpu"
    assert result["metadata"]["dtype"] == "float64"
    assert result["options"]["sizes"] == [64, 33]
    assert set(result["benchmarks"]) == {"64", "33"}
    assert result["solver_availability"]["serial"]["available"] is True
    for size in ("64", "33"):
        serial = result["benchmarks"][size]["serial"]
        assert set(serial) == {
            "log_likelihood", "likelihood_gradient", "training_prediction"
        }
        assert serial["log_likelihood"]["compile_seconds"] >= 0
        if result["solver_availability"]["parallel"]["available"]:
            assert "parallel" in result["benchmarks"][size]
            assert set(result["agreement"][size]) == {
                "log_likelihood", "likelihood_gradient", "training_prediction"
            }
            for difference in result["agreement"][size].values():
                assert difference["max_abs"] < 1e-8
                assert difference["max_rel"] < 1e-8
        else:
            assert "parallel" not in result["benchmarks"][size]

    assert result["dense_reference"]["size"] == 128
    assert "likelihood_gradient" in result["dense_reference"]["solvers"]["serial"]
    for difference in result["dense_reference"]["solvers"]["serial"].values():
        assert difference["max_abs"] < 1e-8
        assert difference["max_rel"] < 1e-8
