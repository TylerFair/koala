"""Stage inputs must never silently default a missing name to None."""

from pathlib import Path


def test_pipeline_uses_explicit_stage_inputs():
    source = (Path(__file__).resolve().parents[1] / "koala" / "pipeline.py").read_text()
    assert "locals()" not in source
