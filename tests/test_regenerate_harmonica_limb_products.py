import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import pickle
from pathlib import Path

import numpy as np
import pytest

from tools.regenerate_harmonica_limb_products import (
    discover_chunk_specs,
    load_limb_samples,
    regenerate_limb_products,
    resolve_wavelength_axis,
    schema_v2_output_paths,
)


def _write_pickle(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def _chunk_payload(start, end, draws=3, include_a0=False):
    width = end - start
    channel_values = np.arange(start, end, dtype=float)[None, :, None]
    payload = {
        "rors": np.broadcast_to(0.1 + channel_values * 1e-3, (draws, width, 1)).copy(),
        "a1": np.broadcast_to(channel_values * 1e-4, (draws, width, 1)).copy(),
        "unused": np.zeros((draws, width)),
    }
    if include_a0:
        payload["a0"] = np.broadcast_to(
            0.09 + channel_values * 1e-3, (draws, width, 1)
        ).copy()
    return payload


def test_chunk_discovery_is_numeric_and_limb_fields_concatenate(tmp_path):
    chunks = tmp_path / "chunks"
    prefix = "planet_instrument_R50"
    _write_pickle(chunks / f"{prefix}_chunk_2_5.pkl", _chunk_payload(2, 5))
    _write_pickle(chunks / f"{prefix}_chunk_0_2.pkl", _chunk_payload(0, 2))

    specs = discover_chunk_specs(tmp_path)
    assert [(spec.start, spec.end) for spec in specs] == [(0, 2), (2, 5)]
    samples = load_limb_samples(specs)

    assert tuple(samples) == ("rors", "a1")
    assert samples["rors"].shape == (3, 5, 1)
    np.testing.assert_allclose(samples["rors"][0, :, 0], 0.1 + np.arange(5) * 1e-3)
    np.testing.assert_allclose(samples["a1"][0, :, 0], np.arange(5) * 1e-4)


def test_chunk_discovery_rejects_incomplete_coverage(tmp_path):
    chunks = tmp_path / "chunks"
    prefix = "planet_R50"
    _write_pickle(chunks / f"{prefix}_chunk_0_2.pkl", {})
    _write_pickle(chunks / f"{prefix}_chunk_3_4.pkl", {})

    with pytest.raises(ValueError, match="gap"):
        discover_chunk_specs(tmp_path)


def test_half_area_chunks_preserve_deterministic_a0_for_limb_geometry(tmp_path):
    chunks = tmp_path / "chunks"
    prefix = "planet_half_area_R50"
    _write_pickle(
        chunks / f"{prefix}_chunk_0_2.pkl",
        _chunk_payload(0, 2, include_a0=True),
    )

    samples = load_limb_samples(discover_chunk_specs(tmp_path))
    assert tuple(samples) == ("rors", "a0", "a1")
    np.testing.assert_allclose(samples["a0"][0, :, 0], [0.09, 0.091])


def test_wavelength_axis_auto_requires_unique_channel_match():
    data = {
        "wavelengths_hr": np.linspace(1.0, 2.0, 5),
        "wavelengths_err_hr": np.full(5, 0.01),
        "wavelengths_lr": np.linspace(1.0, 2.0, 2),
        "wavelengths_err_lr": np.full(2, 0.03),
    }
    axis, wavelength, wavelength_err = resolve_wavelength_axis(data, 5)
    assert axis == "hr"
    assert wavelength.shape == wavelength_err.shape == (5,)

    data["wavelengths_lr"] = data["wavelengths_hr"]
    data["wavelengths_err_lr"] = data["wavelengths_err_hr"]
    with pytest.raises(ValueError, match="uniquely infer"):
        resolve_wavelength_axis(data, 5)


def test_dry_run_uses_new_names_and_refuses_existing_schema_product(tmp_path):
    prefix = "planet_instrument_R50"
    _write_pickle(
        tmp_path / "chunks" / f"{prefix}_chunk_0_2.pkl",
        _chunk_payload(0, 2),
    )
    _write_pickle(
        tmp_path / "planet_spectroscopy_data.pkl",
        {
            "wavelengths_hr": np.array([1.1, 1.2]),
            "wavelengths_err_hr": np.array([0.01, 0.01]),
        },
    )
    # A legacy product is intentionally ignored and remains protected by the
    # distinct schema-v2 output suffix.
    (tmp_path / f"{prefix}_limb_spectra.csv").write_text("legacy\n")

    summary = regenerate_limb_products(tmp_path, dry_run=True)
    expected = schema_v2_output_paths(tmp_path, prefix)
    assert summary["output_paths"] == expected
    assert all("schema_v2" in path.name for path in expected.values())

    expected["csv"].write_text("already generated\n")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        regenerate_limb_products(tmp_path, dry_run=True)


def test_full_regeneration_writes_schema_v2_once(tmp_path):
    prefix = "planet_R50"
    _write_pickle(
        tmp_path / "chunks" / f"{prefix}_chunk_0_2.pkl",
        _chunk_payload(0, 2, draws=5),
    )
    _write_pickle(
        tmp_path / "planet_spectroscopy_data.pkl",
        {
            "wavelengths_hr": np.array([1.1, 1.2]),
            "wavelengths_err_hr": np.array([0.01, 0.01]),
        },
    )

    summary = regenerate_limb_products(tmp_path, random_seed=42)
    for path in summary["output_paths"].values():
        assert path.is_file()
        assert path.stat().st_size > 0
    csv_text = summary["output_paths"]["csv"].read_text()
    assert "limb_product_schema_version" in csv_text.splitlines()[0]
    assert ",2," in csv_text.splitlines()[1]
    with np.load(summary["output_paths"]["posterior_samples"], allow_pickle=False) as saved:
        assert saved["sample_axes"].item() == "draw,wavelength"
        assert saved["depth_evening"].shape == (5, 2)
        assert saved["depth_morning"].shape == (5, 2)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        regenerate_limb_products(tmp_path)
