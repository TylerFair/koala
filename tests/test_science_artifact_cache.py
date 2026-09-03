import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

import fit_jwst
from fit_jwst import (
    SCIENCE_ARTIFACT_SCHEMA_VERSION,
    _build_axis,
    _directory_metadata_identity,
    _file_content_identity,
    _science_artifact_fingerprint,
    _science_artifact_manifest_matches,
    _write_science_artifact_manifest,
)


def test_science_artifact_fingerprint_tracks_data_and_config(tmp_path):
    base = {
        "config": {"flags": {"ld_profile": "quadratic"}},
        "time": np.array([1.0, 2.0]),
        "flux": np.array([0.99, 1.0]),
    }
    reference = _science_artifact_fingerprint("whitelight", base)
    changed_data = _science_artifact_fingerprint(
        "whitelight", {**base, "flux": np.array([0.98, 1.0])}
    )
    changed_config = _science_artifact_fingerprint(
        "whitelight",
        {**base, "config": {"flags": {"ld_profile": "power2"}}},
    )
    assert reference != changed_data
    assert reference != changed_config

    manifest = tmp_path / "artifacts.manifest.json"
    assert not _science_artifact_manifest_matches(manifest, reference)
    _write_science_artifact_manifest(manifest, "whitelight", reference)
    assert _science_artifact_manifest_matches(manifest, reference)
    assert not _science_artifact_manifest_matches(manifest, changed_data)
    payload = json.loads(manifest.read_text())
    assert payload["schema_version"] == SCIENCE_ARTIFACT_SCHEMA_VERSION
    assert not list(tmp_path.glob("*.tmp.*"))


def test_whitelight_fingerprint_tracks_trend_coordinates_and_ordering():
    payload = {
        "config": {
            "flags": {
                "detrending_type": "2spot",
                "whitelight_trend_parameterization": "physical",
                "whitelight_2spot_ordering": "legacy",
            }
        },
        "time": np.array([1.0, 2.0]),
        "flux": np.array([0.99, 1.0]),
    }
    reference = _science_artifact_fingerprint("whitelight", payload)
    cadence = {
        **payload,
        "config": {
            "flags": {
                **payload["config"]["flags"],
                "whitelight_trend_parameterization": "cadence",
            }
        },
    }
    ordered = {
        **payload,
        "config": {
            "flags": {
                **payload["config"]["flags"],
                "whitelight_2spot_ordering": "ordered",
            }
        },
    }
    assert reference != _science_artifact_fingerprint("whitelight", cadence)
    assert reference != _science_artifact_fingerprint("whitelight", ordered)


def test_file_content_identity_detects_same_size_edit(tmp_path):
    source = tmp_path / "source.fits"
    source.write_bytes(b"abcdefgh")
    fixed_mtime_ns = (source.stat().st_mtime_ns // 1_000_000_000) * 1_000_000_000
    os.utime(source, ns=(fixed_mtime_ns, fixed_mtime_ns))
    original = _file_content_identity(source)

    source.write_bytes(b"abcdWXYZ")
    os.utime(source, ns=(fixed_mtime_ns, fixed_mtime_ns))
    changed = _file_content_identity(source)

    assert original["size"] == changed["size"]
    assert original["sha256"] != changed["sha256"]


def test_directory_metadata_identity_tracks_grid_revision(tmp_path):
    grid = tmp_path / "grid"
    grid.mkdir()
    model = grid / "model.dat"
    model.write_bytes(b"old")
    original = _directory_metadata_identity(grid)
    model.write_bytes(b"new-grid")
    changed = _directory_metadata_identity(grid)
    assert original["metadata_sha256"] != changed["metadata_sha256"]


def test_power2_ld_prior_fresh_and_cached_sigma_floor_match(monkeypatch, tmp_path):
    class FakeStellarLimbDarkening:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        fit_jwst, "StellarLimbDarkening", FakeStellarLimbDarkening
    )
    monkeypatch.setattr(
        fit_jwst,
        "get_limb_darkening",
        lambda *args, **kwargs: np.array([0.3, 0.7]),
    )
    grid = tmp_path / "grid"
    grid.mkdir()
    stellar = {
        "teff": 5000.0,
        "logg": 4.5,
        "feh": 0.0,
        "teff_sigma": 0.0,
        "logg_sigma": 0.0,
        "feh_sigma": 0.0,
        "ld_prior_n_grid": 1,
        "ld_prior_min_sigma": 1e-4,
        "ld_data_path": str(grid),
        "ld_prior_cache_dir": str(tmp_path / "ld_prior_cache"),
    }
    kwargs = dict(
        stellar_cfg=stellar,
        wavelengths=np.array([1.1, 1.2]),
        wavelength_err=0.05,
        instrument="NIRISS/SOSS",
        order=1,
        output_dir=str(tmp_path),
        cache_label="parity",
    )

    fresh_mu, fresh_sigma = fit_jwst.get_or_build_power2_ld_prior(**kwargs)
    cached_mu, cached_sigma = fit_jwst.get_or_build_power2_ld_prior(**kwargs)

    np.testing.assert_array_equal(np.asarray(fresh_mu), np.asarray(cached_mu))
    np.testing.assert_array_equal(
        np.asarray(fresh_sigma), np.asarray(cached_sigma)
    )
    np.testing.assert_array_equal(np.asarray(fresh_sigma), np.full(2, 1e-4))


def test_single_point_stellar_axis_is_centered_and_invalid_grid_rejected(
    monkeypatch, tmp_path
):
    np.testing.assert_array_equal(_build_axis(5000.0, 100.0, 1, 3.0), [5000.0])

    stellar = {
        "teff": 5000.0,
        "logg": 4.5,
        "feh": 0.0,
        "teff_sigma": 100.0,
        "ld_prior_n_grid": 0,
        "ld_data_path": str(tmp_path),
    }
    with __import__("pytest").raises(ValueError, match="ld_prior_n_grid"):
        fit_jwst.get_or_build_power2_ld_prior(
            stellar,
            np.array([1.1, 1.2]),
            0.05,
            "NIRISS/SOSS",
            order=1,
            output_dir=str(tmp_path),
        )
