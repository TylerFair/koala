import numpy as np
import pytest

import createdatacube


def test_process_filters_invalid_integrations_globally_before_binning(monkeypatch):
    time = np.arange(8, dtype=float) + 100.0
    wavelengths = np.array([np.nan, 1.0, 2.0, 3.0])
    wavelength_err = np.array([np.nan, 0.01, 0.01, 0.01])
    flux = np.ones((8, 4), dtype=float)
    flux_err = np.full((8, 4), 0.1, dtype=float)

    # Structural padding must be excluded before validation.
    flux_err[:, 0] = 0.0
    # Each remaining bad row exercises a distinct invalid likelihood input.
    flux[1, 1] = np.nan
    flux[2, 2] = np.inf
    flux_err[3, 1] = 0.0
    flux_err[4, 2] = -0.1
    flux_err[5, 3] = np.nan
    flux_err[6, 1] = np.inf

    def fake_unpack(*args, **kwargs):
        return wavelengths, wavelength_err, time, flux, flux_err

    captured = {}

    def fake_bin(wave, wave_err, native_flux, native_flux_err, cfg, oot_mask):
        captured["time_count"] = native_flux.shape[0]
        captured["flux"] = native_flux.copy()
        captured["flux_err"] = native_flux_err.copy()
        captured["oot_mask"] = oot_mask.copy()
        transposed_flux = native_flux.T.copy()
        transposed_err = native_flux_err.T.copy()
        return {
            "wavelengths_lr": wave.copy(),
            "wavelengths_err_lr": wave_err.copy(),
            "flux_lr": transposed_flux.copy(),
            "flux_err_lr": transposed_err.copy(),
            "wavelengths_hr": wave.copy(),
            "wavelengths_err_hr": wave_err.copy(),
            "flux_hr": transposed_flux.copy(),
            "flux_err_hr": transposed_err.copy(),
        }

    monkeypatch.setattr(createdatacube, "unpack_nirspec_exotedrf", fake_unpack)
    monkeypatch.setattr(createdatacube, "bin_spectroscopy_data", fake_bin)

    cfg = {
        "instrument": "NIRSPEC/PRISM",
        "nrs": 1,
        "planet": {"t0": [0.0], "duration": [0.1]},
    }
    data = createdatacube.process_spectroscopy_data(
        "NIRSPEC/PRISM", "", "", "test", cfg, "unused.fits"
    )

    expected_time = time[[0, 7]]
    np.testing.assert_array_equal(np.asarray(data.time), expected_time)
    np.testing.assert_array_equal(np.asarray(data.wl_time), expected_time)
    assert np.asarray(data.flux_unbinned).shape == (2, 3)
    assert np.asarray(data.flux_lr).shape == (3, 2)
    assert np.asarray(data.flux_hr).shape == (3, 2)
    assert captured["time_count"] == 2
    assert captured["oot_mask"].shape == (2,)
    assert np.isfinite(captured["flux"]).all()
    assert np.isfinite(captured["flux_err"]).all()
    assert np.all(captured["flux_err"] > 0)


def test_filter_invalid_integrations_raises_if_none_remain():
    time = np.array([1.0, 2.0])
    flux = np.ones((2, 2))
    flux_err = np.array([[0.0, 0.1], [np.nan, 0.1]])

    with pytest.raises(ValueError, match="No valid integrations remain"):
        createdatacube._filter_invalid_spectroscopic_integrations(time, flux, flux_err)


def test_binned_validation_rejects_nonpositive_uncertainty():
    binned = {
        "flux_lr": np.ones((1, 2)),
        "flux_err_lr": np.array([[0.1, 0.0]]),
        "flux_hr": np.ones((1, 2)),
        "flux_err_hr": np.full((1, 2), 0.1),
    }

    with pytest.raises(ValueError, match="uncertainties must all be finite and > 0"):
        createdatacube._validate_binned_spectroscopy(binned, expected_n_time=2)
