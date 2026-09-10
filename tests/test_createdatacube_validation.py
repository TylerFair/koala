import numpy as np
import pytest

import createdatacube


@pytest.mark.parametrize('instrument', [
    'NIRISS/SOSS', 'NIRSPEC/G395H', 'NIRSPEC/G235M', 'NIRSPEC/G140M-F070',
    'NIRCAM/F322W2', 'NIRCAM/F444W', 'MIRI/LRS',
])
def test_fits_readers_return_native_endian_arrays_for_jax(tmp_path, instrument):
    from astropy.io import fits
    import jax.numpy as jnp
    from koala.instruments import data_wavelength_range

    lower, upper = (3., 4.) if instrument == 'NIRISS/SOSS' else data_wavelength_range(instrument)
    wave = np.linspace(lower, upper, 30)
    flux = np.full((4, 30), 100.)
    hdus = [fits.PrimaryHDU()] + [fits.ImageHDU(a) for a in (
        wave, np.full(30, .01), flux, flux * .01)]
    if instrument == 'NIRISS/SOSS':
        hdus += [fits.ImageHDU(np.zeros(1)) for _ in range(4)]
    hdus.append(fits.ImageHDU(np.arange(4.) + 60000.))
    path = tmp_path / 'spectrum.fits'
    fits.HDUList(hdus).writeto(path)
    assert not fits.getdata(path, 1).dtype.isnative
    if instrument == 'NIRISS/SOSS':
        result = createdatacube.unpack_niriss_exotedrf(path, 1, None, None)
    elif instrument == 'MIRI/LRS':
        result = createdatacube.unpack_miri_exotedrf(path, None, None)
    else:
        result = createdatacube.unpack_nirspec_exotedrf(path, instrument, None, None)
    for values in result:
        assert values.dtype.isnative
        np.testing.assert_array_equal(np.asarray(jnp.asarray(values)), values)
    assert len(result[0]) == 20, 'the 5-pixel edge trim and instrument window keep the interior'


def test_instrument_registry_matches_exotic_ld_modes():
    from koala.instruments import (
        SUPPORTED_INSTRUMENTS, ld_mode_and_bounds, normalize_instrument,
        resolve_detector, detector_label,
    )

    expected = {
        'JWST_NIRSpec_Prism', 'JWST_NIRSpec_G395H', 'JWST_NIRSpec_G395M',
        'JWST_NIRSpec_G235H', 'JWST_NIRSpec_G235M', 'JWST_NIRSpec_G140H-f100',
        'JWST_NIRSpec_G140M-f100', 'JWST_NIRSpec_G140H-f070',
        'JWST_NIRSpec_G140M-f070', 'JWST_NIRISS_SOSSo1', 'JWST_NIRISS_SOSSo2',
        'JWST_NIRCam_F322W2', 'JWST_NIRCam_F444', 'JWST_MIRI_LRS',
    }
    modes = set()
    for name in SUPPORTED_INSTRUMENTS:
        orders = (1, 2) if name == 'NIRISS/SOSS' else (None,)
        for order in orders:
            mode, lo, hi = ld_mode_and_bounds(name, order)
            assert lo < hi
            modes.add(mode)
    assert modes == expected

    assert normalize_instrument('nirspec/g140h-f100') == 'NIRSPEC/G140H'
    assert normalize_instrument('NIRCAM/F444') == 'NIRCAM/F444W'
    with pytest.raises(ValueError):
        normalize_instrument('MIRI/MRS')
    assert resolve_detector('NIRSPEC/G235M', {'nrs': 2}) == (2, None)
    assert resolve_detector('NIRISS/SOSS', {'order': 2}) == (None, 2)
    assert resolve_detector('NIRCAM/F322W2', {}) == (None, None)
    with pytest.raises(KeyError):
        resolve_detector('NIRSPEC/G395H', {})
    assert detector_label('NIRCAM/F444W') == ''
    assert detector_label('NIRSPEC/PRISM', nrs=1) == 'nrs1'


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

    monkeypatch.setattr(createdatacube, "unpack_exotedrf_spectra", fake_unpack)
    monkeypatch.setattr(createdatacube, "bin_spectroscopy_data", fake_bin)

    cfg = {
        "instrument": "NIRSPEC/PRISM",
        "nrs": 1,
        "planet": {
            "period": {"value": 1000.0, "prior": "fixed"},
            "t0": {"value": 0.0, "prior": "fixed"},
            "duration": {"value": 0.1, "prior": "fixed"},
        },
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
