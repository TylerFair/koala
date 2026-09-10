import pickle

import numpy as np
import pytest
from astropy.io import fits

import createdatacube
from koala import readers


def _exotedrf_fits(path, n_int=6, n_wave=30, lower=2.5, upper=3.9):
    wave = np.linspace(lower, upper, n_wave)
    flux = np.full((n_int, n_wave), 100.0)
    hdus = [fits.PrimaryHDU()] + [fits.ImageHDU(a) for a in (wave, np.full(n_wave, .01), flux, flux * .01)]
    hdus.append(fits.ImageHDU(np.arange(n_int, dtype=float) + 60000.))
    fits.HDUList(hdus).writeto(path)
    return path


def _sparta_pickle(path, n_int=8, n_wave=12):
    wave = np.linspace(2.45, 3.99, n_wave)[::-1]  # descending, like a raw column solution
    data = {
        'wavelengths': wave,
        'times': np.arange(n_int, dtype=float) / 1000. + 59883.6,
        'data': np.tile(np.arange(n_wave, dtype=float)[::-1] + 1000., (n_int, 1)),
        'errors': np.full((n_int, n_wave), 2.0),
        'uncut_wavelengths': wave,
        'uncut_times': np.arange(n_int + 2, dtype=float) / 1000. + 59883.6,
        'uncut_data': np.ones((n_int + 2, n_wave)),
        'uncut_errors': np.ones((n_int + 2, n_wave)),
    }
    data['errors'][3, 4] = 2.0e10  # SPARTA HIGH_ERROR sentinel
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return path


def _eureka_s3(path, n_int=5, n_x=10):
    xr = pytest.importorskip('xarray')
    pytest.importorskip('h5netcdf')
    time = np.arange(n_int, dtype=float) / 100. + 59883.6
    wave = np.concatenate([[np.nan], np.linspace(3.9, 4.9, n_x - 2), [np.nan]])
    spec = np.tile(np.arange(n_x, dtype=float) + 50., (n_int, 1))
    mask = np.zeros((n_int, n_x), dtype=bool)
    mask[2, 4] = True
    spec[2, 4] = np.nan
    ds = xr.Dataset(
        {
            'optspec': (('time', 'x'), spec, {'flux_units': 'ELECTRONS'}),
            'opterr': (('time', 'x'), np.full((n_int, n_x), 0.5)),
            'optmask': (('time', 'x'), mask),
            'wave_1d': (('x',), wave, {'wave_units': 'microns'}),
        },
        coords={'time': ('time', time, {'time_units': 'BMJD_TDB'}), 'x': np.arange(n_x)},
    )
    ds.to_netcdf(path, engine='h5netcdf', invalid_netcdf=True)
    return path


def _eureka_s4(path, n_int=5, n_chan=4):
    xr = pytest.importorskip('xarray')
    pytest.importorskip('h5netcdf')
    time = np.arange(n_int, dtype=float) / 100. + 59883.6
    edges = np.linspace(3.9, 4.9, n_chan + 1)
    mid, err = 0.5 * (edges[1:] + edges[:-1]), 0.5 * np.diff(edges)
    ds = xr.Dataset(
        {
            'data': (('wavelength', 'time'), np.full((n_chan, n_int), 7.0)),
            'err': (('wavelength', 'time'), np.full((n_chan, n_int), 0.1)),
            'mask': (('wavelength', 'time'), np.zeros((n_chan, n_int), dtype=bool)),
            'wave_low': (('wavelength',), edges[:-1]),
            'wave_hi': (('wavelength',), edges[1:]),
            'wave_mid': (('wavelength',), mid, {'wave_units': 'microns'}),
            'wave_err': (('wavelength',), err),
        },
        coords={'time': ('time', time, {'time_units': 'BMJD_TDB'}), 'wavelength': mid},
    )
    ds.to_netcdf(path, engine='h5netcdf', invalid_netcdf=True)
    return path


def test_detect_input_format(tmp_path):
    assert readers.detect_input_format(_exotedrf_fits(tmp_path / 'a.fits')) == 'exotedrf'
    assert readers.detect_input_format(_sparta_pickle(tmp_path / 'weird_name.dat')) == 'sparta'
    assert readers.detect_input_format(_eureka_s3(tmp_path / 'S3_x_SpecData.h5')) == 'eureka'
    assert readers.detect_input_format(_eureka_s4(tmp_path / 'S4_x_LCData.h5')) == 'eureka'
    with open(tmp_path / 'junk.pkl', 'wb') as f:
        pickle.dump({'foo': 1}, f)
    with pytest.raises(ValueError):
        readers.detect_input_format(tmp_path / 'junk.pkl')


def test_read_sparta_sorts_and_derives_half_widths(tmp_path):
    wave, wave_err, time, flux, flux_err = readers.read_sparta(_sparta_pickle(tmp_path / 'd.pkl'))
    assert np.all(np.diff(wave) > 0)
    assert flux.shape == (8, 12) and time.shape == (8,)
    np.testing.assert_allclose(flux[0], np.arange(12, dtype=float) + 1000.)
    np.testing.assert_allclose(wave_err, np.abs(np.gradient(wave)) / 2)
    assert flux_err.max() == 2.0e10, 'sentinel errors are preserved'
    wave_u, _, time_u, flux_u, _ = readers.read_sparta(tmp_path / 'd.pkl', use_uncut=True)
    assert flux_u.shape == (10, 12) and time_u.shape == (10,)


def test_read_eureka_stage3_drops_nan_columns_and_neutralizes_mask(tmp_path):
    wave, wave_err, time, flux, flux_err = readers.read_eureka(_eureka_s3(tmp_path / 'S3.h5'))
    assert wave.shape == (8,) and flux.shape == (5, 8)
    assert np.isfinite(flux).all() and np.isfinite(flux_err).all()
    assert flux[2, 3] == pytest.approx(54.0), 'masked pixel takes the channel median'
    assert flux_err[2, 3] == pytest.approx(0.5 * readers.BAD_PIXEL_ERROR_FACTOR)
    assert flux_err[0, 3] == 0.5
    np.testing.assert_allclose(time, np.arange(5) / 100. + 59883.6)


def test_read_eureka_stage4_uses_stored_bin_edges(tmp_path):
    wave, wave_err, time, flux, flux_err = readers.read_eureka(_eureka_s4(tmp_path / 'S4.h5'))
    assert flux.shape == (5, 4)
    np.testing.assert_allclose(wave_err, 0.125)
    np.testing.assert_allclose(flux, 7.0)


def test_full_jd_times_are_converted_to_bmjd(tmp_path):
    path = _sparta_pickle(tmp_path / 'd.pkl')
    with open(path, 'rb') as f:
        d = pickle.load(f)
    d['times'] = d['times'] + readers.MJD_OFFSET
    with open(path, 'wb') as f:
        pickle.dump(d, f)
    _, _, time, _, _ = readers.read_sparta(path)
    assert time[0] == pytest.approx(59883.6)


def test_load_spectra_applies_instrument_window_to_every_format(tmp_path):
    for path in (_exotedrf_fits(tmp_path / 'a.fits', lower=2.0, upper=4.5), _sparta_pickle(tmp_path / 'b.pkl')):
        wave, _, time, flux, _ = createdatacube.load_spectra(path, 'NIRCAM/F322W2', 1, 1, input_format='auto')
        assert wave.min() >= 2.4 and wave.max() <= 4.1
        assert np.all(np.diff(wave) > 0)
        assert flux.shape == (time.size, wave.size)
    _, _, time, _, _ = createdatacube.load_spectra(tmp_path / 'b.pkl', 'NIRCAM/F322W2', 1, 1)
    assert time.shape == (6,)


def test_process_spectroscopy_data_honours_input_format(tmp_path, monkeypatch):
    seen = {}

    def fake_load(infile, instrument, trim_start, trim_end, order=None, input_format='auto', **kwargs):
        seen['format'] = input_format
        raise RuntimeError('stop here')

    monkeypatch.setattr(createdatacube, 'load_spectra', fake_load)
    cfg = {'instrument': 'NIRCAM/F444W', 'input_format': 'sparta', 'planet': {'period': {'value': 1.0, 'prior': 'fixed'}, 't0': {'value': 0.0, 'prior': 'fixed'}, 'duration': {'value': 0.1, 'prior': 'fixed'}}}
    with pytest.raises(RuntimeError):
        createdatacube.process_spectroscopy_data('NIRCAM/F444W', '', '', 'x', cfg, 'unused.pkl')
    assert seen['format'] == 'sparta'
