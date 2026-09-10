"""Readers for extracted spectroscopic light-curve products.

Every reader returns the same tuple::

    (wavelength, wavelength_err, time, flux, flux_err)

* ``wavelength``     float64 (n_wave,)  bin or pixel centres in microns
* ``wavelength_err`` float64 (n_wave,)  half-widths in microns
* ``time``           float64 (n_int,)   BMJD_TDB (BJD_TDB - 2400000.5), Koala's
                                        time convention throughout
* ``flux``           float64 (n_int, n_wave)
* ``flux_err``       float64 (n_int, n_wave)

Supported products:

* ``exotedrf`` - exoTEDRF ``*_box_spectra_fullres.fits`` (wavelength, wavelength
  error, flux, flux error, time as FITS extensions 1-5; NIRISS/SOSS files repeat
  the four spectral extensions per order with the time array in extension 9).
* ``sparta`` - the ``data.pkl`` dictionary written by SPARTA's
  ``gather_and_filter.py`` (``wavelengths``, ``times``, ``data``, ``errors`` keys,
  plus the ``uncut_*`` copies made before integration and column rejection).
* ``eureka`` - Eureka! Stage 3 ``*_SpecData.h5`` (native per-column ``optspec``)
  or Stage 4 ``*_LCData.h5`` (binned ``data``) xarray datasets.

SPARTA and Eureka! copy the JWST ``INT_TIMES['int_mid_BJD_TDB']`` column, which
is already in the modified form, so times pass through unchanged. A full Julian
date (above 1e6) is converted.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
from astropy.io import fits

INPUT_FORMATS = ('exotedrf', 'sparta', 'eureka')

MJD_OFFSET = 2400000.5
_MJD_JD_SPLIT = 1.0e6


def _to_bmjd(t, label):
    t = np.asarray(t, dtype=np.float64)
    if t.size and np.nanmedian(t) > _MJD_JD_SPLIT:
        print(f"[{label}] time array is a full JD; subtracting {MJD_OFFSET} to obtain BMJD_TDB")
        return t - MJD_OFFSET
    return t


def _half_widths_from_centres(wave):
    """Half-widths for a centre-only grid: half the local spacing between centres."""
    wave = np.asarray(wave, dtype=np.float64)
    if wave.size < 2:
        return np.zeros_like(wave)
    return np.abs(np.gradient(wave)) / 2.0


BAD_PIXEL_ERROR_FACTOR = 1.0e6


def neutralize_bad_pixels(flux, flux_err, bad):
    """Give flagged pixels the channel median flux and a huge error.

    Koala drops any integration containing a non-finite value, so flagged
    pixels cannot be left as NaN. Instead each one takes the median of its
    wavelength channel over time and an error ``BAD_PIXEL_ERROR_FACTOR`` times
    the median finite error, which removes its weight from the likelihood
    (this mirrors SPARTA's own interpolate-and-inflate treatment).
    """
    flux = np.array(flux, dtype=np.float64, copy=True)
    flux_err = np.array(flux_err, dtype=np.float64, copy=True)
    bad = np.asarray(bad, dtype=bool) | ~np.isfinite(flux) | ~np.isfinite(flux_err) | (flux_err <= 0)
    if not bad.any():
        return flux, flux_err
    good_err = flux_err[~bad]
    fill_err = BAD_PIXEL_ERROR_FACTOR * (np.median(good_err) if good_err.size else 1.0)
    with np.errstate(all='ignore'):
        channel_median = np.nanmedian(np.where(bad, np.nan, flux), axis=0)
    channel_median = np.where(np.isfinite(channel_median), channel_median, 0.0)
    rows, cols = np.nonzero(bad)
    flux[rows, cols] = channel_median[cols]
    flux_err[rows, cols] = fill_err
    print(f"[readers] {bad.sum()} flagged pixels replaced by channel medians with error {fill_err:.3g}")
    return flux, flux_err


def _sort_by_wavelength(wave, wave_err, flux, flux_err):
    order = np.argsort(wave, kind='stable')
    return wave[order], wave_err[order], flux[:, order], flux_err[:, order]


def _as_float64(*arrays):
    # FITS stores big-endian arrays; JAX requires native-endian numeric input.
    return tuple(np.asarray(a, dtype=np.float64) for a in arrays)


# --------------------------------------------------------------------------- #
# exoTEDRF
# --------------------------------------------------------------------------- #
EXOTEDRF_EDGE_TRIM = 5


def read_exotedrf(path, order=None):
    """Read an exoTEDRF extracted-spectra FITS file.

    ``order`` selects the NIRISS/SOSS order (1 or 2) in files that carry both;
    leave it ``None`` for the NIRSpec, NIRCam, and MIRI layout. The five padding
    pixels exoTEDRF leaves at each end of the wavelength axis are removed.
    """
    if order is None:
        time_ext, base = 5, 0
    else:
        time_ext, base = 9, 4 * (int(order) - 1)
    bjd = fits.getdata(path, time_ext)
    wave = fits.getdata(path, 1 + base)
    wave_err = fits.getdata(path, 2 + base)
    flux = fits.getdata(path, 3 + base)
    flux_err = fits.getdata(path, 4 + base)
    k = EXOTEDRF_EDGE_TRIM
    wave, wave_err = wave[k:-k], wave_err[k:-k]
    flux, flux_err = flux[:, k:-k], flux_err[:, k:-k]
    return _as_float64(wave, wave_err, bjd, flux, flux_err)


# --------------------------------------------------------------------------- #
# SPARTA
# --------------------------------------------------------------------------- #
SPARTA_REQUIRED_KEYS = ('wavelengths', 'times', 'data', 'errors')


def read_sparta(path, use_uncut=False, verbose=True):
    """Read a SPARTA ``gather_and_filter.py`` pickle.

    The dictionary holds per-column wavelength centres in microns
    (``wavelengths``), ``times`` in BMJD_TDB, and ``data``/``errors`` cubes of
    shape (n_int, n_wave) in electrons per group. ``use_uncut`` returns the
    ``uncut_*`` arrays taken before SPARTA's integration and column rejection
    and pixel cleaning. SPARTA stores no bin edges, so half-widths are half the
    spacing between neighbouring columns. Bad pixels keep SPARTA's very large
    error sentinel and are therefore down-weighted rather than dropped.
    """
    with open(path, 'rb') as f:
        d = pickle.load(f)
    if not isinstance(d, dict):
        raise TypeError(f"{path}: expected a dict SPARTA pickle, got {type(d)}")
    prefix = 'uncut_' if use_uncut else ''
    missing = [k for k in SPARTA_REQUIRED_KEYS if prefix + k not in d]
    if missing:
        raise KeyError(f"{path}: SPARTA pickle missing keys {missing}")

    wave = np.asarray(d[prefix + 'wavelengths'], dtype=np.float64)
    time = _to_bmjd(d[prefix + 'times'], 'sparta')
    flux = np.asarray(d[prefix + 'data'], dtype=np.float64)
    flux_err = np.asarray(d[prefix + 'errors'], dtype=np.float64)
    if flux.shape != (time.size, wave.size):
        raise ValueError(f"{path}: flux shape {flux.shape} != (n_int={time.size}, n_wave={wave.size})")
    if flux_err.shape != flux.shape:
        raise ValueError(f"{path}: errors shape {flux_err.shape} != data shape {flux.shape}")
    if np.nanmax(wave) < 0.1:
        raise ValueError(f"{path}: wavelengths look like metres, expected microns")

    wave_err = _half_widths_from_centres(wave)
    wave, wave_err, flux, flux_err = _sort_by_wavelength(wave, wave_err, flux, flux_err)
    if verbose:
        print(f"[sparta] {os.path.basename(path)}: n_int={time.size} n_wave={wave.size} "
              f"wave {wave.min():.4f}-{wave.max():.4f} um, t {time.min():.5f}-{time.max():.5f} BMJD_TDB")
    return wave, wave_err, time, flux, flux_err


# --------------------------------------------------------------------------- #
# Eureka!
# --------------------------------------------------------------------------- #
def _open_eureka(path):
    import xarray as xr
    # Eureka! writes with engine='h5netcdf' and invalid_netcdf=True, which the
    # netCDF4 engine may refuse; h5netcdf reads the files back cleanly.
    try:
        return xr.open_dataset(path, engine='h5netcdf')
    except Exception:
        return xr.open_dataset(path)


def read_eureka(path, order=None, verbose=True):
    """Read an Eureka! Stage 3 ``*_SpecData.h5`` or Stage 4 ``*_LCData.h5`` file.

    Stage 3 provides ``optspec``/``opterr``/``optmask`` with dimensions
    (time, x) and per-column centres ``wave_1d``; the mask sets bad pixels to
    NaN and half-widths come from the column spacing. Stage 4 provides binned
    ``data``/``err``/``mask`` with dimensions (wavelength, time) and exact
    ``wave_mid``/``wave_err``. ``order`` selects the NIRISS/SOSS order when the
    Stage 3 file carries an ``order`` dimension. Times are ``BMJD_TDB``.
    """
    ds = _open_eureka(path)
    try:
        if 'optspec' in ds:
            spec, err, mask, wave_da = ds['optspec'], ds['opterr'], ds['optmask'], ds['wave_1d']
            if 'order' in spec.dims:
                if order is None:
                    raise ValueError(f"{path}: has an 'order' dimension; pass order=1 or 2")
                spec, err, wave_da = (a.sel(order=order) for a in (spec, err, wave_da))
                if 'order' in mask.dims:
                    mask = mask.sel(order=order)
            spec, err, mask = (a.transpose('time', 'x') for a in (spec, err, mask))
            wave = np.asarray(wave_da.values, dtype=np.float64)
            wave_err = None
            flux = np.asarray(spec.values, dtype=np.float64)
            flux_err = np.asarray(err.values, dtype=np.float64)
            bad = np.asarray(mask.values, dtype=bool)
            time = _to_bmjd(spec['time'].values, 'eureka-S3')
            units_da, stage = wave_da, 'S3'
        elif 'data' in ds and 'wave_low' in ds:
            spec = ds['data'].transpose('wavelength', 'time')
            err = ds['err'].transpose('wavelength', 'time')
            wave = np.asarray(ds['wave_mid'].values, dtype=np.float64)
            wave_err = np.asarray(ds['wave_err'].values, dtype=np.float64)
            flux = np.asarray(spec.values, dtype=np.float64).T
            flux_err = np.asarray(err.values, dtype=np.float64).T
            bad = (np.asarray(ds['mask'].transpose('wavelength', 'time').values, dtype=bool).T
                   if 'mask' in ds else np.zeros(flux.shape, dtype=bool))
            time = _to_bmjd(spec['time'].values, 'eureka-S4')
            units_da, stage = ds['wave_mid'], 'S4'
        else:
            raise ValueError(f"{path}: no Eureka! optspec (Stage 3) or data/wave_low (Stage 4) variables found")
        wave_units = str(units_da.attrs.get('wave_units', 'microns')).lower()
    finally:
        ds.close()

    if wave_units.startswith('m') and not wave_units.startswith('micro'):
        wave = wave * 1e6
        if wave_err is not None:
            wave_err = wave_err * 1e6
    keep = np.isfinite(wave)
    wave, flux, flux_err = wave[keep], flux[:, keep], flux_err[:, keep]
    if wave_err is None:
        wave_err = _half_widths_from_centres(wave)
    else:
        wave_err = wave_err[keep]
    bad = bad[:, keep]
    sort_idx = np.argsort(wave, kind='stable')
    wave, wave_err, flux, flux_err, bad = wave[sort_idx], wave_err[sort_idx], flux[:, sort_idx], flux_err[:, sort_idx], bad[:, sort_idx]
    if flux.shape != (time.size, wave.size):
        raise ValueError(f"{path}: flux {flux.shape} != (n_int={time.size}, n_wave={wave.size})")
    flux, flux_err = neutralize_bad_pixels(flux, flux_err, bad)
    if verbose:
        print(f"[eureka-{stage}] {os.path.basename(path)}: n_int={time.size} n_wave={wave.size} "
              f"wave {wave.min():.4f}-{wave.max():.4f} um, t {time.min():.5f}-{time.max():.5f} BMJD_TDB")
    return wave, wave_err, time, flux, flux_err


# --------------------------------------------------------------------------- #
# Format detection and dispatch
# --------------------------------------------------------------------------- #
_FITS_MAGIC = b'SIMPLE  ='
_HDF5_MAGIC = b'\x89HDF\r\n\x1a\n'


def detect_input_format(path):
    """Return ``'exotedrf'``, ``'sparta'``, or ``'eureka'`` for a spectra file.

    The file signature is checked first so a misnamed file is still recognised,
    then the expected layout is confirmed.
    """
    with open(path, 'rb') as f:
        head = f.read(16)
    ext = os.path.splitext(path)[1].lower()

    if head.startswith(_FITS_MAGIC) or ext in ('.fits', '.fit', '.fts'):
        with fits.open(path, memmap=True) as hdul:
            if len(hdul) >= 5:
                flux, wave = hdul[3].data, hdul[1].data
                if (flux is not None and wave is not None and flux.ndim == 2
                        and wave.ndim == 1 and flux.shape[1] == wave.shape[0]):
                    return 'exotedrf'
        raise ValueError(f"{path}: FITS file does not have the exoTEDRF extension layout")

    if head.startswith(_HDF5_MAGIC) or ext in ('.h5', '.hdf5', '.nc'):
        ds = _open_eureka(path)
        try:
            names = set(ds.variables)
        finally:
            ds.close()
        if {'optspec', 'wave_1d'} <= names or {'data', 'wave_low', 'wave_hi'} <= names:
            return 'eureka'
        raise ValueError(f"{path}: HDF5 file lacks Eureka! Stage 3 or Stage 4 variables")

    if head.startswith(b'\x80') or ext in ('.pkl', '.pickle', '.p'):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and all(k in obj for k in SPARTA_REQUIRED_KEYS):
            return 'sparta'
        raise ValueError(f"{path}: pickle is not a SPARTA gather_and_filter output")

    raise ValueError(f"{path}: unrecognised input format (extension {ext!r})")


def read_spectra(path, input_format='auto', order=None, **kwargs):
    """Read ``path`` with the reader for ``input_format`` (``'auto'`` detects it).

    Returns ``(input_format, (wavelength, wavelength_err, time, flux, flux_err))``.
    """
    fmt = str(input_format or 'auto').strip().lower()
    if fmt == 'auto':
        fmt = detect_input_format(path)
        print(f"Detected input format '{fmt}' for {os.path.basename(path)}")
    if fmt == 'exotedrf':
        return fmt, read_exotedrf(path, order=order)
    if fmt == 'sparta':
        return fmt, read_sparta(path, **kwargs)
    if fmt == 'eureka':
        return fmt, read_eureka(path, order=order, **kwargs)
    raise ValueError(f"Unknown input_format {input_format!r}; expected one of {INPUT_FORMATS} or 'auto'.")
