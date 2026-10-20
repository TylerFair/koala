"""Read, normalize, and bin spectroscopy for white-light and channel fits."""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import pickle
from .binning import bin_at_resolution, bin_at_pixel
from .readers import read_spectra
from .exclusions import (
    resolve_exclusions,
    integration_keep_mask,
    time_keep_mask,
    has_cut_phase_directive,
    _pairs_from_legacy_trim,
)
from .instruments import (
    normalize_instrument, resolve_detector, detector_key, data_wavelength_range,
)
jax.config.update('jax_enable_x64', True)

def apply_wavelength_masks(wave, wave_err, fluxcube, fluxcube_err, mask_ranges):
    """Exclude inclusive wavelength intervals supplied as flat min/max pairs."""
    if mask_ranges is None or len(mask_ranges) == 0:
        return wave, wave_err, fluxcube, fluxcube_err

    if len(mask_ranges) % 2 != 0:
        raise ValueError(f"wavelength_masks must have even length (pairs of min/max), got length {len(mask_ranges)}")

    keep_mask = np.ones(len(wave), dtype=bool)

    for i in range(0, len(mask_ranges), 2):
        wl_min = mask_ranges[i]
        wl_max = mask_ranges[i + 1]

        if wl_min >= wl_max:
            raise ValueError(f"Invalid wavelength mask range: min={wl_min} >= max={wl_max}")

        in_mask_range = (wave >= wl_min) & (wave <= wl_max)
        keep_mask &= ~in_mask_range

        print(f"  Masking wavelength range: {wl_min:.3f} - {wl_max:.3f} µm ({np.sum(in_mask_range)} pixels)")

    n_masked = np.sum(~keep_mask)
    if n_masked > 0:
        print(f"  Total masked: {n_masked}/{len(wave)} wavelength pixels")
        wave = wave[keep_mask]
        wave_err = wave_err[keep_mask]
        fluxcube = fluxcube[:, keep_mask]
        fluxcube_err = fluxcube_err[:, keep_mask]

    return wave, wave_err, fluxcube, fluxcube_err

SOSS_ORDER2_RANGE = (0.6, 0.85)


def _finalize_spectra(wave, wave_err, bjd, fluxcube, fluxcube_err, instrument, order,
                      trim_start, trim_end, wl_min=None, wl_max=None, wavelength_masks=None,
                      exclude_integrations=None):
    """Apply Koala's common post-read steps to a spectra cube.

    Integrations are dropped with ``exclude_integrations`` (a list of
    inclusive ``(first, last)`` index pairs; see ``koala.exclusions``) and the
    legacy head/tail counts ``trim_start``/``trim_end``. Wavelengths are
    restricted to finite values inside the instrument window
    (``SOSS_ORDER2_RANGE`` for SOSS order 2) and to the optional
    ``wl_min``/``wl_max`` filter, ``wavelength_masks`` are
    removed, and everything is returned as ascending, native-endian float64.
    """
    ranges = list(exclude_integrations or []) + _pairs_from_legacy_trim(trim_start, trim_end)
    if ranges:
        keep = integration_keep_mask(len(bjd), ranges)
        dropped = int((~keep).sum())
        print(f"Excluding {dropped} of {len(bjd)} integrations by index.")
        fluxcube, fluxcube_err = fluxcube[keep, :], fluxcube_err[keep, :]
        bjd = bjd[keep]

    if order == 2:
        data_range = SOSS_ORDER2_RANGE
    elif order is not None:
        data_range = None
    else:
        data_range = data_wavelength_range(instrument)
    keep = np.isfinite(wave)
    if data_range is not None:
        keep &= (wave >= data_range[0]) & (wave <= data_range[1])
    ii = np.flatnonzero(keep)
    fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:, ii]
    wave, wave_err = wave[ii], wave_err[ii]
    ii = np.argsort(wave, kind='stable')
    wave, wave_err = wave[ii], wave_err[ii]
    fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:, ii]

    if wl_min is not None and wl_max is not None:
        ii = np.where((wave >= wl_min) & (wave <= wl_max))[0]
        fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:, ii]
        wave, wave_err = wave[ii], wave_err[ii]

    if wavelength_masks is not None and len(wavelength_masks) > 0:
        label = f"NIRISS order {order}" if order is not None else instrument
        print(f"Applying wavelength masks for {label}:")
        wave, wave_err, fluxcube, fluxcube_err = apply_wavelength_masks(
            wave, wave_err, fluxcube, fluxcube_err, wavelength_masks
        )

    return tuple(np.asarray(value, dtype=np.float64)
                 for value in (wave, wave_err, bjd, fluxcube, fluxcube_err))


def load_spectra(infile, instrument, trim_start=None, trim_end=None, order=None, input_format='auto',
                 wl_min=None, wl_max=None, wavelength_masks=None, exclude_integrations=None):
    """Read ``infile`` in any supported product format and post-process it.

    ``input_format`` is ``'auto'`` (sniff the file), ``'exotedrf'``, ``'sparta'``,
    or ``'eureka'``; see ``koala.readers``. Returns the same five arrays as
    ``unpack_exotedrf_spectra``.
    """
    _, raw = read_spectra(infile, input_format=input_format, order=order)
    return _finalize_spectra(*raw, instrument, order, trim_start, trim_end,
                             wl_min=wl_min, wl_max=wl_max, wavelength_masks=wavelength_masks,
                             exclude_integrations=exclude_integrations)


def unpack_niriss_exotedrf(
    infile, order, trim_start=None, trim_end=None, wl_min_o1=None, wl_max_o1=None,
    wl_min_o2=None, wl_max_o2=None, wavelength_masks=None, input_format='exotedrf',
    exclude_integrations=None,
):
    """Read one NIRISS/SOSS order from an extracted-spectra product."""
    wl_min, wl_max = (wl_min_o1, wl_max_o1) if order == 1 else (wl_min_o2, wl_max_o2)
    return load_spectra(infile, 'NIRISS/SOSS', trim_start, trim_end, order=order,
                        input_format=input_format, wl_min=wl_min, wl_max=wl_max,
                        wavelength_masks=wavelength_masks,
                        exclude_integrations=exclude_integrations)


def unpack_exotedrf_spectra(
    infile, instrument, trim_start=None, trim_end=None, wl_min=None, wl_max=None,
    wavelength_masks=None, input_format='exotedrf', exclude_integrations=None,
):
    """Read a NIRSpec, NIRCam, or MIRI extracted-spectra product.

    The wavelength window kept for ``instrument`` comes from ``koala.instruments``.
    """
    return load_spectra(infile, instrument, trim_start, trim_end, order=None,
                        input_format=input_format, wl_min=wl_min, wl_max=wl_max,
                        wavelength_masks=wavelength_masks,
                        exclude_integrations=exclude_integrations)


class SpectroData:
    """Simple container for dot notation access."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return _SpectroDataUnpickler(f).load()

    def save_whitelight_csv(self, output_path):
        df = pd.DataFrame({'time': self.wl_time, 'flux': self.wl_flux, 'flux_err': self.wl_flux_err})
        df.to_csv(output_path, index=False)


class _SpectroDataUnpickler(pickle.Unpickler):
    """Read the historical container name without installing an alias module."""

    def find_class(self, module, name):
        if (module, name) == ('createdatacube', 'SpectroData'):
            return SpectroData
        return super().find_class(module, name)


def _filter_invalid_spectroscopic_integrations(time, flux, flux_err):
    """Remove integrations that cannot define a valid Gaussian likelihood.

    The same row mask is applied to every wavelength so the native, white-light,
    low-resolution, and high-resolution time axes remain identical.  Wavelength
    padding must be removed before calling this function: a non-positive
    uncertainty at a retained science pixel is treated as invalid data, not as
    padding.

    Returns
    -------
    time, flux, flux_err, keep
        Filtered arrays and the Boolean integration mask applied to them.
    """
    time = np.asarray(time)
    flux = np.asarray(flux)
    flux_err = np.asarray(flux_err)

    if time.ndim != 1:
        raise ValueError(f"Spectroscopic time must be one-dimensional; got shape {time.shape}.")
    if flux.ndim != 2 or flux_err.ndim != 2:
        raise ValueError(
            "Spectroscopic flux and uncertainty arrays must be two-dimensional; "
            f"got {flux.shape} and {flux_err.shape}."
        )
    if flux.shape != flux_err.shape or flux.shape[0] != time.size:
        raise ValueError(
            "Spectroscopic time/flux/uncertainty arrays are misaligned: "
            f"time={time.shape}, flux={flux.shape}, flux_err={flux_err.shape}."
        )
    if flux.shape[1] == 0:
        raise ValueError("No science wavelength columns remain for spectroscopy.")

    bad_time = ~np.isfinite(time)
    bad_flux = ~np.isfinite(flux).all(axis=1)
    bad_err_nonfinite = ~np.isfinite(flux_err).all(axis=1)
    bad_err_nonpositive = np.any(np.isfinite(flux_err) & (flux_err <= 0), axis=1)
    keep = ~(bad_time | bad_flux | bad_err_nonfinite | bad_err_nonpositive)

    n_bad = int(np.count_nonzero(~keep))
    if n_bad:
        bad_indices = np.flatnonzero(~keep)
        preview = ", ".join(map(str, bad_indices[:10]))
        if bad_indices.size > 10:
            preview += ", ..."
        reason_summary = (
            f"non-finite time rows={np.count_nonzero(bad_time)}, "
            f"non-finite flux rows={np.count_nonzero(bad_flux)}, "
            f"non-finite uncertainty rows={np.count_nonzero(bad_err_nonfinite)}, "
            f"non-positive uncertainty rows={np.count_nonzero(bad_err_nonpositive)}"
        )
        message = (
            f"Found {n_bad}/{time.size} invalid spectroscopic integrations before "
            f"wavelength binning ({reason_summary}); pre-binning row indices: {preview}."
        )
        if not np.any(keep):
            raise ValueError(message + " No valid integrations remain.")
        print(message + " Removing them globally from every wavelength channel.")

    return time[keep], flux[keep, :], flux_err[keep, :], keep


def _validate_binned_spectroscopy(binned_data, expected_n_time):
    """Fail clearly if binning creates invalid likelihood inputs or loses alignment."""
    for resolution in ("lr", "hr"):
        flux = np.asarray(binned_data[f"flux_{resolution}"])
        flux_err = np.asarray(binned_data[f"flux_err_{resolution}"])
        if flux.ndim != 2 or flux_err.shape != flux.shape:
            raise ValueError(
                f"Binned {resolution.upper()} flux/error arrays are misaligned: "
                f"flux={flux.shape}, flux_err={flux_err.shape}."
            )
        if flux.shape[1] != expected_n_time:
            raise ValueError(
                f"Binned {resolution.upper()} time axis has {flux.shape[1]} integrations, "
                f"but the validated native cube has {expected_n_time}."
            )
        if not np.isfinite(flux).all():
            raise ValueError(f"Binned {resolution.upper()} flux contains non-finite values.")
        if not np.isfinite(flux_err).all() or np.any(flux_err <= 0):
            raise ValueError(
                f"Binned {resolution.upper()} uncertainties must all be finite and > 0."
            )


def normalize_flux(flux, flux_err, norm_range):
    """Normalize flux arrays by median of first 50 points."""
    flux = np.array(flux)
    flux_err = np.array(flux_err)
    flux_norm = flux * 1.0
    flux_err_norm = flux_err * 1.0

    for i in range(flux.shape[0]):
        if norm_range is None or np.sum(norm_range) == 0:
            norm_range_fallback = slice(0,150)
            norm = np.nanmedian(flux[i, norm_range_fallback])
        else:
            norm = np.nanmedian(flux[i, norm_range])
        if norm > 0:
            flux_norm[i, :] /= norm
            flux_err_norm[i,:] /= norm
    return flux_norm, flux_err_norm

def _empty_low_resolution(n_time):
    """Placeholder low-resolution arrays when no coarse grid is configured.

    Zero channels keep every downstream shape check valid, and the pipeline
    skips the low-resolution stage when it sees no ``low`` grid.
    """
    print("No low-resolution grid configured; skipping low-resolution binning.")
    empty_axis = np.zeros(0, dtype=float)
    empty_cube = np.zeros((0, int(n_time)), dtype=float)
    return empty_axis, empty_axis.copy(), empty_cube, empty_cube.copy()


def _bin_spectroscopic_grid(wavelengths, widths, flux, error, grid, level, mode):
    """Select a native, reference, resolving-power, or pixel grid."""
    setting = grid.get(level)
    if level == 'low' and setting is None:
        return _empty_low_resolution(flux.shape[1])
    if level == 'high' and setting == 'native':
        return wavelengths, widths, flux, error
    reference_key = 'reference_grid_lr' if level == 'low' else 'reference_grid'
    reference = grid.get(reference_key)
    if setting == 'reference' and reference is not None:
        from .reference_grid import bin_to_reference_grid_simple
        return bin_to_reference_grid_simple(
            wavelengths, flux, error, reference, trim_to_overlap=True, method='average',
        )
    if mode == 'pixels':
        return bin_at_pixel(wavelengths, flux, error, setting)
    if not isinstance(setting, (int, float)):
        choices = "a number or 'reference'" if level == 'low' else "a number, 'native', or 'reference'"
        raise ValueError(f"resolution.{level} must be {choices}. Got: {setting}")
    return bin_at_resolution(wavelengths, flux, error, setting, method='average')


def bin_spectroscopy_data(
    wavelengths, wavelengths_err, flux_unbinned, flux_err_unbinned, cfg, oot_mask,
):
    """Bin and normalize both grids, retaining a common valid time axis."""
    mode = 'resolution' if cfg.get('resolution') is not None else 'pixels'
    grid = cfg.get(mode)
    if grid is None:
        raise ValueError('Must specify pixels or resolution')
    for level, reference_key in (('high', 'reference_grid'), ('low', 'reference_grid_lr')):
        if grid.get(level) == 'reference':
            reference = grid.get(reference_key)
            if reference is None and mode == 'resolution':
                raise ValueError(
                    f"resolution.{level}='reference' requires resolution.{reference_key} "
                    "to be set in config!"
                )
            if reference is not None:
                print(f"Using reference wavelength grid for {level.upper()} resolution from: {reference}")
    # Binners use [wavelength, time]; readers supply [time, wavelength].
    flux = jnp.array((flux_unbinned * 1.0).T)
    error = jnp.array((flux_err_unbinned * 1.0).T)
    result = {}
    keep_time = np.ones(flux.shape[1], dtype=bool)
    for level, suffix in (('low', 'lr'), ('high', 'hr')):
        wave, width, binned, binned_error = _bin_spectroscopic_grid(
            wavelengths, wavelengths_err, flux, error, grid, level, mode,
        )
        valid = _instrument_wavelength_mask(cfg['instrument'], wave)
        wave, width = wave[valid], width[valid]
        binned, binned_error = binned[valid], binned_error[valid]
        count = min(len(wave), len(width), binned.shape[0], binned_error.shape[0])
        wave, width = wave[:count], width[:count]
        binned, binned_error = normalize_flux(
            binned[:count, :], binned_error[:count, :], norm_range=oot_mask,
        )
        valid = np.isfinite(binned).all(axis=1) & np.isfinite(binned_error).all(axis=1)
        wave, width = wave[valid], width[valid]
        binned, binned_error = binned[valid, :], binned_error[valid, :]
        keep_time &= np.isfinite(binned).all(axis=0) & np.isfinite(binned_error).all(axis=0)
        assert len(wave) == len(width) == len(binned) == len(binned_error), "Channels misaligned"
        result.update({
            f'wavelengths_{suffix}': wave, f'wavelengths_err_{suffix}': width,
            f'flux_{suffix}': binned, f'flux_err_{suffix}': binned_error,
        })
    for suffix in ('lr', 'hr'):
        for name in ('flux', 'flux_err'):
            key = f'{name}_{suffix}'
            result[key] = result[key][:, keep_time]
    return result


def transit_epoch_mask(time, t0, half_window, period=None):
    """Boolean mask of ``time`` within ``half_window`` of any transit epoch.

    With a finite positive ``period`` every epoch ``t0 + n * period`` that
    falls inside the time series is masked, so a single time series that
    spans several transits (a multi-visit stack) is handled; without one only
    the ``t0`` epoch is used.
    """
    time = np.asarray(time, dtype=float)
    t0 = float(t0)
    half_window = float(half_window)
    if time.size == 0:
        return np.zeros(0, dtype=bool)
    if period is None or not np.isfinite(period) or float(period) <= 0.0:
        centers = np.array([t0])
    else:
        period = float(period)
        n_low = int(np.floor((np.min(time) - half_window - t0) / period))
        n_high = int(np.ceil((np.max(time) + half_window - t0) / period))
        centers = t0 + period * np.arange(n_low, n_high + 1)
        centers = centers[
            (centers >= np.min(time) - half_window)
            & (centers <= np.max(time) + half_window)
        ]
        if centers.size == 0:
            centers = np.array([t0])
    mask = np.zeros(time.shape, dtype=bool)
    for center in centers:
        mask |= (time >= center - half_window) & (time <= center + half_window)
    return mask


def _resolve_transit_ephemeris(planet_cfg, transit_ephemeris=None):
    """Per-planet ``(t0, duration, period)`` centre values for the data masks."""
    if transit_ephemeris is None:
        from .config import parse_planet_parameter_specs, planet_parameter_centers
        specs = parse_planet_parameter_specs(planet_cfg)
        if 't0' not in specs or 'duration' not in specs:
            raise KeyError(
                'planet.t0 and planet.duration are required to build the out-of-transit mask.',
            )
        t0s = planet_parameter_centers(specs, 't0')
        durations = planet_parameter_centers(specs, 'duration')
        periods = (
            planet_parameter_centers(specs, 'period')
            if 'period' in specs else np.full(t0s.shape, np.nan)
        )
    else:
        t0s = np.atleast_1d(np.asarray(transit_ephemeris['t0'], dtype=float))
        durations = np.atleast_1d(np.asarray(transit_ephemeris['duration'], dtype=float))
        periods = transit_ephemeris.get('period')
        periods = (
            np.full(t0s.shape, np.nan) if periods is None
            else np.atleast_1d(np.asarray(periods, dtype=float))
        )
    if not (t0s.shape == durations.shape == periods.shape):
        raise ValueError(
            "Transit ephemeris arrays must share one length per planet; "
            f"received t0 {t0s.shape}, duration {durations.shape}, "
            f"period {periods.shape}."
        )
    return t0s, durations, periods


def _instrument_wavelength_mask(instrument, wavelengths):
    """Boolean mask keeping binned channels inside the instrument's wavelength window."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    data_range = data_wavelength_range(instrument)
    if data_range is None:
        return np.ones(len(wavelengths), dtype=bool)
    return (wavelengths >= data_range[0]) & (wavelengths <= data_range[1])


def unpack_nirspec_exotedrf(infile, instrument, trim_start, trim_end, **kwargs):
    """Backwards-compatible alias of :func:`unpack_exotedrf_spectra`."""
    return unpack_exotedrf_spectra(infile, instrument, trim_start, trim_end, **kwargs)


def unpack_miri_exotedrf(infile, trim_start, trim_end, **kwargs):
    """Backwards-compatible alias of :func:`unpack_exotedrf_spectra` for MIRI/LRS."""
    return unpack_exotedrf_spectra(infile, 'MIRI/LRS', trim_start, trim_end, **kwargs)


def process_spectroscopy_data(
    instrument, input_dir, output_dir, planet_str, cfg, fits_file, mask_start=None,
    mask_end=None, mask_integrations_start=None, mask_integrations_end=None,
    transit_ephemeris=None, exclude_times=None, exclude_integrations=None,
):
    """Main function to process spectroscopy data.

    ``transit_ephemeris`` optionally supplies ``{'t0', 'duration', 'period'}``
    centre values per planet; otherwise they are read from ``cfg['planet']``.
    Every transit epoch inside the time series is masked as in-transit.

    ``exclude_times`` and ``exclude_integrations`` are lists of inclusive
    ``(start, end)`` pairs (see ``koala.exclusions``). When ``None`` they are
    resolved from ``cfg`` and merged with the legacy ``mask_*`` arguments.
    """
    if exclude_times is None or exclude_integrations is None:
        cfg_times, cfg_integrations = resolve_exclusions(
            cfg, mask_start=mask_start, mask_end=mask_end,
            mask_integrations_start=mask_integrations_start,
            mask_integrations_end=mask_integrations_end,
        )
        if exclude_times is None:
            exclude_times = cfg_times
        if exclude_integrations is None:
            exclude_integrations = cfg_integrations
    prior_t0s, prior_durations, prior_periods = _resolve_transit_ephemeris(
        cfg['planet'], transit_ephemeris
    )
    wl_filt_cfg = cfg.get('wavelength_filter', {})
    wl_min = wl_filt_cfg.get('wl_min')
    wl_max = wl_filt_cfg.get('wl_max')
    wl_min_o1 = wl_filt_cfg.get('wl_min_o1')
    wl_max_o1 = wl_filt_cfg.get('wl_max_o1')
    wl_min_o2 = wl_filt_cfg.get('wl_min_o2')
    wl_max_o2 = wl_filt_cfg.get('wl_max_o2')

    wavelength_masks = cfg.get('wavelength_masks', None)

    instrument = normalize_instrument(instrument)
    nrs, order = resolve_detector(instrument, cfg)
    input_format = cfg.get('input_format', 'auto')
    if detector_key(instrument) == 'order':
        wavelengths, wavelengths_err, time, flux_unbinned, flux_err_unbinned = unpack_niriss_exotedrf(fits_file, order, None, None, exclude_integrations=exclude_integrations, wl_min_o1=wl_min_o1, wl_max_o1=wl_max_o1, wl_min_o2=wl_min_o2, wl_max_o2=wl_max_o2, wavelength_masks=wavelength_masks, input_format=input_format)
        mini_instrument = order
    else:
        wavelengths, wavelengths_err, time, flux_unbinned, flux_err_unbinned = unpack_exotedrf_spectra(fits_file, instrument, None, None, exclude_integrations=exclude_integrations, wl_min=wl_min, wl_max=wl_max, wavelength_masks=wavelength_masks, input_format=input_format)
        mini_instrument = nrs if nrs is not None else ''

    wavelengths = np.array(wavelengths)
    wavelengths_err = np.array(wavelengths_err)
    time = np.array(time)
    flux_unbinned = np.array(flux_unbinned)  # Shape: (n_time, n_wavelength)
    flux_err_unbinned = np.array(flux_err_unbinned)
    if wavelengths.ndim != 1 or wavelengths_err.shape != wavelengths.shape:
        raise ValueError(
            "Spectroscopic wavelength arrays are misaligned: "
            f"wavelengths={wavelengths.shape}, wavelength_err={wavelengths_err.shape}."
        )
    if flux_unbinned.ndim != 2 or flux_err_unbinned.shape != flux_unbinned.shape:
        raise ValueError(
            "Spectroscopic flux/error arrays are misaligned: "
            f"flux={flux_unbinned.shape}, flux_err={flux_err_unbinned.shape}."
        )
    if flux_unbinned.shape != (time.size, wavelengths.size):
        raise ValueError(
            "Spectroscopic wavelength/time/flux arrays are misaligned: "
            f"time={time.shape}, wavelengths={wavelengths.shape}, flux={flux_unbinned.shape}."
        )

    # Remove only structural/non-science columns here. In particular, retain a
    # finite-wavelength column with zero errors so the integration validator
    # below reports it instead of silently treating it as padding.
    invalid_wavelength = (
        ~np.isfinite(wavelengths)
        | ~np.isfinite(wavelengths_err)
        | np.all(np.isnan(flux_unbinned), axis=0)
    )
    if np.any(invalid_wavelength):
        print(
            f"Removing {np.count_nonzero(invalid_wavelength)}/{wavelengths.size} "
            "non-science wavelength columns before integration validation."
        )
    wavelengths = wavelengths[~invalid_wavelength]
    wavelengths_err = wavelengths_err[~invalid_wavelength]
    flux_unbinned = flux_unbinned[:, ~invalid_wavelength]
    flux_err_unbinned = flux_err_unbinned[:, ~invalid_wavelength]

    # Time-range exclusions (spot crossings, tilt events, ramps) and the
    # optional "cut_phase_to_transit" directive; see koala.exclusions.
    if has_cut_phase_directive(exclude_times):
        # Keep t0 +/- 3 hours around every transit epoch in the series.
        window = 3.0 / 24.0
        keep = np.zeros_like(time, dtype=bool)
        for t0, period in zip(prior_t0s, prior_periods):
            keep |= transit_epoch_mask(time, t0, window, period)
        time = time[keep]
        flux_unbinned = flux_unbinned[keep, :]
        flux_err_unbinned = flux_err_unbinned[keep, :]

    keep = time_keep_mask(time, exclude_times)
    if not np.all(keep):
        print(f"Excluding {int((~keep).sum())} of {time.size} integrations by time.")
        time = time[keep]
        flux_unbinned = flux_unbinned[keep, :]
        flux_err_unbinned = flux_err_unbinned[keep, :]

    # A Normal likelihood requires a finite, strictly positive scale. Apply one
    # union mask across all retained science wavelengths before constructing the
    # white light curve or either spectroscopic resolution.
    time, flux_unbinned, flux_err_unbinned, _ = _filter_invalid_spectroscopic_integrations(
        time, flux_unbinned, flux_err_unbinned
    )

    in_transit_mask = np.zeros_like(time, dtype=bool)
    for t0, duration, period in zip(prior_t0s, prior_durations, prior_periods):
        in_transit_mask |= transit_epoch_mask(time, t0, 0.6 * duration, period)

    oot_mask = ~in_transit_mask
    if not np.any(oot_mask):
        raise ValueError(
            "Every integration falls inside a transit window; the "
            "out-of-transit normalization mask is empty."
        )

    binned_data = bin_spectroscopy_data(
        wavelengths, wavelengths_err, flux_unbinned, flux_err_unbinned, cfg, oot_mask
    )
    _validate_binned_spectroscopy(binned_data, expected_n_time=time.size)

    wlc = np.nansum(flux_unbinned, axis=1)
    wl_flux = wlc/np.nanmedian(wlc[oot_mask], axis=0)
    wl_flux_err = np.nanmedian(np.abs(np.diff(wl_flux)))

    return SpectroData(
        time=jnp.array(time), wavelengths_unbinned=jnp.array(wavelengths),
        flux_unbinned=jnp.array(flux_unbinned), wl_time=jnp.array(time), wl_flux=wl_flux,
        wl_flux_err=wl_flux_err, wavelengths_lr=jnp.array(binned_data['wavelengths_lr']),
        wavelengths_err_lr=jnp.array(binned_data['wavelengths_err_lr']),
        flux_lr=jnp.array(binned_data['flux_lr']),
        flux_err_lr=jnp.array(binned_data['flux_err_lr']),
        wavelengths_hr=jnp.array(binned_data['wavelengths_hr']),
        wavelengths_err_hr=jnp.array(binned_data['wavelengths_err_hr']),
        flux_hr=jnp.array(binned_data['flux_hr']),
        flux_err_hr=jnp.array(binned_data['flux_err_hr']), instrument=instrument,
        planet=planet_str, mini_instrument=mini_instrument,
    )
