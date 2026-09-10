import os, sys, glob
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import pickle
from koala.binning import bin_at_resolution, bin_at_pixel
from astropy.io import fits
import matplotlib.pyplot as plt
jax.config.update('jax_enable_x64', True)

def apply_wavelength_masks(wave, wave_err, fluxcube, fluxcube_err, mask_ranges):
    """
    Apply wavelength range masks to exclude specified regions.

    Parameters:
    -----------
    wave : array
        Wavelength array
    wave_err : array
        Wavelength error array
    fluxcube : array
        Flux cube (n_time, n_wavelength)
    fluxcube_err : array
        Flux error cube (n_time, n_wavelength)
    mask_ranges : list or None
        List of wavelength ranges to mask out, formatted as [wl_min1, wl_max1, wl_min2, wl_max2, ...]
        Must have even length (pairs of start/end values)

    Returns:
    --------
    wave, wave_err, fluxcube, fluxcube_err : filtered arrays with masked wavelengths removed
    """
    if mask_ranges is None or len(mask_ranges) == 0:
        return wave, wave_err, fluxcube, fluxcube_err

    # Validate input
    if len(mask_ranges) % 2 != 0:
        raise ValueError(f"wavelength_masks must have even length (pairs of min/max), got length {len(mask_ranges)}")

    # Create mask: True = keep, False = exclude
    keep_mask = np.ones(len(wave), dtype=bool)

    # Process each pair of (min, max) ranges
    for i in range(0, len(mask_ranges), 2):
        wl_min = mask_ranges[i]
        wl_max = mask_ranges[i + 1]

        if wl_min >= wl_max:
            raise ValueError(f"Invalid wavelength mask range: min={wl_min} >= max={wl_max}")

        # Mark wavelengths in this range for exclusion
        in_mask_range = (wave >= wl_min) & (wave <= wl_max)
        keep_mask &= ~in_mask_range

        print(f"  Masking wavelength range: {wl_min:.3f} - {wl_max:.3f} µm ({np.sum(in_mask_range)} pixels)")

    # Apply the mask
    n_masked = np.sum(~keep_mask)
    if n_masked > 0:
        print(f"  Total masked: {n_masked}/{len(wave)} wavelength pixels")
        wave = wave[keep_mask]
        wave_err = wave_err[keep_mask]
        fluxcube = fluxcube[:, keep_mask]
        fluxcube_err = fluxcube_err[:, keep_mask]

    return wave, wave_err, fluxcube, fluxcube_err

def unpack_niriss_exotedrf(infile, order, trim_start, trim_end, wl_min_o1=None, wl_max_o1=None, wl_min_o2=None, wl_max_o2=None, wavelength_masks=None):    

    bjd = fits.getdata(infile, 9)
    wave = fits.getdata(infile, 1 + 4 * (order - 1))
    wave_err = fits.getdata(infile, 2 + 4 * (order - 1))
    fluxcube = fits.getdata(infile, 3 + 4 * (order - 1))
    fluxcube_err = fits.getdata(infile, 4 + 4 * (order -1))
    wave = wave[5:-5]
    wave_err = wave_err[5:-5]
    fluxcube = fluxcube[:, 5:-5]
    fluxcube_err = fluxcube_err[:, 5:-5]
    
    start = 0 if (trim_start is None) else int(trim_start)
    stop  = None if (trim_end in (None, 0)) else -int(trim_end)

    fluxcube     = fluxcube[start:stop, :]
    fluxcube_err = fluxcube_err[start:stop, :]
    bjd            = bjd[start:stop]   # keep time aligned!


    if order == 2:
        ii = np.where((wave >= 0.6) & (wave <= 0.85))[0]
        fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]

    if wl_min_o1 is not None and wl_max_o1 is not None and order == 1:
        ii = np.where((wave >= wl_min_o1) & (wave <= wl_max_o1))[0]
        fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]
    
    if wl_min_o2 is not None and wl_max_o2 is not None and order == 2:
        ii = np.where((wave >= wl_min_o2) & (wave <= wl_max_o2))[0]
        fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]

    # Apply custom wavelength masks
    if wavelength_masks is not None and len(wavelength_masks) > 0:
        print(f"Applying wavelength masks for NIRISS order {order}:")
        wave, wave_err, fluxcube, fluxcube_err = apply_wavelength_masks(
            wave, wave_err, fluxcube, fluxcube_err, wavelength_masks
        )

    # FITS stores big-endian arrays; JAX requires native-endian numeric input.
    wavelength = np.asarray(wave, dtype=np.float64)
    wavelength_err = np.asarray(wave_err, dtype=np.float64)
    t = np.asarray(bjd, dtype=np.float64)
    fluxcube = np.asarray(fluxcube, dtype=np.float64)
    fluxcube_err = np.asarray(fluxcube_err, dtype=np.float64)

    return wavelength,wavelength_err, t, fluxcube, fluxcube_err

def unpack_nirspec_exotedrf(infile, instrument, trim_start, trim_end, wl_min=None, wl_max=None, wavelength_masks=None):
    bjd = fits.getdata(infile, 5)
    wave = fits.getdata(infile, 1)
    wave_err = fits.getdata(infile, 2)
    fluxcube = fits.getdata(infile, 3)
    fluxcube_err = fits.getdata(infile, 4)
    wave = wave[5:-5]
    wave_err = wave_err[5:-5]
    fluxcube = fluxcube[:, 5:-5]
    fluxcube_err = fluxcube_err[:, 5:-5]

    start = 0 if (trim_start is None) else int(trim_start)
    stop  = None if (trim_end in (None, 0)) else -int(trim_end)

    fluxcube     = fluxcube[start:stop, :]
    fluxcube_err = fluxcube_err[start:stop, :]
    bjd            = bjd[start:stop]   # keep time aligned!


    if instrument == 'NIRSPEC/G395M' or instrument == 'NIRSPEC/G395H':
        ii = np.where((wave >= 2.9) & (wave <= 5.0))[0]
        fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]
    if instrument == 'NIRSPEC/PRISM':
        ii = np.where((wave >= 0.6) & (wave <= 5.0))[0]
        fluxcube, fluxcube_err = fluxcube[:,ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]
    if instrument == 'NIRSPEC/G140H':
        ii = np.where((wave >= 1.0) & (wave <= 1.8))[0]
        fluxcube, fluxcube_err = fluxcube[:,ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]

    if wl_min is not None and wl_max is not None:
        ii = np.where((wave >= wl_min) & (wave <= wl_max))[0]
        fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]

    # Apply custom wavelength masks
    if wavelength_masks is not None and len(wavelength_masks) > 0:
        print(f"Applying wavelength masks for {instrument}:")
        wave, wave_err, fluxcube, fluxcube_err = apply_wavelength_masks(
            wave, wave_err, fluxcube, fluxcube_err, wavelength_masks
        )

    # FITS stores big-endian arrays; JAX requires native-endian numeric input.
    wavelength = np.asarray(wave, dtype=np.float64)
    wavelength_err = np.asarray(wave_err, dtype=np.float64)
    t = np.asarray(bjd, dtype=np.float64)
    fluxcube = np.asarray(fluxcube, dtype=np.float64)
    fluxcube_err = np.asarray(fluxcube_err, dtype=np.float64)
    return wavelength, wavelength_err,  t, fluxcube, fluxcube_err

def unpack_miri_exotedrf(infile, trim_start, trim_end, wl_min=None, wl_max=None, wavelength_masks=None):

    bjd = fits.getdata(infile, 5)
    wave = fits.getdata(infile, 1)
    wave_err = fits.getdata(infile, 2)
    fluxcube = fits.getdata(infile, 3)
    fluxcube_err = fits.getdata(infile, 4)
    wave = wave[5:-5]
    wave_err = wave_err[5:-5]
    fluxcube = fluxcube[:, 5:-5]
    fluxcube_err = fluxcube_err[:, 5:-5]


    start = 0 if (trim_start is None) else int(trim_start)
    stop  = None if (trim_end in (None, 0)) else -int(trim_end)

    fluxcube     = fluxcube[start:stop, :]
    fluxcube_err = fluxcube_err[start:stop, :]
    bjd            = bjd[start:stop]   # keep time aligned!


    ii = np.where((wave > 5) & (wave <= 12))[0]
    fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:,ii]
    wave, wave_err = wave[ii], wave_err[ii]
    ii = np.argsort(wave)
    wave, wave_err = wave[ii], wave_err[ii]
    fluxcube, fluxcube_err = fluxcube[:,ii], fluxcube_err[:,ii]

    if wl_min is not None and wl_max is not None:
        ii = np.where((wave >= wl_min) & (wave <= wl_max))[0]
        fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]

    # Apply custom wavelength masks
    if wavelength_masks is not None and len(wavelength_masks) > 0:
        print(f"Applying wavelength masks for MIRI/LRS:")
        wave, wave_err, fluxcube, fluxcube_err = apply_wavelength_masks(
            wave, wave_err, fluxcube, fluxcube_err, wavelength_masks
        )

    # FITS stores big-endian arrays; JAX requires native-endian numeric input.
    wavelength = np.asarray(wave, dtype=np.float64)
    wavelength_err = np.asarray(wave_err, dtype=np.float64)
    t = np.asarray(bjd, dtype=np.float64)
    fluxcube = np.asarray(fluxcube, dtype=np.float64)
    fluxcube_err = np.asarray(fluxcube_err, dtype=np.float64)
    return wavelength, wavelength_err, t, fluxcube, fluxcube_err

class SpectroData:
    """Simple container for dot notation access."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def save_whitelight_csv(self, output_path):
        df = pd.DataFrame({'time': self.wl_time, 'flux': self.wl_flux, 'flux_err': self.wl_flux_err})
        df.to_csv(output_path, index=False)


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


def bin_spectroscopy_data(wavelengths, wavelengths_err, flux_unbinned, flux_err_unbinned, cfg, oot_mask):
    """Handle all the binning logic in one place."""
    resolution = cfg.get('resolution', None)
    pixels = cfg.get('pixels', None)
    nrs = cfg.get('nrs', None)

    flux_unbinned_copy = flux_unbinned * 1.0
    flux_err_unbinned_copy = flux_err_unbinned * 1.0

    # Transpose flux for binning: (n_time, n_wavelength) -> (n_wavelength, n_time)
    flux_transposed = jnp.array(flux_unbinned_copy.T)
    flux_err_transposed = jnp.array(flux_err_unbinned_copy.T)

    # Check if using reference grid for high or low resolution
    use_reference_grid_hr = False
    use_reference_grid_lr = False
    if resolution is not None:
        if resolution.get('high') == 'reference':
            if resolution.get('reference_grid') is None:
                raise ValueError("resolution.high='reference' requires resolution.reference_grid to be set in config!")
            use_reference_grid_hr = True
            reference_grid_path_hr = resolution.get('reference_grid')
            print(f"Using reference wavelength grid for HIGH resolution from: {reference_grid_path_hr}")
        if resolution.get('low') == 'reference':
            if resolution.get('reference_grid_lr') is None:
                raise ValueError("resolution.low='reference' requires resolution.reference_grid_lr to be set in config!")
            use_reference_grid_lr = True
            reference_grid_path_lr = resolution.get('reference_grid_lr')
            print(f"Using reference wavelength grid for LOW resolution from: {reference_grid_path_lr}")

    # Low resolution binning (optional: omit resolution.low to skip it)
    if resolution is not None:
        if resolution.get('low') is None:
            wl_lr, wl_err_lr, flux_lr, flux_err_lr = _empty_low_resolution(
                flux_transposed.shape[1]
            )
        elif use_reference_grid_lr:
            # Bin to reference grid for low resolution
            # flux_transposed is already (wavelength, time) format
            from bin_to_reference_grid import bin_to_reference_grid_simple
            wl_lr, wl_err_lr, flux_lr, flux_err_lr = bin_to_reference_grid_simple(
                wavelengths, flux_transposed, flux_err_transposed,
                reference_grid_path_lr, trim_to_overlap=True, method='average',
            )
        else:
            # Regular resolution binning - make sure it's a number
            low_res = resolution.get('low')
            if not isinstance(low_res, (int, float)):
                raise ValueError(f"resolution.low must be a number or 'reference'. Got: {low_res}")
            wl_lr, wl_err_lr, flux_lr, flux_err_lr = bin_at_resolution(
                wavelengths, flux_transposed, flux_err_transposed, low_res, method='average'
            )
        if cfg['instrument'] == 'NIRSPEC/G395M' or cfg['instrument'] == 'NIRSPEC/G395H':
            # Trim edge wavelengths for low-res based on detector
            if nrs == 1:
                # NRS1: clip wavelengths < 2.9 microns
                valid_lr = (wl_lr >= 2.9) & (wl_lr <= 5.0)
            elif nrs == 2:
                # NRS2: clip wavelengths > 5.0 microns
                valid_lr = wl_lr <= 5.0
        elif cfg['instrument'] == 'NIRSPEC/PRISM':
            valid_lr = (wl_lr >= 0.5) & (wl_lr <= 5.0)
        elif cfg['instrument'] == 'NIRSPEC/G140H':
            valid_lr = (wl_lr >= 1.0) & (wl_lr <= 1.8)
        else:
            valid_lr = np.ones(len(wl_lr), dtype=bool)
        wl_lr = wl_lr[valid_lr]
        wl_err_lr = wl_err_lr[valid_lr]
        flux_lr = flux_lr[valid_lr]
        flux_err_lr = flux_err_lr[valid_lr]

        n_lr = min(len(wl_lr), flux_lr.shape[0], flux_err_lr.shape[0], len(wl_err_lr))
        wl_lr, wl_err_lr = wl_lr[:n_lr], wl_err_lr[:n_lr]
        flux_lr, flux_err_lr = flux_lr[:n_lr, :], flux_err_lr[:n_lr, :]

        # High resolution binning
        if resolution.get('high') == 'native':
            wl_hr, wl_err_hr, flux_hr, flux_err_hr = wavelengths, wavelengths_err, flux_transposed, flux_err_transposed
        elif use_reference_grid_hr:
            # Bin to reference grid
            # flux_transposed is already (wavelength, time) format
            from bin_to_reference_grid import bin_to_reference_grid_simple
            wl_hr, wl_err_hr, flux_hr, flux_err_hr = bin_to_reference_grid_simple(
                wavelengths, flux_transposed, flux_err_transposed,
                reference_grid_path_hr, trim_to_overlap=True, method='average',
            )
        else:
            # Regular resolution binning - make sure it's a number
            high_res = resolution.get('high')
            if not isinstance(high_res, (int, float)):
                raise ValueError(f"resolution.high must be a number, 'native', or 'reference'. Got: {high_res}")
            wl_hr, wl_err_hr, flux_hr, flux_err_hr = bin_at_resolution(
                wavelengths, flux_transposed, flux_err_transposed, high_res, method='average'
            )
        if cfg['instrument'] == 'NIRSPEC/G395H' or cfg['instrument'] == 'NIRSPEC/G395M':
            # Trim edge wavelengths for high-res based on detector
            if nrs == 1:
                # NRS1: clip wavelengths < 2.9 microns
                valid_hr = (wl_hr >= 2.9) & (wl_hr<= 5.0)
            elif nrs == 2:
                # NRS2: clip wavelengths > 5.0 microns
                valid_hr = wl_hr <= 5.0
        elif cfg['instrument'] == 'NIRSPEC/PRISM':
            valid_hr = (wl_hr >= 0.5) & (wl_hr <= 5.0)
        elif cfg['instrument'] == 'NIRSPEC/G140H':
            valid_hr = (wl_hr >= 1.0) & (wl_hr <= 1.8)
        else:
            valid_hr = np.ones(len(wl_hr), dtype=bool) 
        wl_hr = wl_hr[valid_hr]
        wl_err_hr = wl_err_hr[valid_hr]
        flux_hr = flux_hr[valid_hr]
        flux_err_hr = flux_err_hr[valid_hr]

        n_hr = min(len(wl_hr), flux_hr.shape[0], flux_err_hr.shape[0], len(wl_err_hr))
        wl_hr, wl_err_hr = wl_hr[:n_hr], wl_err_hr[:n_hr]
        flux_hr, flux_err_hr = flux_hr[:n_hr, :], flux_err_hr[:n_hr, :]
        flux_lr, flux_err_lr = normalize_flux(flux_lr, flux_err_lr, norm_range=oot_mask)
        flux_hr, flux_err_hr = normalize_flux(flux_hr, flux_err_hr, norm_range=oot_mask)

        keep_wl_lr = np.isfinite(flux_lr).all(axis=1) & np.isfinite(flux_err_lr).all(axis=1)
        wl_lr, wl_err_lr = wl_lr[keep_wl_lr], wl_err_lr[keep_wl_lr]
        flux_lr, flux_err_lr = flux_lr[keep_wl_lr, :], flux_err_lr[keep_wl_lr, :]
    
        keep_wl_hr = np.isfinite(flux_hr).all(axis=1) & np.isfinite(flux_err_hr).all(axis=1)
        wl_hr, wl_err_hr = wl_hr[keep_wl_hr], wl_err_hr[keep_wl_hr]
        flux_hr, flux_err_hr = flux_hr[keep_wl_hr, :], flux_err_hr[keep_wl_hr, :]
    
        keep_t_lr = np.isfinite(flux_lr).all(axis=0) & np.isfinite(flux_err_lr).all(axis=0)
        keep_t_hr = np.isfinite(flux_hr).all(axis=0) & np.isfinite(flux_err_hr).all(axis=0)
        keep_t_post = keep_t_lr & keep_t_hr
    
        flux_lr, flux_err_lr = flux_lr[:, keep_t_post], flux_err_lr[:, keep_t_post]
        flux_hr, flux_err_hr = flux_hr[:, keep_t_post], flux_err_hr[:, keep_t_post]

        assert wl_lr.shape[0] == flux_lr.shape[0] == flux_err_lr.shape[0] == wl_err_lr.shape[0], "LR channels misaligned"
        assert wl_hr.shape[0] == flux_hr.shape[0] == flux_err_hr.shape[0] == wl_err_hr.shape[0], "HR channels misaligned"
    elif cfg.get('pixels', None) is not None:
        # Check if using reference grid for pixels mode too
        use_reference_grid_pixels_hr = False
        use_reference_grid_pixels_lr = False
        if pixels.get('high') == 'reference' and pixels.get('reference_grid') is not None:
            use_reference_grid_pixels_hr = True
            reference_grid_path_hr = pixels.get('reference_grid')
            print(f"Using reference wavelength grid for HIGH resolution from: {reference_grid_path_hr}")
        if pixels.get('low') == 'reference' and pixels.get('reference_grid_lr') is not None:
            use_reference_grid_pixels_lr = True
            reference_grid_path_lr = pixels.get('reference_grid_lr')
            print(f"Using reference wavelength grid for LOW resolution from: {reference_grid_path_lr}")

        # Low resolution binning (optional: omit pixels.low to skip it)
        if pixels.get('low') is None:
            wl_lr, wl_err_lr, flux_lr, flux_err_lr = _empty_low_resolution(
                flux_transposed.shape[1]
            )
        elif use_reference_grid_pixels_lr:
            from bin_to_reference_grid import bin_to_reference_grid_simple
            wl_lr, wl_err_lr, flux_lr, flux_err_lr = bin_to_reference_grid_simple(
                wavelengths, flux_transposed, flux_err_transposed,
                reference_grid_path_lr, trim_to_overlap=True, method='average',
            )
        else:
            wl_lr, wl_err_lr, flux_lr, flux_err_lr = bin_at_pixel(
            wavelengths, flux_transposed, flux_err_transposed, pixels.get('low'))
                  # Trim edge wavelengths based on detector
        if cfg['instrument'] == 'NIRSPEC/G395H' or cfg['instrument'] == 'NIRSPEC/G395M':
            if nrs == 1:
            # NRS1: clip wavelengths < 2.9 microns
                valid_hr = (wl_hr >= 2.9) & (wl_hr <= 5.0)
                wl_hr, wl_err_hr = wl_hr[valid_hr], wl_err_hr[valid_hr]
                flux_hr, flux_err_hr = flux_hr[valid_hr], flux_err_hr[valid_hr]

                valid_lr = (wl_lr >= 2.9) & (wl_lr <= 5.0)
                wl_lr, wl_err_lr = wl_lr[valid_lr], wl_err_lr[valid_lr]
                flux_lr, flux_err_lr = flux_lr[valid_lr], flux_err_lr[valid_lr]
            elif nrs == 2:
          # NRS2: clip wavelengths > 5.0 microns
                valid_hr = wl_hr <= 5.0
                wl_hr, wl_err_hr = wl_hr[valid_hr], wl_err_hr[valid_hr]
                flux_hr, flux_err_hr = flux_hr[valid_hr], flux_err_hr[valid_hr]

                valid_lr = wl_lr <= 5.0
                wl_lr, wl_err_lr = wl_lr[valid_lr], wl_err_lr[valid_lr]
        elif cfg['instrument'] == 'NIRSPEC/PRISM':
            valid_hr = (wl_hr >= 0.5) & (wl_hr <= 5.0)
            wl_hr, wl_err_hr = wl_hr[valid_hr], wl_err_hr[valid_hr]
            flux_hr, flux_err_hr = flux_hr[valid_hr], flux_err_hr[valid_hr]

            valid_lr = (wl_lr >= 0.5) & (wl_lr <= 5.0)
            wl_lr, wl_err_lr = wl_lr[valid_lr], wl_err_lr[valid_lr]
            flux_lr, flux_err_lr = flux_lr[valid_lr], flux_err_lr[valid_lr] 
       
        elif cfg['instrument'] == 'NIRSPEC/G140H':
            valid_hr = (wl_hr >= 1.0) & (wl_hr <= 1.8)
            wl_hr, wl_err_hr = wl_hr[valid_hr], wl_err_hr[valid_hr]
            flux_hr, flux_err_hr = flux_hr[valid_hr], flux_err_hr[valid_hr]

            valid_lr = (wl_lr >= 1.0) & (wl_lr <= 1.8)
            wl_lr, wl_err_lr = wl_lr[valid_lr], wl_err_lr[valid_lr]
            flux_lr, flux_err_lr = flux_lr[valid_lr], flux_err_lr[valid_lr]
        
        n_lr = min(len(wl_lr), flux_lr.shape[0], flux_err_lr.shape[0], len(wl_err_lr))
        wl_lr, wl_err_lr = wl_lr[:n_lr], wl_err_lr[:n_lr]
        flux_lr, flux_err_lr = flux_lr[:n_lr, :], flux_err_lr[:n_lr, :]

        # High resolution binning
        if pixels.get('high') == 'native':
            wl_hr, wl_err_hr, flux_hr, flux_err_hr = wavelengths, wavelengths_err, flux_transposed, flux_err_transposed
        elif use_reference_grid_pixels_hr:
            # Bin to reference grid
            from bin_to_reference_grid import bin_to_reference_grid_simple
            wl_hr, wl_err_hr, flux_hr, flux_err_hr = bin_to_reference_grid_simple(
                wavelengths, flux_transposed, flux_err_transposed,
                reference_grid_path_hr, trim_to_overlap=True, method='average',
            )
        else:
            wl_hr, wl_err_hr, flux_hr, flux_err_hr = bin_at_pixel(
                wavelengths, flux_transposed, flux_err_transposed, pixels.get('high'))
            
        n_hr = min(len(wl_hr), flux_hr.shape[0], flux_err_hr.shape[0], len(wl_err_hr))
        wl_hr, wl_err_hr = wl_hr[:n_hr], wl_err_hr[:n_hr]
        flux_hr, flux_err_hr = flux_hr[:n_hr, :], flux_err_hr[:n_hr, :]
   
        flux_lr, flux_err_lr = normalize_flux(flux_lr, flux_err_lr, norm_range=oot_mask)
        flux_hr, flux_err_hr = normalize_flux(flux_hr, flux_err_hr, norm_range=oot_mask)
    
        keep_wl_lr = np.isfinite(flux_lr).all(axis=1) & np.isfinite(flux_err_lr).all(axis=1)
        wl_lr, wl_err_lr = wl_lr[keep_wl_lr], wl_err_lr[keep_wl_lr]
        flux_lr, flux_err_lr = flux_lr[keep_wl_lr, :], flux_err_lr[keep_wl_lr, :]
    
        keep_wl_hr = np.isfinite(flux_hr).all(axis=1) & np.isfinite(flux_err_hr).all(axis=1)
        wl_hr, wl_err_hr = wl_hr[keep_wl_hr], wl_err_hr[keep_wl_hr]
        flux_hr, flux_err_hr = flux_hr[keep_wl_hr, :], flux_err_hr[keep_wl_hr, :]
    
        keep_t_lr = np.isfinite(flux_lr).all(axis=0) & np.isfinite(flux_err_lr).all(axis=0)
        keep_t_hr = np.isfinite(flux_hr).all(axis=0) & np.isfinite(flux_err_hr).all(axis=0)
        keep_t_post = keep_t_lr & keep_t_hr
    
        flux_lr, flux_err_lr = flux_lr[:, keep_t_post], flux_err_lr[:, keep_t_post]
        flux_hr, flux_err_hr = flux_hr[:, keep_t_post], flux_err_hr[:, keep_t_post]
    
        assert wl_lr.shape[0] == flux_lr.shape[0] == flux_err_lr.shape[0] == wl_err_lr.shape[0], "Low Pixel channels misaligned"
        assert wl_hr.shape[0] == flux_hr.shape[0] == flux_err_hr.shape[0] == wl_err_hr.shape[0], "High Pixel channels misaligned"
    else:
        raise ValueError('Must specify pixels or resolution')
    return {
        'wavelengths_lr': wl_lr, 'wavelengths_err_lr': wl_err_lr, 
        'flux_lr': flux_lr, 'flux_err_lr': flux_err_lr,
        'wavelengths_hr': wl_hr, 'wavelengths_err_hr': wl_err_hr, 
        'flux_hr': flux_hr, 'flux_err_hr': flux_err_hr
    }


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
        from koala.config import parse_planet_parameter_specs, planet_parameter_centers
        specs = parse_planet_parameter_specs(planet_cfg)
        if 't0' not in specs or 'duration' not in specs:
            raise KeyError(
                "planet.t0 and planet.duration are required to build the "
                "out-of-transit mask."
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


def process_spectroscopy_data(instrument, input_dir, output_dir, planet_str, cfg, fits_file, mask_start=None, mask_end=None, mask_integrations_start=None, mask_integrations_end=None, transit_ephemeris=None):
    """Main function to process spectroscopy data.

    ``transit_ephemeris`` optionally supplies ``{'t0', 'duration', 'period'}``
    centre values per planet; otherwise they are read from ``cfg['planet']``.
    Every transit epoch inside the time series is masked as in-transit.
    """
    prior_t0s, prior_durations, prior_periods = _resolve_transit_ephemeris(
        cfg['planet'], transit_ephemeris
    )
    # Unpack data based on instrument
    wl_filt_cfg = cfg.get('wavelength_filter', {})
    wl_min = wl_filt_cfg.get('wl_min')
    wl_max = wl_filt_cfg.get('wl_max')
    wl_min_o1 = wl_filt_cfg.get('wl_min_o1')
    wl_max_o1 = wl_filt_cfg.get('wl_max_o1')
    wl_min_o2 = wl_filt_cfg.get('wl_min_o2')
    wl_max_o2 = wl_filt_cfg.get('wl_max_o2')

    # Get wavelength masks from config
    wavelength_masks = cfg.get('wavelength_masks', None)

    if instrument == 'NIRSPEC/G395H' or instrument == 'NIRSPEC/G395M' or instrument == 'NIRSPEC/PRISM' or instrument == 'NIRSPEC/G140H':
        nrs = cfg['nrs']
        planet_cfg = cfg['planet']
        prior_duration = planet_cfg['duration']
        prior_t0 = planet_cfg['t0']
        wavelengths, wavelengths_err, time, flux_unbinned, flux_err_unbinned = unpack_nirspec_exotedrf(fits_file, instrument, mask_integrations_start, mask_integrations_end, wl_min=wl_min, wl_max=wl_max, wavelength_masks=wavelength_masks)
        mini_instrument = nrs
    elif instrument == 'NIRISS/SOSS':
        order = cfg['order']
        wavelengths, wavelengths_err, time, flux_unbinned, flux_err_unbinned = unpack_niriss_exotedrf(fits_file, order, mask_integrations_start, mask_integrations_end, wl_min_o1=wl_min_o1, wl_max_o1=wl_max_o1, wl_min_o2=wl_min_o2, wl_max_o2=wl_max_o2, wavelength_masks=wavelength_masks)
        mini_instrument = order
    elif instrument == 'MIRI/LRS':
        wavelengths, wavelengths_err, time, flux_unbinned, flux_err_unbinned = unpack_miri_exotedrf(fits_file, mask_integrations_start, mask_integrations_end, wl_min=wl_min, wl_max=wl_max, wavelength_masks=wavelength_masks)
        mini_instrument = ''
    else:
        raise NotImplementedError(f'Instrument {instrument} not implemented yet')
    
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

    # Apply time masking criteria (useful for spot-crossings) and optional "cut" directives.
    #
    # Supported inputs:
    # - mask_start/mask_end can be scalars, strings (expressions), or same-length lists.
    # - None values are treated as open-ended (min(time) or max(time)).
    # - Special directive "cut_phase_to_transit": keep only t0 ± 3 hours.

    def evaluate_mask_value(value, time):
        """Evaluate a mask value that could be a number, None, or a string expression."""
        if value is None:
            return None
        if isinstance(value, str):
            v = value.strip()
            # Do not eval special directives
            if v == "cut_phase_to_transit":
                return v
            namespace = {
                'jnp': np,
                'np': np,
                't': time,
                'time': time,
                'min': np.min,
                'max': np.max,
            }
            return eval(v, {"__builtins__": {}}, namespace)
        return value

    def _to_pairs(mask_start, mask_end):
        """Return list of (start,end) pairs from scalar/list inputs."""
        if mask_start is None and mask_end is None:
            return []

        # If one side is missing, treat as open-ended
        if mask_end is None and mask_start is not None:
            if hasattr(mask_start, '__len__') and not isinstance(mask_start, str):
                return [(s, None) for s in mask_start]
            return [(mask_start, None)]

        if mask_start is None and mask_end is not None:
            if hasattr(mask_end, '__len__') and not isinstance(mask_end, str):
                return [(None, e) for e in mask_end]
            return [(None, mask_end)]

        # Both provided
        if hasattr(mask_start, '__len__') and not isinstance(mask_start, str):
            if not (hasattr(mask_end, '__len__') and not isinstance(mask_end, str)):
                raise ValueError("mask_start is a list but mask_end is not.")
            if len(mask_start) != len(mask_end):
                raise ValueError("mask_start and mask_end lists must be the same length.")
            return list(zip(mask_start, mask_end))

        return [(mask_start, mask_end)]

    pairs = _to_pairs(mask_start, mask_end)

    # 1) Handle "cut_phase_to_transit": keep only t0 +/- 3 hours (3/24 days)
    has_cut = any(
        (isinstance(s, str) and s.strip() == "cut_phase_to_transit") or
        (isinstance(e, str) and e.strip() == "cut_phase_to_transit")
        for s, e in pairs
    )

    if has_cut:
        # Keep t0 +/- 3 hours around every transit epoch in the series.
        window = 3.0 / 24.0  # 3 hours in days

        keep = np.zeros_like(time, dtype=bool)
        for t0, period in zip(prior_t0s, prior_periods):
            keep |= transit_epoch_mask(time, t0, window, period)

        time = time[keep]
        flux_unbinned = flux_unbinned[keep, :]
        flux_err_unbinned = flux_err_unbinned[keep, :]

        # Remove cut directives so they don't get evaluated below
        pairs = [
            (s, e) for (s, e) in pairs
            if not (
                (isinstance(s, str) and s.strip() == "cut_phase_to_transit") or
                (isinstance(e, str) and e.strip() == "cut_phase_to_transit")
            )
        ]

    # 2) Apply "mask out" time ranges (remove points inside each range)
    if len(pairs) > 0:
        timemask = np.zeros_like(time, dtype=bool)
        tmin = float(np.min(time))
        tmax = float(np.max(time))

        for start, end in pairs:
            start_val = evaluate_mask_value(start, time)
            end_val = evaluate_mask_value(end, time)

            # Skip fully-empty pairs
            if start_val is None and end_val is None:
                continue

            # Support open-ended masks
            if start_val is None:
                start_val = tmin
            if end_val is None:
                end_val = tmax

            # If a directive snuck through, ignore it here
            if start_val == "cut_phase_to_transit" or end_val == "cut_phase_to_transit":
                continue

            timemask |= (time >= float(start_val)) & (time <= float(end_val))

        time = time[~timemask]
        flux_unbinned = flux_unbinned[~timemask, :]
        flux_err_unbinned = flux_err_unbinned[~timemask, :]

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
        time=jnp.array(time),
        wavelengths_unbinned=jnp.array(wavelengths),
        flux_unbinned=jnp.array(flux_unbinned),
        wl_time=jnp.array(time),
        wl_flux=wl_flux,
        wl_flux_err=wl_flux_err,
        wavelengths_lr=jnp.array(binned_data['wavelengths_lr']),
        wavelengths_err_lr=jnp.array(binned_data['wavelengths_err_lr']),
        flux_lr=jnp.array(binned_data['flux_lr']),
        flux_err_lr=jnp.array(binned_data['flux_err_lr']),
        wavelengths_hr=jnp.array(binned_data['wavelengths_hr']),
        wavelengths_err_hr=jnp.array(binned_data['wavelengths_err_hr']),
        flux_hr=jnp.array(binned_data['flux_hr']),
        flux_err_hr=jnp.array(binned_data['flux_err_hr']),
        instrument=instrument,
        planet=planet_str,
        mini_instrument=mini_instrument
    )
