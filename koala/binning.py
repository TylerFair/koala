"""NumPy-only spectral binning used by Koala.

Ported from exoTEDRF 2.2.0 (Michael Radica); see licenses/exotedrf.txt.
Resolution and preset-edge arithmetic retain upstream behavior for compatibility,
including its legacy edge weights and uncertainty conventions. The reference
wrapper computes its own quadrature errors. Pixel binning uses Koala's
wavelength-first layout, matching bin_at_resolution.
"""

import numpy as np


def _half_widths(wave):
    """Half widths from neighboring centers, extrapolating the end bins."""
    spacing = np.diff(wave)
    return np.concatenate(([spacing[0]], (spacing[:-1] + spacing[1:]) / 2,
                           [spacing[-1]])) / 2


def bin_at_bins(inwave_low, inwave_up, flux, err, outwave_low, outwave_up):
    """Similar to both other binning functions, except this one will bin the flux data to preset
    bin edges.

    Parameters
    ----------
    inwave_low : array-like(float)
        Lower edge of flux wavelength bins.
    inwave_up : array-like(float)
        Upper edge of flux wavelength bins.
    flux : array-like(float)
        2D Flux to bin.
    err : array-like(float)
        2D Flux error to bin.
    outwave_low : array-like(float)
        Lower edge of bins to which to bin flux.
    outwave_up : array-like(float)
        Upper edge of bins to which to bin flux.

    Returns
    -------
    binlow : ndarray(float)
        Lower edge of wavelength bins.
    binup : ndarray(float)
        Upper edge of wavelength bins.
    binspec : ndarray(float)
        Binned flux.
    binerr : ndarray(float)
        Binner errors.
    """

    nints, ncols = np.shape(flux)
    wave = np.nanmean([inwave_low, inwave_up], axis=0)

    # Set up output arrays.
    binspec = np.zeros((nints, len(outwave_up)))
    binerr = np.zeros((nints, len(outwave_up)))

    # Loop over all input wavelength bins and bin to output bins.
    for j in range(len(outwave_up)):
        low = outwave_low[j]
        up = outwave_up[j]
        for i in range(ncols):
            w = wave[i]
            if low <= w < up:
                binspec[:, j] += flux[:, i]
                binerr[:, j] += err[:, i]

    # Broadcast to 2D.
    binlow = np.repeat(outwave_low[np.newaxis, :], nints, axis=0)
    binup = np.repeat(outwave_up[np.newaxis, :], nints, axis=0)

    return binlow, binup, binspec, binerr


def bin_at_pixel(wave, flux, error, npix):
    """Sum adjacent pixels, with wavelength on axis 0 and quadrature errors.

    Trim incomplete groups symmetrically (the extra pixel is cut from the end),
    as upstream does. Return centers, half widths, flux, and errors. A single
    output bin spans the retained input pixel edges.
    """
    wave = np.asarray(wave)
    flux, error = np.asarray(flux), np.asarray(error)
    if wave.ndim != 1 or wave.size < 2 or not np.all(np.isfinite(wave)):
        raise ValueError('Wavelengths must be a finite 1D array with at least two points.')
    if np.any(np.diff(wave) <= 0):
        raise ValueError('Wavelengths must be strictly increasing.')
    if flux.ndim not in (1, 2) or flux.shape[0] != wave.size or error.shape != flux.shape:
        raise ValueError('Flux and errors must match and have wavelength on the first axis.')
    if isinstance(npix, (bool, np.bool_)) or not isinstance(npix, (int, np.integer)) or not 1 <= npix <= wave.size:
        raise ValueError('npix must be an integer between 1 and the number of wavelengths.')
    cut = wave.size % npix
    start = cut // 2
    stop = wave.size - (cut - start)
    retained_wave = wave[start:stop]
    nbin = retained_wave.size // npix
    shape = (nbin, npix) + flux.shape[1:]
    flux_bin = np.nansum(flux[start:stop].reshape(shape), axis=1)
    err_bin = np.sqrt(np.nansum(error[start:stop].reshape(shape)**2, axis=1))
    wave_bin = np.nanmean(retained_wave.reshape(nbin, npix), axis=1)
    if nbin == 1:
        half = _half_widths(wave)
        wave_err = np.array([(wave[stop - 1] + half[stop - 1]
                              - wave[start] + half[start]) / 2])
    else:
        wave_err = _half_widths(wave_bin)
    return wave_bin, wave_err, flux_bin, err_bin


def bin_at_resolution(wave, flux, flux_err, res, method='sum'):
    """Function that bins input wavelengths and transit depths (or any other observable, like flux)
    to a given resolution "res". Can handle 1D or 2D flux arrays.

    Parameters
    ----------
    wave : array-like[float]
        Input wavelength axis. Must be 1D.
    flux : array-like[float]
        Flux values at each wavelength. Can be 1D or 2D. If 2D, the first axis must be the one
        corresponding to wavelength.
    flux_err : array-like[float]
        Errors corresponding to each flux measurement. Must be the same shape as flux.
    res : int
        Target resolution at which to bin.
    method : str
        Method to bin depths. Either "sum" or "average".

    Returns
    -------
    binned_waves : array-like[float]
        Wavelength of the given bin at the desired resolution.
    binned_werr : array-like[float]
        Half-width of the wavelength bin.
    binned_flux : array-like[float]
        Binned flux.
    binned_ferr : array-like[float]
        Error on binned flux.
    """

    wave = np.asarray(wave)
    flux, flux_err = np.asarray(flux), np.asarray(flux_err)
    if wave.ndim != 1 or wave.size < 2:
        raise ValueError('Input wavelength array must be 1D with at least two points.')
    if not np.all(np.isfinite(wave)) or np.any(wave <= 0):
        raise ValueError('Wavelengths must be finite and positive.')
    if np.any(np.diff(np.sort(wave)) <= 0):
        raise ValueError('Wavelengths must be distinct.')
    if flux.ndim not in (1, 2) or flux.shape[0] != wave.size or flux_err.shape != flux.shape:
        raise ValueError('Flux and errors must match and have wavelength on the first axis.')
    if not np.isfinite(res) or res <= 0:
        raise ValueError('Resolution must be finite and positive.')
    if method not in ('sum', 'average'):
        raise ValueError('Unknown method.')

    # Sort quantities in order of increasing wavelength.
    if np.ndim(wave) > 1:
        raise ValueError('Input wavelength array must be 1D.')
    ii = np.argsort(wave)
    waves, flux, flux_err = wave[ii], flux[ii], flux_err[ii]
    werr = _half_widths(waves)
    inwave_low, inwave_up = waves - werr, waves + werr
    # Calculate the input resolution and check that we are not trying to bin to a higher R.
    average_input_res = np.mean(waves[1:] / np.diff(waves))
    if res > average_input_res:
        raise ValueError('You are trying to bin at a higher resolution than the input.')
    else:
        print('Binning from an average resolution of R={:.0f} to R={}'
                   .format(average_input_res, res))

    # Create binned wavelength axis at resolution res.
    dlog_wl = 1.0/res
    nbins = (np.log(waves[-1]) - np.log(waves[0])) / dlog_wl
    nbins = np.around(nbins).astype(np.int64)
    if nbins < 2:
        raise ValueError('Resolution must produce at least two wavelength bins.')
    log_wave_bin = np.linspace(np.log(waves[0]), np.log(waves[-1]), nbins)
    binned_waves = np.exp(log_wave_bin)
    binned_werr = _half_widths(binned_waves)
    outwave_low = binned_waves - binned_werr
    outwave_up = binned_waves + binned_werr

    # Loop over all wavelengths in the input and bin flux and error into the
    # new wavelength grid.
    ii = 0
    for wl, wu in zip(outwave_low, outwave_up):
        first_time, count = True, 0
        current_flux = np.ones_like(flux[ii]) * np.nan
        current_ferr = np.ones_like(flux_err[ii]) * np.nan
        weight = []
        for i in range(ii, len(waves)):
            # If the wavelength is fully within the bin, append the flux and error to the current
            # bin info.
            if inwave_low[i] >= wl and inwave_up[i] < wu:
                if np.ndim(flux) == 1:
                    current_flux = np.hstack([flux[i], current_flux])
                    current_ferr = np.hstack([flux_err[i], current_ferr])
                else:
                    current_flux = np.vstack([flux[i], current_flux])
                    current_ferr = np.vstack([flux_err[i], current_ferr])
                count += 1
                weight.append(1)
            # For edge cases where one of the input bins falls on the edge of the binned wavelength
            # grid, linearly interpolate the flux into the new bins.
            # Upper edge split.
            elif inwave_low[i] < wu <= inwave_up[i]:
                inbin_width = inwave_up[i] - inwave_low[i]
                in_frac = (inwave_up[i] - wu) / inbin_width
                weight.append(in_frac)
                if np.ndim(flux) == 1:
                    current_flux = np.hstack([flux[i], current_flux])
                    current_ferr = np.hstack([flux_err[i], current_ferr])
                else:
                    current_flux = np.vstack([flux[i], current_flux])
                    current_ferr = np.vstack([flux_err[i], current_ferr])
                count += 1
            # Lower edge split.
            elif inwave_low[i] < wl <= inwave_up[i]:
                inbin_width = inwave_up[i] - inwave_low[i]
                in_frac = (wl - inwave_low[i]) / inbin_width
                weight.append(in_frac)
                if np.ndim(flux) == 1:
                    current_flux = np.hstack([flux[i], current_flux])
                    current_ferr = np.hstack([flux_err[i], current_ferr])
                else:
                    current_flux = np.vstack([flux[i], current_flux])
                    current_ferr = np.vstack([flux_err[i], current_ferr])
                count += 1
            # Since wavelengths are in increasing order, once we exit the bin completely we're done.
            if inwave_low[i] >= wu or i == len(waves)-1:
                if count != 0:
                    # If something was put into this bin, bin it using the requested method.
                    weight.append(0)
                    weight = np.array(weight)
                    if method == 'sum':
                        if np.ndim(current_flux) != 1:
                            thisflux = np.nansum(current_flux * weight[:, None], axis=0)
                        else:
                            thisflux = np.nansum(current_flux * weight, axis=0)
                        thisferr = np.sqrt(np.nansum(current_ferr**2, axis=0))
                    elif method == 'average':
                        if np.ndim(current_flux) != 1:
                            thisflux = np.nansum(current_flux * weight[:, None], axis=0)
                            thisflux /= np.nansum(weight)
                        else:
                            thisflux = np.nansum(current_flux * weight, axis=0)
                            thisflux /= np.nansum(weight)
                        thisferr = np.sqrt(np.nansum(current_ferr**2, axis=0))
                        thisferr /= np.nansum(weight)
                    else:
                        raise ValueError('Unknown method.')
                else:
                    # If nothing is in the bin (can happen if the output reslution is higher than
                    # the local input resolution), append NaNs
                    if np.ndim(flux) == 1:
                        thisflux, thisferr = np.nan, np.nan
                    else:
                        thisflux = np.ones_like(flux[0]) * np.nan
                        thisferr = np.ones_like(flux[0]) * np.nan
                # Store the binned quantities.
                if ii == 0:
                    binned_flux = thisflux
                    binned_ferr = thisferr
                else:
                    binned_flux = np.vstack([binned_flux, thisflux])
                    binned_ferr = np.vstack([binned_ferr, thisferr])
                # Move to the next bin.
                ii = i-1
                break

    # If the input was 1D, reformat to match.
    if np.ndim(flux) == 1:
        binned_flux = binned_flux[:, 0]
        binned_ferr = binned_ferr[:, 0]

    return binned_waves, binned_werr, binned_flux, binned_ferr
