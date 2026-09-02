#!/usr/bin/env python
"""
Bin spectroscopy data to match a reference wavelength grid (e.g., NIRSpec/PRISM native resolution).

This module provides utilities to bin any instrument's data to match the exact wavelength bins
from a reference fit.

Usage:
    from bin_to_reference_grid import load_reference_grid, bin_to_reference_grid

    # Load reference grid from PRISM native fit
    wavelengths, wavelength_errs = load_reference_grid('prism_native_bestfit_params.csv')

    # Bin NIRISS data to match PRISM grid
    wl_binned, wl_err_binned, flux_binned, err_binned = bin_to_reference_grid(
        niriss_wavelengths, niriss_flux, niriss_err, wavelengths, wavelength_errs
    )
"""

import numpy as np
import pandas as pd
from exotedrf.stage4 import bin_at_bins


def load_reference_grid(reference_params_csv):
    """
    Load wavelength grid from a reference fit's bestfit_params.csv file.

    Parameters
    ----------
    reference_params_csv : str
        Path to the reference bestfit_params.csv file (e.g., from NIRSpec/PRISM native fit)

    Returns
    -------
    wavelengths : ndarray
        Central wavelength of each bin
    wavelength_errs : ndarray
        Half-width of each wavelength bin
    """
    df = pd.read_csv(reference_params_csv)
    wavelengths = df['wavelength'].values
    wavelength_errs = df['wavelength_err'].values

    print(f"Loaded reference grid: {len(wavelengths)} bins")
    print(f"  Wavelength range: {wavelengths.min():.4f} - {wavelengths.max():.4f} μm")
    print(f"  Median resolution: R ~ {np.median(wavelengths / (2 * wavelength_errs)):.0f}")

    return wavelengths, wavelength_errs


def bin_to_reference_grid(input_wavelengths, input_flux, input_err,
                          ref_wavelengths, ref_wavelength_errs,
                          trim_to_overlap=True, method='average'):
    """
    Bin input spectroscopy data to match a reference wavelength grid.

    This function uses exotedrf.stage4.bin_at_bins to bin the input data to the exact
    wavelength bins defined by the reference grid.

    Parameters
    ----------
    input_wavelengths : ndarray, shape (n_input_wavelengths,)
        Input wavelength array (1D)
    input_flux : ndarray, shape (n_input_wavelengths, n_times)
    input_err : ndarray, same shape as input_flux
        Input flux errors
    ref_wavelengths : ndarray
        Reference wavelength bin centers
    ref_wavelength_errs : ndarray
        Reference wavelength bin half-widths
    trim_to_overlap : bool, optional
        If True, only return bins that overlap with input wavelength range
    method : str, optional
        Binning method - 'sum' or 'average' (default: 'average')

    Returns
    -------
    wavelengths_out : ndarray
        Output wavelength bin centers
    wavelength_errs_out : ndarray
        Output wavelength bin half-widths
    flux_out : ndarray, shape (n_output_wavelengths, n_times)
        Binned flux
    err_out : ndarray, same shape as flux_out
        Binned errors

    Notes
    -----
    The input flux can be in either (wavelength, time) or (time, wavelength) format.
    The function will detect the format and ensure output is always (wavelength, time).
    """
  
    # Ensure wavelengths is 1D
    input_wavelengths = np.asarray(input_wavelengths).ravel()

    # Calculate input wavelength bin edges
    # Assume input bins are uniform or calculate from neighboring points
    input_werr = np.diff(input_wavelengths) / 2
    input_werr = np.append(input_werr, input_werr[-1])
    input_werr = np.insert(input_werr, 0, input_werr[0])
    input_werr = np.convolve(input_werr, [0.5, 0.5], mode='valid')


    input_wave_low = input_wavelengths - input_werr
    input_wave_up = input_wavelengths + input_werr

    # Calculate reference wavelength bin edges
    ref_wave_low = ref_wavelengths - ref_wavelength_errs
    ref_wave_up = ref_wavelengths + ref_wavelength_errs

    # Optionally trim to overlapping region
    if trim_to_overlap:
        overlap_min = max(input_wavelengths.min(), ref_wavelengths.min())
        overlap_max = min(input_wavelengths.max(), ref_wavelengths.max())

        mask = (ref_wavelengths >= overlap_min) & (ref_wavelengths <= overlap_max)
        ref_wavelengths = ref_wavelengths[mask]
        ref_wavelength_errs = ref_wavelength_errs[mask]
        ref_wave_low = ref_wave_low[mask]
        ref_wave_up = ref_wave_up[mask]

        print(f"Trimming to overlap region: {overlap_min:.4f} - {overlap_max:.4f} μm")
        print(f"  Output bins: {len(ref_wavelengths)}")

    # Check if there's any overlap
    if len(ref_wavelengths) == 0:
        raise ValueError("No overlap between input and reference wavelength ranges!")

    # Use bin_at_bins from exotedrf
    # Note: bin_at_bins expects flux shape (n_times, n_wavelengths) for the input
    # and returns shape (n_times, n_output_wavelengths)

    input_flux_2d = input_flux.T
    input_err_2d = input_err.T

    print(f"Binning {len(input_wavelengths)} input bins to {len(ref_wavelengths)} reference bins...")

    # Call bin_at_bins
    # Call bin_at_bins for flux (we'll recalculate errors properly)
    binned_wave_low, binned_wave_up, binned_flux, _ = bin_at_bins(
      input_wave_low, input_wave_up,
      input_flux_2d, input_err_2d,
      ref_wave_low, ref_wave_up
  )

  # Recalculate errors properly using quadrature
    binned_err = np.zeros_like(binned_flux)
    for j in range(len(ref_wavelengths)):
        low = ref_wave_low[j]
        up = ref_wave_up[j]
        mask = (input_wavelengths >= low) & (input_wavelengths < up)
        if np.any(mask):
            # Proper quadrature: sqrt(sum of squares)
            binned_err[:, j] = np.sqrt(np.nansum(input_err_2d[:, mask]**2, axis=1))

  # bin_at_bins returns 2D arrays even for wavelengths, take first row
    binned_wave_low = binned_wave_low[0, :]
    binned_wave_up = binned_wave_up[0, :]

  # Calculate output wavelengths and errors
    wavelengths_out = (binned_wave_low + binned_wave_up) / 2
    wavelength_errs_out = (binned_wave_up - binned_wave_low) / 2

  # Transpose back to (wavelength, time)
    if binned_flux.ndim == 2:
        binned_flux = binned_flux.T
        binned_err = binned_err.T
    
    print(f"\n=== AFTER AVERAGING ===")
    print(f"binned_flux range: {np.min(binned_flux):.1f} - {np.max(binned_flux):.1f}")
    print(f"binned_err range: {np.min(binned_err):.6f} - {np.max(binned_err):.6f}")
    print(f"binned_err / binned_flux: {np.median(binned_err / binned_flux):.6f}")
    print(f"Binning complete!")
    print(f"  Output shape: {binned_flux.shape}")
    # Filter out bins with no valid data
    valid_mask = (np.nansum(binned_flux, axis=1) > 0) & (np.nansum(binned_err, axis=1) > 0)
    n_removed = np.sum(~valid_mask)
    if n_removed > 0:
        print(f"  Removing {n_removed} bins with no valid data")
        wavelengths_out = wavelengths_out[valid_mask]
        wavelength_errs_out = wavelength_errs_out[valid_mask]
        binned_flux = binned_flux[valid_mask]
        binned_err = binned_err[valid_mask]

    print(f"Binning complete!")
    print(f"  Output shape: {binned_flux.shape}")

    return wavelengths_out, wavelength_errs_out, binned_flux, binned_err

def bin_to_reference_grid_simple(input_wavelengths, input_flux, input_err,
                                 reference_params_csv, **kwargs):
    """
    Convenience function that loads reference grid and bins in one step.

    Parameters
    ----------
    input_wavelengths : ndarray
        Input wavelength array
    input_flux : ndarray
        Input flux array
    input_err : ndarray
        Input flux errors
    reference_params_csv : str
        Path to reference bestfit_params.csv file
    **kwargs : dict
        Additional arguments passed to bin_to_reference_grid()

    Returns
    -------
    wavelengths_out, wavelength_errs_out, flux_out, err_out
        Binned data matching reference grid
    """
    ref_wavelengths, ref_wavelength_errs = load_reference_grid(reference_params_csv)

    return bin_to_reference_grid(
        input_wavelengths, input_flux, input_err,
        ref_wavelengths, ref_wavelength_errs,
        **kwargs
    )


# Example usage
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Bin spectroscopy data to match a reference wavelength grid',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # In Python:
  from bin_to_reference_grid import bin_to_reference_grid_simple

  # Load your data (e.g., NIRISS)
  wavelengths, flux, err = load_my_data()

  # Bin to match PRISM native grid
  wl_out, wl_err_out, flux_out, err_out = bin_to_reference_grid_simple(
      wavelengths, flux, err,
      'output/planet_NIRSPEC_PRISM_Rnative_bestfit_params.csv'
  )
        """
    )

    parser.add_argument('reference_csv', help='Reference bestfit_params.csv file')
    args = parser.parse_args()

    # Just load and show info
    wavelengths, wavelength_errs = load_reference_grid(args.reference_csv)

    print("\nReference grid loaded successfully!")
    print("Use this grid in your Python code with:")
    print(f"  from bin_to_reference_grid import load_reference_grid, bin_to_reference_grid")
    print(f"  ref_wl, ref_wl_err = load_reference_grid('{args.reference_csv}')")
    print(f"  wl_out, wl_err_out, flux_out, err_out = bin_to_reference_grid(")
    print(f"      your_wavelengths, your_flux, your_err, ref_wl, ref_wl_err)")

