#!/usr/bin/env python
"""
Create a reference wavelength grid from instrument data WITHOUT fitting.

Just unpacks the wavelengths and saves them to a CSV template.

Usage:
    python create_reference_grid.py NIRSPEC/PRISM prism_data.fits prism_template.csv
    python create_reference_grid.py NIRISS/SOSS niriss_data.fits niriss_template.csv --order 1
"""

import argparse
import pandas as pd
import numpy as np
import new_unpack


def create_reference_grid(instrument, fits_file, output_csv, order=None, nrs=None):
    """
    Extract wavelengths from FITS file and save as reference grid template.

    Parameters
    ----------
    instrument : str
        Instrument name (e.g., 'NIRSPEC/PRISM', 'NIRISS/SOSS', 'MIRI/LRS')
    fits_file : str
        Path to FITS file
    output_csv : str
        Output CSV file path
    order : int, optional
        NIRISS order (1 or 2)
    nrs : int, optional
        NIRSpec NRS (1 or 2)
    """

    print(f"Extracting wavelength grid from {instrument}")
    print(f"FITS file: {fits_file}")

    # Unpack wavelengths based on instrument
    if instrument in ['NIRSPEC/PRISM', 'NIRSPEC/G395H', 'NIRSPEC/G395M']:
        wavelengths, wavelength_errs, time, flux, flux_err = new_unpack.unpack_nirspec_exoted(
            fits_file, instrument, trim_start=None, trim_end=None
        )
    elif instrument == 'NIRISS/SOSS':
        if order is None:
            raise ValueError("Must specify --order (1 or 2) for NIRISS/SOSS")
        wavelengths, wavelength_errs, time, flux, flux_err = new_unpack.unpack_niriss_exoted(
            fits_file, order, trim_start=None, trim_end=None
        )
    elif instrument == 'MIRI/LRS':
        wavelengths, wavelength_errs, time, flux, flux_err = new_unpack.unpack_miri_exoted(
            fits_file, trim_start=None, trim_end=None
        )
    else:
        raise ValueError(f"Unknown instrument: {instrument}")

    # Remove NaN wavelengths
    mask = np.isfinite(wavelengths) & np.isfinite(wavelength_errs)
    wavelengths = wavelengths[mask]
    wavelength_errs = wavelength_errs[mask]

    # Create DataFrame
    df = pd.DataFrame({
        'wavelength': wavelengths,
        'wavelength_err': wavelength_errs
    })

    # Save to CSV
    df.to_csv(output_csv, index=False)

    print(f"\n✅ Reference grid saved to: {output_csv}")
    print(f"   Number of bins: {len(wavelengths)}")
    print(f"   Wavelength range: {wavelengths.min():.4f} - {wavelengths.max():.4f} μm")

    # Calculate resolution
    bin_widths = 2 * wavelength_errs
    resolutions = wavelengths / bin_widths
    print(f"   Median resolution: R ~ {np.median(resolutions):.0f}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Create reference wavelength grid template from FITS file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create PRISM native grid template
  python create_reference_grid.py NIRSPEC/PRISM prism_data.fits prism_native_template.csv

  # Create NIRISS Order 1 native grid template
  python create_reference_grid.py NIRISS/SOSS niriss_data.fits niriss_o1_template.csv --order 1

  # Create NIRISS Order 2 native grid template
  python create_reference_grid.py NIRISS/SOSS niriss_data.fits niriss_o2_template.csv --order 2

  # Create MIRI/LRS native grid template
  python create_reference_grid.py MIRI/LRS miri_data.fits miri_template.csv

Then in your config.yaml, use:
  resolution:
    high: 'reference'
    low: 40
    reference_grid: 'prism_native_template.csv'
        """
    )

    parser.add_argument('instrument',
                       choices=['NIRSPEC/PRISM', 'NIRSPEC/G395H', 'NIRSPEC/G395M',
                               'NIRISS/SOSS', 'MIRI/LRS'],
                       help='Instrument name')
    parser.add_argument('fits_file', help='Path to FITS file')
    parser.add_argument('output_csv', help='Output CSV template file')
    parser.add_argument('--order', type=int, choices=[1, 2],
                       help='NIRISS order (1 or 2)')
    parser.add_argument('--nrs', type=int, choices=[1, 2],
                       help='NIRSpec NRS (1 or 2) - usually not needed')

    args = parser.parse_args()

    create_reference_grid(
        args.instrument,
        args.fits_file,
        args.output_csv,
        order=args.order,
        nrs=args.nrs
    )


if __name__ == '__main__':
    main()
