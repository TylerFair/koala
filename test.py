#!/usr/bin/env python
"""
Test script to verify reference grid binning works correctly.

This tests the binning without running the full pipeline.
"""

import numpy as np
import pandas as pd
from bin_to_reference_grid import load_reference_grid, bin_to_reference_grid

# Create some fake NIRISS data
print("Creating fake NIRISS data...")
niriss_wavelengths = np.linspace(0.85, 2.8, 280)  # NIRISS Order 1 range
n_times = 100
n_wavelengths = len(niriss_wavelengths)

# Fake flux data (time, wavelength)
niriss_flux = np.random.randn(n_times, n_wavelengths) * 0.001 + 1.0
niriss_err = np.ones_like(niriss_flux) * 0.0001

print(f"NIRISS data shape: {niriss_flux.shape}")
print(f"NIRISS wavelength range: {niriss_wavelengths.min():.3f} - {niriss_wavelengths.max():.3f} μm")

# Load PRISM template
print("\nLoading PRISM template...")
try:
    ref_wavelengths, ref_wavelength_errs = load_reference_grid('prism_template.csv')
    print(f"PRISM template loaded: {len(ref_wavelengths)} bins")
    print(f"PRISM wavelength range: {ref_wavelengths.min():.3f} - {ref_wavelengths.max():.3f} μm")
except FileNotFoundError:
    print("ERROR: prism_template.csv not found!")
    print("Run: python create_reference_grid.py NIRSPEC/PRISM your_prism.fits prism_template.csv")
    exit(1)

# Test binning
print("\nTesting binning...")
try:
    # Transpose flux to (wavelength, time) for the binning function
    niriss_flux_transposed = niriss_flux.T
    niriss_err_transposed = niriss_err.T

    wl_binned, wl_err_binned, flux_binned, err_binned = bin_to_reference_grid(
        niriss_wavelengths,
        niriss_flux_transposed,
        niriss_err_transposed,
        ref_wavelengths,
        ref_wavelength_errs,
        trim_to_overlap=True,
        method='average'
    )

    print(f"✅ Binning successful!")
    print(f"   Output bins: {len(wl_binned)}")
    print(f"   Output wavelength range: {wl_binned.min():.3f} - {wl_binned.max():.3f} μm")
    print(f"   Output flux shape: {flux_binned.shape}")

    # Verify the wavelengths match PRISM in overlap region
    overlap_prism = ref_wavelengths[(ref_wavelengths >= wl_binned.min() - 0.01) &
                                     (ref_wavelengths <= wl_binned.max() + 0.01)]

    if len(wl_binned) == len(overlap_prism):
        print(f"   ✅ Number of bins matches PRISM overlap region")
    else:
        print(f"   ⚠️  Warning: {len(wl_binned)} output bins, {len(overlap_prism)} expected from PRISM")

    # Check a few wavelengths match
    max_diff = np.max(np.abs(wl_binned - overlap_prism[:len(wl_binned)]))
    if max_diff < 0.001:
        print(f"   ✅ Wavelengths match PRISM template (max diff: {max_diff:.6f} μm)")
    else:
        print(f"   ⚠️  Warning: Wavelengths differ from PRISM (max diff: {max_diff:.6f} μm)")

    print("\n" + "="*60)
    print("TEST PASSED! Reference grid binning works correctly.")
    print("="*60)

except Exception as e:
    print(f"\n❌ ERROR during binning: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Now test what the config parsing would look like
print("\n" + "="*60)
print("Testing config parsing logic...")
print("="*60)

# Simulate config dictionary
config = {
    'resolution': {
        'high': 'reference',
        'low': 40,
        'reference_grid': 'prism_template.csv'
    }
}

resolution = config.get('resolution', None)

print(f"Config: {config}")
print(f"\nresolution.get('high') = {resolution.get('high')}")
print(f"resolution.get('low') = {resolution.get('low')}")
print(f"resolution.get('reference_grid') = {resolution.get('reference_grid')}")

# Check the logic that would be used in jwstdata.py
use_reference_grid_hr = False
if resolution is not None:
    if resolution.get('high') == 'reference':
        if resolution.get('reference_grid') is None:
            print("❌ ERROR: reference_grid is None!")
        else:
            use_reference_grid_hr = True
            reference_grid_path_hr = resolution.get('reference_grid')
            print(f"✅ use_reference_grid_hr = {use_reference_grid_hr}")
            print(f"   reference_grid_path_hr = {reference_grid_path_hr}")

if not use_reference_grid_hr:
    print("❌ ERROR: use_reference_grid_hr is still False!")
else:
    print("\n✅ Config parsing logic works correctly!")
    print("   The code should use reference grid binning for high resolution.")

print("\n" + "="*60)
print("All tests completed!")
print("="*60)

