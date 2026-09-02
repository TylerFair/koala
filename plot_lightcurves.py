#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from pathlib import Path
import re
from scipy import stats


# Configuration
fits_dir = '/scratch/midway3/tfairnington/FITS'
output_dir = '/scratch/midway3/tfairnington/lightcurve_plots'
os.makedirs(output_dir, exist_ok=True)

# Manual masking rules
MASKING_RULES = {
    'Kepler-12_NRS1_PRISM_V2': {
        'flux_cut': 0.98,
        'description': 'Mask all flux < 0.98'
    },
    'LTT-9779_NRS1_G395H': {
        'flux_cut': 0.9965,
        'time_cut_before': 1000,
        'description': 'Flux cut < 0.9965, drop data before int 1000'
    },
    'LTT-9779_NRS2_G395H': {
        'flux_cut': 0.997,
        'description': 'Mask flux < 0.997'
    },
    'TOI-3757_NRS1_PRISM_V1': {
        'flux_cut': 0.995,
        'flux_cut_after': 15000,
        'description': 'Flux cut < 0.995 from int 15000 onwards'
    },
    'TOI-3757_NRS1_PRISM_V2': {
        'flux_cut': 0.96,
        'description': 'Mask flux < 0.96'
    },
    'TOI-4010_NRS1_PRISM': {
        'time_cut_after': 17500,
        'description': 'Cut data after int 17500'
    },
    'WASP-19_NRS1_PRISM': {
        'time_range_cut': (12000, 12500),
        'description': 'Mask integrations 12000-12500'
    },
    'WASP-39_NRS1_PRISM': {
            'flux_cut_after': 20000,
        'description': 'Mask integrations post 20000'
    }

}

def apply_masking(wl_flux, bjd_times, detector_key):
    """Apply manual masking rules to light curve data."""
    mask = np.ones(len(wl_flux), dtype=bool)
    
    if detector_key not in MASKING_RULES:
        return wl_flux, bjd_times, mask, None
    
    rules = MASKING_RULES[detector_key]
    description = rules.get('description', '')
    
    # Time cut before
    if 'time_cut_before' in rules:
        mask[:rules['time_cut_before']] = False
        print(f"  Applied: masking before integration {rules['time_cut_before']}")
    
    # Time cut after
    if 'time_cut_after' in rules:
        mask[rules['time_cut_after']:] = False
        print(f"  Applied: masking after integration {rules['time_cut_after']}")
    
    # Time range cut
    if 'time_range_cut' in rules:
        start, end = rules['time_range_cut']
        mask[start:end] = False
        print(f"  Applied: masking integrations {start}-{end}")
    
    # Flux cut (global or conditional)
    if 'flux_cut' in rules:
        flux_threshold = rules['flux_cut']
        
        if 'flux_cut_after' in rules:
            # Only apply flux cut after certain integration
            cut_start = rules['flux_cut_after']
            flux_mask = wl_flux >= flux_threshold
            flux_mask[:cut_start] = True  # Don't mask before cut_start
            mask &= flux_mask
            n_masked = np.sum(~flux_mask[cut_start:])
            print(f"  Applied: flux cut < {flux_threshold} after int {cut_start} (masked {n_masked} points)")
        else:
            # Global flux cut
            flux_mask = wl_flux >= flux_threshold
            mask &= flux_mask
            n_masked = np.sum(~flux_mask)
            print(f"  Applied: flux cut < {flux_threshold} (masked {n_masked} points)")
    
    # Create masked arrays
    masked_flux = wl_flux.copy()
    masked_flux[~mask] = np.nan
    
    masked_times = bjd_times.copy()
    masked_times[~mask] = np.nan
    
    return masked_flux, masked_times, mask, description

def determine_adaptive_bin_size(n_points, target_binned_points=100):
    """
    Determine adaptive bin size based on number of data points.
    - Sparse data (< 500 points): bin by 1-3
    - Medium data (500-2000 points): bin by 5-10
    - Dense data (> 2000 points): bin by 10-50
    """
    if n_points < 500:
        # Sparse: minimal binning
        bin_size = max(1, n_points // target_binned_points)
        bin_size = min(bin_size, 3)
    elif n_points < 2000:
        # Medium: moderate binning
        bin_size = max(5, n_points // target_binned_points)
        bin_size = min(bin_size, 10)
    else:
        # Dense: aggressive binning
        bin_size = max(10, n_points // target_binned_points)
        bin_size = min(bin_size, 50)
    
    return int(bin_size)

def bin_lightcurve(times, data, bin_size=10):
    """Bin down the light curve, handling NaNs properly."""
    # Remove NaNs for binning
    valid_mask = ~np.isnan(data) & ~np.isnan(times)
    valid_times = times[valid_mask]
    valid_data = data[valid_mask]
    
    if len(valid_data) == 0:
        return np.array([]), np.array([])
    
    n_bins = len(valid_data) // bin_size
    if n_bins == 0:
        return np.array([]), np.array([])
    
    binned = np.zeros(n_bins)
    binned_times = np.zeros(n_bins)
    
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size
        binned[i] = np.nanmean(valid_data[start_idx:end_idx])
        binned_times[i] = np.mean(valid_times[start_idx:end_idx])
    
    return binned_times, binned

def plot_white_light(wl_flux, bjd_times, planet_name, detector, output_path, mask_description=None):
    """Plot full and adaptively binned white light curves."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Count valid points
    valid_points = np.sum(~np.isnan(wl_flux))
    
    # Full resolution (showing masked points)
    ax.plot(bjd_times, wl_flux, 'k.', alpha=0.5, markersize=3, label=f'Full res ({valid_points} points)')
    ax.set_ylabel('Normalized Flux')
    
    title = f'{planet_name} - {detector} - White Light Curve'
    if mask_description:
        title += f'\n[{mask_description}]'
    ax.set_title(title)
    
    # Adaptive binning
    bin_size = determine_adaptive_bin_size(valid_points)
    binned_times, binned = bin_lightcurve(bjd_times, wl_flux, bin_size=bin_size)
    
    if len(binned) > 0:
        ax.scatter(binned_times, binned, c='royalblue', alpha=1, s=50, zorder=3,
                  label=f'Binned (bin={bin_size}, {len(binned)} points)')
    
    ax.set_xlabel('BJD Time')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def stitch_kepler167_segments(fits_files):
    """Stitch together Kepler-167 PRISM segments and return segment boundaries."""
    # Sort by segment number
    segments = []
    for f in fits_files:
        if 'PRISM' in f:
            match = re.search(r'PRISM(\d+)', f)
            if match:
                segment_num = int(match.group(1))
                segments.append((segment_num, f))
    
    segments.sort()
    
    # Load and concatenate all segments
    all_flux = []
    all_times = []
    segment_lengths = []
    for seg_num, filepath in segments:
        try:
            with fits.open(filepath) as spec:
                spec_data = spec[3].data  # (n_time, n_wavelength)
                bjd_times = spec[5].data  # BJD times for NIRSpec
                all_flux.append(spec_data)
                all_times.append(bjd_times)
                segment_lengths.append(spec_data.shape[0])
                print(f"  Loaded segment {seg_num}: shape {spec_data.shape}, time range {bjd_times[0]:.4f}-{bjd_times[-1]:.4f}")
        except Exception as e:
            print(f"  Warning: Could not load segment {seg_num}: {e}")
    
    if not all_flux:
        return None, None, None
    
    # Concatenate along time axis
    stitched_flux = np.concatenate(all_flux, axis=0)
    stitched_times = np.concatenate(all_times, axis=0)
    
    print(f"  Stitched total shape: {stitched_flux.shape}, time range {stitched_times[0]:.4f}-{stitched_times[-1]:.4f}")
    return stitched_flux, stitched_times, segment_lengths

def process_niriss(fits_file, planet_name, version='', grating=''):
    """Process NIRISS data."""
    try:
        with fits.open(fits_file) as spec:
            order1 = spec[3].data[:, 5:-5]  # Order 1 spectra
            bjd_times = spec[9].data  # BJD times for NIRISS
        
        # White light curve: sum over wavelengths
        wlc = np.nansum(order1, axis=1)
        # Normalize by out-of-transit baseline
        base_idx = np.arange(100)
        wl_flux = wlc / np.nanmedian(wlc[base_idx], axis=0)
        
        # Build detector key and apply masking
        version_str = f'_{version}' if version else ''
        grating_str = f'_{grating}' if grating else ''
        detector_key = f'{planet_name}_NIRISS{grating_str}{version_str}'.replace('__', '_')
        
        wl_flux, bjd_times, mask, description = apply_masking(wl_flux, bjd_times, detector_key)
        
        output_path = os.path.join(output_dir, f'{detector_key}_whitelight.png')
        plot_white_light(wl_flux, bjd_times, planet_name, f'NIRISS{grating_str}{version_str}', 
                        output_path, description)
        
    except Exception as e:
        print(f"Error processing NIRISS {planet_name}{version}: {e}")

def process_nirspec(nrs_file, planet_name, detector, version='', grating=''):
    """Process NIRSpec data."""
    try:
        with fits.open(nrs_file) as spec:
            spec_data = spec[3].data  # (n_time, n_wavelength)
            bjd_times = spec[5].data  # BJD times for NIRSpec
        
        # White light curve: sum over wavelengths
        wlc = np.nanmean(spec_data, axis=1)

        # Normalize by first 100 integrations
        base_idx = np.arange(100).astype(int)
        wl_flux = wlc / np.nanmedian(wlc[base_idx], axis=0)
        
        # Build detector key and apply masking
        version_str = f'_{version}' if version else ''
        grating_str = f'_{grating}' if grating else ''
        detector_key = f'{planet_name}_{detector}{grating_str}{version_str}'.replace('__', '_')
        
        wl_flux, bjd_times, mask, description = apply_masking(wl_flux, bjd_times, detector_key)
            
        output_path = os.path.join(output_dir, f'{detector_key}_whitelight.png')
        plot_white_light(wl_flux, bjd_times, planet_name, f'{detector}{grating_str}{version_str}', 
                        output_path, description)
        
    except Exception as e:
        print(f"Error processing {detector} {planet_name}{version}: {e}")

def process_kepler167_stitched(fits_files, planet_name):
    """Process stitched Kepler-167 PRISM data with 4th segment correction after normalization."""
    try:
        stitched_flux, stitched_times, segment_lengths = stitch_kepler167_segments(fits_files)
        
        if stitched_flux is None:
            print(f"  Could not stitch Kepler-167 segments")
            return
        
        # White light curve: sum over wavelengths
        wlc = np.nanmean(stitched_flux, axis=1)
        
        # Normalize by first 100 integrations
        base_idx = np.arange(100).astype(int)
        wl_flux = wlc / np.nanmedian(wlc[base_idx])
        
        # NOW apply correction to 4th segment after normalization (shift down by 0.0175)
        if segment_lengths and len(segment_lengths) >= 4:
            seg4_start = sum(segment_lengths[:3])
            seg4_end = sum(segment_lengths[:4])
            print(f"  Applying -0.0175 correction to segment 4 after normalization (integrations {seg4_start}-{seg4_end})")
            wl_flux[seg4_start:seg4_end] -= 0.0175
        
        detector_key = 'Kepler-167_NRS1_PRISM_stitched'
        description = '4th segment shifted down by 0.0175 (after normalization)'
        
        output_path = os.path.join(output_dir, f'{detector_key}_whitelight.png')
        plot_white_light(wl_flux, stitched_times, planet_name, 'NRS1_PRISM (stitched)', 
                        output_path, description)
        
    except Exception as e:
        print(f"Error processing stitched Kepler-167: {e}")

def main():
    # Get all FITS files
    fits_files = list(Path(fits_dir).glob('*_box_spectra_fullres*.fits'))
    
    # Group by planet, detector, grating, and version
    planets = {}
    for f in fits_files:
        filename = f.name
        
        # Extract grating
        grating = ''
        for g in ['G395H', 'G395M', 'G140H', 'G235H', 'PRISM']:
            if g in filename:
                grating = g
                break
        
        # Extract version if present
        version = ''
        if '_V1.fits' in filename:
            version = 'V1'
        elif '_V2.fits' in filename:
            version = 'V2'
        
        # Extract segment for PRISM
        segment = ''
        if 'PRISM' in filename:
            match = re.search(r'PRISM(\d+)', filename)
            if match:
                segment = match.group(1)
        
        # Determine planet name
        planet = filename.split('_')[0]
        
        # Determine detector
        if '_nrs1_' in filename:
            detector = 'NRS1'
        elif '_nrs2_' in filename:
            detector = 'NRS2'
        else:
            detector = 'NIRISS'
        
        # Store file info
        if planet not in planets:
            planets[planet] = []
        
        planets[planet].append({
            'file': str(f),
            'detector': detector,
            'version': version,
            'grating': grating,
            'segment': segment
        })
    
    # Process each planet's observations
    for planet_name in sorted(planets.keys()):
        print(f"\nProcessing {planet_name}...")
        
        # Special handling for Kepler-167 PRISM segments
        if planet_name == 'Kepler-167':
            prism_segments = [obs['file'] for obs in planets[planet_name] 
                            if 'PRISM' in obs['grating'] and obs['segment']]
            if prism_segments:
                print(f"  Found {len(prism_segments)} PRISM segments to stitch")
                process_kepler167_stitched(prism_segments, planet_name)
                continue
        
        # Regular processing for all other observations
        for obs in planets[planet_name]:
            if obs['detector'] == 'NIRISS':
                process_niriss(obs['file'], planet_name, obs['version'], obs['grating'])
            else:
                process_nirspec(obs['file'], planet_name, obs['detector'], obs['version'], obs['grating'])

if __name__ == '__main__':
    main()
