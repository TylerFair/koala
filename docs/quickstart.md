# Fit one dataset

Put an extracted box-spectrum FITS file in `FITS/`, then save this as `config.yaml`:

```yaml
planet: {name: WASP-39, period: 4.05528043, duration: 0.11693087, t0: 59787.055, b: 0.4498, rprs: 0.1457}
stellar: {feh: 0.04, teff: 5509, logg: 4.22, teff_sigma: 28, logg_sigma: 0.07, feh_sigma: 0.02, ld_model: stagger, ld_data_path: ../exotic_ld_data}
instrument: NIRISS/SOSS
order: 1
path: .
input_dir: FITS
output_dir: WASP-39_RESULTS
fits_file: WASP-39_box_spectra_fullres.fits
resolution: {high: reference, low: 20, reference_grid: prism_template.csv}
flags: {detrending_type: linear, ld_profile: power2, ld_prior: informed, need_lowres: true, spectro_sampler: independent_nuts, spectro_mass_matrix: laplace}
outlier_clip: {whitelight_sigma: 5, spectroscopic_sigma: 5}
host_device: gpu
```

Run all stages:

```bash
export JAX_ENABLE_X64=1
python fit_jwst.py -c config.yaml
```

The output directory contains numbered white-light plots and CSV time series, `chunks/` checkpoints and diagnostics, low- and high-resolution transmission-spectrum CSV files, detailed best-fit parameter and light-curve tables, noise-binning CSV/PNG products, and summary PNGs. Exact stems include the target, instrument, detector or order, and resolution.

The spectrum CSV columns are `wavelength`, `wavelength_err`, and, for planet zero, `depth00`, `depth_err00`, `depth_ppm00`, and `depth_err_ppm00`.

```python
import pandas as pd
import matplotlib.pyplot as plt

s = pd.read_csv("WASP-39_RESULTS/WASP-39_NIRISS_SOSS_order1_R20.csv")
plt.errorbar(s.wavelength, s.depth_ppm00, xerr=s.wavelength_err,
             yerr=s.depth_err_ppm00, fmt=".")
plt.xlabel("Wavelength [micron]")
plt.ylabel("Transit depth [ppm]")
plt.show()
```

