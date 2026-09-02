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

The output directory contains numbered white-light plots and CSV time series, `chunks/` checkpoints and diagnostics, low- and high-resolution transmission-spectrum CSV files, detailed best-fit parameter and light-curve tables, noise-binning CSV/PNG products, and summary PNGs. Exact stems include the target, instrument, detector or order, and resolution. The spectrum CSV columns are `wavelength`, `wavelength_err`, and, for planet zero, `depth00`, `depth_err00`, `depth_ppm00`, and `depth_err_ppm00`.

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

## Before running

Confirm that `FITS/WASP-39_box_spectra_fullres.fits` exists relative to the repository. Confirm that `../exotic_ld_data` resolves from the run directory. Change `output_dir` to a new directory for each scientifically distinct setup.

The supplied orbital time and FITS time array must use the same convention.

## What happens in order

The fitter reads the extracted spectral time series. It applies configured time and wavelength masks. It sums a white-light curve and performs an initial numerical optimization.

It samples the white-light posterior and checks ESS and divergences. It writes the posterior-median geometry handoff. It constructs R=20 wavelength channels because `resolution.low: 20`.

It samples those channels in independent GPU lanes. It constructs the `reference_grid` channels. It samples the high-resolution chunks and writes each checkpoint immediately.

It concatenates accepted chunks in wavelength order. It writes spectra, parameter tables, time-series tables, and plots.

## First files to inspect

Open `00_*_preopt_init_check.png` first. A misplaced transit usually indicates an inconsistent `t0`. Open `11_*_whitelightmodel.png` and `12_*_whitelightresidual.png` next.

The baseline should be described outside transit without obvious coherent structure. Open `14_*_whitelightdetrended.png` to inspect the transit after subtracting the fitted trend. Open `15_*_whitelight_summary.png` for the combined overview.

Only then inspect `24_*_R20_spectrum_00.png` and the high-resolution spectrum.

## Check the log

```text
Fitting whitelight for outliers and bestfit parameters
Building jaxoplanet whitelight model: detrend='linear', ld='informed'
Checkpoint directory: .../chunks
  chunk 0:... - COMPUTING
  chunk 0:... - SAVED checkpoint
Transmission spectroscopy data saved to ...csv
Analysis complete!
```

The builder line confirms the requested trend and LD prior. The checkpoint line makes an interrupted run resumable. Read all gate messages; completion alone is not a convergence statement.

## Inspect values numerically

```python
from pathlib import Path
import pandas as pd

out = Path("WASP-39_RESULTS")
white = next(out.glob("*_whitelight_timeseries.csv"))
print(pd.read_csv(white).describe())

spectra = [p for p in out.glob("*_R*.csv") if "noisebin" not in p.name]
for path in sorted(spectra):
    frame = pd.read_csv(path)
    if "depth_ppm00" in frame:
        print(path.name, len(frame), frame.depth_ppm00.median())
```

The light-curve table should have finite flux, uncertainty, and model columns. The spectrum length should agree with the requested grid.

## Resume

If the process stops, run the identical command again. Matching checkpoints load automatically. The fingerprint includes data arrays, priors, model settings, and sampler controls.

Changing any of them creates a distinct checkpoint family.

## Next choices

Use the [Concepts](concepts.md) page to understand the staged fit. Use [Limb darkening](guides/limb_darkening.md) to choose a prior. Use [Systematics trends](guides/trends.md) to match visit behavior.

Use [Samplers](guides/samplers.md) to interpret gate and swap messages. Use [Outputs](guides/outputs.md) for every table column.

## Common first-run problems

**Missing config argument:** use `-c config.yaml`. **FITS file not found:** remember that `fits_file` is joined to `path/input_dir`. **No LD data:** point `stellar.ld_data_path` to the Stagger data tree.

**GPU out of memory:** reduce `flags.vmap_chunk`. **No high-resolution files:** check `analysis_stage` and `resolution.high`. **No low-resolution files:** set `need_lowres: true`.

**Gate failure:** inspect the reported channel before changing sampler thresholds.
