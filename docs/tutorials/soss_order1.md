# NIRISS/SOSS order 1

Start from the checked-in WASP-39 configuration:

```bash
cp configs_fiducial_stellarinformed/WASP-39_soss_order1_config.yaml wasp39.yaml
python fit_jwst.py -c wasp39.yaml
```

Its defining entries are:

```yaml
instrument: NIRISS/SOSS
order: 1
fits_file: WASP-39_box_spectra_fullres.fits
resolution: {high: reference, low: 20, reference_grid: prism_template.csv}
flags:
  need_lowres: true
  ld_profile: power2
  ld_prior: stellarprior
  detrending_type: linear
```

Use `order: 2` with the corresponding order-2 configuration and extraction. Orders are fitted separately. The reference grid is used for high-resolution binning when `high: reference`; `low: 20` supplies the calibration stage.

## Dataset and complete configuration

This example is the WASP-39 b NIRISS/SOSS order-1 visit.

The extracted file is `WASP-39_box_spectra_fullres.fits`.

The bridge spectrum is binned to $R=20$ and the final spectrum follows `prism_template.csv`.

```yaml
planet:
  name: WASP-39                 # Output prefix.
  period: 4.05528043            # Days.
  duration: 0.11693087083333333 # Days.
  t0: 59787.055                 # Same time system as the FITS table.
  b: 0.4498                     # Initial impact parameter.
  rprs: 0.1457                  # Initial radius ratio.
stellar:
  feh: 0.04                     # Metallicity [M/H].
  teff: 5509                    # Effective temperature, K.
  logg: 4.22                    # Surface gravity.
  teff_sigma: 28                # Uncertainty propagated into LD.
  logg_sigma: 0.07
  feh_sigma: 0.02
  ld_model: stagger             # ExoTiC-LD atmosphere grid.
  ld_data_path: ../exotic_ld_data
  ld_prior_min_sigma: 1.0e-4    # Coefficient-prior floor.
instrument: NIRISS/SOSS
order: 1                        # Fit order 1 only.
path: /scratch/midway3/tfairnington/
input_dir: FITS                 # Relative to path.
output_dir: WASP-39_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR
fits_file: WASP-39_box_spectra_fullres.fits
resolution:
  high: reference               # Final grid comes from the next file.
  low: 20                       # Low-resolution bridge.
  reference_grid: prism_template.csv
flags:
  vmap_chunk: 40                # Resident GPU lanes.
  detrending_type: linear       # c + v(t-tmin).
  interpolate_trend: false      # Fit channel trends directly.
  interpolate_ld: false         # Build channel LD priors directly.
  ld_prior: stellarprior        # Alias for informed.
  need_lowres: true
  ld_profile: power2
  spectro_sampler: independent_nuts
  spectro_mass_matrix: laplace
  spectro_jitter_prior: lognormal
  whitelight_geometry_estimator: posterior_median
  mask_start: jnp.min(t)        # Mask initial settling.
  mask_end: jnp.min(t) + 0.007
  mask_integrations_start: null
outlier_clip:
  whitelight_sigma: 5
  spectroscopic_sigma: 5
host_device: gpu
```

Change `path`, `input_dir`, and `output_dir` for your filesystem.

The ephemeris and FITS times must use the same time convention.

## Run

```bash
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=gpu
python fit_jwst.py -c wasp39.yaml
```

The fit masks the initial settling interval, builds the white-light curve, constructs the stellar LD prior, and samples white light.

It then runs the R=20 bridge and the reference-grid channels.

## Log walkthrough

This trimmed excerpt comes from a completed run:

```text
[LD prior] Building power2 grid for NIRISS/SOSS (whitelight)
               with 125 stellar combinations using model=stagger
Fitting whitelight for outliers and bestfit parameters
Building jaxoplanet whitelight model: detrend='linear',
               ld='informed', ld_profile='power2'
Running chunked MCMC: 20 channels in blocks of 40 (mode=serial)
Checkpoint directory: .../chunks
Transmission spectroscopy data saved to ..._R20.csv
Transmission spectroscopy data saved to ..._Rreference.csv
Analysis complete!
```

The 125 combinations propagate the configured stellar uncertainties.

`COMPUTING` marks a new chunk.

`LOADING from checkpoint` marks a fingerprint-compatible resume.

`reusing compiled ... runner` means an equal-width executable was reused.

Read the ESS and divergence lines before accepting a channel.

## White-light fit

```{image} ../_static/soss_wasp39_whitelight.png
:alt: WASP-39 SOSS order-1 white-light fit
:width: 760px
:align: center
```

Inspect ingress, egress, and the out-of-transit baseline.

The separate residual and detrended plots make low-frequency structure easier to see.

Resolve a poor white-light baseline before interpreting spectral features.

## Transmission spectrum

```{image} ../_static/soss_wasp39_spectrum.png
:alt: WASP-39 SOSS order-1 transmission spectrum
:width: 760px
:align: center
```

Depth is calculated as `rors**2` for every draw and reported in fractional units and ppm.

## Output checklist

- `00_*_preopt_init_check.png`: model at initialization.

- `00_*_init_vs_opt_check.png`: initialization compared with optimization.

- `11_*_whitelightmodel.png`: normalized white-light data and model.

- `12_*_whitelightresidual.png`: white-light residuals.

- `14_*_whitelightdetrended.png`: trend-removed white light.

- `15_*_whitelight_summary.png`: combined white-light panels.

- `22_*_R20_summary.png`: offset low-resolution channel fits.

- `24_*_R20_spectrum_00.png`: low-resolution transmission spectrum.

- `31_*_Rreference_spectrum_00.png`: final transmission spectrum.

- `34_*_Rreference_summary.png`: offset final channel fits.

- `36_*_Rreference_noisebin.png`: residual RMS versus bin size.

- `*_whitelight_timeseries.csv`: white-light data/model table.

- `*_R20.csv`, `*_Rreference.csv`: spectral depth tables.

- `chunks/`: posterior checkpoints and diagnostics.

## Plot the CSV

```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

out = Path("/scratch/midway3/tfairnington/WASP-39_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR")
s = pd.read_csv(out / "WASP-39_NIRISS_SOSS_order1_Rreference.csv")
good = s[["wavelength", "depth_ppm00", "depth_err_ppm00"]].notna().all(axis=1)
fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(s.loc[good, "wavelength"], s.loc[good, "depth_ppm00"],
            xerr=s.loc[good, "wavelength_err"],
            yerr=s.loc[good, "depth_err_ppm00"], fmt=".", color="k")
ax.set(xlabel="Wavelength [micron]", ylabel="Transit depth [ppm]")
fig.tight_layout()
plt.show()
```

## Common problems

**NaN channels:** inspect the extraction and wavelength masks; never replace missing flux with zero.

**Order contamination:** fit order 2 separately with `order: 2` and its matching extraction.

**Missing bridge products:** retain `need_lowres: true` when the R=20 stage is required.

**Interrupted job:** rerun the identical command; matching chunks load automatically.

**Repeated gate failure:** inspect the named wavelength light curve, uncertainties, trend, and masks. Reduce `vmap_chunk` to isolate it.

**Out of memory:** lower `vmap_chunk`; the spectral grid and posterior model do not change.
