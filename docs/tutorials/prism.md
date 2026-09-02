# NIRSpec/PRISM

This tutorial fits the HAT-P-65 b visit-1 NIRSpec/PRISM NRS1 extraction. You will model the broadband ramp, run an R=20 bridge, and retain the native high-resolution channels. At the end, you will have white-light ramp diagnostics, resumable native-channel posteriors, and both low- and high-resolution spectra.

```bash
cp configs_fiducial_stellarinformed/HAT-P-65_nrs1_prism_v1_config.yaml prism.yaml
python fit_jwst.py -c prism.yaml
```

```yaml
instrument: NIRSPEC/PRISM
nrs: 1
fits_file: HAT-P-65_nrs1_box_spectra_fullres_PRISM_V1.fits
resolution: {high: native, low: 20}
flags:
  detrending_type: explinear
  need_lowres: true
```

PRISM spans a large dynamic range. Remove saturated wavelengths in the extraction or with the configured wavelength filters/masks before fitting. Use `resolution.high: 20`, `100`, or `300` to bin at constant resolving power, or `native` to retain input channels. `need_lowres: true` runs the R=20 stage before the high-resolution stage. For staged cluster runs, use `analysis_stage: prep` to finish preparation/low resolution and `analysis_stage: highres` for the final stage.

## Dataset and complete configuration

This example fits HAT-P-65 b, visit 1, with NIRSpec/PRISM NRS1. The final grid is native and the bridge grid is $R=20$.

```yaml
planet:
  name: HAT-P-65                # Output prefix.
  period: 2.60544751            # Days.
  duration: 0.17688375          # Days.
  t0: 60470.72                  # Same convention as FITS times.
  b: 0.464
  rprs: 0.1006
stellar:
  feh: 0.1
  teff: 5835
  logg: 4.18
  teff_sigma: 51
  logg_sigma: 0.1
  feh_sigma: 0.08
  ld_model: stagger
  ld_data_path: ../exotic_ld_data
  ld_prior_min_sigma: 1.0e-4
instrument: NIRSPEC/PRISM
order: null                     # Unused for NIRSpec.
nrs: 1
path: /scratch/midway3/tfairnington/
input_dir: FITS
output_dir: HAT-P-65_PRISM_NRS1_V1_STELLARINFORMEDLD_POWER2_EXPLINEAR
fits_file: HAT-P-65_nrs1_box_spectra_fullres_PRISM_V1.fits
resolution:
  high: native                  # Retain extracted channel grid.
  low: 20                       # Bridge fit.
flags:
  vmap_chunk: 40
  detrending_type: explinear    # Linear baseline plus exponential ramp.
  spectro_fixed_timescale_trends: true
  interpolate_trend: false
  interpolate_ld: false
  ld_prior: stellarprior
  need_lowres: true
  ld_profile: power2
  spectro_sampler: independent_nuts
  spectro_mass_matrix: laplace
  spectro_jitter_prior: lognormal
  whitelight_geometry_estimator: posterior_median
  mask_integrations_start: null
outlier_clip:
  whitelight_sigma: 4
  spectroscopic_sigma: 4
host_device: gpu
```

The white-light stage fits the exponential amplitude and decay time. The channel stage fixes the decay time to the white-light posterior median and fits the amplitude.

## Run

```bash
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=gpu
python fit_jwst.py -c prism.yaml
```

For a split allocation, first set `analysis_stage: prep`, then use `analysis_stage: highres` with compatible prepared products.

## Log walkthrough

This excerpt is from a completed HAT-P-65 PRISM run:

```text
[LD prior] Loading cached power2 grid for NIRSPEC/PRISM (whitelight)
Fitting whitelight for outliers and bestfit parameters
Building jaxoplanet whitelight model: detrend='explinear',
               ld='informed', ld_profile='power2'
Saved white-light time series to ..._whitelight_timeseries.csv
[LD prior] Loading cached power2 grid for NIRSPEC/PRISM (R20)
Transmission spectroscopy data saved to ..._R20.csv
[LD prior] Loading cached power2 grid for NIRSPEC/PRISM (Rnative)
Running chunked MCMC: 369 channels in blocks of 100 (mode=serial)
Checkpoint directory: .../chunks
  chunk 0:100 - COMPUTING (100 channels)
```

`Loading cached power2 grid` means the fingerprinted stellar grid already exists. The 369-channel message makes native PRISM memory requirements explicit. Use a smaller `vmap_chunk` than the historical width shown in this excerpt when device memory is limited.

## White-light fit

```{image} ../_static/prism_hatp65_whitelight.png
:alt: HAT-P-65 PRISM white-light fit
:width: 760px
:align: center
```

The early-time curvature is the feature modeled by the exponential ramp. Inspect the out-of-transit baseline and the residuals before accepting the decay time for channel fits.

## Transmission spectrum

```{image} ../_static/prism_hatp65_spectrum.png
:alt: HAT-P-65 PRISM native transmission spectrum
:width: 760px
:align: center
```

Native PRISM channels vary strongly in photon count and saturation risk across wavelength. The spectrum should be inspected together with its uncertainty and depth-error outlier columns.

## Resolution choices

`high: native` preserves the extraction grid. `high: 20`, `100`, or `300` constructs constant-$R$ bins. `high: reference` uses `reference_grid`.

The R=20 bridge is independent of the high-resolution choice. Use coarser bins when native channels are dominated by low ESS or weakly constrained systematics.

## Output checklist

- `11_*_whitelightmodel.png`: broadband data and explinear model.

- `12_*_whitelightresidual.png`: broadband residuals.

- `14_*_whitelightdetrended.png`: transit after ramp/baseline removal.

- `15_*_whitelight_summary.png`: combined broadband diagnostics.

- `24_*_R20_spectrum_00.png`: bridge spectrum.

- `25_*_R20_noisebin.png`: bridge residual RMS.

- `31_*_Rnative_spectrum_00.png`: native final spectrum.

- `34_*_Rnative_summary.png`: native offset light curves.

- `36_*_Rnative_noisebin.png`: native residual RMS.

- `*_bestfit_params.csv`: includes channel `A`; `tau` is present where sampled.

- `chunks/`: resumable native-channel posterior files.

## Plot and filter finite channels

```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

out = Path("/scratch/midway3/tfairnington/HAT-P-65_PRISM_NRS1_V1_STELLARINFORMEDLD_POWER2_EXPLINEAR")
s = pd.read_csv(out / "HAT-P-65_NIRSPEC_PRISM_nrs1_Rnative.csv")
good = s.depth_ppm00.notna() & s.depth_err_ppm00.notna()
fig, ax = plt.subplots(figsize=(9, 4))
ax.errorbar(s.wavelength[good], s.depth_ppm00[good],
            xerr=s.wavelength_err[good], yerr=s.depth_err_ppm00[good], fmt="k.")
ax.set(xlabel="Wavelength [micron]", ylabel="Transit depth [ppm]")
fig.tight_layout()
plt.show()
```

## Common problems

**Saturation:** exclude saturated integrations or wavelength bins during extraction/preparation. A finite placeholder is not a substitute for a valid measurement. **Too many native channels:** choose R=100 or R=300, or reduce `vmap_chunk` to lower GPU memory.

**Ramp-depth degeneracy:** verify that pre-transit baseline constrains `tau`; inspect white-light residuals and compare a simpler trend when appropriate. **No R=20 files:** set `need_lowres: true` and use `analysis_stage: all` or `prep`. **Resume rejected:** a changed data array or model option changes the fingerprint. Keep the old checkpoints and let the new configuration create its own family.

**Gate repeatedly fails:** inspect saturation and uncertainties first, then the ramp model. Isolate the channel with a smaller width.

## Next steps

Read [Systematics trends](../guides/trends.md) for the explinear equation and its fixed channel timescale, then consult [GPUs and clusters](../guides/gpu_and_clusters.md) before scheduling a large native-grid run. Use [Samplers](../guides/samplers.md) for repeated gate failures and [Outputs](../guides/outputs.md) for the native-spectrum and noise-binning columns.
