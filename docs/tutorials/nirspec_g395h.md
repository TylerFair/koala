# NIRSpec/G395H

This tutorial fits the WASP-121 b NIRSpec/G395H NRS1 extraction; NRS1 and NRS2 remain separate inputs and fits. You will use a quadratic baseline, an R=20 bridge, and a reference wavelength grid. At the end, you will have NRS1 white-light diagnostics, checkpointed channel posteriors, and a detector-labeled transmission spectrum.

```bash
cp configs_fiducial_stellarinformed/WASP-121_nrs1_g395h_config.yaml wasp121_nrs1.yaml
python fit_jwst.py -c wasp121_nrs1.yaml
```

```yaml
instrument: NIRSPEC/G395H
nrs: 1
fits_file: WASP-121_nrs1_box_spectra_fullres_G395H.fits
resolution: {high: reference, low: 20, reference_grid: prism_template.csv}
flags:
  detrending_type: quadratic
  need_lowres: true
  ld_profile: power2
  ld_prior: stellarprior
```

Set `nrs: 2` and use the NRS2 FITS file for the long-wavelength detector. Do not combine detector files in one configuration. G395M and G140H use the same detector key and stage structure with their corresponding `instrument` strings.

## Dataset and complete configuration

This example fits WASP-121 b on the NIRSpec/G395H NRS1 detector. The checked configuration uses a quadratic baseline, an R=20 bridge, and a final reference grid.

```yaml
planet:
  name: WASP-121                 # Output prefix.
  period: 1.274924762            # Days.
  duration: 0.11595833333333333  # Days.
  t0: 59867.1426                 # FITS time convention.
  b: 0.1                         # Initial impact parameter.
  rprs: 0.122551                 # Initial radius ratio.
stellar:
  feh: 0.46
  teff: 6827
  logg: 4.64
  teff_sigma: 72
  logg_sigma: 0.06
  feh_sigma: 0.05
  ld_model: stagger
  ld_data_path: ../exotic_ld_data
  ld_prior_min_sigma: 1.0e-4
instrument: NIRSPEC/G395H
order: null                      # SOSS-only field; unused here.
nrs: 1                           # NRS1 detector segment.
path: /scratch/midway3/tfairnington/
input_dir: FITS
output_dir: WASP-121_G395H_NRS1_STELLARINFORMEDLD_POWER2_QUADRATIC
fits_file: WASP-121_nrs1_box_spectra_fullres_G395H.fits
resolution:
  high: reference
  low: 20
  reference_grid: prism_template.csv
flags:
  vmap_chunk: 40
  detrending_type: quadratic     # c + vt + v2 t^2.
  ld_prior: stellarprior
  need_lowres: true
  ld_profile: power2
  spectro_sampler: independent_nuts
  mask_start: cut_phase_to_transit
  mask_end: cut_phase_to_transit
outlier_clip:
  whitelight_sigma: 5
  spectroscopic_sigma: 5
host_device: gpu
```

The symbolic `cut_phase_to_transit` mask is used by the fitter's established configuration path for this visit. Use the checked NRS2 configuration rather than changing only the detector number: its filename and output name also differ.

## Run

```bash
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=gpu
python fit_jwst.py -c wasp121_nrs1.yaml
```

## Log walkthrough

```text
[LD prior] Building power2 grid for NIRSPEC/G395H (whitelight)
               with 125 stellar combinations using model=stagger
Fitting whitelight for outliers and bestfit parameters
Building jaxoplanet whitelight model: detrend='quadratic',
               ld='informed', ld_profile='power2'
Checkpoint directory: .../chunks
  chunk 0:40 - COMPUTING (40 channels)
  chunk 0:40 - SAVED checkpoint
  chunk 40:80 - reusing compiled independent-NUTS runner (width 40)
Transmission spectroscopy data saved to ..._Rreference.csv
Analysis complete!
```

The detector label in every stem should read `nrs1`. If it reads `nrs2`, confirm both the configuration and the extraction. The quadratic coefficient `v2` appears in the detailed parameter CSV.

The per-chunk diagnostic JSON records divergences and radius-ratio ESS.

## White-light fit

```{image} ../_static/g395h_wasp121_whitelight.png
:alt: WASP-121 G395H NRS1 white-light fit
:width: 760px
:align: center
```

The broadband G395H curve constrains the common chord used by all NRS1 wavelength bins. Inspect the baseline on both sides of transit and verify that a quadratic is warranted.

## Transmission spectrum

```{image} ../_static/g395h_wasp121_spectrum.png
:alt: WASP-121 G395H NRS1 transmission spectrum
:width: 760px
:align: center
```

This displayed product is an existing WASP-121 NRS1 R=100 spectrum. The tutorial configuration writes the grid named by `resolution.high`; use the matching CSV stem when plotting.

## Files to inspect

- `00_*_preopt_init_check.png` catches a poor ephemeris or normalization before sampling.

- `00_*_init_vs_opt_check.png` shows the numerical optimization result.

- `11_*_whitelightmodel.png` is the primary broadband fit.

- `12_*_whitelightresidual.png` reveals remaining visit structure.

- `14_*_whitelightdetrended.png` isolates the transit after subtracting the trend.

- `15_*_whitelight_summary.png` collects white-light diagnostics.

- `22_*_R20_summary.png` and `24_*_R20_spectrum_00.png` describe the bridge stage.

- `34_*_Rreference_summary.png` and `31_*_Rreference_spectrum_00.png` describe the final stage.

- `*_bestfit_params.csv` contains `c`, `v`, and `v2` summaries for every channel.

- `*_noisebin.csv` provides numerical residual RMS curves.

- `chunks/*.pkl` contains posterior draws.

- `chunks/*.diagnostics.json` contains gate inputs.

## Plot NRS1

```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

out = Path("/scratch/midway3/tfairnington/WASP-121_G395H_NRS1_STELLARINFORMEDLD_POWER2_QUADRATIC")
files = sorted(out.glob("WASP-121_NIRSPEC_G395H_nrs1_R*.csv"))
files = [p for p in files if "noisebin" not in p.name]
s = pd.read_csv(files[-1])
fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(s.wavelength, s.depth_ppm00,
            xerr=s.wavelength_err, yerr=s.depth_err_ppm00, fmt="k.")
ax.set(xlabel="Wavelength [micron]", ylabel="Transit depth [ppm]")
fig.tight_layout()
plt.show()
```

## Join NRS1 and NRS2 after fitting

Read the two final CSVs separately. Add a detector column before concatenation. Do not discard an offset between detector segments without checking their white-light depths, limb assumptions, and trends.

Do not assume NRS1 and NRS2 share bad-pixel masks.

## Common problems

**Tilt event or jump:** use `linear_discontinuity` when the time series has a detector step. Initialize `t_jump_guess` near the event. **NaNs at detector edges:** mask or remove those wavelength bins in preparation. Zero-filled channels produce misleading likelihoods.

**Wrong detector:** `nrs`, `fits_file`, and the physical extraction must agree. **Low resolution omitted:** keep `need_lowres: true` for the bridge products or interpolation. **Interrupted run:** repeat the command unchanged to load matching checkpoints.

**Gate repeatedly fails:** inspect the named channel, reduce `vmap_chunk`, and test whether the trend or discontinuity model is missing structure. **Memory pressure:** NRS1 and NRS2 channel counts differ; tune the width for each detector rather than assuming one maximum.

## Next steps

Read [Systematics trends](../guides/trends.md) before introducing a detector-discontinuity model, and use [Samplers](../guides/samplers.md) to interpret a NUTS-to-HMC swap. The [Configuration reference](../guides/configuration.md) lists the NRS detector and resolution controls, while [Outputs](../guides/outputs.md) explains how to combine detector-labeled tables safely.
