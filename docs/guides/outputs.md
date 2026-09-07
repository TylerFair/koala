# Reading the results

The two files most users need are the transmission-spectrum CSV and the
white-light time-series CSV. The filenames begin with the target and
instrument, so inspect a run with:

```bash
find OUTPUT -maxdepth 1 -type f | sort
```

## Transmission spectrum

The main spectrum file ends in the fitted resolution, for example
`WASP-39_NIRISS_SOSS_order1_R100.csv`.

```python
import pandas as pd

spectrum = pd.read_csv("OUTPUT/WASP-39_NIRISS_SOSS_order1_R100.csv")
wavelength = spectrum["wavelength"]
depth_ppm = spectrum["depth_ppm00"]
uncertainty_ppm = spectrum["depth_err_ppm00"]
```

`wavelength_err` is the wavelength half-width/error in microns. The `00`
suffix denotes the first planet; additional planets use `01`, `02`, and so on.
Use `depth00` and `depth_err00` for fractional rather than ppm units.

The companion `*_bestfit_params.csv` contains radius ratio, limb-darkening,
trend, and jitter summaries for each channel. It is useful for diagnostics;
the shorter spectrum CSV is the clean input for atmospheric retrievals and
plots.

## White-light fit

`*_whitelight_timeseries.csv` contains the input time and flux alongside the
fitted model:

| Column | Meaning |
|---|---|
| `time_bjd` | Input time |
| `time_from_t0_hr` | Hours from the fitted reference transit time |
| `flux`, `flux_err` | Normalized flux and reported uncertainty |
| `bestfit_model` | Posterior-summary light-curve model |
| `residual`, `residual_ppm` | Data minus model |
| `is_outlier` | One where the cadence was clipped |
| `transit_model`, `trend_model`, `detrended_flux` | Model components, when available |

`*_whitelight_bestfit_params.csv` summarizes the fitted geometry and baseline.
`*_whitelight_geometry_handoff.json` records the fixed geometry passed to the
wavelength channels. Keep both with a published spectrum.

## Diagnostics and restart files

Numbered PNG files show the white-light fit, residuals, transmission spectrum,
and noise-binning checks. The exact set depends on the selected stages and
models. Use the CSV files for numerical work.

The `chunks/` directory contains posterior checkpoints and JSON diagnostics.
An unchanged run resumes from compatible checkpoints automatically. The
manifest fingerprints the data and model settings, so a changed analysis does
not silently reuse an incompatible chain.

For each accepted channel, inspect effective sample size, divergences, and the
recorded sampler. A saved checkpoint only means the computation finished; the
quality-gate result in its diagnostic JSON determines whether the posterior was
accepted or retried.

Checkpoint files are Python pickles. They are an implementation and restart
format, so load only files from a trusted run. For most downstream work, prefer
the stable CSV products.

## Light curves and Harmonica products

`*_wavelengths.csv` maps channel number to wavelength. The matching
`*_lightcurves_wide.csv` stores `time`, followed by `flux_NNN` and
`flux_err_NNN` pairs. Channel `NNN` corresponds to the same row in the
wavelength file.

Harmonica runs add `*_limb_spectra.csv`, transmission-string figures, and
`*_limb_posterior_samples.npz`. The CSV reports indexed terminator halves; it
does not by itself identify physical morning and evening limbs. Preserve the
NPZ file when the correlation between limb quantities matters.
