# Output files

Everything koala makes goes into `output_dir`.
After the first fit, it looks like this:

```text
results/WASP-39_SOSS_ORDER1/
  koala.log
  WASP-39_NIRISS_SOSS_order1_R100_spectrum.csv
  WASP-39_NIRISS_SOSS_order1_R100_params.csv
  WASP-39_NIRISS_SOSS_order1_R100_lightcurves.csv
  WASP-39_NIRISS_SOSS_order1_whitelight_params.csv
  WASP-39_NIRISS_SOSS_order1_whitelight_lightcurve.csv
  figures/
  state/
```

The tables start with the target, instrument and detector or order, followed by the stage: `whitelight`, or the wavelength grid of the channel fits (`R100`, `Rnative`, `Rreference`, ...).
In the tables below, `*` stands for the part before the stage.

## The spectrum

`*_R100_spectrum.csv` (or whichever grid you chose) is the spectrum, one row per wavelength channel:

| Column | |
|---|---|
| `wavelength`, `wavelength_err` | channel centre and half-width, in µm |
| `depth00`, `depth_err00` | transit depth $(R_p/R_*)^2$ and its uncertainty |
| `depth_ppm00`, `depth_err_ppm00` | the same in ppm |

The `00` is the planet number, so a second planet in the same fit has columns ending in `01`.

`*_R100_params.csv` has the fitted value and uncertainties of everything else in each channel: radius ratio, limb-darkening coefficients and trend coefficients, each with `_err`, `_err_low` and `_err_high` columns.

`*_R100_lightcurves.csv` has the light curve of every channel side by side: `time`, then `flux_000`, `flux_err_000`, `flux_001`, ...
Channel `000` is the first row of the spectrum, `001` the second, and so on.

## The white-light fit

`*_whitelight_params.csv` has the fitted orbit and every other white-light parameter, with uncertainties.
`t0` is in BJD_TDB − 2400000, and `t0_bjd_tdb` is the same time as a full BJD_TDB.

`*_whitelight_lightcurve.csv` has the white-light curve with the model for every integration: `time_bjd` (in BJD_TDB − 2400000), `time_from_t0_hr`, `flux`, `flux_err`, `bestfit_model`, `transit_model`, `trend_model`, `detrended_flux`, `residual`, `residual_ppm` and `is_outlier`.

## Other kinds of fit

Some fits write one more table for each stage:

| File | When | |
|---|---|---|
| `*_R100_emission.csv` | eclipses and phase curves | the planet's emission (see the [eclipse case study](../case_studies/rocky_eclipse.md)); the depth columns of the spectrum still describe $(R_p/R_*)^2$ |
| `*_R100_timing.csv` | `spectro_free_t0` | the depth and transit time of every channel (see the [timing case study](../case_studies/timing_spectra.md)), with the sign convention in `*_R100_timing.json` |
| `*_R100_limbs.csv` | Harmonica | the depths of the two halves of the limb, with their posterior draws in `*_R100_limb_samples.npz` (see the [Harmonica case study](../case_studies/harmonica.md)) |
| `*_R100_stellar_spots.csv` | a spotted star | the spot contrast in every channel |
| `*_R100_joint_geometry.csv` | `spectro_joint_geometry` | the orbit fitted together with all the channels |

## Figures

The figures are in `figures/`:

| File | |
|---|---|
| `whitelight_start.png` | the white-light data with the model koala started from, before and after optimizing it; the first thing to look at if a fit goes wrong |
| `whitelight.png` | the white-light fit: the data with the full model, the data with the trend divided out, and the residuals |
| `whitelight_corner.png` | the white-light posterior |
| `R100_spectrum.png` | the spectrum |
| `R100_lightcurves.png` | a dozen channels spread across the wavelength range, with their residuals |
| `R100_residual_map.png` | the residuals of every channel against time, which is a good way to spot leftover systematics |
| `R100_limb_darkening.png` | the fitted limb-darkening coefficients against wavelength, with the stellar-model values they started from |
| `R100_noise.png` | how the scatter of the residuals goes down as you average more integrations together, compared with white noise |

Eclipses, timing, Harmonica and spotted-star fits add a figure with the same name as their table.
With `flags.plots: minimal`, koala only makes `whitelight.png` and the spectrum (and the emission or timing figure, when there is one).

The binned points in the figures are only there to make them easier to read.
The fits always use every integration, and for numbers you should read the tables.

The figures are drawn at the width of a journal page, 7 inches, so they can go straight into a paper: with `\includegraphics[width=\textwidth]{...}` every label prints at 11 pt or larger in an AASTeX, MNRAS or A&A article.
To draw your own figures the same way, call `koala.plotting_style.apply_publication_style()` first.

## Joint fits

A [joint fit](../case_studies/transmission_soss.md) writes the shared orbit at the top and gives every observation its own directory with the usual files:

```text
results/WASP-39_joint/
  koala.log
  joint_params.csv
  joint_spectrum.csv
  figures/        joint_corner.png, joint_spectrum.png, and a spectrum for each observation
  state/
  joint_soss_order1/
  joint_soss_order2/
```

`joint_params.csv` has the shared orbit and the transit time of each visit, and `joint_spectrum.csv` has the spectra of every observation on one wavelength axis (`joint_spectrum_binned.csv` is the same, binned for the figure).
When the channel times are free, there's also `joint_limbs.csv`, with the morning and evening limbs (see the [timing case study](../case_studies/timing_spectra.md)).

## Everything else

`koala.log` has every message from the run, including the details koala doesn't print to the screen.

`state/` holds what koala needs to pick up an interrupted fit or reuse a finished one: the posterior samples of each batch of channels and their sampling diagnostics in `state/chunks/`, the white-light samples, the outlier masks, the orbit passed from the white-light fit to the channels, and the files that record which data and settings they came from.
You don't need to read any of it, but keep it with the rest of the directory if you might want to rerun the fit.
