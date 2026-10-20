# Output files

Let's open the spectrum from the bundled SOSS fit.
The filename includes the target, instrument, detector or order, and wavelength grid:

```python
from pathlib import Path
import pandas as pd

out = Path("results/WASP-39_SOSS_ORDER1")
spectrum = pd.read_csv(next(out.glob("*_R100.csv")))
print(spectrum[["wavelength", "depth_ppm00", "depth_err_ppm00"]])
```

The wavelength is in microns, and these depth columns are in ppm.
The `00` suffix identifies the first planet.
For an eclipse, open `*_R*_emission.csv` and read `eclipse_depth_ppm` instead.
The generic transit-depth columns still describe $(R_p/R_*)^2$ in an eclipse run.

Next, open `*_whitelight_timeseries.csv` to compare the data, full model, trend, and residuals.
The wide channel table contains the input channel curves, so use the posterior products when you need the fitted model.
Excluded integrations are removed before fitting.
Outlier flags describe clipping among the retained integrations.

Keep CSV and NPZ products for analysis and checkpoints for resuming a fit.
For a limb difference, preserve the paired Harmonica draws so you retain the correlation between the two halves.
The {download}`example inventory <_static/data/runs/inventory.json>` lists files, CSV columns, and saved-array metadata from the three documentation runs.

## Reference

Here `*` stands for the filename prefix and `R*` for a grid such as `R20`, `R100`, `Rnative`, or `Rreference`.
Optional models and stages add their own products.

### Spectra and fitted parameters

| File | Columns or contents |
|---|---|
| `*_R*.csv` | `wavelength`, `wavelength_err` (µm). `depth00`, `depth_err00` (fractional). `depth_ppm00`, `depth_err_ppm00` (ppm). `sampler_used` when recorded. Planet indices continue `01`, `02`, etc. |
| `*_R*_bestfit_params.csv` | Wavelength and half-width. `rors` (radius ratio), `depth`, `depth_ppm`. Active limb-darkening and trend coefficients. Parameter errors use `_err`, `_err_low`, `_err_high`, with the ppm suffix where applicable. |
| `*_whitelight_bestfit_params.csv` | `planet_id`, `period`, `duration`, `t0`, `t0_bjd_tdb`, `b`, `rors`, `depths`. `a_rs`, `cos_i`, `ecc`, `omega` when applicable. Active trend, limb-darkening, and emission parameters. |
| `*_R*_emission.csv` | `wavelength`, `wavelength_err`, `planet_index`. `eclipse_depth_ppm`, plus for `phase_curve` `phase_c1_ppm`, `phase_d1_ppm` and the derived `nightside_flux_ppm`, `phase_amplitude_ppm`, `phase_offset_deg`. Each has `_err_low` and `_err_high`. May include `sampler_used`. |
| `*_whitelight_bestfit_params_emission.csv` | The same emission quantities for the broadband fit. |
| `*_stellar_spots.csv` | `wavelength`, `wavelength_err`, `spot_index`, `contrast`, `contrast_err_low`, `contrast_err_high`. |
| `*_joint_geometry.csv` | `parameter`, `planet`, `median`, `std`, `err_low`, `err_high`, `prior_center`, `prior_sigma`, `whitelight_median`, `whitelight_std`, `shift_in_whitelight_sigma`. |

In the general parameter tables, `t0` and `eclipse_time`, when present, are BMJD_TDB.
Periods and durations are in d.
`t0_bjd_tdb` is full BJD_TDB.
In the general parameter table, unsuffixed emission flux parameters (`eclipse_depth`, `phase_c1`, `phase_d1`, `nightside_flux`, `phase_amplitude`) are fractional and `phase_offset_deg` is in degrees.
Use the emission CSV for ppm.
The added timing-spectrum columns instead use full BJD_TDB (see [timing columns](tutorials/time_shift.md#timing-columns)).

`depths` and `depth` describe `(Rp/R*)²`, including in eclipse runs.
They are not eclipse depths.
The asymmetric errors give the distances from the reported median to the lower and upper posterior quantiles.
`depth_error_ratio_to_median` and `depth_error_outlier_gt5x`, when present, compare a channel's depth uncertainty with the median channel uncertainty.

### Light curves

| File | Columns |
|---|---|
| `*_whitelight_timeseries.csv` | Table described below. |
| `*_R*_lightcurves_wide.csv` | `time`, followed by `flux_NNN`, `flux_err_NNN` for each channel. |
| `*_R*_wavelengths.csv` | `wavelength_center`, `wavelength_err` in µm. Row `NNN` matches that light-curve channel. |
| `*_whitelight_GP_database.csv` | GP flux, uncertainty, and trend in `gp_flux`, `gp_err`, `gp_trend`, with the time coordinate. |
| `*_noisebin.csv` | `bin_size_points`, `measured_rms`, `measured_rms_p16`, `measured_rms_p84`, `expected_white_rms`. RMS values are in normalized-flux units. |

| White-light column | Meaning |
|---|---|
| `time_bjd` | Loaded time coordinate in BMJD_TDB, despite the historical column name. |
| `time_from_t0_hr` | Hours from the reference primary-transit epoch. Not from eclipse. |
| `flux`, `flux_err` | Normalized input flux and uncertainty. |
| `bestfit_model` | Fitted full model. |
| `residual`, `residual_ppm` | Data minus model, fractional and ppm. |
| `is_outlier` | Clipping mask. |
| `transit_model` | Astrophysical model component, when exported. |
| `trend_model` | Instrument trend. |
| `detrended_flux` | Flux after removing the fitted trend. |

### Figures

| Filename pattern | Plot |
|---|---|
| `00_*_preopt_init_check.png`, `00_*_init_vs_opt_check.png` | Initial and optimized white-light models. |
| `11_*`, `12_*`, `13_*`, `14_*`, `15_*` PNGs | White-light model, residuals, corner plot, detrended curve, and summary. |
| `2*_*`, `3*_*` PNGs | Coarse- and final-grid spectra, light-curve grids, summaries, and noise-binning plots. Suffixes identify the content. |
| `*_emission.png`, `*_stellar_spots.png` | Emission parameters or stellar-spot contrasts against wavelength. |
| `*_limb_spectra.png`, `*_transmission_strings.png`, `*_transmission_string_posterior.png` | Harmonica limb spectra and boundary plots. |

`flags.plots: minimal` skips the larger noise, polynomial, and light-curve grids.
Read numerical values from the tables, not the PNGs.

### Harmonica

`*_limb_spectra.csv` records `wavelength`, `wavelength_err`, `planet_index`, `limb_product_schema_version`, and `limb_product_convention`, followed by the summaries below.
Summary columns end in `_median`, `_err_lo`, and `_err_hi`.

| Quantity | Meaning and unit |
|---|---|
| `rp_one`, `rp_two` | Area-equivalent radius ratios for the two halves. |
| `depth_one`, `depth_two` | Representative half-area depths in ppm. |
| `depth_total_area` | Total area-equivalent depth in ppm. |
| `rp_one_endpoint`, `rp_two_endpoint` | Boundary radius ratios at the two endpoints. |
| `depth_one_endpoint`, `depth_two_endpoint` | Squared endpoint radii in ppm. |
| `asymmetry_coefficient`, `endpoint_delta_r` | Area-asymmetry coefficient and endpoint radius difference, in stellar-radius units. |
| `depth_a0` | Squared mean boundary radius in ppm. Not the total silhouette area. |
| `bandpass_min`, `bandpass_max` | Broadband wavelength limits in µm, when present (no summary suffix). |
| `a0`, `a1`, higher active harmonics | Boundary coefficients in stellar-radius units. |

The two halves are angular indices, not physical morning/evening labels.
Older products can use those physical labels.
Read the stored convention before assigning an orientation.

`*_limb_posterior_samples.npz` preserves paired draws, wavelength information, and the schema/convention metadata.
Draw keys use the quantity names in the table without summary suffixes, with shape `(draw, wavelength)` recorded by `sample_axes`.
The archive also records `depth_units` and `wavelength_units`.
Depths in the NPZ are fractional, while depth summaries in the CSV are ppm.
Preserve the NPZ when computing limb differences.
See the [Harmonica example](tutorials/harmonica.md#read-the-two-limbs).

### Saved state

| File or directory | Contents |
|---|---|
| `*_whitelight_geometry_handoff.json` | White-light orbital solution used in spectroscopy. |
| `*_whitelight_outlier_mask.npy`, `*_R*_spectroscopic_outlier_mask.npy` | Boolean clipping masks. |
| `*_R*_poly_coeffs.npz` | Polynomial comparison coefficients when those plots are requested. |
| `*_spectroscopy_data_*LR_*HR.pkl` | Prepared grids and light curves for reuse. |
| `*_artifacts.manifest.json`, `*.manifest.json` | Data/model identities for cached products. |
| `chunks/*.pkl` | Posterior batch checkpoints. |
| `chunks/*.diagnostics*.json`, `chunks/*.gradient.json` | Sampling and initialization records. See [diagnostics](troubleshooting.md#sampling-diagnostics). |
| `whitelight_mcmc_diagnostics.json` | White-light sampling record when written. |
| `ld_prior_cache/` and the configured LD cache | Atmosphere-derived prior tables and cached arrays. |

The exact restart files depend on the sampler and version.
Keep them together when resuming a run.
Use the CSV/NPZ science products for analysis.

### Model-stacking products

`*_stacked.csv` contains the mixture's `depth`, `depth_err_lo`, `depth_err_hi`, unaligned `absolute_depth` and errors, each candidate's depth and `*_weight`, achromatic offsets, and disagreement summaries.
Depths and offsets are fractional.
The accompanying diagnostics JSON records the predictive comparison.
The PNG shows the mixture and weights.
See [model stacking](case_studies/model_stacking.md).

### Timing and model-grid products

Free channel epochs add timing columns and a separate `*_time_shift.csv`.
The [timing tutorial](tutorials/time_shift.md#timing-columns) lists their units and conventions.

The `koala.marginalization` workflow writes `marginalized/spectrum.csv`, `posterior.npz`, and `diagnostics.json` inside its workspace.
It retains absolute depth offsets in the mixture.
See [combining models](tutorials/combining_models.md) before comparing these products with the aligned stacking outputs above.
