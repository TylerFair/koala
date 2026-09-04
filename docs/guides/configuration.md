# Configuration

A normal configuration contains the dataset facts and three model choices. The
sampler, convergence gates, parameterizations, caches, and compilation strategy
already have validated defaults and do not need to be copied into each YAML
file.

This complete NIRISS/SOSS configuration runs after its path fields point to an
extracted box-spectrum FITS file and limb-darkening grid:

```yaml
planet:
  name: Target
  period: 3.0          # days
  duration: 0.10       # days
  t0: 60000.0          # BMJD_TDB
  b: 0.4
  rprs: 0.1

stellar:
  teff: 5500
  logg: 4.4
  feh: 0.0
  teff_sigma: 50
  logg_sigma: 0.1
  feh_sigma: 0.1
  ld_model: stagger
  ld_data_path: /path/to/exotic_ld_data

instrument: NIRISS/SOSS
order: 1

path: /path/to/work
input_dir: FITS
output_dir: TARGET_SOSS_ORDER1
fits_file: target_box_spectra_fullres.fits

resolution:
  high: 100
  low: 20

flags:
  detrending_type: linear
  ld_profile: power2
  ld_prior: informed

outlier_clip:
  whitelight_sigma: 5
  spectroscopic_sigma: 5

host_device: gpu
```

Run it with:

```bash
python fit_jwst.py -c config.yaml
```

For NIRSpec, replace `order` with `nrs: 1` or `nrs: 2` and select an
instrument such as `NIRSPEC/G395H`, `NIRSPEC/G395M`, or `NIRSPEC/PRISM`.

## Public configuration

These top-level sections describe the observation. They belong in ordinary
configs whenever they apply.

| Key | What to provide |
|---|---|
| `planet` | Required `name`, `period`, `t0`, `b`, and `rprs`, plus `duration` (or an explicit `a_rs` geometry). `ecc` and `omega` are optional. Scalars and same-length arrays support one or more planets. |
| `stellar` | Required `teff`, `logg`, `feh`, `ld_model`, and `ld_data_path`. Add `teff_sigma`, `logg_sigma`, and `feh_sigma` when using a stellar-informed limb-darkening prior. |
| `instrument` | `NIRISS/SOSS` or a supported NIRSpec mode. Add `order` for SOSS or `nrs` for NIRSpec. |
| `path`, `input_dir`, `fits_file` | Base directory, extraction directory, and input box-spectrum FITS filename. |
| `output_dir` | A new result directory for this scientifically distinct fit. |
| `resolution` or `pixels` | One binning mapping with `high` and, when the low-resolution stage is used, `low`. A value may be an integer, `native`, or `reference`; reference binning also needs `reference_grid`. |
| `outlier_clip` | White-light and spectroscopic sigma thresholds. Integration-index masks also live in this section. |
| `time_binning` | Optional cadence binning: `enabled`, `dt_seconds`, `method`, `whitelight`, and `spectroscopic`. Leave it absent to retain the input cadence. |
| `wavelength_filter`, `wavelength_masks` | Optional wavelength inclusion limits and excluded intervals applied during data preparation. |
| `host_device` | `gpu` for production or `cpu` for a small diagnostic run. |

The following PUBLIC `flags` entries are the science and dataset-specific
choices. Omit only those whose defaults are appropriate.

| `flags` key | Default | What it controls and when to change it |
|---|---|---|
| `detrending_type` | `linear` | Visit-systematics model. Start with the simplest baseline supported by the out-of-transit data; common choices include `none`, polynomial families, `explinear`, `spot`, `2spot`, discontinuity combinations, and GP variants. See [Trend models](trends.md). |
| `ld_profile` | `quadratic` | Limb-darkening law: `quadratic` or `power2`. |
| `ld_prior` | inferred | Limb-darkening treatment: `fixed`, `widegaussian`, `informed` (aliases `stellar` and `stellarprior`), `sing`, or `uniform`. See [Limb darkening](limb_darkening.md). |
| `mask_start`, `mask_end` | `false` | Start/end times in BMJD_TDB for a bad cadence interval. Equal-length lists mask several intervals. Leave both absent when no time interval should be removed. |
| `spot_amp`, `spot_center`, `spot_width` | `0` | Initial amplitude, central time, and width in days for a `spot` or `2spot` trend. These are visit-specific inputs, not sampler tuning. |
| `spot_amp2`, `spot_center2`, `spot_width2` | `0` | Initial values for the second feature in a `2spot` trend. The legacy spellings `spot_amp_2`, `spot_center_2`, and `spot_width_2` remain accepted. |
| `t_jump_guess`, `jump_guess` | midpoint, `0` | Initial time in BMJD_TDB and flux amplitude for a discontinuity trend. Set both from the observed jump. |

## Advanced flags

Most users can omit this section. These are the few supported controls worth
changing for reproducibility, memory limits, staged runs, or a different
forward model.

| `flags` key | Default | Change it when... |
|---|---|---|
| `random_seed` | `555` | You need an explicitly recorded seed for a reproducible analysis. An environment seed override still takes precedence. |
| `need_lowres` | `true` | You intentionally want to skip the low-resolution spectrum and no calibration or interpolation requires it. |
| `analysis_stage` | `all` | A cluster workflow must stop after `whitelight` or `prep`, or resume at `highres`. Ordinary runs should stay on `all`. |
| `spectro_chunk_size`, `vmap_chunk` | 40 lanes | GPU memory is insufficient, or a native-cadence PRISM run benefits from fewer resident channels. Use a positive integer; the preferred `spectro_chunk_size` also accepts `auto`. If both are present, it wins over the legacy `vmap_chunk` spelling. This changes execution only, not wavelength binning. |
| `spectro_sampler` | `independent_nuts` | You deliberately select `independent_hmc`, legacy `joint_nuts`, or approximate `laplace_is` after reviewing [Samplers and convergence](samplers.md). The validated exact default needs no companion tuning flags. |
| `transit_engine` | `jaxoplanet` | The science case requires Harmonica's asymmetric transmission-string model. Set it to `harmonica`; otherwise leave it absent. |
| `harmonica_max_order` | `1` | Harmonica data support additional odd boundary modes. Higher orders add model flexibility and cost. |
| `harmonica_spectro_parameterization` | `delta_r` | A Harmonica analysis intentionally uses its alternative odd-mode coordinates. |
| `harmonica_spectro_fit_jitter` | `true` | A Harmonica likelihood deliberately fixes rather than fits extra white noise. |
| `harmonica_spectro_odd_frac_sigma` | `0.1` | Prior knowledge justifies a different fractional scale for the asymmetric modes. |

## Internal settings

Everything else under `flags` is internal tuning with validated defaults. Old
keys remain accepted for backward compatibility, but they are intentionally
absent from normal configurations and startup summaries. The authoritative
tier table and executable defaults are recorded in `fit_jwst.py`; the
validation rationale is recorded in
`acceleration_reports/ORCHESTRATOR_SUMMARY.md` and its linked acceleration
reports. An unrecognized key produces a warning with the closest supported
spelling and is otherwise ignored.
