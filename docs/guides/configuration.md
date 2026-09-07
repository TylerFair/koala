# Configuration

A fit is described by one YAML file. Start from the nearest file in the
[`examples` directory](https://github.com/TylerFair/jwst-lightcurves/tree/main/examples),
change the target and file paths, and keep the rest small. The sampler and
convergence checks already have production defaults.

## A minimal SOSS configuration

```yaml
planet:
  name: WASP-39
  period: 4.05528043      # days
  duration: 0.11693087    # days
  t0: 59787.055           # BMJD_TDB
  b: 0.4498
  rprs: 0.1457

stellar:
  teff: 5509
  logg: 4.22
  feh: 0.04
  teff_sigma: 28
  logg_sigma: 0.07
  feh_sigma: 0.02
  ld_model: stagger
  ld_data_path: /path/to/exotic_ld_data

instrument: NIRISS/SOSS
order: 1

path: /path/to/analysis
input_dir: FITS
fits_file: WASP-39_box_spectra_fullres.fits
output_dir: WASP-39_SOSS_ORDER1

resolution:
  high: 100
  low: 20

flags:
  detrending_type: linear
  ld_profile: power2
  ld_prior: stellarprior

outlier_clip:
  whitelight_sigma: 5
  spectroscopic_sigma: 5

host_device: gpu
```

Run it from the repository checkout:

```bash
python fit_jwst.py -c config.yaml
```

For NIRSpec, use an instrument such as `NIRSPEC/G395H` or
`NIRSPEC/PRISM`, remove `order`, and add `nrs: 1` or `nrs: 2`.

## The settings most users change

| Setting | Meaning |
|---|---|
| `planet` | Transit ephemeris and initial geometry. Supply `period`, `t0`, `b`, `rprs`, and either `duration` or `a_rs`. |
| `stellar` | Stellar parameters used to calculate limb darkening. The three uncertainty fields are required for `ld_prior: stellarprior`. |
| `instrument` with `order` or `nrs` | The observing mode and SOSS order or NIRSpec detector. |
| `path`, `input_dir`, `fits_file` | The input is read from `path/input_dir/fits_file`. Absolute paths are easiest on a cluster. |
| `output_dir` | Result directory, interpreted relative to `path` unless absolute. Use a fresh directory for a distinct analysis. |
| `resolution` | Required `high` and `low` wavelength grids. A grid can be an integer resolving power, `native`, or `reference`; the last also needs `reference_grid`. Keep `low` present even when a staged run will not fit that grid. |
| `flags.detrending_type` | The visit baseline. Begin with `linear` and change it only when the out-of-transit data support another model. See [Trend models](trends.md). |
| `flags.ld_profile` | `power2` or `quadratic`. |
| `flags.ld_prior` | `uniform`, `gaussian`, `sing`, `stellarprior`, or `fixed`. `gaussian` has width 0.2, `stellarprior` requires power-2, and `sing` requires quadratic. See [Limb darkening](limb_darkening.md). |
| `flags.light_curve_model` | `transit` (default), `eclipse`, or `phase_curve`. Eclipse and phase-curve fits use JAXoplanet Keplerian geometry; see [Eclipses, phase curves, and stellar spots](phase_curves.md). |
| `flags.fit_geometry` | Whether to infer orbital geometry in a JAXoplanet surface fit. Eclipse-only fits default to `false`; set it only when the observation constrains the transit geometry. |
| `host_device` | `gpu` for a full run; `cpu` is useful for imports and small tests. |

The example files also show wavelength masks, cadence masks, reference grids,
and PRISM settings in context.

## Useful optional controls

These controls affect execution rather than the scientific model:

```yaml
flags:
  random_seed: 555
  spectro_chunk_size: 20
  analysis_stage: all
```

`spectro_chunk_size` is the number of wavelength channels resident on the GPU.
The default is 40 for the independent samplers; lower it after an out-of-memory
error. It does not change the wavelength bins. The older spelling `vmap_chunk`
is still accepted.

`analysis_stage` can be `all`, `whitelight`, `prep`, or `highres`. A normal
run uses `all`; staged cluster runs are described in [GPUs and clusters](gpu_and_clusters.md).

The default spectroscopic sampler is `independent_nuts`. Leave it unchanged
unless you have a reason to compare backends; see [Samplers](samplers.md).

Advanced execution controls are `chunk_mode`, `chunk_parallel_job_count`,
`chunk_parallel_job_index`, `spectro_sampler`, `spectro_min_depth_ess`,
`spectro_max_divergences`, `jax_compilation_cache_dir`, and `plots`. Scientific
advanced controls are `transit_engine`, `trend_inference`, `ld_uniform_basis`,
`harmonica_max_order`, `harmonica_spectro_parameterization`,
`harmonica_spectro_fit_jitter`, and `harmonica_spectro_odd_frac_sigma`.
The compatibility spelling `vmap_chunk` remains accepted for
`spectro_chunk_size`; `need_lowres` controls whether the coarse stage runs.

## Dataset-specific masks

Mask a bad time interval in BMJD_TDB with:

```yaml
flags:
  mask_start: 60795.15
  mask_end: 60795.26
```

For a spot crossing or detector jump, choose the matching trend and provide an
initial location from the light curve. The trend tutorial gives complete
examples. The corresponding initial-guess keys are `spot_amp`, `spot_center`,
`spot_width`, `spot_amp2`, `spot_center2`, `spot_width2`, and `jump_guess` or
`t_jump_guess`; the compatibility aliases `spot_amp_2`, `spot_center_2`, and
`spot_width_2` remain accepted. Unknown keys under `flags` emit a warning and are ignored, so treat
that warning as a likely typo.
