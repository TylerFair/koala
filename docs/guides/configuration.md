# Configuration

A fit is described by one YAML file. Start from the nearest file in the
[`examples` directory](https://github.com/TylerFair/koala/tree/main/examples)
and change the target and paths; the sampler and convergence checks already
have production defaults.

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

```bash
python fit_jwst.py -c config.yaml
```

| Setting | Meaning |
|---|---|
| `planet` | `period`, `t0`, `b`, `rprs`, and either `duration` or `a_rs`. |
| `stellar` | Stellar parameters for limb darkening; the three `*_sigma` fields are required for `ld_prior: stellarprior`. |
| `instrument` | `NIRISS/SOSS` with `order: 1` or `2`; `NIRSPEC/G395H`, `G395M`, `PRISM`, `G140H`, or `G235H` with `nrs: 1` or `2`; or `MIRI/LRS` with neither. |
| `path`, `input_dir`, `fits_file` | The input is read from `path/input_dir/fits_file`. |
| `output_dir` | Result directory, relative to `path` unless absolute. |
| `resolution` | `high` is the final grid: a resolving power, `native`, or `reference` (with `reference_grid`). `low` is an optional coarse stage; omit it to skip that stage. |
| `flags.detrending_type` | The systematics trend; see the [Introduction](../concepts.md) for the list. |
| `flags.ld_profile`, `flags.ld_prior` | Limb-darkening law and prior; see [Limb darkening](limb_darkening.md). |
| `flags.light_curve_model` | `transit`, `eclipse`, or `phase_curve`; see [Eclipses, phase curves, and stellar spots](phase_curves.md). |
| `host_device` | `gpu` for a full run; `cpu` for checks and the bundled example. |

Mask a time interval with `flags.mask_start` and `flags.mask_end`. The
template trends take initial guesses through `spot_amp`, `spot_center`,
`spot_width` (`spot_amp2`, `spot_center2`, `spot_width2`, or the aliases
`spot_amp_2`, `spot_center_2`, `spot_width_2`, for a second spot) and
`jump_guess` or `t_jump_guess`.

## Parameter priors

Every orbital entry of `planet` (`period`, `t0`, `b`, `rprs`, `duration`
or `a_rs`, `ecc`, `omega`) accepts one of three forms:

```yaml
planet:
  name: WASP-39
  period: 4.05528043                          # bare number: today's default
  t0: [free, uniform, 59786.9, 59787.2]       # list form
  b: {mode: free, prior: gaussian, mu: 0.45, sigma: 0.05}   # mapping form
  duration: [free, truncated_gaussian, 0.117, 0.005, 0.05, 0.3]
  rprs: 0.1457
  ecc: [fixed, 0.0]
```

A bare number keeps the historical behaviour: `period`, `ecc`, and
`omega` are fixed, the others are free with the built-in wide priors and
the number is the starting point. The list forms are `[fixed, value]`,
`[free, uniform, low, high]`, `[free, gaussian, mu, sigma]`, and
`[free, truncated_gaussian, mu, sigma, low, high]`; the mapping form uses
the same names (`mode`, `prior`, `value`, `mu`, `sigma`, `low`, `high`).
For several planets give one entry per planet. `ecc` and `omega` may only
be fixed, and explicit priors need `flags.transit_engine: jaxoplanet`.

A free `period` is sampled in the white-light fit (it appears in the
best-fit CSV and the corner plot) and then held at its posterior median
for the spectroscopic stages, like `t0`, `b`, and `duration`. When one
time series spans several transits every epoch `t0 + n * period` is
masked as in transit, so a free period constrains the ephemeris:

```yaml
planet:
  period: [free, gaussian, 4.05528043, 0.001]
  t0: [free, uniform, 59786.9, 59787.2]
```

Stacking separate FITS files into one series is not done by Koala; the
input must already be a single time series.

Other accepted `flags` keys, shown in context by the example files:
`fit_geometry`, `transit_engine`, `analysis_stage`, `random_seed`,
`spectro_chunk_size` (alias `vmap_chunk`), `need_lowres`,
`trend_inference`, `ld_uniform_basis`, `spectro_sampler`,
`spectro_min_depth_ess`, `spectro_max_divergences`,
`spectro_cadence_reduction`, `spectro_transit_grid`,
`spectro_transit_grid_nodes`, `chunk_mode`, `chunk_parallel_job_count`,
`chunk_parallel_job_index`, `jax_compilation_cache_dir`, `plots`,
`harmonica_max_order`, `harmonica_spectro_parameterization`,
`harmonica_spectro_fit_jitter`, and `harmonica_spectro_odd_frac_sigma`.
Unknown keys under `flags` produce a warning and are ignored.
