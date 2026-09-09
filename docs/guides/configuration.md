# Configuration

A fit is described by one YAML file. Start from the nearest file in the
[`examples` directory](https://github.com/TylerFair/koala/tree/main/examples),
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
`NIRSPEC/PRISM`, remove `order`, and add `nrs: 1` or `nrs: 2`. For MIRI,
use `instrument: MIRI/LRS` and omit both `order` and `nrs`.

## The settings most users change

| Setting | Meaning |
|---|---|
| `planet` | Transit ephemeris and initial geometry. Supply `period`, `t0`, `b`, `rprs`, and either `duration` or `a_rs`. |
| `stellar` | Stellar parameters used to calculate limb darkening. The three uncertainty fields are required for `ld_prior: stellarprior`. |
| `instrument` with `order` or `nrs` | The observing mode (`NIRISS/SOSS`, a `NIRSPEC/*` disperser, or `MIRI/LRS`) and the SOSS order or NIRSpec detector. |
| `path`, `input_dir`, `fits_file` | The input is read from `path/input_dir/fits_file`. Absolute paths are easiest on a cluster. |
| `output_dir` | Result directory, interpreted relative to `path` unless absolute. Use a fresh directory for a distinct analysis. |
| `resolution` | The final `high` wavelength grid and an optional coarse `low` grid. A grid can be an integer resolving power, `native`, or `reference`; the last also needs `reference_grid`. Omit `low` to skip the low-resolution bridge stage entirely. |
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
  spectro_cadence_reduction: auto
  spectro_transit_grid: auto
  spectro_transit_grid_nodes: 769
  analysis_stage: all
```

`spectro_chunk_size` is the number of wavelength channels resident on the GPU.
The default is 40 for the independent samplers; lower it after an out-of-memory
error. It does not change the wavelength bins. The older spelling `vmap_chunk`
is still accepted.

`spectro_cadence_reduction` can be `auto` (the default) or `off`. For a
compatible additive trend with more than 5,000 cadences, `auto` evaluates the
likelihood directly in the transit windows and uses exact, reported-error-
grouped sufficient statistics outside them. Eligible sampled trends are
linear through quartic polynomials, fixed-timescale exponential-linear,
fixed-shape one- and two-spot templates, quadratic plus a fixed spot, a
fixed-shape linear discontinuity, and fixed spot plus discontinuity. Masked
cadences are excluded from both pieces. Free-timescale exponential trends,
GP likelihoods, Gaussian-marginalized trend inference, and surface models use
the direct path.

`spectro_transit_grid` can also be `auto` (the default) or `off`. On compatible
duration-parameterized power-2 or quadratic-LD transit stages with more than
5,000 cadences, `auto` evaluates the limb-darkened model on a contact-aware
Chebyshev grid and interpolates a separate model value for every transit-
window cadence. It is not time binning: all observed fluxes, errors, and
likelihood residuals remain separate. The grid splits at second/third contact
when those contacts exist and uses one segment for a grazing transit.
`spectro_transit_grid_nodes` sets the total nodes (default `769`); conservative
power-2 grazing/near-grazing handoffs are automatically raised to the audited
`2049` nodes. A handoff close enough to outer contact that the full radius-
ratio prior is not covered safely stays on the direct kernel. Short light
curves, Keplerian geometry, interpolated LD, and surface models also retain the
original path.

`analysis_stage` can be `all`, `whitelight`, `prep`, or `highres`. A normal
run uses `all`; staged cluster runs are described in [GPUs and clusters](gpu_and_clusters.md).

The default spectroscopic sampler is `independent_nuts`. Leave it unchanged
unless you have a reason to compare backends; see [Samplers](samplers.md).

Gaussian-process white-light trends have one process-level execution control,
`KOALA_GP_SOLVER`. It accepts `auto`, `serial`, or `parallel`. The default
`auto` selects parallel on a GPU or TPU when the installed tinygp supports it,
and serial on a CPU. The optional top-level configuration key `gp_solver`
takes the same values and overrides the environment variable for one run. `serial` is useful for matched validation; explicitly
requesting unavailable parallel support fails with installation guidance.
Library callers can make the same choice with `gp_solver=` and can pass
`gp_assume_sorted=True` only after validating that timestamps are
nondecreasing. See [Gaussian processes](gaussian_processes.md).

Advanced execution controls are `chunk_mode`, `chunk_parallel_job_count`,
`chunk_parallel_job_index`, `spectro_sampler`, `spectro_min_depth_ess`,
`spectro_max_divergences`, `spectro_cadence_reduction`,
`spectro_transit_grid`, `spectro_transit_grid_nodes`,
`jax_compilation_cache_dir`, and `plots`. Scientific
advanced controls are `transit_engine`, `trend_inference`, `ld_uniform_basis`,
`harmonica_max_order`, `harmonica_spectro_parameterization`,
`harmonica_spectro_fit_jitter`, and `harmonica_spectro_odd_frac_sigma`.
The compatibility spelling `vmap_chunk` remains accepted for
`spectro_chunk_size`; `need_lowres` controls whether the coarse stage runs
when `resolution.low` is set (omitting `resolution.low` skips it regardless).

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
