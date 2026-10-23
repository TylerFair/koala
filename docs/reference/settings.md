# Settings reference

This page lists every setting koala reads from a YAML file.
You won't need most of them: [configuring a fit](../guide/configuration.md) covers the ones that come up in practice.
Anything under `flags` that koala doesn't recognize gets a warning and is otherwise ignored.

## Top level

| Key | Default | |
|---|---|---|
| `planet` | required | the orbit, see below |
| `stellar` | required | the star, see below |
| `instrument` | required | see [instruments](../guide/instruments.md) |
| `order` | | SOSS order, 1 or 2 |
| `nrs` | | NIRSpec detector, 1 or 2 |
| `input_file` | required | the extracted spectra (`fits_file` also works) |
| `input_dir` | | directory of `input_file`, relative to `path` |
| `input_format` | `auto` | `exotedrf`, `eureka` or `sparta` |
| `output_dir` | `<name>_RESULTS` | where the results go, relative to `path` |
| `path` | `.` | base directory for `input_dir` and `output_dir` |
| `host_device` | automatic | `cpu` or `gpu` |
| `resolution` | | the wavelength channels, see below |
| `pixels` | | channels of a fixed number of detector columns instead of a resolving power, with the same `high` and `low` keys (`reference` for a supplied grid) |
| `wavelength_filter` | instrument range | `wl_min` and `wl_max` in µm (`wl_min_o1`, `wl_max_o1`, `wl_min_o2`, `wl_max_o2` for SOSS) |
| `wavelength_masks` | | wavelength ranges to leave out, as (start, end) pairs in µm |
| `exclude_times`, `exclude_integrations` | | integrations to leave out, see [configuring a fit](../guide/configuration.md#leaving-out-bad-data) (also accepted under `flags`) |
| `outlier_clip` | | clipping thresholds, see below |
| `trend_priors` | | white-light trend priors, see below |
| `gp_solver` | `auto` | `serial` or `parallel` Gaussian-process solver (`auto` picks parallel on a GPU) |

## Planet

Every entry except `name` is `{value, prior, ...}` with one of the priors `fixed`, `uniform` (`low`, `high`), `log_uniform` (`low`, `high`) or `gaussian` (`sigma`, and optionally `low`, `high`).
For several planets, give a list with one entry per planet.

| Key | Units | |
|---|---|---|
| `name` | | target name, used in file names |
| `period` | days | orbital period |
| `t0` | BMJD_TDB | mid-transit time |
| `eclipse_time` | BMJD_TDB | mid-eclipse time, instead of `t0` (circular orbit, fixed period) |
| `duration` | days | transit duration |
| `a_rs` | | scaled semi-major axis $a/R_*$, instead of `duration` |
| `b` | | impact parameter |
| `rprs` | | radius ratio $R_p/R_*$ |
| `ecc`, `omega` | radians for `omega` | eccentricity and argument of periastron, fixed only (default 0) |
| `eclipse_depth_ppm` | ppm | planet flux at eclipse, for `eclipse` and `phase_curve` |
| `phase_c1_ppm`, `phase_d1_ppm` | ppm | cosine and sine terms of a `phase_curve` |

## Star

| Key | Default | |
|---|---|---|
| `teff`, `logg`, `feh` | | effective temperature (K), surface gravity ($\log_{10}$ cgs) and metallicity (dex) |
| `teff_sigma`, `logg_sigma`, `feh_sigma` | | their uncertainties, used by `ld_prior: stellarprior` |
| `ld_model` | `mps1` | ExoTiC-LD stellar-model grid (`mps1`, `mps2`, `stagger`, `kurucz`, `phoenix`) |
| `ld_data_path` | `~/.cache/koala/exotic_ld_data` | where the stellar-model files are kept |
| `ld_interpolate_type` | `trilinear` | how ExoTiC-LD interpolates the grid |
| `ld_coefficients` | | your own fixed coefficients (for `ld_prior: fixed` or `fixed_u1`) |
| `ld_prior_model` | `ld_model`, or `stagger` if that isn't set | stellar-model grid for `stellarprior` |
| `ld_prior_n_grid` | 9 | grid points per stellar parameter for `stellarprior` |
| `ld_prior_nsigma` | 5 | half-width of that grid, in standard deviations |
| `ld_prior_min_sigma` | 1e-4 | smallest allowed prior width |
| `ld_prior_cache`, `ld_prior_cache_dir` | `true`, `.koala_cache/ld_prior` | cache the `stellarprior` grid |
| `ld_mu_min` | | leave out the extreme limb when calibrating `sing` |
| `rotation_period` | days | rotation period of a spotted star |
| `spots` | | a list of spots, each with `latitude_deg`, `longitude_deg`, `radius_deg`, `contrast` and optionally `contrast_prior_width` |

## Wavelength channels and clipping

| Key | Default | |
|---|---|---|
| `resolution.high` | | a resolving power, `native`, or `reference` |
| `resolution.low` | | an optional coarser stage, with the same choices |
| `resolution.reference_grid`, `resolution.reference_grid_lr` | | CSV files of bin edges for `reference` |
| `outlier_clip.whitelight_sigma` | 5 | clipping threshold for the white-light fit; 0 turns it off |
| `outlier_clip.spectroscopic_sigma` | 0 | clipping threshold applied to the `resolution.low` channels before the fine stage; 0 turns it off |
| `outlier_clip.mask_integrations_start`, `outlier_clip.mask_integrations_end` | | number of integrations to leave out at the start and end |

## Trend priors

The `trend_priors` block sets the priors on the white-light trend, in the same `{value, prior, ...}` form as the planet.
The defaults are:

| Key | Default prior | |
|---|---|---|
| `c` | uniform(0.9, 1.1) | normalization |
| `v`, `v2`, `v3`, `v4` | uniform(−0.1, 0.1) | polynomial coefficients, per day to the power of the order |
| `A` | uniform(−0.1, 0.1) | ramp amplitude |
| `tau` | log-uniform(0.001, 0.1) days | ramp timescale |
| `jitter` | log-uniform(1e-5, 1e-2) | extra white noise, added in quadrature to the flux uncertainties |

Time is measured from the first integration.
`jitter: {value: 0.0, prior: fixed}` uses the reported uncertainties alone.
In the channel fits, the polynomial coefficients have uniform(−10, 10) priors on a time axis scaled to run over about one unit across the observation.

## Flags: the model

| Key | Default | |
|---|---|---|
| `light_curve_model` | `transit` | `transit`, `eclipse` or `phase_curve` |
| `transit_engine` | `jaxoplanet` | `jaxoplanet` or `harmonica` |
| `detrending_type` | `linear` | see [trends](../guide/trends.md) |
| `ld_profile` | `quadratic` | `quadratic`, `power2` or `kipping`, see [limb darkening](../guide/limb_darkening.md) |
| `ld_prior` | `uniform` | `uniform`, `stellarprior`, `gaussian`, `sing`, `fixed` or `fixed_u1` |
| `ld_gaussian_sigma` | 0.2 | width of the `gaussian` limb-darkening prior |
| `ld_uniform_basis` | `uplus_uminus` | uniform quadratic priors on $u_1 \pm u_2$, or `coefficients` for a box in $u_1, u_2$ |
| `ld_uniform_coefficient_bounds` | `[0, 1]` | that box, which also bounds the free $u_2$ of `fixed_u1` |
| `ld_parameterization` | `auto` | power-2 sampling coordinates: `maxted_smooth`, `decorrelated` or `coefficients` |
| `spot_center`, `spot_width`, `spot_amp` | | starting guesses for a spot crossing (BMJD_TDB, days, flux) |
| `spot_center2`, `spot_width2`, `spot_amp2` | | the same for a second spot (`spot_center_2`, `spot_width_2`, `spot_amp_2` also work) |
| `t_jump_guess`, `jump_guess` | | starting time and size of a step |
| `mask_start`, `mask_end` | | an older way of leaving out time ranges, combined with `exclude_times` |
| `spectro_fixed_timescale_trends` | `true` | keep the white-light ramp timescale in the channel fits |
| `spectro_trend_basis` | | `elapsed_days` to put channel polynomial priors on time since the first integration |
| `spectro_trend_bounds` | | per-coefficient `[low, high]` bounds for the channel trends, for example `A_jump: [-10, 10]` |
| `trend_inference` | `sampled_uniform` | `gaussian_marginalized` integrates the linear trend coefficients analytically under Gaussian priors |
| `trend_prior_scale` | 0.1 | the width of those Gaussian priors |
| `spectro_free_t0` | `false` | fit a transit time in every channel, see [timing spectra](../case_studies/timing_spectra.md) |
| `spectro_t0_half_width_s` | 1800 | how far (in seconds) each channel's transit time can move |
| `spectro_joint_geometry` | `false` | fit the orbit together with all the channels |
| `spectro_joint_geometry_prior_inflation` | 3 | how much wider than the white-light uncertainties its priors are |
| `harmonica_max_order` | 1 | see [Harmonica](../case_studies/harmonica.md) |
| `harmonica_spectro_parameterization` | `delta_r` | `delta_r`, `fractional` or `half_area` |
| `harmonica_spectro_odd_frac_sigma` | 0.1 | prior width on the asymmetric outline terms |
| `harmonica_spectro_fit_jitter` | `true` | fit jitter in each Harmonica channel |

## Flags: sampling

| Key | Default | |
|---|---|---|
| `num_chains` | 4 | chains per fit |
| `whitelight_num_chains`, `lowres_num_chains`, `highres_num_chains` | | chains for one stage |
| `whitelight_num_warmup`, `lowres_num_warmup`, `highres_num_warmup` | | warm-up steps for one stage |
| `whitelight_num_samples`, `lowres_num_samples`, `highres_num_samples` | 1000 | draws kept per chain |
| `random_seed` | 555 | random seed (the `FIT_JWST_SEED` environment variable overrides it) |
| `spectro_min_depth_ess` | 400 | effective samples needed for each channel's depth, summed over chains |
| `spectro_max_divergences` | 0 | divergences allowed per channel |
| `spectro_sampler` | `independent_nuts` | `independent_hmc`, or `joint_nuts` (needed for `spectro_joint_geometry`) |
| `whitelight_mass_matrix` | `auto` | `laplace` or `adaptive` |
| `spectro_mass_matrix` | `laplace` | `adaptive` learns a dense mass matrix for each channel |
| `whitelight_reuse_samples` | | an NPZ file of white-light draws to use instead of fitting (skips the convergence checks, so use with care) |

## Flags: running

| Key | Default | |
|---|---|---|
| `analysis_stage` | `all` | `whitelight`, `prep` (prepare the channels and stop) or `highres` |
| `need_lowres` | `true` | run the coarse stage when `resolution.low` is set |
| `plots` | `full` | `minimal` skips the larger diagnostic figures |
| `cpu_host_devices` | cores available | number of CPU devices to split channels across (`1` for one) |
| `spectro_chunk_size` | `auto` | channels fitted at the same time (`vmap_chunk` also works) |
| `chunk_mode` | `serial` | `parallel` to split the channels over several jobs (for example a Slurm array), then `combine` |
| `chunk_parallel_job_count`, `chunk_parallel_job_index` | from Slurm | the number of jobs and this job's index for `parallel` |
| `jax_compilation_cache_dir` | `.koala_cache/jax` | where compiled models are cached (`false` turns it off) |
| `spectro_cadence_reduction` | `auto` | `off` turns off the exact speed-up for integrations far from transit |
| `spectro_transit_grid` | `off` | `auto` interpolates the transit on a time grid where it can (an approximation, so it's off by default) |
| `spectro_transit_grid_nodes` | 769 | number of points in that interpolation |
| `grid_prime_samplers` | `true` on GPUs | compile the model grid's samplers ahead of time |
| `grid_cache_memory_mb` | | memory budget for compiled model-grid programs (`0` for no limit) |

## Joint fits

A file with a top-level `joint` block runs a [joint fit](../case_studies/transmission_soss.md):

| Key | Default | |
|---|---|---|
| `joint.members` | required | the configuration files to fit together, relative to the joint file |
| `joint.apply_to` | `[]` | configuration files that get the joint orbit without contributing to it |
| `joint.shared` | `[a_rs, b]` | the shared orbit (the only choice for now) |
| `joint.visits` | `auto` | `auto` groups members with overlapping times, or give lists of member file names |
| `planet.a_rs`, `planet.b` | from the first member | priors on the shared orbit |
| `output_dir` | `results/joint` | relative to the joint file |
| `host_device` | automatic | used for every member |
