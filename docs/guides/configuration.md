# Configuration reference

Run a configuration with:

```bash
python fit_jwst.py -c config.yaml
```

## Dataset and target

| Key | Type / default | Meaning |
|---|---|---|
| `planet` | mapping / required | `name`, `period`, `duration`, `t0`, `b`, `rprs`; optional `a_rs`, `ecc`, `omega` |
| `stellar` | mapping / required | `teff`, `logg`, `feh`, LD grid settings and optional uncertainties |
| `instrument` | string / required | Verified strings include NIRISS/SOSS and NIRSPEC G395H, G395M, G140H, G235H, PRISM |
| `order` | integer / required for SOSS | SOSS spectral order |
| `nrs` | integer / required for NIRSpec | Detector number |
| `path` | path / `.` | Base path |
| `input_dir` | path / target-derived | Input directory below `path` |
| `output_dir` | path / `<target>_RESULTS` | Output directory below `path` |
| `fits_file` | path / required | Box-spectrum FITS filename |
| `resolution` or `pixels` | mapping / one required | `high`, `low`, and optional `reference_grid` bins |
| `outlier_clip` | mapping / `{}` | `whitelight_sigma`, `spectroscopic_sigma`, optional integration masks |
| `time_binning` | mapping / disabled | `enabled`, `dt_seconds`, `method`, `whitelight`, `spectroscopic` |
| `wavelength_filter`, `wavelength_masks` | mapping/list / none | Include ranges and mask wavelength intervals during cube preparation |
| `host_device` | string / read from config | Requested host device label |

## Model flags

| `flags` key | Type / default | Meaning |
|---|---|---|
| `detrending_type` | string / `linear` | Trend family |
| `transit_engine` | string / `jaxoplanet` | `jaxoplanet` or `harmonica` |
| `param_method` | string / `duration` | Transit geometry parameterization; Harmonica defaults to `a_rs` |
| `ld_profile` | string / `quadratic` | `quadratic` or `power2` |
| `ld_prior` | string / inferred | `fixed`, `widegaussian`, `informed`, `sing`, or `uniform` |
| `fix_ld` | bool / false | Legacy fixed-LD switch |
| `interpolate_ld`, `interpolate_trend` | bool / false | Interpolate low-resolution LD or linear trend to high resolution |
| `need_lowres` | bool / true | Run low-resolution analysis |
| `spectro_fixed_timescale_trends` | bool / true (production default) | Fix spectroscopic explinear `tau` to white-light median |
| `trend_inference` | string / `sampled_uniform` | Sample coefficients or use `gaussian_marginalized` |
| `trend_prior_means`, `trend_prior_scales` | mappings / model defaults | Gaussian marginalization priors |
| `spot_amp`, `spot_center`, `spot_width` | numbers / 0 | First spot initialization/template |
| `spot_amp2`, `spot_center2`, `spot_width2` | numbers / 0 | Second spot settings; `_2` aliases exist |
| `t_jump_guess`, `jump_guess` | number / none, 0 | Discontinuity initialization |
| `mask_start`, `mask_end` | number/expression / false | Time mask boundaries |
| `hr_custom_ld_path` | path / none | Custom high-resolution power-2 LD table |
| `hr_custom_ld_smooth_window` | int / 1 | Custom-LD smoothing width |
| `transit_window_optimization` | string / `auto` | `auto` or `off` |
| `jaxoplanet_kernel` | string / `auto` | Kernel selector; verified alternatives are `stock`, `streamed`, `fused`, `quadratic_specialized`, `quadratic_local_jvp`, `native_power2` |

## Sampling and execution flags

| `flags` key | Type / default | Meaning |
|---|---|---|
| `spectro_sampler` | string / `independent_nuts` (production default) | `independent_nuts`, `independent_hmc`, `joint_nuts`, or `laplace_is` |
| `spectro_mass_matrix` | string / `laplace` (production default) | `laplace` or `adaptive` |
| `spectro_sampling_mode` | string / `auto` | Legacy `auto`, `joint`, or `independent` routing |
| `spectro_jitter_prior` | string / `lognormal` | `lognormal` or `log_uniform` |
| `spectro_jitter_prior_center`, `spectro_jitter_prior_scale` | float / 0.5, 2.0 | Lognormal jitter controls |
| `vmap_chunk` | int/bool / 40 for independent samplers | Resident channel width |
| `random_seed` | int / 555 | Master PRNG seed |
| `analysis_stage` | string / `all` | `all`, `whitelight`, `prep`, or `highres` |
| `chunk_mode` | string / `serial` | `serial`, `parallel`, or `combine` |
| `chunk_parallel_job_count`, `chunk_parallel_job_index` | int / none | Parallel chunk partition |
| `compile_box` | bool / false | Enable persistent JAX compilation cache |
| `jax_compilation_cache_dir` | path / `/scratch/midway3/tfairnington/jax_cache` | Cache used by `compile_box` |
| `save_whitelight_trace` | bool / false | Save white-light trace plot |
| `whitelight_geometry_estimator` | string / `posterior_median` | Geometry handoff estimator |
| `whitelight_log_likelihood_batch_size` | int / 64 | Draw batch for likelihood evaluation |
| `<stage>_num_warmup`, `<stage>_num_samples` | int / 1000, 1000 | `whitelight`, `lowres`, or `highres` MCMC lengths |
| `spectro_max_tree_depth`, `spectro_target_accept` | int,float / 10, model default | Generic NUTS controls; stage prefixes override |
| `spectro_hmc_num_steps`, `spectro_hmc_trajectory_jitter` | int,float / 16, 0 | HMC trajectory controls; stage prefixes override |
| `spectro_gradient_diagnostic` | string / `first` | `off`, `first`, or `each` |
| `spectro_gradient_diagnostic_strict` | bool / false | Abort on gradient diagnostic failure |
| `spectro_min_depth_ess`, `spectro_max_divergences` | int / 400, 0 | Spectroscopic quality gate |
| `whitelight_min_ess`, `whitelight_max_divergences` | int / 400, 0 | White-light quality gate |
| `whitelight_max_extra_blocks` | int / 3 | Additional white-light sampling blocks allowed by the gate |
| `spectro_width_selection`, `lowres_width_selection`, `highres_width_selection` | path / none | Measured width manifest |
| `spectro_batch_plan`, `lowres_batch_plan`, `highres_batch_plan` | path / none | Difficulty-aware batch plan |
| `bin_time`, `bin_dt_seconds`, `bin_method` | bool,float,string / false, config value, `mean` | Flag-level time-binning controls |
| `bin_whitelight`, `bin_spectroscopic` | bool / inherited | Apply time binning to either stage |

Laplace metric controls accept generic `spectro_laplace_*` and stage-specific `lowres_laplace_*`/`highres_laplace_*`: `warmup` (150), `target_accept` (0.95 for NUTS), `max_tree_depth` (10), `start_at_map` (false), `hessian_method` (`exact`), `fd_relative_step` (2e-4), `fuse_program` (false), `trust_radius` (5), and `map_decrement_tolerance` (1e-4). White-light equivalents are `whitelight_mass_matrix`, `whitelight_laplace_warmup` (200), `target_accept` (0.9), `max_tree_depth` (10), `trust_radius` (5), and `hessian_method` (`finite_difference`).

Laplace importance sampling accepts generic, `spectro_`, `lowres_`, or `highres_` prefixed `laplace_is_*` keys: `output` (`imh`), `num_draws` (4096), `rounds` (2), `draw_chunk_size` (256), `student_df` (3), `scale_inflation` (1.5), `wide_fraction` (0), `wide_scale` (3), `map_maxiter` (200), `map_tol` (1e-4), `trust_radius` (5), `khat_threshold` (0.7), `min_ess` (400), `min_ess_fraction` (0.2), `min_imh_acceptance` (0.2), `imh_thin` (8), `fallback` (true), and `force` (false).

Harmonica adds `harmonica_max_order` (1), `harmonica_spectro_parameterization` (`delta_r`), `harmonica_spectro_fit_jitter` (true), `harmonica_spectro_odd_frac_sigma` (0.1), and legacy `harmonica_wl_parameterization`. Stage prefixes `harmonica_wl`, `harmonica_lr`, and `harmonica_hr` accept `dense_mass`, `regularize_mass_matrix`, `max_tree_depth`, and `target_accept`. Sing adds `ld_sing_offset`, `ld_sing_offset_path`, `ld_sing_calibration_warmup` (1000), `ld_sing_calibration_samples` (1000), and `ld_sing_calibration_min_ess`.
