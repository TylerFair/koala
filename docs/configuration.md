# Configuration

One YAML file describes a fit. For a complete starting example, see [your first fit](quickstart.md).
The tables below list configuration keys; [planet parameters](tutorials/planet_parameters.md) explains priors and time conventions.

## Choose the wavelength grid

The final grid is set by `resolution.high`.
A number requests constant resolving power, `native` keeps the input channels, and `reference` uses a supplied grid.
The optional `resolution.low` adds a coarse fit before the final stage.
Remove it to go straight to the final grid.
These are wavelength bins.
koala fits every integration and never averages integrations in time.

## Sampling and resuming

By default, each fit uses NUTS with four chains started from a Laplace approximation.
Each chain retains 1,000 draws.
Multi-chain checks require rank-normalized split R-hat below 1.01 and at least 400 combined bulk and tail effective samples for the checked parameters.
The default permits no divergences.
See [sampling diagnostics](troubleshooting.md#sampling-diagnostics) for which parameters are checked at each stage.

Independent spectral chains that miss the mixing threshold without divergences can continue for up to three more draw blocks before a sampler fallback.
The fit retains 1,000 evenly spaced draws per chain from the extended history and checks those draws again.
White-light extensions retain their additional draws.
The diagnostic record gives the generated and retained counts.

Repeat the same command to resume compatible completed batches.
koala checks the input and model identity before reusing results.
Changing the spectroscopic batch size or compilation-cache directory does not invalidate an otherwise matching white-light posterior.

The automatic batch size balances memory use against the cost of waiting for the slowest channel.
An explicit `spectro_chunk_size` counts wavelength channels, each with its own chains.
Independent chains share compilation and deterministic preparation, while keeping separate random keys, initial offsets, and step-size adaptation.
Larger batches can reduce dispatch overhead but need more memory for gradients.

## Reuse a white-light posterior

You can explicitly import `whitelight_samples_by_chain.npz` with `flags.whitelight_reuse_samples`.
It must have leading chain and draw axes for each parameter.
This bypasses both white-light sampling and its convergence check.
Use an independently validated posterior with matching data, model, native geometry parameters, and priors.
Ordinary checkpoint reuse still checks its saved scientific identity.

## Reference

The tables below group the settings by purpose.
Unknown names under `flags` produce a warning and are ignored.

### Joint white light

A top-level `joint` mapping selects the joint pipeline.
See [joint white light](tutorials/planet_parameters.md#joint-white-light-across-observations) for a complete example.

| Key | Default | Meaning |
|---|---|---|
| `joint.members` | Required list | Ordinary configuration paths, relative to the joint file. File stems must be unique. |
| `joint.apply_to` | `[]` | Additional configurations receiving the joint orbit without entering its likelihood. |
| `joint.shared` | `[a_rs, b]` | The only supported shared orbital parameter set. Fixed circular period required. |
| `joint.visits` | `auto` | Group overlapping observation ranges, or give lists of member file stems covering every member once. |
| `planet.a_rs`, `planet.b` | First member | Shared parameter priors. A free duration prior requires an explicit joint `a_rs` prior. |
| `output_dir` | `results/joint` | Relative to the joint file. Joint products go in `joint/` and member products in each file-stem directory. |
| `host_device` | `gpu` | Device applied to all configurations. |

### Data and execution

| Top-level key | Value or default | Meaning |
|---|---|---|
| `planet` | Mapping | Target name and parameter priors. Table below. |
| `stellar` | Mapping | Limb-darkening inputs and optional surface spots. |
| `instrument` | Required | Instrument/mode from the table below. |
| `order` | 1 or 2 | Required SOSS order. |
| `nrs` | 1 or 2 | NIRSpec detector. |
| `path` | `.` | Base directory. |
| `input_dir` | `<planet.name>_NIRSPEC` | Input directory relative to `path`. |
| `input_file` | Required | Extracted spectra filename. Alias `fits_file`. |
| `input_format` | `auto` | `auto`, `exotedrf`, `sparta`, or `eureka`. |
| `output_dir` | `<planet.name>_RESULTS` | Results directory relative to `path`, or absolute. |
| `host_device` | `gpu` | `cpu` or `gpu`. Set explicitly for a portable example. |
| `gp_solver` | `auto` | `auto`, `serial`, or `parallel`. [GP details](tutorials/systematics.md#gaussian-processes). |
| `exclude_times`, `exclude_integrations` | No exclusions | Also accepted under `flags`. [range syntax](tutorials/systematics.md#exclude-time-ranges). |
| `wavelength_filter` | Instrument window | Mapping of wavelength limits in µm. |
| `wavelength_filter.wl_min`, `wavelength_filter.wl_max` | µm | Lower/upper limits outside SOSS. |
| `wavelength_filter.wl_min_o1`, `wavelength_filter.wl_max_o1`, `wavelength_filter.wl_min_o2`, `wavelength_filter.wl_max_o2` | µm | SOSS order-specific limits. |
| `wavelength_masks` | No masks | Excluded wavelength bounds in µm, as a flat sequence of min/max pairs. |

#### Input formats

| `input_format` | Extracted product |
|---|---|
| `exotedrf` | FITS wavelength, wavelength error, flux, flux error, and time arrays. SOSS orders occupy separate extensions. |
| `sparta` | Filtered pickle from `gather_and_filter.py`, containing `wavelengths`, `times`, `data`, and `errors`. |
| `eureka` | Stage 3 `*_SpecData.h5` or Stage 4 `*_LCData.h5`. Stage 4 is already binned. |

`auto` identifies the format from its contents.
Data should be barycentric TDB times.
Full BJD arrays are converted to BMJD. [YAML epochs need the matching origin](tutorials/planet_parameters.md#transit-and-eclipse-times).

#### Instrument modes

| `instrument` | Selection |
|---|---|
| `NIRISS/SOSS` | `order: 1` or `2` |
| `NIRSPEC/PRISM` | `nrs: 1` or `2` |
| `NIRSPEC/G395H`, `NIRSPEC/G395M` | `nrs: 1` or `2` |
| `NIRSPEC/G235H`, `NIRSPEC/G235M` | `nrs: 1` or `2` |
| `NIRSPEC/G140H`, `NIRSPEC/G140M` | F100LP. `-F100` aliases accepted |
| `NIRSPEC/G140H-F070`, `NIRSPEC/G140M-F070` | F070LP |
| `NIRCAM/F322W2`, `NIRCAM/F444W` | No detector key |
| `MIRI/LRS` | No detector key |

Names are case-insensitive.
Each mode limits wavelengths to its throughput window.
MIRI/MRS and imaging modes are not supported.

### Wavelength grids and clipping

| Key | Value | Meaning |
|---|---|---|
| `resolution.high` | Number, `native`, `reference` | Final resolving power, input grid, or supplied grid. |
| `resolution.low` | Optional. Same choices | Coarse grid before the final stage. |
| `resolution.reference_grid` | CSV path | Grid for `high: reference`. |
| `resolution.reference_grid_lr` | CSV path | Grid for `low: reference`. |
| `pixels.high`, `pixels.low` | Pixel count or `reference` | Alternative binning by detector-column count when `resolution` is absent. |
| `pixels.reference_grid`, `pixels.reference_grid_lr` | CSV paths | Reference grids for pixel-mode stages. |
| `outlier_clip.whitelight_sigma` | Default 4 | White-light clipping threshold. |
| `outlier_clip.spectroscopic_sigma` | Default 4 | Spectroscopic clipping threshold. |
| `outlier_clip.mask_integrations_start` | Count | Remove integrations from the start. |
| `outlier_clip.mask_integrations_end` | Count | Remove integrations from the end. |

### Planet

Every entry except `name` is a `{value, prior, ...}` mapping.
See [planet parameters](tutorials/planet_parameters.md) for prior fields, multiple planets, and timing.

| `planet` key | Unit | Role |
|---|---|---|
| `name` | String | Target label used in filenames. |
| `period` | d | Orbital period. Fixed or free. |
| `t0` | BMJD_TDB | Primary-transit epoch. |
| `eclipse_time` | BMJD_TDB | Alternative to `t0` for a circular eclipse/phase model with fixed period. |
| `duration` | d | Transit-duration geometry. |
| `a_rs` | Dimensionless | Scaled semimajor axis for Keplerian, emission, or Harmonica geometry. |
| `b` | Dimensionless | Impact parameter. |
| `rprs` | Dimensionless | Planet-to-star radius ratio. |
| `ecc` | Dimensionless | Eccentricity. Fixed only, default 0. |
| `omega` | Radians | Argument of periastron. Fixed only, default 0. |
| `eclipse_depth_ppm` | ppm | Planet flux at secondary eclipse ($f_d$): the uniform emission of `eclipse`, and the eclipse depth of `phase_curve`. |
| `phase_c1_ppm` | ppm | `phase_curve` cosine coefficient $C_1$. Any sign. |
| `phase_d1_ppm` | ppm | `phase_curve` sine coefficient $D_1$. Any sign. |

For the sinusoidal phase-curve definition and sign convention, see [eclipses and phase curves](tutorials/eclipses_and_phase_curves.md#sinusoidal-phase-curve).

Removed `*_prior_width_*`, `a_rs_prior_min`, and `a_rs_prior_max` settings are replaced by the corresponding parameter's `sigma`, `low`, and `high`.
`flags.fit_geometry` is replaced by the parameter priors.
The removed phase-map keys `dayside_flux_ppm`, `nightside_flux_ppm`, and `hotspot_offset_deg` raise an error naming `eclipse_depth_ppm`, `phase_c1_ppm`, and `phase_d1_ppm`.

### Trend priors

The optional top-level `trend_priors` block changes white-light priors on `c`, `v`, `v2`, `v3`, `v4`, `A`, `tau`, and `jitter`.
Each entry uses `{prior, low, high, sigma, value}` as appropriate.
Omitted entries keep their defaults.
See [white-light trend priors](tutorials/systematics.md#white-light-trend-priors) for examples and units.

### Stellar inputs

| `stellar` key | Value or unit | Meaning |
|---|---|---|
| `teff`, `teff_sigma` | K | Effective temperature and uncertainty. |
| `logg`, `logg_sigma` | log₁₀(cm s⁻²) | Surface gravity and uncertainty. |
| `feh`, `feh_sigma` | dex | Metallicity and uncertainty. |
| `ld_model` | Atmosphere grid. Default `mps1` | ExoTiC-LD grid, e.g. `stagger`. |
| `ld_data_path` | Directory | Downloaded atmosphere and throughput data. |
| `ld_interpolate_type` | Default `trilinear` | ExoTiC-LD interpolation choice. |
| `ld_coefficients` | Two numbers | Explicit coefficients for the chosen law. |
| `ld_prior_model` | `ld_model`, or `stagger` if absent | Atmosphere grid for stellar-uncertainty priors. |
| `ld_prior_n_grid` | Default 9 | Grid points per stellar parameter for atmosphere-derived priors. |
| `ld_prior_nsigma` | Default 5 | Half-range of that grid in stellar standard deviations. |
| `ld_prior_min_sigma` | Default `1e-4` | Minimum width for atmosphere-derived marginal coefficient priors. |
| `ld_mu_min` | Minimum µ | Cut the extreme limb from Sing profile calibration. |
| `ld_prior_cache` | Default `true` | Cache atmosphere-derived priors. |
| `ld_prior_cache_dir` | Default `.koala_cache/ld_prior` | Prior-cache directory. |
| `rotation_period` | d | Rotation period for a spotted stellar surface. |
| `spots` | List of mappings | Surface spots. Fields below. |
| `spots[].latitude_deg`, `longitude_deg`, `radius_deg` | Degrees | Fixed spot latitude, longitude at `t0`, and angular radius. |
| `spots[].contrast` | 0 to 1 | Photosphere-to-dark contrast. |
| `spots[].contrast_prior_width` | Dimensionless | Positive width frees the spot contrast. |

### Model flags

| `flags` key | Value or default | Meaning |
|---|---|---|
| `transit_engine` | `jaxoplanet`, `harmonica` | Light-curve engine. Default jaxoplanet. |
| `light_curve_model` | `transit`, `eclipse`, `phase_curve` | Astrophysical signal. Default transit. |
| `detrending_type` | Default `linear` | [Trend names and shapes](tutorials/systematics.md#trend-names). |
| `ld_profile` | `quadratic`, `power2`, `kipping` | Intensity law. Default `quadratic`. `kipping` is the quadratic law sampled in Kipping (2013) $(q_1, q_2)$. |
| `ld_prior` | `fixed`, `gaussian`, `uniform`, `stellarprior`, `sing`, `fixed_u1` | [Coefficient prior](tutorials/limb_darkening.md). Default `uniform` for quadratic. For explicit power-2, an omitted prior resolves to `stellarprior` when all three stellar uncertainties are supplied, otherwise `gaussian`. |
| `ld_parameterization` | `auto`, `maxted_smooth`, `coefficients`, `decorrelated` | Sampling coordinates for power-2 limb darkening. `auto` uses `maxted_smooth` for `gaussian`/`uniform` priors. |
| `ld_uniform_basis` | `uplus_uminus`, `coefficients` | Coordinates for uniform quadratic coefficients. Default `uplus_uminus`, u+ ~ U(-1,2), u- ~ U(-2,2). |
| `ld_uniform_coefficient_bounds` | Default `[0, 1]` | Finite increasing `[low, high]` for both direct quadratic coefficients with `ld_prior: uniform` and `ld_uniform_basis: coefficients`. E.g. `[-2, 2]` or `[-3, 3]`. Also sets the free u2 bounds for `fixed_u1`. Applied in white light and spectroscopy. Not a physical LD-triangle constraint. |
| `trend_inference` | `sampled_uniform`, `gaussian_marginalized` | Sample or integrate linear trend coefficients. |
| `spectro_free_t0` | Default `false` | Fit an independent transit epoch in every spectroscopic bin with the white-light orbit fixed. [timing outputs and conventions](tutorials/time_shift.md). Requires an unspotted transit and fixed geometry. |
| `spectro_t0_half_width_s` | Default 1800 seconds | Half-width of the uniform epoch-offset prior when `spectro_free_t0` is enabled. Must be finite and positive. |
| `spectro_trend_basis` | `batman_normalized` for sampled constant through quartic trends, otherwise `elapsed_days` | Normalized time is x=(t−mean(observed t))/span(observed t). The command-line sampled-polynomial defaults are independent U(−10,10) priors, including the intercept. Output coefficients retain elapsed-day units. |
| `spectro_trend_bounds` | Mapping of coefficient names to `[lower, upper]` | Bounds for sampled `c`, `v`, `v2`, `v3`, `v4`, `A`, `log_tau`, `A_spot`, `A_spot2`, `A_jump`, and `A_gp` in the selected basis. These do not truncate Gaussian priors. |
| `spectro_fixed_timescale_trends` | Default `true` | Reuse the white-light ramp shape in spectroscopy. Set `false` to sample a channel timescale. Harmonica requires `false` for a ramp. |
| `ld_gaussian_sigma` | Default 0.2 | Positive Gaussian coefficient-prior width, used in white light and spectroscopy where supported. Kipping uses propagated widths in q coordinates. |
| `trend_prior_scale` | Default 0.1 per coefficient | Standard deviation of every Gaussian trend-coefficient prior for `gaussian_marginalized` (time in days since the first integration). |
| `exclude_times`, `exclude_integrations` | Lists of ranges | Inclusive time or integration exclusions. |
| `mask_start`, `mask_end` | Legacy time bounds | Merged with exclusions. `cut_phase_to_transit` keeps ±3 h around transits. |
| `spot_amp`, `spot_center`, `spot_width` | Flux, BMJD_TDB, d | First template's initial guesses. |
| `spot_amp2`, `spot_center2`, `spot_width2` | Same units | Second template. Aliases `spot_amp_2`, `spot_center_2`, `spot_width_2`. |
| `jump_guess`, `t_jump_guess` | Flux, BMJD_TDB | Step amplitude and time guesses. |

### Execution flags

| `flags` key | Value or default | Meaning |
|---|---|---|
| `analysis_stage` | `all`, `whitelight`, `prep`, `highres` | Full fit, white light only, prepare spectral inputs, or final grid. Later stages need preceding products. |
| `need_lowres` | Default `true` | Allow the coarse stage when a low grid is supplied. |
| `plots` | `full`, `minimal` | Full plots or a reduced set. |
| `random_seed` | Default 555 | Random seed. `FIT_JWST_SEED` overrides it. |
| `num_chains` | Default 4 | Independent chains vectorized on one device. |
| `whitelight_num_chains`, `lowres_num_chains`, `highres_num_chains` | Positive integer | Override the chain count for a stage. |
| `spectro_chunk_size` | `auto` or positive integer | Channels resident together. Alias `vmap_chunk`. GPU jaxoplanet independent sampling defaults to `auto`, balances batch padding, and retains memory headroom. CPU independent sampling defaults to one channel per host device (`cpu_host_devices`), or 40 on one device. |
| `cpu_host_devices` | Allocated cores | CPU runs of `koala`, `python -m koala` or `koala.fit` give JAX one host device per core and split independent-sampler channels across them. `1` keeps a single device. GPU runs are unchanged. |
| `chunk_mode` | `serial`, `parallel`, `combine` | Process batches locally, split across jobs, or combine completed batches. |
| `chunk_parallel_job_count` | Integer | Number of jobs for parallel chunk mode. |
| `chunk_parallel_job_index` | Integer | This job's zero-based index. |
| `jax_compilation_cache_dir` | Directory or `false` | Persistent JAX compilation cache. Default `.koala_cache/jax`, `false` disables it. |
| `grid_prime_samplers` | Boolean. GPU grids default `true` | Prime the exact batched sampling executable while compiling and running initialization. No disposable chains are sampled. |
| `grid_cache_memory_mb` | Nonnegative number. `0` disables eviction | Resident-memory budget for completed grid compilation caches. Preparation and persistent executable caches remain available. |
| `spectro_sampler` | `independent_nuts`, `independent_hmc`, `joint_nuts` | Spectroscopic sampler. Independent NUTS is the default. |
| `whitelight_reuse_samples` | NPZ path. Unset by default | Reuse saved white-light samples with leading chain and draw axes for each parameter. Bypasses sampling and its convergence gate. Use only an independently validated posterior with matching model and data. |
| `whitelight_mass_matrix` | `auto`, `laplace`, `adaptive` | Metric for the white-light NUTS chain. `auto` picks from the trend model. |
| `spectro_mass_matrix` | `laplace`, `adaptive` | Metric for independent spectroscopic NUTS. `adaptive` learns a dense mass matrix per channel (useful when limb darkening sits on a prior edge). |
| `spectro_cadence_reduction` | `auto`, `off` | Use exact sufficient statistics for eligible out-of-transit data. Every integration still contributes. |
| `spectro_transit_grid` | `auto`, `off` | Interpolate eligible transit calculations on a time grid. |
| `spectro_transit_grid_nodes` | Default 769 | Transit interpolation nodes. |
| `spectro_joint_geometry` | Default `false` | Share free geometry across a stage's channels. |
| `spectro_joint_geometry_prior_inflation` | Default 3.0 | Multiply white-light geometry uncertainties by this factor. |
| `spectro_min_depth_ess`, `spectro_max_divergences` | Defaults 400, 0 | Combined bulk and tail ESS across chains, and total allowed divergences per channel. [diagnostics](troubleshooting.md#sampling-diagnostics). |
| `whitelight_num_warmup`, `lowres_num_warmup`, `highres_num_warmup` | Nonnegative integer | Stage warmup overrides. |
| `whitelight_num_samples`, `lowres_num_samples`, `highres_num_samples` | Default 1000 | Retained draws per chain for each stage. |

### Harmonica flags

| `flags` key | Value or default | Meaning |
|---|---|---|
| `harmonica_max_order` | Default 1 | Maximum boundary harmonic order. |
| `harmonica_spectro_parameterization` | `delta_r`, `fractional`, `half_area` | Coordinates for the asymmetric boundary. |
| `harmonica_spectro_fit_jitter` | Default `true` | Fit spectroscopic extra white noise. |
| `harmonica_spectro_odd_frac_sigma` | Default 0.1 | Width of the odd-harmonic fractional prior. |

The [Harmonica reference](tutorials/harmonica.md) describes boundary and limb products.
