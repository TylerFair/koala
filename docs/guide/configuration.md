# Configuring a fit

Every koala fit is described by one YAML file.
[Your first fit](../quickstart.md) used the smallest file that works, and this page goes through each part of it in a bit more detail, along with the settings you're most likely to change.
The [settings reference](../reference/settings.md) lists everything else.

## The planet

Each property of the orbit has a starting value and a prior:

```yaml
planet:
  name: WASP-39
  period: {value: 4.05528043, prior: fixed}
  t0: {value: 59787.055, prior: uniform, low: 59787.005, high: 59787.105}
  duration: {value: 0.117, prior: log_uniform, low: 0.01, high: 1.0}
  b: {value: 0.45, prior: uniform, low: 0.0, high: 1.5}
  rprs: {value: 0.146, prior: uniform, low: 0.01, high: 0.5}
```

There are four kinds of prior:

| `prior` | Also needs | |
|---|---|---|
| `fixed` | | held at `value` |
| `uniform` | `low`, `high` | flat between the bounds |
| `log_uniform` | `low`, `high` | flat in the logarithm, with `low` > 0 |
| `gaussian` | `sigma` | centred on `value`, optionally cut at `low` and `high` |

The starting value has to sit inside the prior.
`name` is only used to name the output files.

The orbit can be described with either the transit `duration` (in days) or the scaled semi-major axis `a_rs` ($a/R_*$).
Eccentricity `ecc` and argument of periastron `omega` (in radians) can be given as fixed values, and default to a circular orbit.
If you let the period vary, it's fitted in white light and then held fixed for the wavelength channels.
If your observation has more than one transit of the same planet, all of them are fitted together.

Times are in BMJD_TDB, that is BJD_TDB − 2400000.5.
Values in papers are often quoted as BJD − 2400000, which is half a day later, so watch out for that.
koala stops with an error if no transit allowed by your priors falls inside the data, and points out the half-day offset when that's the cause.

For a secondary eclipse on a circular orbit, you can give `eclipse_time` instead of `t0`, and koala works out the transit time from it.
The [rocky-planet eclipse](../case_studies/rocky_eclipse.md) is an example.

To fit two planets that transit in the same light curve, give a list with one entry per planet:

```yaml
planet:
  period: [{value: 3.0782, prior: fixed}, {value: 7.4203, prior: fixed}]
  t0: [{value: 60010.214, prior: uniform, low: 60010.18, high: 60010.25},
       {value: 60010.831, prior: uniform, low: 60010.80, high: 60010.86}]
```

A single entry instead of a list gives every planet the same starting value and prior.

## The star

```yaml
stellar:
  teff: 5509
  logg: 4.22
  feh: 0.04
```

These are the star's effective temperature (K), surface gravity ($\log_{10}$ cgs) and metallicity (dex).
koala uses [ExoTiC-LD](https://exotic-ld.readthedocs.io) to work out the limb darkening at each wavelength from a grid of stellar models.
The default grid is MPS-ATLAS (`ld_model: mps1`), and `ld_model` can also be `mps2`, `stagger`, `kurucz` or `phoenix`.
If you want the limb-darkening priors to follow the uncertainty on the stellar parameters, add `teff_sigma`, `logg_sigma` and `feh_sigma` as well (see [limb darkening](limb_darkening.md)).

The stellar-model files are downloaded the first time they're needed, into `~/.cache/koala/exotic_ld_data` unless you set `ld_data_path`.

## Wavelength channels

The `resolution` block sets the wavelength channels for the spectroscopic fit:

```yaml
resolution:
  high: 100
```

A number gives bins with that resolving power $R = \lambda/\Delta\lambda$.
`native` fits each wavelength in the input file as its own channel.
`reference` uses the bin edges in a CSV file you supply as `resolution.reference_grid`, which is handy when you want to match another analysis.
If you also give a `low` entry, koala first fits a coarser grid at that resolution before the final one.
That's optional, and most fits don't need it.

To use only part of the wavelength range, or to leave out a band such as a detector artifact, use:

```yaml
wavelength_filter:
  wl_min: 2.9
  wl_max: 5.0
wavelength_masks: [3.95, 4.05, 4.45, 4.50]
```

`wavelength_masks` is a list of (start, end) pairs, in µm.
For SOSS, the limits are per order: `wl_min_o1`, `wl_max_o1`, `wl_min_o2` and `wl_max_o2`.

## Leaving out bad data

Often the first few minutes of an observation are affected by the detector settling, or there's a bad stretch you'd rather not fit.
You can leave out integrations by time or by number:

```yaml
exclude_times: [[null, "min(t) + 10 / 1440"]]
exclude_integrations: [[400, 405], [-5, null]]
```

This drops the first 10 minutes, integrations 400 to 405, and the last five integrations.
Ranges include both ends, integrations count from zero, negative numbers count from the end, and `null` leaves that end open.
Times are in BMJD_TDB days, and `t` in an expression is the array of integration times.

Separately, koala clips single outlying integrations, at 4σ from the fitted model by default.
You can change the threshold for the white-light and spectroscopic fits:

```yaml
outlier_clip:
  whitelight_sigma: 5
  spectroscopic_sigma: 5
```

## Model choices

The model choices go in the `flags` block.
The ones you're most likely to change are the trend, the limb darkening and the kind of light curve:

```yaml
flags:
  detrending_type: quadratic
  ld_profile: power2
  ld_prior: stellarprior
  light_curve_model: transit
```

The options for each are on their own pages: [trends](trends.md) and [limb darkening](limb_darkening.md).
For `light_curve_model`, the choices are `transit` (the default), `eclipse` and `phase_curve`, and the [eclipse](../case_studies/rocky_eclipse.md) and [phase-curve](../case_studies/phase_curve.md) case studies show how to set them up.

## Where and how it runs

koala uses a GPU if JAX can see one and the CPU otherwise.
To choose yourself, set `host_device: cpu` or `host_device: gpu`.
On a CPU, koala splits the wavelength channels across all the cores it's been given (on a Slurm node, the cores in your allocation).
`flags.cpu_host_devices` changes that number.

koala saves its progress as it goes.
If a fit stops part way through, run the same command again and it picks up where it left off, reusing the white-light fit and any finished batches of channels.
It checks that the data and settings haven't changed before reusing anything, so if you change the model, use a new `output_dir` (or koala will start the affected parts again).

A few other settings that come up:

- `flags.analysis_stage: whitelight` stops after the white-light fit, which is a quick way to check a new setup.
- `flags.spectro_chunk_size` is the number of channels fitted at the same time. Lower it if a GPU runs out of memory.
- `flags.random_seed` sets the random seed (the default is 555), so a fit can be repeated exactly.
- `flags.whitelight_num_samples` and `flags.highres_num_samples` set the number of posterior draws per chain (the default is 1000).
