# Frequently asked questions

## What is the shortest working command?

```bash
export JAX_ENABLE_X64=1
python fit_jwst.py -c config.yaml
```

The command requires a YAML configuration and an extracted box-spectrum FITS file.

The CLI requires `-c` or `--config`; the configuration is not positional.

## Which Python environment should I use?

Use Python 3.11 for the Read the Docs build.

For fitting, use an environment containing compatible JAX, NumPyro, jaxoplanet, ExoTiC-LD, Astropy, pandas, matplotlib, jaxopt, tinygp, and ArviZ versions.

JAX must be configured for 64-bit calculations.

Set `JAX_ENABLE_X64=1` before importing JAX.

## Why can ExoTiC-LD not find its models?

The Python package and its model-data tree are separate requirements.

Point `stellar.ld_data_path` at the local `exotic_ld_data` directory.

The path is interpreted in the normal filesystem context of the running job.

Compute nodes must be able to read it.

## Why does the GPU run out of memory?

The resident channel width is usually the first control to change.

Reduce `flags.vmap_chunk` from 40 to 20, 10, or 4.

This changes concurrency, not wavelength binning or the scientific model.

Long PRISM time series use more memory than short SOSS time series at the same width.

Harmonica and complex trends can also increase the device footprint.

## Why does my fit take forever?

First distinguish compilation from sampling.

The first call for a new chunk width can spend roughly 1–1.5 minutes compiling on a V100 or A100.

Later equal-width chunks should report much shorter sampling times.

If every chunk recompiles, check that its static width and model settings are identical.

If sampling is slow, inspect tree depths, divergences, and ESS.

A difficult channel can consume much more NUTS work than its neighbors.

Use checkpoint diagnostics to identify it.

Reducing the channel width can isolate difficult lanes.

## What does `need_lowres` do?

`need_lowres: true` runs the low-resolution spectroscopic stage before the high-resolution stage.

The stage is also activated when trend or limb-darkening interpolation requires it.

The fitted Sing gray-offset calibration requires low-resolution information.

Set it false only when the high-resolution model has everything it needs directly.

## How do I resume an interrupted run?

Run the same command with the same configuration and input arrays.

{{ project }} fingerprints the checkpoint inputs.

Matching completed chunks are loaded from `output_dir/chunks/`.

Missing chunks are sampled.

Changing data, priors, stage settings, or relevant model sources changes the fingerprint and prevents an unsafe reuse.

## The gate keeps failing. What should I inspect?

Read the chunk diagnostic JSON first.

Locate the affected wavelength index and check the corresponding flux and uncertainty arrays for NaNs, saturation, discontinuities, or extremely small reported errors.

Inspect the white-light geometry handoff.

Check whether the selected trend describes the visit.

For a detector tilt event, try the discontinuity model with an informed initial time.

For an early visit ramp, use `explinear`.

Reduce `vmap_chunk` to isolate the lane.

Do not treat a low-ESS or divergent posterior as a measured depth.

## Why do my depths look biased?

Start with limb darkening and systematics, because both can trade against transit depth.

Compare the white-light residuals before interpreting a spectral offset.

Confirm that times, flux normalization, and uncertainty units match the expected extraction.

Check saturation and wavelength masks.

Compare fixed, stellar-informed, and wider limb-darkening assumptions when scientifically appropriate.

Use the same geometry handoff for the comparisons.

Avoid interpreting isolated high-uncertainty channels; the detailed parameter table flags depth errors larger than five times the channel median.

## What is the difference between NRS1 and NRS2?

They are the two NIRSpec detector segments.

Set `nrs: 1` or `nrs: 2` and provide the corresponding extracted FITS file.

Fit them with separate configurations.

Their wavelength ranges, bad pixels, saturation behavior, and systematics can differ.

Join spectra only after inspecting each detector result.

## How do I bin to a reference grid?

Set `resolution.high: reference` and supply `resolution.reference_grid`.

```yaml
resolution:
  high: reference
  low: 20
  reference_grid: prism_template.csv
```

The reference file must be available to the job.

Use a numerical value such as 20, 100, or 300 for constant resolving-power bins.

Use `native` to retain the input channel grid.

## Can I use my own extraction?

Yes, if it is written in the box-spectrum FITS structure consumed by `SpectroData` and `process_spectroscopy_data`.

Use the same time, wavelength, flux, and uncertainty conventions as the checked inputs.

Make invalid or saturated samples explicit rather than assigning implausibly small errors.

Run the white-light stage first and inspect its time-series CSV and plots before committing to the spectroscopic run.

## How do I run only one stage?

Use `flags.analysis_stage` or the `JWSTJAXFIT_ANALYSIS_STAGE` environment variable.

Accepted values are `all`, `whitelight`, `prep`, and `highres`.

`prep` completes data preparation and any needed low-resolution work, then stops before high resolution.

`highres` consumes compatible prepared products.

## How do I make a reproducible run?

Set `flags.random_seed` and archive the exact YAML.

`FIT_JWST_SEED` overrides the YAML, so record that environment variable when used.

Record the JAX, NumPyro, and jaxoplanet versions and GPU model.

Keep the geometry handoff, checkpoint manifests, diagnostics, and final CSVs together.

## How should I cite the fit?

Use the placeholder package entry on the [Citing](citing.md) page until archival metadata is assigned.

Also cite jaxoplanet, NumPyro, ExoTiC-LD when used, Harmonica when used, and Sing et al. for the Sing prior.

State the limb-darkening law, prior, trend family, sampler, detector or order, and resolving-power grid in the methods section.

