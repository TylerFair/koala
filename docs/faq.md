# Frequently asked questions

## What data does Koala expect?

An extracted box-spectrum FITS time series containing time, wavelength, flux,
and uncertainty information in the structure read by `SpectroData`. The code
fits light curves; it does not begin from raw detector ramps. Start from the
example for your observing mode and set `path`, `input_dir`, and `fits_file`.

## Do I need a GPU?

A CPU is enough to check imports, read a configuration, and build the docs. A
full spectroscopic run is intended for an NVIDIA GPU. Run
`python -c "import jax; print(jax.devices())"` inside your allocation before a
long job. If memory is tight, reduce `flags.vmap_chunk`; that reduces the
number of simultaneous channels without changing the analysis.

## Which trend and limb-darkening model should I use?

Start with the simplest trend supported by the baseline. A linear trend is a
reasonable first look for many SOSS and G395 observations; PRISM often shows
an early exponential ramp. Use a step only for a real visit-wide transition
and a spot template only for a localized coherent feature.

Power-2 limb darkening with `stellarprior` is the standard supplied example. It
propagates uncertainty in stellar parameters rather than fixing the
coefficients. The right robustness check depends on the science case; the
[trend](guides/trends.md) and [limb-darkening](guides/limb_darkening.md)
tutorials show the available choices.

## Why can ExoTiC-LD not find its models?

Installing `exotic-ld` does not install its atmosphere grids. Download the data
separately and set `stellar.ld_data_path` to the directory that contains the
requested grid. Prefer an absolute path on a cluster and confirm that the
compute node can read it.

## Can I resume an interrupted fit?

Yes. Run the same `python fit_jwst.py -c config.yaml` command again. Compatible
completed batches load from `output_dir/chunks/`, and missing batches continue.
Changes to the data or relevant model settings produce a different fingerprint
so an incompatible posterior is not silently reused.

## What should I inspect before using the spectrum?

Inspect the white-light model and residual plots first. The residuals should
not contain an obvious ramp, step, or localized event left by the chosen trend.
Then read the channel diagnostic JSON files for effective sample size and
divergences, and inspect channels with unusually large depth errors. A finished
command alone is not a convergence check.

The [Quickstart](quickstart.md) identifies the first plots, and
[Output files](guides/outputs.md) documents the tables.

## How do I compare several reasonable models?

Keep the input data and sampling policy fixed, fit each scientific alternative
into its own output directory, and use the
[model-stacking tutorial](guides/model_stacking.md). Stacking combines
posterior draws using out-of-sample predictive performance per wavelength
channel, allowing model disagreement to widen the reported uncertainty.

## What should I save for a reproducible analysis?

Archive the YAML file, input-data identity, software environment, random seed,
white-light geometry handoff, chunk diagnostics, and final tables together.
Record the detector or SOSS order, wavelength grid, trend, limb-darkening
treatment, and sampler in the methods. See [Citing](citing.md) for the
underlying software and methods to cite.
