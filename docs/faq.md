# Frequently asked questions

## What data does Koala expect?

An extracted spectroscopic time series from exoTEDRF (box-spectrum FITS),
SPARTA (pickle), or Eureka! (Stage 3 or 4 HDF5); the format is detected
automatically. Koala fits light curves; it does not start from raw detector
ramps. Start from the example for your observing mode and set `path`,
`input_dir`, and `input_file`.

## Do I need a GPU?

A CPU is enough to check a configuration, run the bundled example, and build
the docs. A full spectroscopic run is intended for an NVIDIA GPU. Run
`python -c "import jax; print(jax.devices())"` inside your allocation before
a long job. If memory is tight, reduce `flags.spectro_chunk_size`; that
lowers the number of simultaneous channels without changing the analysis.

## Which trend and limb-darkening model should I use?

Start with the simplest trend supported by the baseline. A linear trend is a
reasonable first look for many SOSS and G395 observations; PRISM often shows
an early exponential ramp (`explinear`). Use a step only for a real
visit-wide transition and a spot only for a localized coherent feature.
Power-2 limb darkening with `stellarprior` is the standard supplied choice;
the [limb-darkening guide](guides/limb_darkening.md) lists the alternatives.

## Why can ExoTiC-LD not find its models?

Installing `exotic-ld` does not install its atmosphere grids. Set
`stellar.ld_data_path` to a writable directory; the required files are
downloaded there on first use. On an offline cluster, populate the directory
on an internet-connected machine first and prefer an absolute path.

## Can I resume an interrupted fit?

Yes. Run the same `python fit_jwst.py -c config.yaml` command again.
Compatible completed batches load from `output_dir/chunks/`, and missing
batches continue. Changes to the data or model settings produce a different
fingerprint so an incompatible posterior is not silently reused.

## How do I compare several reasonable models?

Keep the input data and sampling policy fixed, fit each scientific
alternative into its own output directory, and follow the
[model-stacking guide](guides/model_stacking.md).
