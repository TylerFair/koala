# Executed notebook examples

These examples preserve their code, tabular previews, fit log tail, measured
wall time, and embedded white-light and transmission-spectrum figures. They
follow the analysis in small narrative steps so the saved result can be read
before installing or rerunning the pipeline.

- {download}`NIRISS/SOSS order-1 notebook <../../examples/niriss_soss_order1.ipynb>`

- {download}`NIRSpec/G395M NRS1 notebook <../../examples/nirspec_g395m.ipynb>`

Both saved notebooks were executed on an NVIDIA V100. Their source
configurations deliberately retain `host_device: "cpu"`; CPU reruns are
supported but slower. Set `PENUMBRA_EXAMPLE_DATA_ROOT` to the directory above
your `FITS/` folder and choose a fresh `PENUMBRA_NOTEBOOK_RESULT_ROOT` so no
existing result is replaced.

The notebooks use an R=20 tutorial spectrum and an internal bounded-demo
runtime profile with reduced warmup and draw counts. The science-facing
configuration cells contain only supported public and advanced choices. The
adjacent YAML files are the production starting points; they use the validated
sampler and convergence defaults automatically.

The plotting cells call the same functions used by the pipeline itself, so the
embedded figures demonstrate the current publication style rather than a
notebook-only reimplementation.
