# Quickstart

This walkthrough fits one extracted JWST transit. It uses NIRISS/SOSS as the
example, but NIRSpec follows the same three steps.

You need an [installed environment](install.md) and the ExoTiC-LD stellar
grids. The repository includes a compact real WASP-39 box-spectrum FITS file,
so this first run needs no separate light-curve download.

## 1. Copy an example

From the repository root:

```bash
cp examples/niriss_soss_order1.yaml config.yaml
```

Use `examples/nirspec_g395m.yaml` or `examples/nirspec_prism.yaml` instead for
those modes.

## 2. Point it at the limb-darkening grids

Open `config.yaml` and change this path:

```yaml
stellar:
  ld_data_path: /path/to/exotic_ld_data
```

The example already reads `examples/data/WASP-39_soss_binned8.fits` and writes
to `results/WASP-39_SOSS_ORDER1`. The FITS file retains every integration and
both SOSS orders from the real observation, with adjacent detector columns
combined to keep the repository small. It is a teaching extraction rather
than a publication data product.

- {download}`Download the example FITS <../examples/data/WASP-39_soss_binned8.fits>`
- {download}`Read its provenance and transformation <../examples/data/README.md>`

When adapting the configuration to your own observation, replace the example
planet and stellar values and set `path`, `input_dir`, `fits_file`, and
`output_dir`. The transit time `planet.t0` and FITS time array must use the same
time system. Give every distinct scientific setup its own output directory.

For a first fit, keep the example's model choices:

```yaml
resolution:
  low: 20
  high: 100

flags:
  detrending_type: linear
  ld_profile: power2
  ld_prior: stellarprior
```

Both resolution keys are required by the current pipeline. The low-resolution
grid is a coarse bridge and check; the high-resolution grid is the final
spectrum.

## 3. Run the fit

The supplied SOSS example uses `host_device: cpu`, so it can start on any
machine. Expect roughly two hours on eight CPU cores: the white-light fit
takes a few minutes, the coarse $R=20$ bridge about a quarter of an hour, and
the 115 channels of the $R=100$ spectrum the rest. On one NVIDIA GPU the same
fit takes ten to twenty minutes. For an installed CUDA-enabled JAX environment
inside a GPU allocation, change that line to `host_device: gpu`.

```bash
python fit_jwst.py -c config.yaml
```

The run first fits the wavelength-summed light curve, then the spectroscopic
channels. The first GPU batch can pause while JAX compiles; later equal-sized
batches reuse that work. Completed channel batches are checkpointed, so the
same command resumes an interrupted run.

## Inspect the result

Start with the white-light model and residual plots in `output_dir`. The exact
filename includes the target and instrument, but the numbered plots make the
order clear:

- `11_*_whitelightmodel.png` shows the data and fitted transit.
- `12_*_whitelightresidual.png` reveals structure left by the trend model.
- `31_*_spectrum_00.png` shows the final high-resolution transmission
  spectrum. A low-resolution bridge stage, when used, writes `24_*`.

The white-light residuals should be centered on zero without a coherent ramp,
step, or localized feature. If they are not, choose an appropriate
[systematics trend](guides/trends.md) before interpreting the spectrum.

The spectrum CSV contains wavelength, bin half-width, transit depth, and depth
uncertainty. Find and plot it without depending on the target-specific stem:

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

out = Path("results/WASP-39_SOSS_ORDER1")
spectrum_path = next(out.glob("*_R100.csv"))
spectrum = pd.read_csv(spectrum_path)

plt.errorbar(
    spectrum["wavelength"],
    spectrum["depth_ppm00"],
    xerr=spectrum["wavelength_err"],
    yerr=spectrum["depth_err_ppm00"],
    fmt=".",
)
plt.xlabel("Wavelength [micron]")
plt.ylabel("Transit depth [ppm]")
plt.show()
```

Do not treat a completed process as the only quality check. Read any ESS or
divergence messages and inspect the JSON diagnostics in `output_dir/chunks/`.
Koala retries failed channels with an alternate exact sampler, but a channel
that still fails needs investigation.

## Make the model yours

Continue with the choice that matters for your dataset:

- [Fit your first transit](tutorials/soss_order1.md) for a fuller SOSS example.
- [Choose a systematics trend](guides/trends.md) for ramps, steps, spots, and
  smooth baselines.
- [Choose limb darkening](guides/limb_darkening.md) for fixed, stellar-prior, or
  weak priors.
- [Marginalize over models](guides/model_stacking.md) when several reasonable
  models give different spectra.

See [Output files](guides/outputs.md) when you are ready to consume the full
tables and diagnostics.
