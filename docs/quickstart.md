# Quickstart

This walkthrough fits one extracted JWST transit. It uses the bundled
NIRISS/SOSS observation of WASP-39 b; every other mode follows the same steps.

Download and extract the GitHub source ZIP, open a terminal in `koala-main`,
and run `python -m pip install -e .` (see [installation](install.md)).

## 1. Copy an example

From the repository root:

```bash
cp examples/niriss_soss_order1.yaml config.yaml
```

Use `examples/nirspec_g395m.yaml`, `examples/nirspec_prism.yaml`, or `examples/nircam_f322w2.yaml` instead for
those modes.

## 2. Check the data and model settings

No edits are needed for the bundled example. It reads
`examples/data/WASP-39_soss_binned8.fits`, writes to
`results/WASP-39_SOSS_ORDER1`, and lets ExoTiC-LD download the
stellar-atmosphere files it needs into a local cache on first use:

```yaml
stellar:
  ld_data_path: exotic_ld_data
```

- {download}`Download the example FITS <../examples/data/WASP-39_soss_binned8.fits>`
- {download}`Read its provenance and transformation <../examples/data/README.md>`

For your own observation, replace the planet and stellar values and set
`path`, `input_dir`, `input_file`, and `output_dir`. Every planet
parameter is a `{value, prior, ...}` mapping (see
[Parameter priors](guides/configuration.md#parameter-priors)). The transit
time `planet.t0` and the FITS time array must use the same time system.

The model choices for a first fit are:

```yaml
resolution:
  low: 20
  high: 100

flags:
  detrending_type: linear
  ld_profile: power2
  ld_prior: stellarprior
```

`resolution.high` is the final spectrum. `resolution.low` is an optional
coarse stage: omit it and Koala goes straight from the white-light fit to the
final grid.

## 3. Run the fit

```bash
python fit_jwst.py -c config.yaml
```

The run first fits the wavelength-summed light curve, then the spectroscopic
channels in batches. Completed batches are checkpointed, so the same command
resumes an interrupted run. The bundled example sets `host_device: cpu` so it
runs anywhere; change it to `gpu` inside a CUDA-enabled environment.

## 4. Plot the spectrum

`output_dir` holds the white-light model and residual plots (`11_*` and
`12_*`), the spectrum plot (`31_*`), and the spectrum CSV:

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

out = Path("results/WASP-39_SOSS_ORDER1")
spectrum = pd.read_csv(next(out.glob("*_R100.csv")))

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

Continue with [Fit your first transit](tutorials/soss_order1.md) for the
result figures, [limb darkening](guides/limb_darkening.md) for the prior
choices, and [model stacking](guides/model_stacking.md) when several
reasonable models give different spectra.

## Input formats

`input_file` may be any of these extracted-spectra products; `input_format`
defaults to `auto`, which recognises the file from its contents.

| `input_format` | Product | Notes |
|---|---|---|
| `exotedrf` | exoTEDRF `*_box_spectra_fullres.fits` | Wavelength, wavelength error, flux, error, and time extensions; both SOSS orders in one file. |
| `sparta` | SPARTA `gather_and_filter.py` pickle | Uses the filtered `wavelengths`, `times`, `data`, `errors` arrays. SPARTA's large error sentinels on bad pixels are kept, which down-weights them. |
| `eureka` | Eureka! Stage 3 `*_SpecData.h5` or Stage 4 `*_LCData.h5` | Stage 3 is the native per-column product and is the natural input for Koala's own binning; Stage 4 files are already binned. Masked pixels are replaced by the channel median over time with a large error, so they carry no weight. |

Times are BMJD_TDB (BJD_TDB − 2,400,000.5) in every product and in every Koala
table; the white-light table also lists `t0_bjd_tdb`. Bin half-widths are taken
from the file when it stores them and otherwise from the spacing between
column centres.

## Supported modes

Koala supports every JWST time-series mode with an ExoTiC-LD throughput.
Names are case-insensitive.

| `instrument` | Detector key | ExoTiC-LD mode |
|---|---|---|
| `NIRISS/SOSS` | `order: 1` or `2` | `JWST_NIRISS_SOSSo1`, `JWST_NIRISS_SOSSo2` |
| `NIRSPEC/PRISM` | `nrs: 1` or `2` | `JWST_NIRSpec_Prism` |
| `NIRSPEC/G395H`, `NIRSPEC/G395M` | `nrs` | `JWST_NIRSpec_G395H`, `JWST_NIRSpec_G395M` |
| `NIRSPEC/G235H`, `NIRSPEC/G235M` | `nrs` | `JWST_NIRSpec_G235H`, `JWST_NIRSpec_G235M` |
| `NIRSPEC/G140H`, `NIRSPEC/G140M` (F100LP filter) | `nrs` | `JWST_NIRSpec_G140H-f100`, `JWST_NIRSpec_G140M-f100` |
| `NIRSPEC/G140H-F070`, `NIRSPEC/G140M-F070` (F070LP filter) | `nrs` | `JWST_NIRSpec_G140H-f070`, `JWST_NIRSpec_G140M-f070` |
| `NIRCAM/F322W2`, `NIRCAM/F444W` | none | `JWST_NIRCam_F322W2`, `JWST_NIRCam_F444` |
| `MIRI/LRS` | none | `JWST_MIRI_LRS` |

`NIRSPEC/G140H-F100` and `NIRSPEC/G140M-F100` are accepted aliases of the
F100LP entries. MIRI/MRS and the imaging modes are not supported. Each mode
keeps only the wavelengths inside its throughput window when the extracted
spectra are read; the windows live in `koala/instruments.py`.
- `host_device: gpu` runs the fit on an NVIDIA GPU with a CUDA-enabled JAX
  build; `cpu` is for configuration checks and the bundled example.
