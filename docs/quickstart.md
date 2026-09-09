# Quickstart

This walkthrough fits one extracted JWST transit. It uses the bundled
NIRISS/SOSS observation of WASP-39 b; NIRSpec and MIRI follow the same steps.

Download and extract the GitHub source ZIP, open a terminal in `koala-main`,
and run `python -m pip install -e .` (see [installation](install.md)).

## 1. Copy an example

From the repository root:

```bash
cp examples/niriss_soss_order1.yaml config.yaml
```

Use `examples/nirspec_g395m.yaml` or `examples/nirspec_prism.yaml` instead for
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
`path`, `input_dir`, `fits_file`, and `output_dir`. The transit time
`planet.t0` and the FITS time array must use the same time system.

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

## Supported modes

- NIRISS/SOSS: `instrument: NIRISS/SOSS` with `order: 1` or `order: 2`.
- NIRSpec G395H, G395M, PRISM, G140H, or G235H: `instrument: NIRSPEC/G395H`
  (and so on) with `nrs: 1` or `nrs: 2`; no `order`.
- MIRI/LRS: `instrument: MIRI/LRS`; no `order` or `nrs`.
- `host_device: gpu` runs the fit on an NVIDIA GPU with a CUDA-enabled JAX
  build; `cpu` is for configuration checks and the bundled example.
