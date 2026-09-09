# Fit your first transit

This tutorial turns the bundled NIRISS/SOSS observation of WASP-39 b into a
transmission spectrum with a linear baseline and power-2 limb darkening from
`stellarprior`. The same workflow applies to the other supported modes.

## Configuration

```bash
cp examples/niriss_soss_order1.yaml wasp39.yaml
```

No edits are needed: the example reads
`examples/data/WASP-39_soss_binned8.fits`, writes to
`results/WASP-39_SOSS_ORDER1`, and downloads the ExoTiC-LD files it needs on
first use. The FITS file is a teaching extraction that combines groups of
eight adjacent detector columns; use your own extraction for science.

- {download}`Download the example FITS <../../examples/data/WASP-39_soss_binned8.fits>`
- {download}`Read its provenance and transformation <../../examples/data/README.md>`

The analysis choices are:

```yaml
instrument: NIRISS/SOSS
order: 1

resolution:
  low: 20
  high: 100

flags:
  detrending_type: linear
  ld_profile: power2
  ld_prior: stellarprior
```

Order 2 is stored in different extensions of the same FITS file; fit it
separately with `order: 2`.

## Run

```bash
python fit_jwst.py -c wasp39.yaml
```

The pipeline fits the white-light curve, fixes the channel geometry to its
posterior-median handoff, then fits the wavelength channels. Re-running the
command resumes from the channel checkpoints. For a GPU, set
`host_device: gpu` and export `JAX_PLATFORMS=gpu` before starting Python.

## White-light fit

```{image} ../_static/soss_wasp39_whitelight.png
:alt: WASP-39 b SOSS order-1 white-light data, fitted transit and residuals
:width: 760px
:align: center
```

Residuals should be centred on zero without a drift or an isolated feature.
If the baseline is curved, try `quadratic`; if it shows early settling, try
`explinear`; fix the white-light fit before trusting any spectral structure.

## Transmission spectrum

```{image} ../_static/soss_wasp39_spectrum.png
:alt: WASP-39 b SOSS order-1 transmission spectrum
:width: 760px
:align: center
```

The final `*_R100.csv` has one row per wavelength bin, with transit depths
computed from the posterior radius-ratio draws:

```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

out = Path("results/WASP-39_SOSS_ORDER1")
spectrum = pd.read_csv(next(out.glob("*_R100.csv")))
good = spectrum[["wavelength", "depth_ppm00", "depth_err_ppm00"]].notna().all(axis=1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(
    spectrum.loc[good, "wavelength"],
    spectrum.loc[good, "depth_ppm00"],
    xerr=spectrum.loc[good, "wavelength_err"],
    yerr=spectrum.loc[good, "depth_err_ppm00"],
    fmt="k.",
)
ax.set(xlabel="Wavelength [micron]", ylabel="Transit depth [ppm]")
fig.tight_layout()
plt.show()
```

## Adapt this example

| Goal | Change |
|---|---|
| Fit SOSS order 2 | `order: 2` |
| Fit NIRSpec or MIRI | Set `instrument`, `nrs`, and `fits_file`; see [Supported modes](../quickstart.md#supported-modes) |
| Use native wavelength bins | `resolution.high: native` |
| Change the resolving power | Set `resolution.high` to an integer such as `20` or `300` |
| Use a supplied wavelength grid | `resolution.high: reference` with `reference_grid` |
| Lower GPU memory use | Reduce `flags.spectro_chunk_size` |
