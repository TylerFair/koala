# Fit your first transit

This tutorial turns an extracted NIRISS/SOSS time series into a transmission spectrum. It uses WASP-39 b, a linear baseline, and power-2 limb darkening with `stellarprior`. The same workflow applies to the other supported JWST modes.

The repository includes a compact real WASP-39 box-spectrum FITS file for this tutorial. After extracting the source ZIP, run `python -m pip install -e .` from its root. ExoTiC-LD downloads the required stellar-atmosphere and instrument files automatically on the first run. Koala fits light curves; it does not run the JWST detector calibration or spectral extraction.

## 1. Start from the example

Copy the small, annotated configuration:

```bash
cp examples/niriss_soss_order1.yaml wasp39.yaml
```

No YAML edits are needed. The example stores automatically downloaded model data here:

```yaml
stellar:
  ld_data_path: exotic_ld_data
```

The example already reads `examples/data/WASP-39_soss_binned8.fits` and writes figures, tables, and resumable checkpoints to `results/WASP-39_SOSS_ORDER1`. The teaching FITS retains the complete observation and both SOSS orders, but combines groups of eight adjacent detector columns. Use your original extraction for a scientific analysis.

- {download}`Download the example FITS <../../examples/data/WASP-39_soss_binned8.fits>`
- {download}`Read its provenance and transformation <../../examples/data/README.md>`

The remaining choices describe the analysis:

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

This prepares a coarse $R=20$ grid and fits SOSS order 1 in final $R=100$ bins. A standard ExoTEDRF box-spectrum FITS stores order 2 in different extensions; select it with `order: 2` and fit it separately.

## 2. Run the fit

```bash
python fit_jwst.py -c wasp39.yaml
```

The example defaults to CPU so it can start on a laptop; the complete fit then takes about two hours on eight cores (a few minutes for the white-light stage, the rest in the $R=100$ channels), against ten to twenty minutes on one GPU. For a GPU run, set `host_device: gpu` in the YAML and make JAX's device choice explicit:

```bash
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=gpu
python fit_jwst.py -c wasp39.yaml
```

The pipeline first fits the wavelength-summed white-light curve. It fixes the wavelength-channel geometry to the selected white-light posterior-median handoff, then fits the transmission spectrum. Rerunning the same command resumes compatible channel checkpoints.

## 3. Check the white-light fit

```{image} ../_static/soss_wasp39_whitelight.png
:alt: WASP-39 b SOSS order-1 white-light data, fitted transit and residuals
:width: 760px
:align: center
```

This archived WASP-39 example shows the same model and validation step; it is not a regenerated output from your $R=100$ run.

Look at ingress, egress, and the out-of-transit baseline. Residuals should be centered on zero without a smooth drift or an isolated feature that the model missed. Fix a poor broadband fit before trusting any spectral structure: wavelength-channel fits inherit its fixed geometry and shared trend information.

The first useful files are:

- `*_whitelight_summary.png` — the fit and residuals at a glance.
- `*_whitelight_timeseries.csv` — time, data, model, trend, and detrended flux.
- `*_bestfit_params.csv` — fitted parameters and uncertainties.

If the residual baseline is curved, compare a quadratic trend. If it has a settling ramp, try `explinear`. The [trend tutorial](../guides/trends.md) shows how to make that choice without adding arbitrary flexibility.

## 4. Read the transmission spectrum

```{image} ../_static/soss_wasp39_spectrum.png
:alt: WASP-39 b SOSS order-1 transmission spectrum
:width: 760px
:align: center
```

This archived spectrum uses the project's supplied reference grid. Your first run writes the same quantities in $R=100$ bins.

The final `*_R100.csv` contains one row per wavelength bin. Transit depths are computed from every posterior radius-ratio draw, so `depth_ppm00` and its uncertainty already preserve the nonlinear transformation $d=(R_p/R_\star)^2$.

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

Before using the spectrum, inspect the channel-fit summary and the sampler diagnostics in `chunks/`. A smooth-looking spectrum does not rescue divergent chains or channels with poor effective sample size.

## Adapt this example

Most analyses change only a few lines:

| Goal | Change |
|---|---|
| Fit SOSS order 2 | Set `order: 2`; the same standard box-spectrum FITS stores it in different extensions |
| Fit NIRSpec | Set the matching `instrument`, `nrs`, and FITS file |
| Use native wavelength bins | Set `resolution.high: native` |
| Change constant resolving power | Set `resolution.high` to an integer such as `20` or `300` |
| Use a supplied wavelength grid | Set `resolution.high: reference` and `reference_grid` |
| Lower GPU memory use | Reduce `flags.vmap_chunk` |

Use the [NIRSpec notes](nirspec_g395h.md) for detector handling and the [PRISM notes](prism.md) for long time series and ramps. Read [limb darkening](../guides/limb_darkening.md) before changing the LD treatment. The [outputs guide](../guides/outputs.md) is the reference for every saved column.

## If the run stops

- **A FITS file is not found:** check the resolved `path/input_dir/fits_file` path.
- **The first model misses transit:** check `t0`, `period`, `duration`, and their time convention.
- **A channel contains NaNs:** inspect the extraction or wavelength mask; do not replace missing flux with zero.
- **The GPU runs out of memory:** lower `flags.vmap_chunk` and rerun.
- **A quality gate repeatedly fails:** inspect that channel's light curve and uncertainties before changing the sampler.
