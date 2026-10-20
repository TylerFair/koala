# Your first fit

Fit the bundled WASP-39 b NIRISS/SOSS order 1 observation from the repository root after [installation](install.md).
koala fits the white-light transit, then holds its orbit fixed while fitting the wavelength channels at $R=100$.
The input has 537 integrations and groups adjacent detector columns in eights. It has no time averaging.
The FITS file is [included in the repository](https://github.com/TylerFair/koala/blob/main/examples/data/README.md) and available as a {download}`download <../examples/data/WASP-39_soss_binned8.fits>`.

## Set the fit

Save this as `config.yaml`. It is also supplied as `examples/niriss_soss_order1.yaml`.

```yaml
planet:
  name: WASP-39
  period: {value: 4.05528043, prior: fixed}
  duration: {value: 0.11693087, prior: log_uniform, low: 0.01, high: 1.0}
  t0: {value: 59787.055, prior: uniform, low: 59787.005, high: 59787.105}
  b: {value: 0.4498, prior: uniform, low: 0.0, high: 1.5}
  rprs: {value: 0.1457, prior: uniform, low: 0.01, high: 0.5}

stellar:
  teff: 5509
  logg: 4.22
  feh: 0.04
  ld_model: "stagger"
  ld_data_path: "exotic_ld_data"

instrument: 'NIRISS/SOSS'
order: 1

input_dir: 'examples/data'
input_file: 'WASP-39_soss_binned8.fits'
output_dir: 'results/WASP-39_SOSS_ORDER1'

resolution:
  high: 100

flags:
  detrending_type: 'linear'

host_device: "cpu"
```

The period is fixed. Duration, epoch, impact parameter and radius ratio are fitted with the stated priors.
Times are BMJD_TDB = BJD_TDB − 2400000.5; durations and periods are in days.
The stellar grid supplies limb-darkening starting values and downloads to `exotic_ld_data` on first use.
The default quadratic coefficients are fitted with independent uniform priors on $u_1+u_2$ in $(-1,2)$ and $u_1-u_2$ in $(-2,2)$.
The linear trend and noise term are fitted too. Everything else uses the defaults.

## Run it

In a fresh notebook kernel, run this before importing JAX:

```python
import koala
koala.fit("config.yaml")
```

This sets up CPU host devices before JAX starts and runs the same pipeline as the command line.
White-light sampling and channel progress appear in the cell output.
On a cluster, start the notebook on an allocated compute node.
The command-line equivalent is:

```bash
koala -c config.yaml
```

This fit took 842.80 s (14 min 3 s) on 16 cores of an Intel Xeon Gold 6248R node, with JAX 0.6.2 and the stellar-model files already downloaded.
That wall time includes an automatic white-light retry after two divergent transitions; the retry passed the convergence checks with zero divergences.
Here is the real output, abridged:

```text
Data length: (537,)
Fitting whitelight for outliers and bestfit parameters
...
White-light Laplace chain had 2 divergent transition(s); rerunning it once with target acceptance 0.99 (smaller steps) before any fallback.
sample: 100%|██████████| 1200/1200 [02:06<00:00,  9.48it/s]
...
Running chunked MCMC: 115 channels in blocks of 16 (mode=serial)
chunk 0:16 sample: 100%|██████████| 1000/1000 [00:42<00:00, 29.77it/s]
...
chunk 112:115 sample: 100%|██████████| 1000/1000 [00:42<00:00, 28.31it/s]
All chunks complete! Final shape: (4000, 115)
```

## Four views of the fit

First, look at the white-light curve and its residuals.
Then compare the wavelength channels and the spectrum.
The larger points in these figures are time bins for display only.
koala fits the individual integrations.
These four figures are from the archived bundled-data fit; clipping and sampler results can differ between runs.

::::{tab-set}

:::{tab-item} White light
```{figure} _static/soss_whitelight.png
:alt: Normalized WASP-39 white-light flux and fitted model

We sum the flux over wavelength to measure the transit geometry.
The black curve is the fitted model.
```
:::

:::{tab-item} Residuals
```{figure} _static/soss_residuals.png
:alt: WASP-39 white-light residuals in parts per million

The white-light data minus the complete model, in ppm.
Look for structure that remains after removing the transit and trend.
```
:::

:::{tab-item} Wavelength channels
```{figure} _static/soss_detrended.png
:alt: Selected WASP-39 wavelength-channel light curves after division by their trends

Selected wavelength channels after division by their fitted linear trends.
The curves are offset vertically so we can see each transit.
```
:::

:::{tab-item} Spectrum
```{figure} _static/soss_spectrum.png
:alt: WASP-39 transmission spectrum at resolving power 100

The transit depths at $R=100$, computed from the radius-ratio posteriors.
Horizontal bars show the wavelength-bin half-widths.
```
:::
::::

The channel fits hold the orbit at the white-light posterior medians.
The spectrum is therefore conditional on that orbit.
If you need to include the uncertainty in the shared geometry, see [joint spectroscopic geometry](tutorials/planet_parameters.md#joint-spectroscopic-geometry).

## Open the results

The output directory is `results/WASP-39_SOSS_ORDER1/`.
Read the spectrum with:

```python
from pathlib import Path
import pandas as pd

out = Path("results/WASP-39_SOSS_ORDER1")
spectrum = pd.read_csv(next(out.glob("*_R100.csv")))
print(spectrum[["wavelength", "depth00", "depth_err00"]])
```

`wavelength` and `wavelength_err` are in µm. `depth00` and `depth_err00` are fractional transit depth and uncertainty; `depth_ppm00` and `depth_err_ppm00` give the same quantities in ppm.
The `00` suffix is the planet index. koala can fit several planets at once, with columns `01`, `02`, and so on.
The white-light timeseries CSV gives the model and residuals; channel plots and posterior files let you inspect the spectral fit.
See [outputs](outputs.md) for the remaining files. Repeat the command to resume compatible completed batches.

Good job, you're ready for the [case studies](case_studies/index.md).
