# Your first fit

Let's fit a transmission spectrum of WASP-39 b using the bundled NIRISS/SOSS order-1 observation.
We'll first fit the white-light curve, then measure the transit depth in each wavelength channel.
The example uses the default quadratic limb darkening and a linear trend.

## Run the example

First, follow the [installation instructions](install.md) and run this command from the repository root:

```bash
koala -c examples/niriss_soss_order1.yaml
```

The configuration reads `examples/data/WASP-39_soss_binned8.fits` and writes the results to `results/WASP-39_SOSS_ORDER1/`.
ExoTiC-LD downloads the stellar models and throughput files it needs on first use.
Those stellar models supply starting values for the limb-darkening coefficients.
The fit uses uniform priors on $u_1+u_2$ in $(-1,2)$ and $u_1-u_2$ in $(-2,2)$.

The input contains all 537 integrations.
Adjacent detector columns have been summed in groups of eight to make a compact teaching example.
No integrations have been averaged in time.
You can {download}`download the FITS file <../examples/data/WASP-39_soss_binned8.fits>` and read its [provenance](https://github.com/TylerFair/koala/blob/main/examples/data/README.md).
For science, use the full extraction and keep a record of how it was reduced.

::::{dropdown} The complete YAML
```{literalinclude} ../examples/niriss_soss_order1.yaml
:language: yaml
```
::::

koala starts four NUTS chains from a Laplace approximation at the best fit.
It checks the sampling diagnostics before accepting the results.
The example runs on a CPU.
For a CUDA JAX installation, set `host_device: gpu` and run inside a GPU allocation.
If the run is interrupted, repeat the command to resume compatible completed batches.

## Four views of the fit

First, look at the white-light curve and its residuals.
Then compare the wavelength channels and the spectrum.
The larger points in these figures are time bins for display only.
koala fits the individual integrations.

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
Here the scatter is 108.6 ppm over the 536 integrations that were not clipped.
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

Start with the spectrum table:

```python
from pathlib import Path
import pandas as pd

out = Path("results/WASP-39_SOSS_ORDER1")
spectrum = pd.read_csv(next(out.glob("*_R100.csv")))
print(spectrum[["wavelength", "depth_ppm00", "depth_err_ppm00"]])
```

Here `00` identifies the first planet.
The [SOSS case study](case_studies/transmission_soss.md) shows how to plot this table and try a different wavelength grid.
For the remaining files and their columns, see [output files](outputs.md).
