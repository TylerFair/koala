# The WASP-39 b transmission spectrum

Let's measure the transmission spectrum of WASP-39 b with the bundled NIRISS/SOSS observation.
We'll use a linear trend and the default quadratic limb darkening with wide uniform priors.
Then we'll plot the depths and try a different wavelength grid.
If this is your first koala fit, start with [your first fit](../quickstart.md) for the installation and input details.

## Fit the observation

From the repository root, run:

```bash
koala -c examples/niriss_soss_order1.yaml
```

The example fits SOSS order 1 at a final resolving power of $R=100$.
The input contains all 537 integrations, with adjacent detector columns summed in groups of eight.
koala fits the white-light curve first and holds its orbital solution fixed when fitting the channels.
Open the [white-light and channel plots](../quickstart.md#four-views-of-the-fit) before interpreting the spectrum.

```{figure} ../_static/soss_spectrum.png
:alt: WASP-39 b NIRISS SOSS order-1 transmission spectrum

The transmission spectrum from the compact eight-column extraction, with the default quadratic limb darkening.
Vertical bars show the exported depth uncertainties, and horizontal bars show bin half-widths.
```

## Plot the spectrum

Now we can read the table and make the same kind of plot:

```python
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

out = Path("results/WASP-39_SOSS_ORDER1")
spectrum = pd.read_csv(next(out.glob("*_R100.csv")))
fig, ax = plt.subplots(figsize=(9, 4))
ax.errorbar(
    spectrum["wavelength"], spectrum["depth_ppm00"],
    xerr=spectrum["wavelength_err"],
    yerr=spectrum["depth_err_ppm00"], fmt=".",
)
ax.set(xlabel="Wavelength (µm)", ylabel="Transit depth (ppm)")
fig.tight_layout()
plt.show()
```

The `00` suffix identifies the first planet.
The depths come from squared radius-ratio draws.
Their uncertainties are conditional on the white-light orbit.
To include shared geometric uncertainty within the spectroscopic fit, see [joint spectroscopic geometry](../tutorials/planet_parameters.md#joint-spectroscopic-geometry).

## Change the wavelength grid

Next, copy the configuration and replace its output directory and resolution block with:

```yaml
output_dir: results/WASP-39_SOSS_NATIVE
resolution:
  high: native
```

With no `low` entry, this skips the coarse stage and fits the input channels directly.
It cannot recover detector-column information already combined in the teaching extraction.
For another resolving power, use a number in `high`.
For externally supplied bin edges, use `high: reference` with `resolution.reference_grid`.

You can also fit order 2, which is stored separately in the same FITS file.
Set `order: 2` and choose another output directory.
Compare the residuals and fitted light curves before attributing structure in either spectrum to the atmosphere.
If the spectrum changes with the model, the [limb-darkening](../tutorials/limb_darkening.md) and [trend](../tutorials/systematics.md) tutorials show how to investigate those choices.
