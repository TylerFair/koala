# Morning and evening limbs with Harmonica

So far we've treated the planet as a circle, with one radius at each wavelength.
The atmosphere at the planet's morning terminator can be quite different from the evening one, though, and the two limbs then block different amounts of starlight.
Ingress is mostly shaped by one limb and egress by the other, so a transit can tell them apart.

[Harmonica](https://github.com/DavoGrant/harmonica) (Grant & Wakeford 2023) models the planet's outline as a radius that varies around the edge of the disk.
With koala, you can fit that outline in every wavelength channel and get a spectrum for each limb.
(If you'd rather measure the effect through the transit time, see [timing spectra](timing_spectra.md).)

Harmonica isn't installed with koala by default.
It needs a C++ compiler (see [installation](../install.md#harmonica)); then, from the `koala` folder, run:

```bash
python -m pip install ".[harmonica]"
```

## Setting up the fit

Start from a copy of the first-fit configuration:

```bash
cp examples/niriss_soss_order1.yaml my_harmonica.yaml
```

and add these settings to it:

```yaml
output_dir: results/WASP-39_HARMONICA
flags:
  transit_engine: harmonica
  harmonica_max_order: 1
  ld_profile: power2
  ld_prior: gaussian
```

`transit_engine: harmonica` swaps the circular planet for a Harmonica outline.
With `harmonica_max_order: 1`, the outline has a mean radius and one pair of terms that make one side bigger than the other, which is the simplest shape that can separate the two limbs.
Harmonica fits one planet at a time, and works with power-2 limb darkening and with the quadratic law with fixed, uniform or fixed-$u_1$ priors.
Here we've chosen power-2 with Gaussian priors.

Then run it as usual:

```bash
koala -c my_harmonica.yaml
```

## Reading the two limbs

The ordinary spectrum file gives the depth of the whole outline.
`*_R100_limbs.csv` adds the two halves (`depth_one` and `depth_two`, in ppm, each with `_median`, `_err_lo` and `_err_hi`), along with the coefficients of the outline.
`*_R100_limb_samples.npz` keeps the posterior draws of both halves together:

```python
from pathlib import Path
import numpy as np
import pandas as pd

out = Path("results/WASP-39_HARMONICA")
limbs = pd.read_csv(next(out.glob("*_R100_limbs.csv")))
with np.load(next(out.glob("*_R100_limb_samples.npz"))) as draws:
    difference = draws["depth_two"] - draws["depth_one"]
```

Use the draws to work out a difference between the limbs, as above.
The depths of the two halves are correlated, so subtracting their error bars as if they were independent gives the wrong uncertainty on the difference.
The halves are labelled by their position around the outline rather than as morning and evening, and which is which depends on the direction of the orbit on the sky.
The convention is stored in each file (`limb_product_convention`), so read it before assigning them.
The depths in the NPZ file are fractional, while those in the CSV are in ppm.

## An example

Here's the same kind of fit for WASP-94 A b with SOSS order 1 at $R = 50$, using Gaussian priors on power-2 limb darkening (that extraction isn't included with koala):

```{figure} ../_static/harmonica_limbs.png
:alt: The WASP-94 b transmission spectrum of the whole planet, and of each of its two halves

The depth of the whole outline (top) and of each half (below).
Because the two halves are correlated, their error bars alone can't tell you whether the limbs differ.
```

## Other settings

| `flags` key | |
|---|---|
| `harmonica_max_order` | the highest order of the outline (default 1) |
| `harmonica_spectro_parameterization` | how the outline is sampled in each channel: `delta_r` (default), `fractional` or `half_area` |
| `harmonica_spectro_odd_frac_sigma` | width of the prior on the terms that make the two sides differ (default 0.1) |
| `harmonica_spectro_fit_jitter` | fit a jitter term in each channel (default `true`) |

A ramp trend with Harmonica needs `spectro_fixed_timescale_trends: false`, so that each channel fits its own ramp timescale.
