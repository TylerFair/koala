# Harmonica limb spectra

So far we've described the planet with one circular radius at each wavelength.
Harmonica instead fits a planetary boundary whose two halves can differ.
Ingress and egress then constrain the two terminator halves.
Let's try this with the bundled WASP-39 b observation.

## Fit the boundary

First, copy the {download}`SOSS starter <../../examples/niriss_soss_order1.yaml>`:

```bash
cp examples/niriss_soss_order1.yaml my_harmonica.yaml
```

Set the output directory and merge these flags into the copy:

```yaml
output_dir: results/WASP-39_HARMONICA
flags:
  transit_engine: harmonica
  harmonica_max_order: 1
  harmonica_spectro_parameterization: delta_r
  harmonica_spectro_odd_frac_sigma: 0.1
  ld_profile: power2
  ld_prior: gaussian
```

Then run the fit:

```bash
koala -c my_harmonica.yaml
```

This uses the bundled observation's planet and stellar parameters.
Harmonica accepts either `duration` or `a_rs` for the shared Keplerian orbit.
It supports one planet with power-2 limb darkening or supported quadratic treatments, including fixed, uniform, and fixed-$u_1$ priors.
The settings above choose Gaussian power-2 priors explicitly.

## Read the two limbs

The ordinary spectrum CSV reports the total area-equivalent transit depth.
The file `*_limb_spectra.csv` adds the two indexed halves and boundary summaries.
The matching `*_limb_posterior_samples.npz` keeps their paired posterior draws.
Use the output directory you selected above:

```python
from pathlib import Path
import numpy as np
import pandas as pd

out = Path("results/WASP-39_HARMONICA")
limbs = pd.read_csv(next(out.glob("*_R100_limb_spectra.csv")))
print(limbs.columns.tolist())
with np.load(next(out.glob("*_R100_limb_posterior_samples.npz"))) as posterior:
    print(posterior.files)
```

The halves follow the boundary convention stored in the product.
Assigning morning and evening requires the orbital orientation.
Use paired draws when computing a limb difference, since the two depths are correlated.
Subtracting independent error bars loses that information.
The [output reference](../outputs.md#harmonica) lists the area and endpoint quantities separately.

## An example spectrum

This archived WASP-94 b fit illustrates the same outputs:

```{figure} ../_static/harmonica_limbs.png
:alt: Archived WASP-94 b total-area transmission spectrum and spectra of its two terminator halves

WASP-94 b with SOSS order 1 at $R=50$, using Gaussian priors on power-2 limb darkening.
The lower panels show the two indexed halves.
Their marginal error bars alone do not establish a limb difference.
```

The WASP-94 extraction is not bundled.
To reproduce this figure, you'll need that target's extraction, planet and stellar parameters, atmosphere-cache path, and an $R=50$ final grid.

The [trend reference](../tutorials/systematics.md) describes polynomial, ramp, Gaussian-process and spot-crossing alternatives.
The [limb-darkening reference](../tutorials/limb_darkening.md) lists quadratic and power-2 laws and their fixed, Gaussian, uniform and atmosphere-based priors, with engine restrictions.
For boundary parameters and their priors, see the [Harmonica reference](../tutorials/harmonica.md).
