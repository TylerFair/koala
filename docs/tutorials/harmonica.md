# Advanced: fit limb asymmetry

The standard transit model gives the planet one radius at each wavelength. The Harmonica engine instead fits an asymmetric boundary, allowing ingress and egress to constrain representative spectra for two halves of the terminator.

Use this only after the ordinary transit, timing, baseline, and limb-darkening model give a satisfactory fit. An ingress/egress residual is not by itself evidence for atmospheric asymmetry.

## Configure the asymmetric model

Start from the checked WASP-94 b configuration:

```bash
cp examples/harmonica_soss_order1.yaml wasp94_harmonica.yaml
python fit_jwst.py -c wasp94_harmonica.yaml
```

The defining settings are:

```yaml
planet:
  a_rs: 7.299998525
  ecc: 0.0
  omega: 0.0

flags:
  transit_engine: harmonica
  harmonica_max_order: 1
  harmonica_spectro_parameterization: delta_r
  harmonica_spectro_odd_frac_sigma: 0.1
  spectro_sampler: independent_nuts
  ld_profile: quadratic
  ld_prior: fixed
```

Harmonica needs the scaled semimajor axis and eccentric geometry. It currently supports one planet with power-2 or quadratic limb darkening; fixed direct quadratic coefficients are the simplest starting point.

## What the model means

For the supported odd-cosine expansion,

$$
r(\theta)=a_0+a_1\cos\theta+a_3\cos3\theta+a_5\cos5\theta.
$$

`harmonica_max_order: 1` fits only $a_0$ and $a_1$. The `delta_r` parameterization samples the endpoint radius difference; at order 1 that difference is $2a_1$. Higher orders add shape flexibility and require stronger data.

The two exported limb spectra are geometric halves defined by the model coordinate system. Koala calls them terminator one and terminator two; assigning them to morning/evening or leading/trailing limbs requires your chosen orbital convention.

## Check the information-bearing data

```{image} ../_static/harmonica_wasp94_whitelight.png
:alt: Example WASP-94 b Harmonica white-light fit
:width: 760px
:align: center
```

Inspect ingress and egress separately. Check timing, cadence masks, limb darkening, and the baseline before interpreting a difference between them.

```{image} ../_static/harmonica_wasp94_limb_spectra_v3.png
:alt: Example representative spectra for the two geometric terminator halves
:width: 760px
:align: center
```

The limb depths are half-area-equivalent quantities, not the depth at a single exposure or the endpoint radius squared. For active odd coefficients define

$$
Q=a_1^2+a_3^2+a_5^2,\qquad
C=a_1-a_3/3+a_5/5.
$$

Then

$$
D_\mathrm{two}=a_0^2+Q/2+4a_0C/\pi,\qquad
D_\mathrm{one}=a_0^2+Q/2-4a_0C/\pi.
$$

Their mean is the total area-equivalent transit depth.

## Use the joint posterior

The CSV is convenient for plotting marginal intervals. Differences between the limbs must use the joint NPZ samples because the two depths are correlated:

```python
from pathlib import Path
import numpy as np
from koala.harmonica_products import load_harmonica_limb_dataframe

out = Path("/data/results/WASP-94_HARMONICA")
limbs = load_harmonica_limb_dataframe(next(out.glob("*_limb_spectra.csv")))
joint = np.load(next(out.glob("*_limb_posterior_samples.npz")))

depth_difference = joint["depth_two"] - joint["depth_one"]
print(limbs[["wavelength", "depth_one_median", "depth_two_median"]].head())
print(depth_difference.shape)  # posterior draw, wavelength
```

The main products are:

- `*_limb_spectra.csv` — marginal radius and depth summaries plus convention metadata;
- `*_limb_posterior_samples.npz` — correlated draws for joint calculations;
- `*_transmission_strings.png` — median boundary by wavelength;
- `*_transmission_string_posterior.png` — boundary draws for a selected channel;
- the ordinary spectrum CSV — the total area-equivalent result.

Nonphysical boundary shapes call for a stronger odd-mode prior or a lower maximum order. Mirror-like limb spectra are expected when the total area is well measured but the asymmetry is weak: report the correlated difference posterior rather than treating the two marginal errors as independent.

See [limb darkening](../guides/limb_darkening.md) before changing the LD treatment and [samplers](../guides/samplers.md) when an asymmetric channel fails the quality gate.
