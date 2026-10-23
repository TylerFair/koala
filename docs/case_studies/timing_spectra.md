# Timing spectra

Usually every wavelength channel is fitted with the transit time held at the white-light value.
If the planet's morning and evening limbs are different sizes at some wavelengths, though, the transit at those wavelengths starts and ends slightly differently, and the fitted transit time moves.
Letting each channel have its own transit time gives a timing spectrum: the time offset from the white-light transit at each wavelength, alongside the usual depths.

Here we'll measure one for WASP-39 b with both SOSS orders, using the joint fit from the [previous case study](transmission_soss.md) at a coarser resolution of $R = 20$.

## Setting up the fit

The only new setting is `spectro_free_t0`.
Let's make two member files from the joint example with that switched on:

```python
from pathlib import Path
import yaml

for order in (1, 2):
    cfg = yaml.safe_load(Path(f"examples/joint_soss_order{order}.yaml").read_text())
    cfg["resolution"] = {"high": 20}
    cfg.setdefault("flags", {})["spectro_free_t0"] = True
    Path(f"timing_order{order}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
```

and a joint file, `timing_joint.yaml`, that uses them:

```yaml
joint:
  members: [timing_order1.yaml, timing_order2.yaml]
planet:
  a_rs: {value: 11.4, prior: uniform, low: 5.0, high: 25.0}
  b: {value: 0.45, prior: uniform, low: 0.0, high: 1.0}
output_dir: results/WASP-39_timing
```

Now run it:

```bash
koala -c timing_joint.yaml
```

The white-light fit is the same as before.
In the channels, the orbit is still fixed at the joint result, but each channel's transit time is free within ±30 minutes of the white-light time (`spectro_t0_half_width_s` changes that).
Both the depth and the time offset of every channel have to pass the convergence checks.
Free channel times work with the jaxoplanet transit model and any of the trends, but not yet with the spotted-star surface model or with `spectro_joint_geometry`.

## Reading the offsets

Each order's directory now has a `*_R20_timing.csv` table and a `figures/R20_timing.png` figure, and the joint `figures/joint_spectrum.png` shows the depths and offsets of both orders on one wavelength axis.

```python
import pandas as pd

out = Path("results/WASP-39_timing")
for order in (1, 2):
    timing = pd.read_csv(next((out / f"timing_order{order}").glob("*_R20_timing.csv")))
    print(timing[["wavelength", "depth", "depth_err", "dt0_s", "t0_err_s", "corr_depth_t0"]])
```

Here's `figures/joint_spectrum.png` from that run:

```{figure} ../_static/timing_spectrum.png
:alt: The WASP-39 b depth spectrum at R = 20 from both SOSS orders, and the transit time offset of each channel

The depths (top) and transit-time offsets (bottom) of every channel from both orders.
In this figure the offsets are measured from the average over 1.22–1.30 µm, between two water bands, so that several visits can be put on the same scale.
```

With 537 integrations, the transit time of each channel is measured to 2–3 s at the brightest wavelengths and 5–13 s at the ends of the orders.
The offsets scatter more than those uncertainties suggest, though: six of the 30 channels are more than 2σ from zero, and the most significant is in the 1.4 µm water band, where the transit is $9.6 \pm 2.4$ s later than in white light.
That could be a real difference between the limbs, but correlated noise in the light curves (which the fit treats as white noise) does the same thing, and this compact example file isn't meant for science.
Before reading anything into a timing spectrum, check the channel residuals for correlated noise.

A positive `dt0_s` is a later transit than the white-light one.
The white-light transit time is itself fitted, so a size difference between the limbs that's the same at every wavelength moves the white-light time too, and doesn't show up in `dt0_s`.
What a timing spectrum measures is how the limb difference changes with wavelength.
The uncertainties don't include the uncertainty in the white-light orbit and transit time, so if you fit timing spectra from several visits in a retrieval, give each visit its own free time offset.
`corr_depth_t0` is the correlation between a channel's depth and its time, which tells you whether the two trade off.

## From a time offset to a limb difference

For each channel, koala also works out how much the fitted transit time would move if one half of the planet were slightly bigger.
That's the `time_shift_coefficient` column, in seconds per stellar radius, and it's calculated for that channel's own radius, limb darkening and trend, with the white-light orbit.
Dividing a time offset by it gives the difference in radius between the two limbs, to first order:

$$
\Delta t_0 \approx \frac{\partial t_0}{\partial R_{\rm trailing}}\,(R_{\rm trailing} - R_{\rm leading}) + \text{visit offset}.
$$

The trailing half of the planet is the evening limb, and a larger leading (morning) half gives an earlier transit.
The full set of timing columns is:

| Column | |
|---|---|
| `t0`, `t0_bjd`, `t0_err_s` | the channel's transit time (full BJD_TDB) and its uncertainty in seconds |
| `dt0_s` | offset from the white-light transit time, in seconds |
| `corr_depth_t0` | correlation between depth and transit time |
| `time_shift_coefficient` | change in transit time per stellar radius of growth of the trailing half, in seconds |
| `d_dt0_dRL_s_per_Rstar`, `d_dt0_dRT_s_per_Rstar` | the same for the leading half and the trailing half on their own |
| `d_depth_dRL`, `d_depth_dRT` | the matching changes in the fitted depth |
| `time_shift_coefficient_analytic` | the simple formula below, for comparison |

For a circular orbit and a planet that crosses the whole star, there's a simple approximation for the coefficient,

$$
\frac{\partial t_0}{\partial R_{\rm trailing}} \approx \frac{C}{v},\qquad
C=\frac{3(\pi^2/4-\theta^2+\cos^2\theta)}{16\cos\theta},\qquad
v=\frac{2\pi(a/R_*)s}{P},
$$

with $s=\sqrt{1-(1-b^2)/(a/R_*)^2}$ and $\theta=\arcsin(b/s)$.
For WASP-39 b the full calculation gives 3,134–3,138 s per stellar radius in every channel, and the formula gives 3,141 s, so they agree to about 0.2%.
The 9.6 s offset in the water band would therefore mean an evening limb about 0.003 $R_*$ bigger, relative to the white light, at those wavelengths.
koala's calculation doesn't need either assumption, and works for eccentric and grazing orbits too.

If you'd rather fit the shape of the planet directly, see [morning and evening limbs with Harmonica](harmonica.md).
