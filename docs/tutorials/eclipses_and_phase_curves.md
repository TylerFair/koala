# Eclipses and phase curves

During secondary eclipse, the star hides the planet's emitted light.
An eclipse fit measures that planet-to-star flux ratio.
A phase-curve fit also measures how the planet's brightness changes around its orbit.
Here we'll look at what those model parameters mean before running the longer case studies.

## Start with an eclipse

The synthetic [MIRI/LRS example](../case_studies/rocky_eclipse.md) uses a uniformly bright planet.
Its eclipse depth is configured in ppm:

```yaml
planet:
  eclipse_depth_ppm: {value: 100.0, prior: uniform, low: 0.0, high: 1000.0}
flags:
  light_curve_model: eclipse
  transit_engine: jaxoplanet
```

This is a fragment of the complete `examples/eclipse.yaml` configuration.
The orbit uses `a_rs`.
For a circular orbit with a fixed period, `eclipse_time` can replace `t0` (see [time conventions](planet_parameters.md#transit-and-eclipse-times)).

Before applying the trend, the flux is 1 during total eclipse and `1 + eclipse_depth_ppm / 1e6` outside eclipse.
The emitted flux is independent of `rprs**2`.
When you open the results, read `eclipse_depth_ppm` from the emission table.
The generic `depths` column still describes the planet's transit area ratio.

## Sinusoidal phase curve

Now suppose the observation covers the planet's changing brightness around its orbit.
We can add a cosine and a sine term:

```yaml
planet:
  eclipse_depth_ppm: {value: 120.0, prior: uniform, low: 0.0, high: 1000.0}
  phase_c1_ppm: {value: 50.0, prior: uniform, low: -1000.0, high: 1000.0}
  phase_d1_ppm: {value: 0.0, prior: uniform, low: -1000.0, high: 1000.0}
flags:
  light_curve_model: phase_curve
```

The planet flux is

$$L(\phi)=f_d+C_1(\cos\phi-1)+D_1\sin\phi,$$

where $f_d$ is the flux at the reference phase and $\phi=2\pi(t-t_0-P/2)/P$.
For a circular orbit, that phase is secondary eclipse.
The full astrophysical model is

$$F(t)-1=\mathrm{transit}(t)+L(\phi)E(t),$$

before multiplying by the instrument trend.
Here $E(t)$ is the visible fraction of a uniform planet disk.
It is 1 outside eclipse and 0 during totality.
The transit uses the chosen quadratic or power-2 limb darkening.

At transit phase, the planet flux is $f_d-2C_1$.
A negative $D_1$ gives a brightness maximum before eclipse when expressed with the output phase-offset convention below.
No positivity or day/night ratio limit is imposed.
For eccentric orbits, $t_0+P/2$ need not be the eclipse time.

Try the [synthetic thermal phase curve](../case_studies/phase_curve.md) to see these coefficients recovered from an observation.
An eclipse-only visit generally cannot constrain both phase coefficients.

## Stellar spots

koala can also model a rotating spotted stellar surface.
The generated example `examples/stellar_spots.yaml` uses fixed spot positions and radii.
A positive contrast-prior width frees the contrast:

```yaml
stellar:
  rotation_period: 25.0
  spots:
    - latitude_deg: -8.5
      longitude_deg: 2.0
      radius_deg: 10.0
      contrast: 0.2
      contrast_prior_width: 0.15
```

Longitude is defined at `t0`.
Contrast runs from 0 for the photosphere to 1 for a dark spot.
Use moderate, nonoverlapping spots because the surface is represented by a smooth spherical-harmonic approximation.
These models require jaxoplanet.
They differ from the empirical spot-crossing [trend templates](systematics.md).

## Reference

### Model choices

| `light_curve_model` | Planet parameters | Meaning |
|---|---|---|
| `transit` | `rprs` | A dark planet blocks the star. |
| `eclipse` | `eclipse_depth_ppm` | A uniformly bright planet disappears behind the star. |
| `phase_curve` | `eclipse_depth_ppm`, `phase_c1_ppm`, `phase_d1_ppm` | A sinusoidally varying uniform planet disk transits and is eclipsed. |

Use fixed, uniform, or Gaussian priors for signed phase coefficients.
With fixed geometry, the model is linear in $f_d$, $C_1$, and $D_1$.
The spectroscopic fit therefore uses an exact precomputed basis.

### Derived emission quantities

| Output | Definition |
|---|---|
| `nightside_flux_ppm` | $f_d-2C_1$, the planet flux at transit phase. |
| `phase_amplitude_ppm` | $\sqrt{C_1^2+D_1^2}$. |
| `phase_offset_deg` | $-\operatorname{atan2}(D_1,C_1)$ in degrees. Positive puts the maximum before secondary eclipse for a circular orbit (an eastward hotspot). |

Without the minus sign, $\operatorname{atan2}(D_1,C_1)$ is the phase of the maximum measured after the reference eclipse phase.
Each emitted-flux quantity has lower and upper posterior-error columns.
See [output files](../outputs.md) for their names and units.
