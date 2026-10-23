# Trends

JWST light curves are rarely flat outside transit.
There's usually a slow drift, sometimes a ramp while the detector settles, and occasionally a jump or the bump of a starspot crossing.
koala fits a trend together with the transit, and the model is the transit multiplied by the trend, $F = (1 + \mathrm{transit}) \times \mathrm{trend}$.
Each wavelength channel gets its own trend of the same form.

## Supported trends

Set the trend with `flags.detrending_type`:

```yaml
flags:
  detrending_type: explinear
```

```{figure} ../_static/trend_gallery.png
:alt: The shape of every supported trend, without a transit

The shape of each trend, with made-up amplitudes and no transit.
```

| `detrending_type` | Trend |
|---|---|
| `linear` (default) | a straight line in time |
| `constant`, `quadratic`, `cubic`, `quartic` | a polynomial of that order |
| `explinear` | an exponential ramp plus a straight line |
| `linear_discontinuity` | a straight line with a smooth step |
| `spot`, `2spot` | a straight line plus one or two spot crossings (Gaussian bumps) |
| `quadratic+spot` | a quadratic plus one spot crossing |
| `spot+linear_discontinuity` | a spot crossing, a straight line and a step |
| `spot+explinear`, `2spot+explinear` | one or two spot crossings with a ramp and a straight line |
| `gp` | a Gaussian process about a constant |
| `linear+gp`, `quadratic+gp`, `cubic+gp`, `quartic+gp`, `explinear+gp` | the named trend plus a Gaussian process |
| `none` | no trend at all |

The ramp, step and spot shapes (the ramp's timescale, the step's time, and the position and width of each spot) are measured from the white-light curve.
In the wavelength channels those shapes stay fixed, and only their amplitudes are fitted, along with the polynomial.

For a spot crossing or a step, give koala a rough idea of where it is (times in BJD_TDB − 2400000, widths in days, amplitudes in normalized flux):

```yaml
flags:
  detrending_type: spot
  spot_center: 59787.57
  spot_width: 0.005
  spot_amp: 0.0005
```

A second spot uses `spot_center2`, `spot_width2` and `spot_amp2`, and a step uses `t_jump_guess` and `jump_guess`.

## Choosing a trend

Start with the default straight line, fit the white-light curve, and look at the residuals.
If there's structure left that clearly isn't noise, try the next simplest trend that could explain it, and compare the spectra as well as the residuals.
Keep the data and the other settings the same, so that the trend is the only thing that changes.

It's worth remembering that a flexible trend can soak up part of the signal you're trying to measure.
A ramp with a free timescale can trade off against a shallow eclipse, and a Gaussian process with a timescale close to the ingress or the transit duration can absorb part of the transit.
If a strong ramp at the start is the problem, it's often better to leave out the first few minutes with `exclude_times` (see [configuring a fit](configuration.md#leaving-out-bad-data)) than to model it.

If you can't decide between a few trends, you don't have to.
The [model stacking case study](../case_studies/model_stacking.md) shows how to fit several and combine their spectra.

## Gaussian processes

The `gp` trends use a Matérn-3/2 Gaussian process from [tinygp](https://tinygp.readthedocs.io), fitted to the white-light curve.
Each wavelength channel then uses the white-light GP curve as a fixed shape with its own amplitude, so the channel fits are conditional on the white-light GP.

## A spotted star

The spot-crossing trends above are simple bumps in the light curve.
koala can also fit a physical model of a rotating star with spots on its surface (using jaxoplanet's starry surface maps), where each spot has a position and size, and a contrast that can vary with wavelength:

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

The positions and sizes are fixed, and a positive `contrast_prior_width` lets the contrast be fitted (0 is the same brightness as the rest of the star, 1 is completely dark).
The longitude is measured at the transit time `t0`.
Keep the spots moderate in size and not overlapping, since the surface is described by a smooth map.
`examples/stellar_spots.yaml` is a complete example on generated data.

## Trend priors

By default, every trend coefficient has a wide uniform prior, so the light curve rather than the prior decides the trend.
The `trend_priors` block changes the priors on the white-light trend:

```yaml
trend_priors:
  v: {prior: gaussian, value: 0.0, sigma: 0.01}
  tau: {prior: uniform, low: 0.001, high: 0.05}
```

This uses the same form as the planet parameters, and any coefficient you leave out keeps its default.
The [settings reference](../reference/settings.md#trend-priors) lists the coefficient names and their default priors.
