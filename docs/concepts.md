# Introduction

Koala turns an extracted JWST time series into a transmission spectrum. It
first fits the **white-light curve**, the sum over wavelength, to learn the
transit time, duration, impact parameter, and visit-wide systematics. It then
fixes that geometry and fits every **wavelength channel** for its own radius
ratio, limb darkening, noise term, and trend coefficients. The squared radius
ratios form the transmission spectrum.

```{image} _static/soss_wasp39_whitelight.png
:alt: White-light transit and fitted model for a JWST NIRISS/SOSS observation
:width: 760px
:align: center
```

Each channel model is a transit plus a systematics trend plus white noise
with a fitted jitter term. The trend is set by `flags.detrending_type`:

- `none`
- `linear`
- `quadratic`
- `cubic`
- `quartic`
- `explinear`
- `linear_discontinuity`
- `spot`
- `2spot`
- `quadratic+spot`
- `spot+linear_discontinuity`
- `spot+explinear`
- `2spot+explinear`

A Gaussian process can be added to a polynomial or `explinear` trend by
appending `+gp` (for example `linear+gp`), or used alone as `gp`. It absorbs
correlated residual structure that the mean trend does not describe.

A thermal phase curve is not a trend. It is a separate light-curve model,
`flags.light_curve_model: phase_curve`, that adds the planet's day-night
brightness and hotspot offset to the transit and eclipse; the trend above
still describes the instrument on top of it. See
[Eclipses, phase curves, and stellar spots](guides/phase_curves.md).

Channels are independent once the geometry is fixed, so Koala fits them in
parallel batches on the GPU and checkpoints each completed batch; re-running
the same configuration resumes from the checkpoints. `resolution.high` sets
the final wavelength grid (a resolving power, `native`, or `reference`). An
optional coarse `resolution.low` stage can run first; leave it out and the
pipeline goes straight from white light to the final grid.

The [Quickstart](quickstart.md) walks through this workflow on the bundled
WASP-39 b example.
