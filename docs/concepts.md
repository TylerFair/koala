# Introduction

Koala turns an extracted JWST time series into a transmission spectrum. It
models the transit and the instrument together, while keeping the workflow
simple: learn what is shared from white light, then measure one transit depth
per wavelength channel.

```{image} _static/soss_wasp39_whitelight.png
:alt: White-light transit and fitted model for a JWST NIRISS/SOSS observation
:width: 760px
:align: center
```

## One observation, two scales

The **white-light curve** is the sum over wavelength. Its high signal-to-noise
constrains the transit time, duration, impact parameter, and visit-wide
systematics. Koala saves this fit, then passes its median orbital geometry
to the spectroscopic stage.

A **spectroscopic channel** is one wavelength-bin light curve. With the shared
geometry fixed, each channel measures its own planet-to-star radius ratio,
limb profile, noise term, and trend coefficients. The squared radius ratio is
the transit depth plotted in the final transmission spectrum.

This division keeps weak channels from confusing a change in orbital geometry
with a change in depth. It also means the channel posteriors condition on the
chosen white-light geometry; they do not independently propagate its full
uncertainty. The exact handoff is recorded in
`*_whitelight_geometry_handoff.json`.

## What the model describes

Each channel combines three ingredients:

$$
\mathrm{flux}(t) = \mathrm{transit}(t) + \mathrm{systematics}(t)
                  + \mathrm{noise}(t).
$$

The transit model contains the wavelength-dependent depth and limb darkening.
The systematics model can be a smooth polynomial, an exponential ramp, a
detector step, a spot-crossing template, or a Gaussian process. The likelihood
uses the input uncertainty with an additional fitted jitter term.

These choices can trade against transit depth, so they are scientific
assumptions rather than cosmetic settings. Begin with the simplest trend that
describes the out-of-transit baseline and inspect the residuals. Use a
`stellarprior` limb prior when its assumptions and atmosphere grid are
appropriate; compare plausible alternatives when the spectrum is sensitive to
the choice.

- [Choose a systematics trend](guides/trends.md)
- [Choose limb darkening](guides/limb_darkening.md)

## From one channel to a spectrum

Channels are independent once the shared geometry is fixed. Koala therefore
runs several chains together on the GPU. `flags.vmap_chunk` controls how many
channels live on the device at once; it changes memory use and throughput, not
the wavelength bins or model.

Each completed group is checkpointed. Re-running the same configuration loads
compatible checkpoints and computes only missing work. The final spectrum
collects the accepted depth posteriors in wavelength order.

An optional low-resolution stage can act as a bridge before the final grid. It
is used when a model needs a smooth wavelength-dependent calibration, including
some limb-darkening treatments. `resolution.high` selects the final grid:

- a number requests constant resolving power;
- `native` keeps the extraction's channelization;
- `reference` uses `resolution.reference_grid`.

## When a fit is trustworthy

Sampling finishing is not the same as sampling succeeding. Koala checks the
effective sample size of the depth posterior and counts divergent transitions.
A failing channel is retried with an alternate exact sampler. The log and the
JSON files under `chunks/` record what was accepted.

Your scientific check is equally important. Inspect the white-light residuals,
look for wavelength-localized failures, and ask whether the trend and limb
model are plausible for the observation. When several plausible models remain,
[model stacking](guides/model_stacking.md) can carry their predictive
disagreement into the reported spectrum.

The [Quickstart](quickstart.md) walks through this workflow on a supplied
configuration.
