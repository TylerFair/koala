# Your first fit

In this tutorial we'll fit a transit of the hot Saturn WASP-39 b observed with NIRISS/SOSS.
We'll start from the extracted spectra and finish with the planet's transmission spectrum.
The data come with koala, so all you need is a working [installation](install.md).
Run everything from the top of the `koala` directory you cloned.

## A first look at the data

The observation is in `examples/data/WASP-39_soss_binned8.fits`.
It's an exoTEDRF extraction of the JWST Early Release Science observation of WASP-39 b (program 1366, observation 21), with 537 integrations spread over 8.2 hours.
To keep the file small, every eight neighbouring detector columns have been added together, which leaves 256 wavelengths in each SOSS order.
The integrations themselves are untouched.

It's always worth looking at the data before fitting anything, so let's open order 1:

```python
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

with fits.open("examples/data/WASP-39_soss_binned8.fits") as hdul:
    time = hdul["TIME"].data          # BMJD_TDB, as in every JWST file
    wave = hdul["WAVE O1"].data       # microns
    flux = hdul["FLUX O1"].data       # shape (integration, wavelength)

hours = (time - time[0]) * 24
white = flux.sum(axis=1)
white /= np.median(white[:100])
image = flux / np.median(flux[:100], axis=0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.9, 5.4), sharex=True)
ax1.plot(hours, white, ".k", ms=2)
ax1.set_ylabel("white-light flux")
ax2.pcolormesh(hours, wave, image.T, vmin=0.97, vmax=1.01, shading="nearest")
ax2.set_xlabel("hours since the first integration")
ax2.set_ylabel("wavelength (µm)")
```

```{figure} _static/first_fit_data.png
:alt: The white-light curve of the WASP-39 b SOSS observation, and the flux at each wavelength against time
```

The top panel is the white-light curve, the sum over all wavelengths, and the transit is the 2% dip in the middle.
The bottom panel shows every wavelength as one row, divided by its flux before the transit.
The transit shows up at every wavelength, and the data get noisier towards 2.8 µm, where there are fewer photons.
What we're after is much harder to see by eye: small changes in the transit depth with wavelength, caused by the planet's atmosphere.

## Setting up the fit

koala reads everything it needs from one YAML file.
Here's the one for this observation, which is also in `examples/niriss_soss_order1.yaml`:

```yaml
planet:
  name: WASP-39
  period: {value: 4.05528043, prior: fixed}
  t0: {value: 59787.555, prior: uniform, low: 59787.505, high: 59787.605}
  duration: {value: 0.117, prior: log_uniform, low: 0.01, high: 1.0}
  b: {value: 0.45, prior: uniform, low: 0.0, high: 1.5}
  rprs: {value: 0.146, prior: uniform, low: 0.01, high: 0.5}

stellar:
  teff: 5509
  logg: 4.22
  feh: 0.04

instrument: NIRISS/SOSS
order: 1

input_file: examples/data/WASP-39_soss_binned8.fits
output_dir: results/WASP-39_SOSS_ORDER1

resolution:
  high: 100
```

The `planet` block gives a starting value and a prior for each property of the orbit.
We hold the period fixed, since one transit can't measure it better than the published ephemeris.
The others are fitted with broad priors: the mid-transit time `t0` anywhere in a 2.4-hour window, the transit duration, the impact parameter `b`, and the radius ratio `rprs` ($R_p/R_*$).
Times are in BJD_TDB − 2400000, the way most papers quote them, and the period and duration are in days.
Literature values often use a different zero point, so it's worth checking this when you set up your own planet.

The `stellar` block is the star's effective temperature, surface gravity and metallicity.
koala uses these to look up the limb darkening at each wavelength in a grid of stellar models.
By default those values are only starting points, and the limb darkening is fitted freely.

`instrument` and `order` say which observing mode this is ([other modes](guide/instruments.md) are set the same way), `input_file` is the extracted spectra, and `output_dir` is where the results go.
Finally, `resolution` sets the wavelength channels: here, bins with a constant resolving power $R = \lambda/\Delta\lambda = 100$.

And that's everything a fit needs.
Everything else has a default that's a good place to start:

- a linear trend in time for each light curve,
- quadratic limb darkening with wide uniform priors on its two coefficients,
- an extra white-noise (jitter) term on top of the reported flux uncertainties,
- integrations more than 5σ from the white-light model are clipped, and left out of the channel fits too,
- four chains of 1000 posterior draws each, which have to agree before koala accepts a fit.

[Configuring a fit](guide/configuration.md) shows how to change these.

## Running koala

Now we can run the fit from a terminal:

```bash
koala -c examples/niriss_soss_order1.yaml
```

or from a notebook (start a fresh kernel, so koala can set up JAX before anything else imports it):

```python
import koala

koala.fit("examples/niriss_soss_order1.yaml")
```

This fit took about 5 minutes on 16 cores of a cluster node (Intel Xeon Gold 6248R).
Here's what koala printed along the way:

```text
koala 0.1.0 | WASP-39 | NIRISS/SOSS order 1 | CPU, 16 cores
Reading examples/data/WASP-39_soss_binned8.fits
  exoTEDRF spectra with time, wavelength, flux and flux error
  537 integrations over 8.2 hours, 246 wavelength columns from 0.89 to 2.79 µm
  Binned to 115 channels at R = 100
Planet parameters:
  period    fixed at 4.05528043
  t0        uniform(59787.505, 59787.605), start 59787.555
  duration  log-uniform(0.01, 1.0), start 0.117
  b         uniform(0.0, 1.5), start 0.45
  rprs      uniform(0.01, 0.5), start 0.146
Limb darkening: quadratic law with uniform priors on u1 + u2 and u1 - u2
Stagger has no models above solar metallicity, so limb darkening uses its [Fe/H] = 0 models.
Downloaded stellar limb-darkening data.
White light: transit with a linear trend, 4 chains of 1000 draws
sample: 100%|██████████████████████████████████████████████████| 1200/1200 [00:37<00:00, 32.05it/s]
  Converged: R-hat <= 1.003, bulk effective sample size >= 1948, no divergences
  rprs             0.14591 ± 0.00018
  b                0.4510 ± 0.0048
  t0               59787.556749 ± 0.000014 BJD_TDB - 2400000
  duration         0.119600 ± 0.000088 d
  Residual scatter 109 ppm; 1 integration clipped at 5σ
Limb darkening from stellar models: 100%|███████████████████████| 115/115 [00:00<00:00, 679.57it/s]
Spectroscopy at R = 100: 115 channels, orbit fixed to the white-light posterior
Channels: 100%|██████████████████████████████████████████████████| 115/115 [03:03<00:00,  1.60s/it]
All 115 channels passed the convergence checks!
Spectrum: results/WASP-39_SOSS_ORDER1/WASP-39_NIRISS_SOSS_order1_R100_spectrum.csv
Figures: results/WASP-39_SOSS_ORDER1/figures/
Full log: results/WASP-39_SOSS_ORDER1/koala.log
Done in 5 min 7 s.
```

Let's go through that.
First koala reads the file and tells us what it found, which is a good moment to check that the instrument, the number of integrations and the wavelength range are what you expect.
It then prints the priors it's going to use, and which stellar models it will use for the limb darkening.
That's the 3D Stagger grid, and since WASP-39 is a little more metal-rich than the Sun, where Stagger stops, koala uses its solar-metallicity models.
The first time you fit a star, it also downloads the stellar models it needs.

Next it fits the white-light curve.
Adding up all the wavelengths gives the most precise light curve we have, so this is where koala measures the orbit: the transit time, duration and impact parameter, along with the white-light radius ratio, limb darkening, trend and jitter.
Once the four chains agreed, it printed the results and how many outlying integrations it clipped.
(If the chains have divergent transitions, which happens when the sampler's steps are too big for the shape of the posterior, koala says so and runs them again with smaller steps.)

Then it looks up the limb darkening at each wavelength and fits each channel with the orbit held fixed at the white-light result.
Each channel gets its own radius ratio, limb darkening, trend and jitter.
The progress bar counts channels, and at the end koala checks every channel's chains and tells us whether they all converged.

If you run the same command again, koala notices the finished work in the output directory and reuses it instead of fitting everything again.

## The white-light fit

Before looking at the spectrum, let's check that the white-light fit is good.
koala makes its own figures in `figures/` in the output directory (this one is `figures/whitelight.png`), but it's easy to make our own from the tables it saves:

```python
from pathlib import Path
import pandas as pd

out = Path("results/WASP-39_SOSS_ORDER1")
wl = pd.read_csv(next(out.glob("*_whitelight_lightcurve.csv")))
good = wl["is_outlier"] == 0
hours = wl["time_from_t0_hr"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.9, 4.6), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax1.plot(hours[good], wl["detrended_flux"][good], ".k", ms=2, alpha=0.5)
ax1.plot(hours, wl["transit_model"], "C1", lw=1.5)
ax1.set_ylabel("detrended flux")
ax2.plot(hours[good], wl["residual_ppm"][good], ".k", ms=2, alpha=0.5)
ax2.axhline(0, color="C1", lw=1.5)
ax2.set_xlabel("hours from mid-transit")
ax2.set_ylabel("residual (ppm)")
```

```{figure} _static/first_fit_whitelight.png
:alt: The detrended white-light curve with the fitted transit model, and the residuals in ppm
```

The model goes right through the data, and the residuals scatter by about 110 ppm around zero with no pattern left in them, which is what we'd hope for.
One integration at the very start of the observation was more than 5σ from the model, so koala clipped it, and it's left out of this plot.

The fitted orbit is in `*_whitelight_params.csv`:

```python
params = pd.read_csv(next(out.glob("*_whitelight_params.csv")))
print(params[["t0", "t0_err", "duration", "duration_err", "b", "b_err", "rors", "rors_err"]].iloc[0])
```

```text
t0              59787.556749
t0_err              0.000014
duration            0.119600
duration_err        0.000088
b                   0.451028
b_err               0.004754
rors                0.145912
rors_err            0.000177
```

The transit time is measured to about a second and the radius ratio to about 0.1%, which is why it pays to fit the orbit once with all the light and then hold it fixed.

## The transmission spectrum

Now for the fun part.
The spectrum is in a CSV file named after the target, instrument and wavelength grid:

```python
spec = pd.read_csv(next(out.glob("*_R100_spectrum.csv")))

fig, ax = plt.subplots(figsize=(6.9, 3.2))
ax.errorbar(spec["wavelength"], spec["depth_ppm00"],
            xerr=spec["wavelength_err"], yerr=spec["depth_err_ppm00"],
            fmt="o", ms=3, color="k")
ax.set_xlabel("wavelength (µm)")
ax.set_ylabel("transit depth (ppm)")
```

```{figure} _static/first_fit_spectrum.png
:alt: The transmission spectrum of WASP-39 b from NIRISS/SOSS order 1 at R = 100
```

That looks pretty good!
The bumps at 1.15, 1.4, 1.9 and 2.7 µm are water absorbing in the planet's atmosphere, the same features seen in these data by [Feinstein et al. (2023)](https://doi.org/10.1038/s41586-022-05674-1).
The `00` at the end of the column names refers to the first planet, since koala can fit several planets in the same light curve.
The depth is $(R_p/R_*)^2$, and its uncertainty doesn't include the uncertainty in the orbit, which is the same for every channel.

## Next steps

To fit your own observation, copy this file and change the planet, the star, the instrument and the input file.
These pages cover what you're likely to need next:

- [configuring a fit](guide/configuration.md): priors, wavelength grids, masking bad integrations, and the other settings,
- [instruments and input files](guide/instruments.md): which modes and file formats koala reads,
- [limb darkening](guide/limb_darkening.md) and [trends](guide/trends.md): the model choices that matter most for a spectrum,
- [output files](guide/outputs.md): everything in the results directory.

The [WASP-39 b case study](case_studies/transmission_soss.md) carries on from here with SOSS order 2 and the full-resolution spectrum.
