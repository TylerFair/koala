# koala

koala fits JWST transit, eclipse and phase-curve light curves.
You give it the extracted spectra and a short YAML file describing the planet.
It fits the white-light curve to measure the orbit, then fits each wavelength channel, and gives you back a spectrum along with the fitted light curves and posteriors.
koala is built on [JAX](https://docs.jax.dev) and [NumPyro](https://num.pyro.ai), so the same fit runs on a laptop or a GPU.
Every integration goes into the fit, and nothing is averaged in time.

```{image} _static/soss_spectrum.png
:alt: Transmission spectrum of WASP-39 b from the NIRISS/SOSS example
:class: hero-figure
```

Some of the things koala can do:

- fit NIRISS/SOSS, NIRSpec (PRISM and the G140, G235 and G395 gratings), NIRCam grism and MIRI/LRS time series,
- read extracted spectra from exoTEDRF, Eureka! and SPARTA directly,
- quadratic and power-2 limb darkening, either free or informed by stellar models,
- polynomial, ramp, step, spot-crossing and Gaussian-process trends,
- transits, secondary eclipses and phase curves,
- one orbit fitted jointly to several orders, detectors or visits,
- a transit time for every wavelength channel, and separate morning and evening limbs,
- combining fits with different trends or limb darkening by their predictive weights.

If you're new to koala, [install it](install.md) and then work through [your first fit](quickstart.md).
That fits a real WASP-39 b observation that comes with the code, from the extracted spectra to the spectrum above.
The user guide covers the choices you'll make for your own data, and the case studies go through longer analyses.

If something doesn't work, or these pages don't answer your question, please [open an issue on GitHub](https://github.com/TylerFair/koala/issues).

```{toctree}
:maxdepth: 1
:caption: Getting started

install
quickstart
```

```{toctree}
:maxdepth: 1
:caption: User guide

guide/configuration
guide/instruments
guide/limb_darkening
guide/trends
guide/outputs
guide/troubleshooting
```

```{toctree}
:maxdepth: 1
:caption: Case studies

case_studies/transmission_soss
case_studies/timing_spectra
case_studies/rocky_eclipse
case_studies/phase_curve
case_studies/harmonica
case_studies/model_stacking
```

```{toctree}
:maxdepth: 1
:caption: Reference

reference/settings
reference/python
reference/model_grids
reference/development
reference/citing
```
