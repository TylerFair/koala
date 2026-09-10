# Koala

```{raw} html
<p class="tagline">Kool exOplAnet Lightcurve Analysis: JWST light-curve fitting, from transits to phase curves</p>
```

Koala fits extracted JWST time-series spectra with [JAX](https://jax.readthedocs.io/)
and [NumPyro](https://num.pyro.ai/). Give it a FITS extraction and a short
YAML file; it fits the white-light transit, carries the shared geometry into
every wavelength channel, and writes a transmission spectrum. JAXoplanet fits
also support [eclipses, phase curves, and stellar spots](guides/phase_curves.md).

Koala supports every JWST time-series mode (NIRISS/SOSS, all NIRSpec gratings
and PRISM, NIRCam grism, MIRI/LRS) and reads exoTEDRF, SPARTA, and Eureka!
products, detecting the format automatically. It offers
GPU-parallel inference, resumable runs, and predictive model stacking.

```{image} _static/soss_wasp39_spectrum.png
:alt: NIRISS/SOSS transmission spectrum of WASP-39 b fitted with Koala
:class: hero-figure
:width: 680px
:align: center
```

## Installation

Install JAX for your hardware (CPU by default, or the CUDA build for NVIDIA
GPUs), then install Koala from the repository:

```bash
git clone https://github.com/TylerFair/koala.git
cd koala
python -m pip install -e .
```

The [installation guide](install.md) covers GPU wheels, the limb-darkening
data, and cluster environments.

:::{admonition} Navigating the docs
:class: tip

- After [installing](install.md) Koala, head to the [Quickstart](quickstart.md)
  to fit the bundled WASP-39 b example.
- The [Introduction](concepts.md) explains what each stage of a fit does.
- The [Tutorials](tutorials/index.md) follow real observations; the guides
  cover configuration, limb darkening, surface models, and model stacking.
- Check the [FAQ](faq.md) if a fit does not behave as expected.
:::

## The workflow

1. Start from the example for your instrument and point it at an extracted
   exoTEDRF, SPARTA, or Eureka! spectral time series.
2. Inspect the white-light fit and residuals before trusting the spectrum.
3. Choose a trend and limb-darkening treatment supported by the data.
4. Resume, compare, or stack models without changing the basic workflow.

## Where to go next

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Fit your first transit
:link: tutorials/soss_order1
:link-type: doc

Run the bundled NIRISS/SOSS example from configuration to spectrum.
:::

:::{grid-item-card} Fit a thermal phase curve
:link: tutorials/phase_curve
:link-type: doc

Recover the day-night contrast and hotspot offset of a synthetic phase curve.
:::

:::{grid-item-card} Choose limb darkening
:link: guides/limb_darkening
:link-type: doc

The two intensity laws and the five `ld_prior` choices.
:::

:::{grid-item-card} Marginalize over models
:link: guides/model_stacking
:link-type: doc

Propagate disagreement between plausible light-curve models into the spectrum.
:::
::::

## Attribution and license

See [Citing Koala](citing.md) for how to reference Koala and the methods your
configuration uses. Koala is distributed under the BSD 3-Clause License;
bundled components keep their own licenses.

```{toctree}
:hidden:
:caption: Getting started

install
quickstart
concepts
faq
```

```{toctree}
:hidden:
:caption: Tutorials

Overview <tutorials/index>
tutorials/soss_order1
tutorials/rocky_eclipse
tutorials/phase_curve
tutorials/harmonica
```

```{toctree}
:hidden:
:caption: Guides

guides/configuration
guides/limb_darkening
guides/phase_curves
guides/model_stacking
guides/gaussian_processes
```

```{toctree}
:hidden:
:caption: Reference

api
citing
```
