# Koala

```{raw} html
<p class="tagline">Kool exOplAnet Lightcurve Analysis: JWST light-curve fitting, from transits to phase curves</p>
```

Koala fits extracted JWST time-series spectra with [JAX](https://jax.readthedocs.io/)
and [NumPyro](https://num.pyro.ai/). Give it a FITS extraction and a short
YAML file; it fits the white-light transit, carries the shared geometry into
every wavelength channel, and produces a transmission spectrum with
diagnostics you can inspect. JAXoplanet fits also support
[eclipses, thermal phase curves, and stellar spots](guides/phase_curves.md).

Koala is built for NIRISS/SOSS and NIRSpec, with GPU-parallel inference,
resumable runs, and predictive model stacking.

```{image} _static/soss_wasp39_spectrum.png
:alt: NIRISS/SOSS transmission spectrum of WASP-39 b fitted with Koala
:class: hero-figure
:width: 680px
:align: center
```

## Installation

Install JAX for your hardware (CPU by default, or the CUDA extra for NVIDIA
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
  to fit the bundled WASP-39 b example in a few minutes.
- The [Introduction](concepts.md) explains what each stage of a fit does
  before you make scientific choices.
- The [Tutorials](tutorials/index.md) follow real observations for each
  instrument mode; the [Guides](reference.md) go deeper on trends, limb
  darkening, samplers, and outputs.
- Check the [FAQ](faq.md) if a fit does not behave as expected.
:::

## The workflow

1. Start from the example for your instrument and point it at an extracted
   box-spectrum FITS file.
2. Inspect the white-light fit and residuals before trusting the spectrum.
3. Choose a trend and limb-darkening treatment supported by the data.
4. Resume, compare, or stack models without changing the basic workflow.

## Where to go next

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Fit your first transit
:link: tutorials/soss_order1
:link-type: doc

Run a NIRISS/SOSS example and learn which output plots matter first.
:::

:::{grid-item-card} Choose a systematics trend
:link: guides/trends
:link-type: doc

Compare linear, polynomial, ramp, step, spot, and GP descriptions.
:::

:::{grid-item-card} Choose limb darkening
:link: guides/limb_darkening
:link-type: doc

Understand the five `ld_prior` choices and when each applies.
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
tutorials/nirspec_g395h
tutorials/prism
tutorials/wasp39_eclipse
tutorials/harmonica
tutorials/synthetic_surfaces
tutorials/executed_notebooks
```

```{toctree}
:hidden:
:caption: Guides

Overview <reference>
guides/trends
guides/limb_darkening
guides/model_stacking
guides/configuration
guides/outputs
guides/samplers
guides/gaussian_processes
guides/phase_curves
guides/gpu_and_clusters
guides/loop_mode
```

```{toctree}
:hidden:
:caption: Reference

api
citing
```
