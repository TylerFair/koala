# Koala

**Kool exOplAnet Lightcurve Analysis**

## JWST light-curve fitting, from transits to phase curves

Koala fits extracted JWST time-series spectra. Give it a FITS extraction
and a short YAML file; it fits the white-light transit, carries the shared
geometry into every wavelength channel, and produces a transmission spectrum
with diagnostics you can inspect. JAXoplanet fits also support
[eclipses, thermal phase curves, and stellar spots](guides/phase_curves.md).

Built for NIRISS/SOSS and NIRSpec, with GPU-parallel inference in JAX and
NumPyro.

```{button-ref} quickstart
:color: primary
:shadow:

Fit your first dataset
```

```{button-ref} install
:color: secondary

Install Koala
```

```{image} _static/soss_wasp39_spectrum.png
:alt: NIRISS/SOSS transmission spectrum of WASP-39 b fitted with Koala
:class: hero-figure
:width: 760px
:align: center
```

## The workflow

1. Start from the example for your instrument and point it at an extracted
   box-spectrum FITS file.
2. Inspect the white-light fit and residuals before trusting the spectrum.
3. Choose a trend and limb-darkening treatment supported by the data.
4. Resume, compare, or stack models without changing the basic workflow.

The numerical machinery stays behind the configuration file. The
[Introduction](concepts.md) explains what the stages mean when you are ready
to make scientific choices.

## Tutorials

::::{grid} 1 2 2 2
:gutter: 2

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

## Other observing modes

- [WASP-39 b G395H secondary eclipse](tutorials/wasp39_eclipse.md) presents a real $R=300$ emission spectrum.
- [Synthetic eclipses, phase curves, and stellar spots](tutorials/synthetic_surfaces.md)
  demonstrates emission spectra, thermal maps, corner plots, and residual checks.
- [NIRSpec/G395H](tutorials/nirspec_g395h.md) covers detector-aware fitting.
- [NIRSpec/PRISM](tutorials/prism.md) covers long time series and ramps.
- [Harmonica](tutorials/harmonica.md) covers asymmetric transit shapes.

```{toctree}
:hidden:
:maxdepth: 1

install
concepts
quickstart
faq
```

```{toctree}
:hidden:
:maxdepth: 1

tutorials/index
tutorials/wasp39_eclipse
reference
```
