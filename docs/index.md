# koala

koala fits extracted JWST time-series spectra with JAX and NumPyro.
We first fit the white-light curve to measure the planet's orbit.
Then we use that orbit to fit each wavelength channel and measure a transmission or emission spectrum.
koala runs on CPUs and GPUs and fits every integration without averaging in time.

These features include:

- Readers for exoTEDRF, SPARTA, and Eureka! extractions from NIRISS/SOSS, NIRSpec, NIRCam, and MIRI/LRS.
- Quadratic and power-2 limb darkening, with several choices of coefficient prior.
- Polynomial trends, exponential ramps, Gaussian processes, and spot-crossing models.
- Transits, secondary eclipses, and thermal phase curves.
- Wavelength-dependent transit times and asymmetric planetary limbs with Harmonica.
- Model comparisons that combine spectra from several plausible fits.

:::{admonition} Where to find what you need
:class: hint

If you're new to koala, start with [installation](install.md) and [your first fit](quickstart.md).
The first fit uses a bundled WASP-39 b observation.

The tutorials below explain one model choice at a time.
For a complete analysis, try a case study using the bundled observation, synthetic data, or your own extraction.

If you're looking for a particular setting or output column, go to [configuration](configuration.md) or [output files](outputs.md).
If a fit goes wrong, start with [troubleshooting](troubleshooting.md).
:::

```{toctree}
:maxdepth: 1
:caption: User guide

install
quickstart
configuration
outputs
troubleshooting
testing
citing
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/transit_model
tutorials/limb_darkening
tutorials/systematics
tutorials/planet_parameters
tutorials/time_shift
tutorials/eclipses_and_phase_curves
tutorials/harmonica
tutorials/combining_models
```

```{toctree}
:maxdepth: 1
:caption: Case studies

case_studies/transmission_soss
case_studies/rocky_eclipse
case_studies/phase_curve
case_studies/model_stacking
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
reference/model_marginalization
```
