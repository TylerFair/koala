# Koala

**Kool exOplAnet Lightcurve Analysis**

**Transit light curves in, transmission spectra out.**

Koala fits extracted JWST time-series spectra with JAX and NumPyro. It
first learns the transit geometry and visit-wide systematics from the
white-light curve, then fits the wavelength channels in parallel and writes a
transmission spectrum with diagnostic plots and per-channel checks.

<p align="center">
  <img src="docs/_static/soss_wasp39_spectrum.png" width="760" alt="A fitted JWST NIRISS/SOSS transmission spectrum for WASP-39 b">
</p>

The code supports NIRISS/SOSS and NIRSpec time series, several limb-darkening
and instrumental-trend models, GPU-parallel exact inference, resumable runs,
and predictive model stacking. Start with the supplied configurations rather
than building a model from scratch.

The JAXoplanet backend also supports eclipses, smooth phase curves, and
rotating stellar spots. See the [surface-model guide](docs/guides/phase_curves.md)
for configuration, physical assumptions, and repeatable synthetic examples.
The [executed surface tutorial](docs/tutorials/synthetic_surfaces.md) includes
eclipse and day/night spectra, a thermal map with uncertainty, stellar-spot
recovery, corner plots, and residual diagnostics, with PNG/PDF downloads.

## Get started

Clone the repository, create an environment, and install the scientific
dependencies:

```bash
git clone https://github.com/TylerFair/jwst-lightcurves.git
cd jwst-lightcurves
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jax numpy numpyro numpyro-ext jaxopt jaxoplanet \
  astropy pandas scipy matplotlib arviz pyyaml tinygp exotic-ld \
  'exotedrf[stage4]'
```

The SOSS example points to a compact real WASP-39 extraction included in the
repository. Set its `stellar.ld_data_path`, then run it from the repository
root:

```bash
cp examples/niriss_soss_order1.yaml config.yaml
# Edit stellar.ld_data_path in config.yaml.
python fit_jwst.py -c config.yaml
```

For an NVIDIA GPU, install the JAX CUDA build appropriate for your system
before running the fit. The `stellarprior` limb-darkening choice also needs the
ExoTiC-LD model-data directory.

The bundled FITS file retains the complete observation but combines adjacent
detector columns to keep the download small. It is intended for learning the
workflow; use your original extraction for science.

Read the **[installation guide](https://jwst-lightcurves.readthedocs.io/en/latest/install.html)**
for those two steps, then follow **[Fit your first transit](https://jwst-lightcurves.readthedocs.io/en/latest/tutorials/soss_order1.html)**.

## Learn by doing

- [Fit your first transit](https://jwst-lightcurves.readthedocs.io/en/latest/tutorials/soss_order1.html)
- [Choose a systematics trend](https://jwst-lightcurves.readthedocs.io/en/latest/guides/trends.html)
- [Choose limb darkening](https://jwst-lightcurves.readthedocs.io/en/latest/guides/limb_darkening.html)
- [Marginalize over models](https://jwst-lightcurves.readthedocs.io/en/latest/guides/model_stacking.html)

The [documentation](https://jwst-lightcurves.readthedocs.io/) also covers
NIRSpec, output files, cluster runs, and the optional Harmonica asymmetric-
transit model.

## Citation and license

See the [citation guide](https://jwst-lightcurves.readthedocs.io/en/latest/citing.html)
for the methods used by your configuration. Koala is distributed under the
[BSD 3-Clause License](LICENSE); vendored components retain their own licenses.
