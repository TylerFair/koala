# Installation

Koala currently runs directly from its repository. A dedicated environment
keeps JAX and its compiled dependencies separate from other analysis code.

## 1. Clone and create an environment

```bash
git clone https://github.com/TylerFair/jwst-lightcurves.git
cd jwst-lightcurves
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Python 3.11 is the recommended version.

## 2. Install the scientific dependencies

For a CPU environment:

```bash
python -m pip install jax numpy numpyro numpyro-ext jaxopt jaxoplanet \
  astropy pandas scipy matplotlib arviz pyyaml tinygp exotic-ld \
  'exotedrf[stage4]'
```

The ExoTEDRF `stage4` extra supplies the box-spectrum reader and binning
utilities used by the input pipeline.

CPU is useful for checking a configuration and building the documentation. A
full spectroscopic fit is designed for an NVIDIA GPU. Install the CUDA-enabled
JAX wheel using the command for your driver and CUDA installation in the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html),
then install the remaining packages above.

Check what JAX can see:

```bash
python -c "import jax; print(jax.devices())"
```

Run this check inside a GPU allocation on a cluster. Seeing only `CpuDevice`
there means the JAX build, driver, or allocation needs attention.

## 3. Add the limb-darkening data

The `exotic-ld` Python package and its stellar-atmosphere grids are separate.
Download the grids following the
[ExoTiC-LD installation guide](https://exotic-ld.readthedocs.io/en/latest/views/installation.html),
then point each `stellarprior` configuration to the data directory:

```yaml
stellar:
  ld_model: stagger
  ld_data_path: /path/to/exotic_ld_data
```

Use an absolute path on a cluster. The compute node must be able to read the
directory.

## 4. Verify the command-line program

Run from the repository root:

```bash
python fit_jwst.py --help
```

You should see the required `-c/--config` option. Then check the numerical
environment:

```bash
python -c "import jax, numpyro, jaxoplanet; print(jax.devices())"
```

Koala enables JAX 64-bit calculations itself. Set the platform before
Python starts when you need to force one:

```bash
export JAX_PLATFORMS=cpu   # configuration checks on a login node
# export JAX_PLATFORMS=gpu # fits inside a GPU allocation
```

You are ready for the [Quickstart](quickstart.md).

The repository includes a 4.45 MB teaching extraction at
`examples/data/WASP-39_soss_binned8.fits`, so the quickstart needs no separate
light-curve download. The `stellarprior` choice still requires the
ExoTiC-LD grids from step 3.

## Documentation only

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -W -b html docs docs/_build/html
```

The repository's `setup.sh` contains a developer-specific environment path;
it is not an installer and is not needed for the steps above.
