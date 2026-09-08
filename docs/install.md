# Installation

Koala supports editable installs, regular installs, and installation from GitHub. A dedicated environment
keeps JAX and its compiled dependencies separate from other analysis code.

## 1. Clone and create an environment

```bash
git clone https://github.com/TylerFair/koala.git
cd koala
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Python 3.11 is the recommended version.

## 2. Install the scientific dependencies

The package metadata in `pyproject.toml` reads runtime dependencies from the
root `requirements.txt`, including JAX, JAXlib, JAXoplanet, NumPyro, and the
scientific and plotting libraries. Harmonica is built into the same distribution.

For a CPU environment:

```bash
python -m pip install -e .
```

Harmonica is bundled in `harmonica_modified/` and compiled into `koala-jwst`.
There is no separate Harmonica installation step. It provides `harmonica.jax.harmonica_transit_power2_ld` and
`harmonica_transit_quad_ld`. Its source build requires a C++ compiler
(Xcode Command Line Tools on macOS, or GCC/Clang on Linux). Pip installs the
Python build dependencies automatically; the required Eigen headers are
included in the repository, with no submodule checkout needed.

Koala reads ExoTEDRF box-spectrum FITS files directly with Astropy and includes
its own NumPy binning utilities; installing ExoTEDRF is not required.

CPU is useful for checking a configuration, running the bundled example
(about two hours on eight cores), and building the documentation. A full
spectroscopic fit at native or high resolution is designed for an NVIDIA GPU. Install the CUDA-enabled
JAX wheel using the command for your driver and CUDA installation in the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html),
then run `python -m pip install -e .`. The requirements do not
force a CPU-only JAX extra. Harmonica GPU builds additionally require the CUDA
toolkit and `HARMONICA_ENABLE_CUDA=1` when installing Koala.

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

Run from any directory after installation:

```bash
koala --help
```

From the repository root, `python fit_jwst.py --help` also remains supported.
Use `python -m pip install .` for a regular installation, or
`python -m pip install "git+https://github.com/TylerFair/koala.git"` to install
directly from GitHub. The distribution name is `koala-jwst`; these commands
install from source without requiring a PyPI release.

You should see the required `-c/--config` option. Then check the numerical
environment:

```bash
python -c "import jax, numpyro, jaxoplanet; from harmonica.jax import harmonica_transit_power2_ld, harmonica_transit_quad_ld; print(jax.devices())"
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

## Documentation dependencies

```bash
python -m pip install -e ".[docs]"
sphinx-build -W -b html docs docs/_build/html
```

The `docs` extra adds Sphinx and its extensions to the runtime install.
Read the Docs uses this same extra. There is no separate docs requirements file.
To install test tools, use `python -m pip install -e ".[test]"`.
