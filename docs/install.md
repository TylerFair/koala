# Installation

Koala supports editable installs, regular installs, and installation from GitHub. A dedicated environment
keeps JAX and its compiled dependencies separate from other analysis code.

## 1. Download the source and create an environment

Download the source ZIP from GitHub and extract it. In a terminal:

```bash
cd koala-main
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

## 3. Run the bundled example

From the extracted repository root, no configuration edits are needed:

```bash
python fit_jwst.py -c examples/niriss_soss_order1.yaml
```

The real WASP-39 FITS file is included. ExoTiC-LD 3.2 or newer downloads the
required stellar-atmosphere and instrument files automatically into the
example's `exotic_ld_data/` directory. Keep an internet connection during the
first run; subsequent runs reuse that cache. Results go to
`results/WASP-39_SOSS_ORDER1/`.

For an offline cluster, populate the cache on an internet-connected machine
first and copy it over. Set `stellar.ld_data_path` to that copied directory.

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
light-curve download. The `stellarprior` model data is fetched automatically
as described in step 3.

## Documentation dependencies

```bash
python -m pip install -e ".[docs]"
sphinx-build -W -b html docs docs/_build/html
```

The `docs` extra adds Sphinx and its extensions to the runtime install.
Read the Docs uses this same extra. There is no separate docs requirements file.
To install test tools, use `python -m pip install -e ".[test]"`.
