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

Koala reads exoTEDRF, SPARTA, and Eureka! products directly and includes its
own binning utilities; none of those pipelines needs to be installed.

Publication-style figures can optionally use SciencePlots and the cmcrameri
colour maps:

```bash
python -m pip install -e ".[plots]"
```

Without them Koala uses a plain Matplotlib serif style. The SciencePlots
style renders text with LaTeX when a TeX install (`latex` and `dvipng`) is
on the PATH and silently uses Matplotlib's own text rendering otherwise.

tinygp is installed from the Koala fork at a validated commit rather than
from PyPI, because no PyPI release yet contains the parallel associative-scan
quasiseparable solver that Koala uses for Gaussian-process trends on GPUs.
The requirement line is:

```
tinygp @ git+https://github.com/TylerFair/tinygp.git@96d110c8fb4350b0daf50f7f1f86a9196b82fcd4
```

This needs `git` on the machine and network access to GitHub. The fork's
`main` tracks upstream `dfm/tinygp` with one packaging change (installation
on Python 3.10); the pinned commit is also the fork's `koala-stable` branch,
which is advanced only after the Koala GP test suites pass against the new
revision. An existing environment that still has tinygp 0.3.x from PyPI keeps
working with the serial solver; requesting the parallel solver there raises
an error with the install line above. See
[Gaussian processes](guides/gaussian_processes.md) for solver selection.


CPU is useful for checking a configuration, running the bundled example,
and building the documentation. A full spectroscopic fit at native or high
resolution is designed for an NVIDIA GPU. Install the CUDA-enabled
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
there means the JAX build, driver, or allocation needs attention. Koala
cannot use a GPU that JAX does not list: with `host_device: gpu` it falls
back to CPU and prints a warning. Common causes on a desktop PC:

- **Windows.** There are no CUDA JAX wheels for native Windows, so
  `pip install "jax[cuda12]"` silently installs the CPU build even though
  `nvidia-smi` works. Run Koala under WSL2 (Ubuntu) and install JAX there.
- **Driver too old for the wheel.** The `cuda12` wheels need a recent NVIDIA
  driver, and Blackwell cards such as the RTX 50 series need driver 570 or
  newer. `import jax` then logs `cuInit` or "Unable to initialize backend
  'cuda'" and continues on CPU. Update the driver.
- **Mismatched JAX packages.** `jax`, `jaxlib`, `jax-cuda12-plugin`, and
  `jax-cuda12-pjrt` must all be the same version. If a later
  `pip install` upgraded `jax` alone, JAX logs a plugin version error and
  falls back to CPU. Fix it with
  `python -m pip install -U "jax[cuda12]"`, then re-run the check above.

## 3. Run the bundled example

From the extracted repository root, no configuration edits are needed:

```bash
python fit_jwst.py -c examples/niriss_soss_order1.yaml
```

The real WASP-39 FITS file is included. ExoTiC-LD (installed from the pinned
upstream commit in `requirements.txt`, because the PyPI 3.2.0 release lacks
the power-2 law) downloads the
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
