# Installation

Create an environment and install the documentation dependencies:

```bash
conda create -n jwst-fit python=3.11
conda activate jwst-fit
pip install -r docs/requirements.txt
```

Install the scientific dependencies used by `fit_jwst.py`, including JAX, NumPyro, jaxoplanet, Astropy, pandas, matplotlib, jaxopt, numpyro-ext, tinygp, ArviZ, PyYAML, and ExoTiC-LD. Follow the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for the CUDA wheel matching the cluster driver. The stellar-informed prescriptions also require a local `exotic_ld_data` tree; set `stellar.ld_data_path` to it.

{{ project }} requires 64-bit JAX:

```bash
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=gpu       # use cpu on a login node
export JAX_COMPILATION_CACHE_DIR=/scratch/$USER/jax_cache
python fit_jwst.py -c config.yaml
```

`FIT_JWST_SEED` overrides `flags.random_seed`. `JWSTJAXFIT_ANALYSIS_STAGE` and `JWSTJAXFIT_CHUNK_MODE` override the corresponding stage controls.

## Verify the environment

Run these checks before submitting a long fit:

```bash
python -c "import jax; print(jax.__version__, jax.config.x64_enabled, jax.devices())"
python -c "import numpyro, jaxoplanet; print(numpyro.__version__, jaxoplanet.__version__)"
python -c "from exotic_ld import StellarLimbDarkening; print('ExoTiC-LD import OK')"
```

The first command must report 64-bit mode as true after `JAX_ENABLE_X64=1` is exported. On a compute node, the device list must include the requested GPU. On a login node, explicitly use `JAX_PLATFORMS=cpu`.

## Scientific dependencies

JAX provides compiled array operations and automatic differentiation. NumPyro provides NUTS and HMC. jaxoplanet evaluates the transit geometry and limb-darkened light curve.

ExoTiC-LD calculates wavelength-dependent stellar intensity coefficients. Astropy reads the FITS extraction. pandas writes result tables.

matplotlib writes the numbered diagnostic figures. jaxopt and numpyro-ext support numerical optimization. tinygp is required by GP trend models.

ArviZ provides posterior summaries and effective-sample-size diagnostics. PyYAML reads the configuration.

## CPU smoke test

A CPU can check imports and configuration parsing, but a full spectroscopic fit is intended for a GPU.

```bash
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=8
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
python fit_jwst.py --help
```

The help output shows the required `-c/--config` argument. Do not set `JAX_PLATFORMS` after importing JAX.

## CUDA installation

Install the JAX CUDA wheel appropriate for the system driver by following the upstream JAX instructions. Do not mix a CPU-only `jaxlib` with a GPU submission and assume it will discover CUDA dynamically. Check `jax.devices()` inside the allocation.

The cluster CUDA module and the JAX wheel must be compatible.

## ExoTiC-LD data

The model-data directory is not embedded in the YAML.

```yaml
stellar:
  ld_model: stagger
  ld_data_path: /shared/reference/exotic_ld_data
```

Use an absolute path on a cluster when the submission working directory may vary. Confirm the compute node can read it. The stellar-informed mode also needs `teff_sigma`, `logg_sigma`, and `feh_sigma`.

## Persistent compilation cache

The environment variable configures JAX generally. The fitter can configure its cache explicitly:

```yaml
flags:
  compile_box: true
  jax_compilation_cache_dir: /scratch/account/user/jax_cache
```

Use node-visible fast storage. Cache entries depend on JAX/XLA versions and static shapes. A cache does not eliminate the first compilation for a new model shape.

## Documentation build

```bash
pip install -r docs/requirements.txt
sphinx-build -W -b html docs docs/_build/html
```

`-W` converts warnings into failures. The Furo theme is configured in `docs/conf.py`.

## Common installation problems

**No GPU appears:** check the allocation, CUDA module, and installed `jaxlib` wheel. **64-bit warning:** export `JAX_ENABLE_X64=1` before Python starts. **ExoTiC-LD file error:** correct `stellar.ld_data_path` and verify node access.

**Import error for tinygp:** install it even when the current configuration does not use a GP, because the main CLI imports the module. **Read the Docs failure:** install only `docs/requirements.txt`; autodoc mocks heavy scientific imports. **Different local and cluster behavior:** print package versions and `jax.devices()` in both environments.

## Reproducible environment record

Save the Python version. Save JAX, jaxlib, NumPyro, and jaxoplanet versions. Record the GPU model and CUDA driver.

Record the ExoTiC-LD model-data revision. Archive the exact YAML and environment variables with the output.
