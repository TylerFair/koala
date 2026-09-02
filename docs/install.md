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
