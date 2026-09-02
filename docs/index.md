# {{ project }}

{{ project }} fits JWST transit light curves from extracted box-spectrum FITS files. It fits a white-light curve first, passes the orbital geometry to wavelength channels, and writes light-curve diagnostics and transmission spectra.

```bash
conda activate jaxoplanet
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=gpu
cp configs_fiducial_stellarinformed/WASP-39_soss_order1_config.yaml config.yaml
# Edit path, input_dir, output_dir, and fits_file in config.yaml.
python fit_jwst.py -c config.yaml
ls /path/from/config/output_dir
python -c "import pandas as pd; print(pd.read_csv('/path/to/spectrum.csv').head())"
```

Features include:

- JWST NIRISS/SOSS orders 1 and 2.
- NIRSpec G395H, G395M, G140H, G235H, and PRISM, including NRS1/NRS2 selection.
- White-light, low-resolution, and high-resolution spectroscopic stages.
- Stellar-informed, Sing, fixed, wide-Gaussian, and free limb darkening.
- Polynomial, spot, discontinuity, exponential-ramp, and GP systematics.
- Laplace-metric HMC samplers implemented in JAX for GPUs.
- Harmonica transmission-string fits for limb asymmetry.

```{toctree}
:maxdepth: 2
:caption: Getting started

install
quickstart
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/soss_order1
tutorials/nirspec_g395h
tutorials/prism
tutorials/harmonica
```

```{toctree}
:maxdepth: 2
:caption: Guides

guides/limb_darkening
guides/trends
guides/samplers
guides/configuration
guides/outputs
guides/gpu_and_clusters
api
citing
```

