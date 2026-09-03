# jwst-lightcurves

Transit light-curve fitting for JWST time-series spectroscopy. Fits the
white-light curve first, passes the orbital geometry to the wavelength
channels, and writes a transmission spectrum with per-channel posteriors.

Built on JAX and NumPyro. Channels are sampled in parallel on the GPU with a
per-channel Laplace metric, so a full spectrum takes minutes rather than hours.

## Features

- NIRISS/SOSS orders 1 and 2; NIRSpec G395H, G395M, G140H and PRISM, with
  detector selection.
- Limb darkening: stellar-informed (ExoTiC-LD), fixed, wide-Gaussian, free,
  and the Sing et al. (2026) offset-corrected quadratic prior.
- Baselines: polynomial, spot crossings, sigmoid discontinuities (mirror tilt
  events), exponential ramps, and Gaussian processes.
- Exact MCMC throughout. A channel that fails its effective-sample-size or
  divergence check is automatically re-run with a different sampler, and the
  sampler used is recorded per channel.
- Harmonica transmission strings for limb asymmetry.
- Bayesian model averaging over light-curve modelling choices, weighted per
  channel by leave-one-out predictive score.

## Installation

```bash
git clone https://github.com/TylerFair/jwst-lightcurves
cd jwst-lightcurves
pip install -e .
```

Requires Python 3.10+, JAX with GPU support, NumPyro and jaxoplanet.
Stellar-informed limb darkening additionally needs the ExoTiC-LD data files.

## Quickstart

```bash
cp examples/niriss_soss_order1.yaml config.yaml
# point path, input_dir and fits_file at your extracted spectra
python fit_jwst.py -c config.yaml
```

The output directory contains the white-light fit and its diagnostics, the
per-channel posterior checkpoints, and the transmission spectrum as a CSV.
See [`examples/`](examples) to get started and the
[documentation](https://jwst-lightcurves.readthedocs.io) for the full
configuration reference.

## Citation

If this code contributes to a publication, please cite the repository and the
underlying tools: jaxoplanet, NumPyro, ExoTiC-LD, and, where relevant,
Harmonica and Sing et al. (2026).

## License

BSD 3-Clause. See [LICENSE](LICENSE). Vendored components keep their own
licences: `models/harmonica/HARMONICA_LICENSE` and
`models/jaxoplanet/JAXOPLANET_LICENSE`.
