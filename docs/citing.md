# Citing Koala

An archival software citation has not yet been assigned. Until it is, cite the
repository and record the exact commit used. Also cite the methods that were
active in your analysis:

- [jaxoplanet](https://jax.exoplanet.codes/) for differentiable transit and
  orbit calculations;
- [NumPyro](https://num.pyro.ai/) for NUTS and HMC;
- [ExoTiC-LD](https://exotic-ld.readthedocs.io/) when it supplied stellar
  limb-darkening coefficients;
- [Harmonica](https://github.com/DavoGrant/harmonica) when using the
  transmission-string model;
- Sing et al. (2026), [arXiv:2609.00263](https://arxiv.org/abs/2609.00263),
  when using `ld_prior: sing`.

## What to report

A reproducible methods section should state:

- the input extraction, instrument, disperser, detector or SOSS order, and
  wavelength grid;
- the white-light and spectroscopic trend model;
- the limb-darkening law and prior;
- that channel fits condition on the white-light geometry handoff;
- the selected sampler, convergence criteria, seed, and software versions.

Archive the YAML, the spectrum and best-fit CSVs,
`*_whitelight_geometry_handoff.json`, and the chunk diagnostics. For Harmonica,
also archive the joint limb-posterior NPZ.

Adapt this prose to the actual run:

> We fitted the wavelength-summed transit first and conditioned the
> spectroscopic channel fits on its posterior-median geometry. Wavelength
> channels were sampled with independent NumPyro NUTS using local Laplace mass
> matrices and were accepted only after the configured effective-sample-size
> and divergence checks.

If the fitter retained another sampler for some channels, say so. For a
power-2 `stellarprior` fit, add that ExoTiC-LD atmosphere-grid intensities and
the stated stellar-parameter uncertainties defined the wavelength-dependent
priors.

Record key versions with:

```bash
git rev-parse HEAD
python -c "import jax, numpyro, jaxoplanet; print(jax.__version__, numpyro.__version__, jaxoplanet.__version__)"
```
