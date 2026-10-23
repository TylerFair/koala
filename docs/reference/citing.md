# Citing koala

If koala helps your research, please cite the [repository](https://github.com/TylerFair/koala) and give the version or commit you used (there isn't a paper to cite yet).
You can record both with:

```bash
git rev-parse HEAD
python -m pip freeze > software_versions.txt
```

koala is built on other people's work, and please cite the parts your fit used as well:

| | |
|---|---|
| [jaxoplanet](https://jax.exoplanet.codes) | the transit, eclipse and spotted-star models |
| [NumPyro](https://num.pyro.ai) and [JAX](https://docs.jax.dev) | the sampler and everything underneath it |
| [ExoTiC-LD](https://exotic-ld.readthedocs.io) | limb darkening from stellar models |
| [Harmonica](https://github.com/DavoGrant/harmonica) | the morning and evening limbs |
| [tinygp](https://tinygp.readthedocs.io) | Gaussian-process trends |
| [Sing et al. (2026)](https://arxiv.org/abs/2609.00263) | `ld_prior: sing` |
| [Yao et al. (2018)](https://arxiv.org/abs/1704.02030) | model stacking |

along with the papers for your observing program and your data reduction.
The example WASP-39 b data come from JWST program 1366, and [their README](https://github.com/TylerFair/koala/blob/main/examples/data/README.md) has the details.

In a paper, the methods section for something like the [first fit](../quickstart.md) might read:

> We fitted the light curves with koala, first fitting the white-light curve for the orbit and then each wavelength channel with the orbit fixed at the white-light posterior median.
> Each channel had its own radius ratio, quadratic limb-darkening coefficients, linear trend in time and white-noise jitter.
> The limb-darkening coefficients had uniform priors on $u_1 + u_2$ in $(-1, 2)$ and $u_1 - u_2$ in $(-2, 2)$.
> We sampled the posterior of every fit with four chains of the No-U-Turn Sampler, started from a Laplace approximation, keeping 1000 draws per chain and fitting every integration.

Change it to match what you actually did: the extraction, instrument and wavelength grid, the time system, any priors you changed, and the versions you used.
