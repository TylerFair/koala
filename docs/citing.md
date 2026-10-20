# Citing koala

If you use koala in a paper, please cite the [repository](https://github.com/TylerFair/koala) and record the commit used for your analysis.
An archival software citation has not yet been assigned.
You can save the commit and installed versions with:

```bash
git rev-parse HEAD
python -m pip freeze > software_versions.txt
```

For the bundled SOSS example, a methods description could read:

> We fitted the white-light transit with koala and held the spectroscopic orbit at the white-light posterior medians.
> Each wavelength channel had its own radius ratio, linear trend, and noise term.
> We used quadratic limb darkening with independent uniform priors on $u_1+u_2$ in $(-1,2)$ and $u_1-u_2$ in $(-2,2)$.
> We fitted the individual integrations using four NUTS chains started from a Laplace approximation.

Adapt this to your actual fit, including joint geometry if you used it.
Report the extraction, instrument and detector or order, wavelength grid, time system, priors, and software versions.
Archive the input YAML, spectrum, white-light results, and posterior products needed to reproduce the analysis.

koala depends on other software and methods that also deserve credit.
Cite the ones used in your fit, together with the observing-program and reduction-pipeline papers for your data.
The bundled WASP-39 extraction has [separate provenance](https://github.com/TylerFair/koala/blob/main/examples/data/README.md).

## Reference

| Software or method | Citation source |
|---|---|
| koala | [TylerFair/koala](https://github.com/TylerFair/koala) and the recorded commit |
| jaxoplanet | [Project documentation](https://jax.exoplanet.codes/) |
| NumPyro | [Project documentation](https://num.pyro.ai/) |
| ExoTiC-LD atmosphere coefficients | [ExoTiC-LD documentation](https://exotic-ld.readthedocs.io/) |
| Harmonica asymmetric boundary | [Harmonica repository](https://github.com/DavoGrant/harmonica) |
| Sing limb-darkening calibration | [Sing et al. (2026)](https://arxiv.org/abs/2609.00263) |
