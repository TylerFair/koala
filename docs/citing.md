# Citing

Use this placeholder until the {{ project }} software paper and archival release are available:

```bibtex
@software{jwst_transit_fitter,
  title = {JWST transit light-curve fitting},
  author = {Package contributors},
  year = {2026},
  note = {Citation metadata to be assigned}
}
```

Also cite the methods used in the analysis:

- [jaxoplanet](https://jax.exoplanet.codes/) for differentiable orbital and light-curve calculations.
- [NumPyro](https://num.pyro.ai/) for NUTS and HMC inference.
- [ExoTiC-LD](https://exotic-ld.readthedocs.io/) for stellar limb-darkening coefficients.
- [Harmonica](https://github.com/DavoGrant/harmonica) for transmission strings and asymmetric transit light curves.
- Sing et al. (2026), [arXiv:2609.00263](https://arxiv.org/abs/2609.00263), when using the Sing limb-darkening prior.

Record the versions, configuration file, random seed, and detector/order with every result.

## Methods checklist

A reproducible methods section should identify the extraction used as input. State the JWST instrument, disperser, detector, and SOSS order where applicable. State the wavelength grid: native, reference, pixel bins, or constant resolving power.

State whether a low-resolution bridge was used. State the white-light trend and the spectroscopic trend. For explinear fits, state that the channel timescale was fixed to the white-light posterior median.

State the limb-darkening law. State whether coefficients were fixed, wide-Gaussian, uniform, stellar-informed, or Sing-informed. For stellar-informed LD, report the atmosphere grid, stellar values and uncertainties, and minimum prior width.

For Sing LD, report `mu_min` and the shared-offset calibration method. State that the spectroscopic geometry was conditioned on the white-light posterior median. State the primary sampler and Laplace metric.

State the alternate exact sampler policy and quality gate. State warmup, retained draws, target acceptance, tree depth or HMC steps, and random seed. State the GPU model and software versions.

## Suggested prose

The following text can be adapted to the actual configuration:

> We fitted the wavelength-summed transit first and conditioned the spectroscopic channel fits on its posterior-median geometry. Each channel was sampled with independent NumPyro NUTS using a local Laplace inverse-Hessian metric. Channels were required to have zero divergences and a bulk effective sample size of at least 400 in transit depth; failed chunks were repeated with eight-step Laplace-metric HMC.

Change the statement when the configuration selected another sampler, gate, trend, or geometry estimator. For a stellar-informed fit, add:

> Power-2 limb-darkening priors were calculated from ExoTiC-LD Stagger-grid intensities. Uncertainties in effective temperature, surface gravity, and metallicity were propagated into wavelength-dependent Gaussian coefficient priors.

For an explinear fit, add:

> The white-light baseline included an exponential ramp and linear term. Spectroscopic channels used the white-light posterior-median ramp timescale and fitted independent ramp amplitudes.

For Harmonica, add:

> We modeled the planet boundary as an odd-cosine transmission string and report full-circle-equivalent depths for two indexed angular half-areas. We supplied the physical hemisphere mapping separately from the fit.

## Software roles

jaxoplanet supplies differentiable transit and orbit calculations. NumPyro supplies Hamiltonian Monte Carlo inference. ExoTiC-LD supplies model-atmosphere limb intensities and fitted coefficients.

Harmonica supplies the asymmetric transmission-string forward model. {{ project }} supplies the JWST data staging, priors, systematics models, sampler routing, diagnostics, and outputs described here. Sing et al. supplies the $(l,\delta)$ limb-darkening prescription and shared-offset framework.

## Archive with a result

Archive the YAML configuration. Archive the input-extraction identifier or checksum. Archive `*_whitelight_geometry_handoff.json`.

Archive spectrum and detailed parameter CSVs. Archive chunk diagnostics or their combined summary. Archive the random seed and environment variables.

Archive dependency versions. For Harmonica, archive the joint limb posterior NPZ, not only marginal CSV errors.

## Citation boundaries

Do not cite jaxoplanet as the sampler; NumPyro provides NUTS/HMC. Do not cite ExoTiC-LD when coefficients were supplied by an unrelated calculation. Cite Harmonica whenever the transmission-string engine is used.

Cite Sing et al. whenever `ld_prior: sing` is used. Describe any custom extraction separately from the light-curve fitter.

Replace the placeholder package entry when a versioned archival citation is assigned.

## Machine-readable version record

```bash
python -c "import jax, numpyro, jaxoplanet; print(jax.__version__, numpyro.__version__, jaxoplanet.__version__)"
```

Store this output with the configuration and final tables.
