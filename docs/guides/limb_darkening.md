# Limb darkening

Two flags control limb darkening: the intensity law (`flags.ld_profile`) and
what is assumed about its coefficients (`flags.ld_prior`).

```yaml
stellar:
  teff: 5509
  teff_sigma: 28
  logg: 4.22
  logg_sigma: 0.07
  feh: 0.04
  feh_sigma: 0.02
  ld_model: stagger
  ld_data_path: exotic_ld_data

flags:
  ld_profile: power2
  ld_prior: stellarprior
```

## `ld_profile`

- `power2`: the two-coefficient power-2 law. This is the default choice and
  the one used by `stellarprior`.
- `quadratic`: the classical two-coefficient quadratic law, required by
  `sing`.

## `ld_prior`

- `stellarprior`: ExoTiC-LD coefficients computed from the `stellar` block,
  with the stated `teff_sigma`, `logg_sigma`, and `feh_sigma` propagated into
  a wavelength-dependent prior. Requires `power2`.
- `fixed`: the ExoTiC-LD coefficients are held at their computed values.
- `gaussian`: a Gaussian prior of width 0.2 centred on the computed
  coefficients.
- `uniform`: broad flat priors on the coefficients, with no stellar-model
  information.
- `sing`: quadratic coefficients calibrated with a shared gray offset to the
  stellar model, following Sing et al. (2026). Requires `quadratic`; set
  `stellar.ld_mu_min` to exclude the extreme limb from the calibration.
