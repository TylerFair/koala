# Limb darkening

Select the law and prior under `flags`:

```yaml
flags: {ld_profile: power2, ld_prior: informed}
```

`power2` uses coefficients `c1,c2`; `quadratic` uses `u1,u2`. Supported prior modes are:

**Stellar-informed.** This is the ExoTiC-LD Stagger-grid prescription and currently requires power-2 plus stellar uncertainties.

```yaml
stellar: {teff: 5509, logg: 4.22, feh: 0.04, teff_sigma: 28, logg_sigma: 0.07, feh_sigma: 0.02, ld_model: stagger, ld_data_path: ../exotic_ld_data, ld_prior_min_sigma: 1.0e-4}
flags: {ld_profile: power2, ld_prior: informed}
```

`stellarprior` and `stellar` are aliases for `informed`.

**Fixed.** Use calculated coefficients without sampling them. `fix_ld: true` is the legacy spelling.

```yaml
flags: {ld_profile: quadratic, ld_prior: fixed}
```

**Wide Gaussian.** Sample around calculated coefficients with the broad model prior.

```yaml
flags: {ld_profile: power2, ld_prior: widegaussian}
```

**Free uniform.** Sample within the model's native physical bounds.

```yaml
flags: {ld_profile: power2, ld_prior: uniform}
```

**Sing.** For quadratic limb darkening, transform `(u1,u2)` to `(l, delta)`. The wavelength-dependent Stagger-grid prediction supplies the deviation prior, while a shared gray `l` offset is calibrated in a low-resolution stage. `mu_min` excludes the extreme stellar limb during the intensity-profile fit.

```yaml
stellar: {ld_model: stagger, ld_data_path: ../exotic_ld_data, ld_mu_min: 0.2}
flags:
  ld_profile: quadratic
  ld_prior: sing
  ld_sing_offset: fit
  ld_sing_calibration_warmup: 1000
  ld_sing_calibration_samples: 1000
  ld_sing_calibration_min_ess: 100
```

Alternatively supply `ld_sing_offset_path` for an existing gray-offset artifact. See Sing et al. (2026), [arXiv:2609.00263](https://arxiv.org/abs/2609.00263).

