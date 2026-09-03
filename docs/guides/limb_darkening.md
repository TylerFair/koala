# Limb darkening

Select the law and prior under `flags`:

```yaml
flags: {ld_profile: power2, ld_prior: informed}
```

`power2` uses coefficients `c1,c2`; `quadratic` uses `u1,u2`. Supported prior modes are: **Stellar-informed.** This is the ExoTiC-LD Stagger-grid prescription and currently requires power-2 plus stellar uncertainties.

```yaml
stellar: {teff: 5509, logg: 4.22, feh: 0.04, teff_sigma: 28, logg_sigma: 0.07, feh_sigma: 0.02, ld_model: stagger, ld_data_path: ../exotic_ld_data, ld_prior_min_sigma: 1.0e-4}
flags: {ld_profile: power2, ld_prior: informed}
```

`stellarprior` and `stellar` are aliases for `informed`. **Fixed.** Use calculated coefficients without sampling them. `fix_ld: true` is the legacy spelling.

```yaml
flags: {ld_profile: quadratic, ld_prior: fixed}
```

**Wide Gaussian.** Sample around calculated coefficients with the broad model prior.

```yaml
flags: {ld_profile: power2, ld_prior: widegaussian}
```

**Free uniform.** For quadratic LD, `uniform` means wide flat priors on
$u_+=u_1+u_2$ and $u_-=u_1-u_2$ following Sing et al. (2026); `sing` means the
offset-corrected informed prior. The wide box is $u_+\in[-1,2]$ and
$u_-\in[-2,2]$; set `ld_uniform_basis: coefficients` for the historical
$u_1,u_2\in[0,1]$ prior.

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

## The two intensity laws

With $\mu=\cos\gamma$, the quadratic law is

$$I(\mu)/I(1)=1-u_1(1-\mu)-u_2(1-\mu)^2.$$

The power-2 law is

$$I(\mu)/I(1)=1-c_1(1-\mu^{c_2}).$$

The disk center is $\mu=1$ and the geometric limb is $\mu=0$. Quadratic LD is required by the Sing prescription. Power-2 is the current stellar-informed path.

## Comparison

| Mode | Assumption | Construction | Use | Depth consequence |
|---|---|---|---|---|
| `fixed` | Atmosphere coefficients are exact | Deterministic coefficients | Controlled comparisons | Does not propagate LD uncertainty into depth |
| `widegaussian` | Atmosphere values are useful centers | Truncated Normal, width 0.2 | Weakly informed fits | Permits broader LD-depth covariance |
| `informed` | Stagger intensities and stellar errors describe the star | Propagated truncated Gaussian | Standard power-2 analysis | Constrains chromatic LD while retaining stellar uncertainty |
| `uniform` | Wide uninformative quadratic LD | Flat $u_+\in[-1,2]$, $u_-\in[-2,2]$ | Sensitivity tests | Can enlarge depth uncertainty |
| `sing` | Stagger shape needs a shared gray correction | Truncated Gaussian in $(l,\delta)$ | Sing et al. quadratic analysis | Separates common and chromatic profile differences |

## Informed prior construction

The fitter evaluates ExoTiC-LD Stagger intensities over the configured stellar-parameter grid. It fits power-2 coefficients in every wavelength bin and stellar grid point. The weighted means become `c1_mean` and `c2_mean`.

The coefficient scatter propagated from `teff_sigma`, `logg_sigma`, and `feh_sigma` becomes the prior width. `ld_prior_min_sigma` floors that width. The cache records total, fit, and stellar-scatter columns for both coefficients.

```yaml
stellar:
  ld_prior_model: stagger
  ld_prior_n_grid: 5
  ld_prior_nsigma: 3
  ld_prior_min_sigma: 1.0e-4
flags: {ld_profile: power2, ld_prior: informed}
```

The power-2 Gaussian support is `c1` from 0 to 1 and `c2` from 0.001 to 1.

## Wide Gaussian and uniform priors

Without propagated widths, `widegaussian` uses a coefficient standard deviation of 0.2. Quadratic coefficients are truncated to $[0,1]$. Power-2 uses the bounds above.

The `free` alias resolves to `widegaussian`. Use `uniform` for the explicit flat coefficient prior:

```yaml
flags: {ld_profile: power2, ld_prior: uniform}
```

For quadratic LD, uniform mode uses independent flat priors on
$u_+=u_1+u_2\in[-1,2]$ and $u_-=u_1-u_2\in[-2,2]$. This deliberately enlarges
the prior relative to the legacy coefficient box; it is not a reparameterized
version of the same prior. Set `ld_uniform_basis: coefficients` to recover flat
$u_1,u_2\in[0,1]$. Wide-Gaussian and uniform/free power-2 priors retain their
existing behavior. Builders save `u1`, `u2`, `l`, `delta`, and `u` for the new
quadratic path so downstream output remains in familiar quantities.

## Sing coordinates and calibration

Define $u_+=u_1+u_2$ and $u_-=u_1-u_2$. The implemented coordinates are

$$l=1-u_+,\qquad \delta=(u_+-u_-)/8=u_2/4.$$

The inverse is

$$u_1=(1-l)-4\delta,\qquad u_2=4\delta.$$

The sampler bounds $l$ to $[0,1]$ and $\delta$ to $[-(1-l)/4,(1-l)/4]$. The low-resolution calibration fits quadratic LD freely. It transforms fitted and Stagger coefficients to $(l,\delta)$.

It computes an inverse-variance-weighted fitted-minus-model gray offset across wavelength. The JSON artifact contains both offsets, uncertainties, channel counts, and a fingerprint. `ld_sing_offset: fit` requests this calibration.

`ld_sing_offset_path` supplies an existing compatible artifact. The code's tabulated fallback offsets are $\Delta l=0.020$ and $\Delta\delta=-0.003$, with scatters 0.031 and 0.016. The log states when it uses that fallback.

## The `mu_min` choice

`stellar.ld_mu_min` excludes Stagger intensity points nearer the extreme limb than the selected $\mu$. For `ld_mu_min: 0.2`, points with $\mu<0.2$ do not enter the coefficient fit. This reduces leverage from the atmosphere grid's most extreme limb.

Changing it changes the calculated prior and must be recorded.

## Checks after fitting

Read `c1,c2` or `u1,u2` from `*_bestfit_params.csv`. Plot them against wavelength with their asymmetric errors. Look for posteriors pressed against truncation bounds.

Inspect ingress and egress residuals before accepting a depth shift as atmospheric. If many informed widths equal the floor, revisit the stellar uncertainties and grid settings. If a uniform fit is unstable, compare it with informed or wide-Gaussian LD on the same geometry.
