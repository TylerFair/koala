# Choose limb darkening

Limb darkening controls the transit shape, especially ingress and egress, and is correlated with radius ratio and impact parameter. Koala separates two choices: the intensity law (`ld_profile`) and what is assumed about its coefficients (`ld_prior`).

- `ld_profile`: `quadratic` or `power2`
- `ld_prior`: `uniform`, `gaussian`, `sing`, `stellarprior`, or `fixed`

The `gaussian` prior has a fixed coefficient width of 0.2. `stellarprior`
requires `power2`, and `sing` requires `quadratic`.

For a first fit, use `power2` with `stellarprior`:

```yaml
stellar:
  teff: 5509
  teff_sigma: 28
  logg: 4.22
  logg_sigma: 0.07
  feh: 0.04
  feh_sigma: 0.02
  ld_model: stagger
  ld_data_path: ../exotic_ld_data

flags:
  ld_profile: power2
  ld_prior: stellarprior
```

This propagates the configured stellar-parameter uncertainties through ExoTiC-LD Stagger intensities into a wavelength-dependent coefficient prior.

The prior is built over a grid of 125 stellar-parameter combinations. By
default Koala integrates each unique ExoTiC-LD grid node once per bin, blends
the node integrals with ExoTiC-LD's own trilinear weights, and fits every
combination and bin in one batched solve; this reproduces the per-call
ExoTiC-LD coefficients to about 1e-8 and takes under a minute even at native
resolution. Finished priors are cached under
`stellar.ld_prior_cache_dir`, keyed by the wavelength grid and stellar inputs.

```{image} ../_static/tutorial_limb_darkening.svg
:alt: Schematic power-2 and quadratic stellar intensity profiles
:width: 820px
:align: center
```

The curves illustrate the role of the law and coefficients; they are not fitted observations.

## Compare two priors on your transit

Start with two copies of the complete first-transit YAML and run only white light. The snippets below replace the named fields in those full configurations; keep the data, stellar, path, and mask fields identical:

```yaml
# wasp39_stellarprior.yaml
output_dir: results/WASP-39_LD_STELLARPRIOR
flags:
  analysis_stage: whitelight
  detrending_type: linear
  ld_profile: power2
  ld_prior: stellarprior
```

```yaml
# wasp39_wide.yaml
output_dir: results/WASP-39_LD_WIDE
flags:
  analysis_stage: whitelight
  detrending_type: linear
  ld_profile: power2
  ld_prior: gaussian
```

```bash
python fit_jwst.py -c wasp39_stellarprior.yaml
python fit_jwst.py -c wasp39_wide.yaml
```

Compare ingress and egress residuals, radius ratio, impact parameter, and the fitted LD coefficients. If both fits pass diagnostics but move the geometry, run both through the wavelength channels with `analysis_stage: all` and treat LD choice as part of the result rather than choosing by eye.

## Which treatment answers my question?

| Goal | Profile and prior | What it assumes |
|---|---|---|
| Standard transmission spectrum | `power2` + `stellarprior` | The atmosphere grid describes the star, with uncertainty |
| Hold calculated LD fixed | either profile + `fixed` | Calculated coefficients are exact |
| Weakly constrain coefficients around a model | either profile + `gaussian` | Model values are useful centers, with width 0.2 |
| Test broad quadratic LD | `quadratic` + `uniform` | Wide flat priors on $u_+$ and $u_-$ |
| Apply the Sing gray-offset calibration | `quadratic` + `sing` | The modeled chromatic shape is useful after a shared correction |

There is no universally best choice. If two defensible treatments move the spectrum, that movement belongs in the uncertainty analysis; the [stacking tutorial](model_stacking.md) shows one way to carry it forward.

## The two laws

The power-2 law is

$$
I(\mu)/I(1)=1-c_1(1-\mu^{c_2}),
$$

and the quadratic law is

$$
I(\mu)/I(1)=1-u_1(1-\mu)-u_2(1-\mu)^2,
$$

where $\mu=1$ is disk center and $\mu=0$ is the stellar limb. The `stellarprior` choice uses power-2, while `sing` requires quadratic limb darkening.

## Useful alternatives

Fix calculated coefficients for a controlled comparison:

```yaml
flags:
  ld_profile: power2
  ld_prior: fixed
```

Use a broad Gaussian around calculated coefficients:

```yaml
flags:
  ld_profile: power2
  ld_prior: gaussian
```

Use broad quadratic coordinates:

```yaml
flags:
  ld_profile: quadratic
  ld_prior: uniform
```

In quadratic uniform mode the defaults are $u_+=u_1+u_2\in[-1,2]$ and $u_-=u_1-u_2\in[-2,2]$. These are deliberately broad priors. They are not equivalent to independent uniform priors on $u_1$ and $u_2$.

For the Sing calibration:

```yaml
stellar:
  ld_model: stagger
  ld_data_path: ../exotic_ld_data
  ld_mu_min: 0.2

flags:
  ld_profile: quadratic
  ld_prior: sing
```

The pipeline first fits broad quadratic LD at low resolution, estimates a shared gray correction to the Stagger prediction, and carries the corrected chromatic prior to the final spectrum. `ld_mu_min` excludes the most extreme stellar-limb intensities when fitting the atmosphere profile; changing it changes the prior and should be reported.

## How to validate the choice

Inspect ingress and egress residuals in the white-light and channel summaries. Then plot the fitted `c1,c2` or `u1,u2` columns from `*_bestfit_params.csv` against wavelength.

Watch for three signals:

- posteriors pressed against prior bounds;
- strong wavelength-to-wavelength coefficient jumps unsupported by the stellar model;
- transit depths or geometry that move substantially between defensible LD choices.

The last case is not resolved by selecting the fit with the smallest residual scatter. Compare predictive performance and propagate the model choice. For published work, report the intensity law, prior mode, stellar grid, stellar parameters and uncertainties, and any `ld_mu_min` choice.
