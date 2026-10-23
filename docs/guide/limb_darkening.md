# Limb darkening

A star is fainter near its edge than at its centre, so the planet blocks less light near the start and end of the transit than in the middle.
That curvature is tied up with the transit depth, which makes the treatment of limb darkening one of the choices that matters most for a spectrum.
koala fits the limb darkening separately in every wavelength channel.

## The choices

There are two settings: the law for the star's brightness profile (`ld_profile`), and the prior on its coefficients (`ld_prior`).

| `ld_profile` | Brightness at $\mu = \cos\theta$ |
|---|---|
| `quadratic` (default) | $1 - u_1(1-\mu) - u_2(1-\mu)^2$ |
| `power2` | $1 - c(1-\mu^\alpha)$ |
| `kipping` | the quadratic law, sampled in the $q_1, q_2$ of Kipping (2013) |

| `ld_prior` | Coefficients |
|---|---|
| `uniform` (default) | free, with wide uniform priors and no input from stellar models |
| `stellarprior` | Gaussian priors from the stellar models, with widths set by the uncertainties on Teff, log g and [Fe/H] |
| `gaussian` | Gaussian priors centred on the stellar-model values, with a fixed width (`ld_gaussian_sigma`, 0.2 by default) |
| `sing` | stellar-model quadratic coefficients shifted by an offset measured from the data, following [Sing et al. (2026)](https://arxiv.org/abs/2609.00263) |
| `fixed` | held at the stellar-model values, or at `stellar.ld_coefficients` if you give them |
| `fixed_u1` | quadratic $u_1$ held at the stellar-model value, $u_2$ free |

Not every pair works: `kipping` can't be used with `stellarprior`, `sing` or `fixed_u1`.

For the default quadratic law, the uniform priors are on $u_1 + u_2$ in $(-1, 2)$ and $u_1 - u_2$ in $(-2, 2)$.
That's a lot wider than the physically allowed region, and deliberately so, because forcing the coefficients to be physical can bias the depths ([Coulombe, Roy & Benneke 2024](https://arxiv.org/abs/2409.03812)).
For power-2, the uniform prior is flat in the $h_1, h_2$ coordinates of Maxted (2018).
If you pick `power2` without saying which prior, koala uses `stellarprior` when you've given all three stellar uncertainties, and `gaussian` otherwise.

Even when the prior is uniform, the stellar models give the starting point for each channel, which is why the `stellar` block needs `teff`, `logg` and `feh`.

## Which one should I use?

We'd suggest starting with the default.
In noise-free injection tests on a NIRSpec/PRISM light curve with 17,850 integrations, it recovered every depth within 0.1σ of the truth, for every trend we tried.
In the same tests, quadratic coefficients with stellar-model priors were within 0.04σ, power-2 reached 0.11σ with Gaussian priors and 0.18σ with uniform ones, and the `kipping` sampling reached 0.29σ.
Priors from stellar models are only as good as the models, though.

Fixing the coefficients deserves the most care.
In similar tests, fixing $u_1$ at the right value made the errors about 10% smaller with no real bias (0.05σ).
Fixing it at the wrong value gave a fit that converged just as well, with errors just as small, but depths off by up to 1.3σ.
A good-looking fit doesn't tell you the stellar model was right.

If you're not sure, fit more than one and compare them, or combine them as in the [model stacking case study](../case_studies/model_stacking.md).
Changing the limb darkening can change the orbit measured from the white-light curve too, so always rerun the whole fit rather than just the channels.

## Examples

Power-2 limb darkening with priors from the stellar models:

```yaml
stellar:
  teff: 5509
  teff_sigma: 28
  logg: 4.22
  logg_sigma: 0.07
  feh: 0.04
  feh_sigma: 0.02
flags:
  ld_profile: power2
  ld_prior: stellarprior
```

koala works out the coefficients over a grid of stellar parameters spanning ±5σ around the values you give, and uses their spread to set the priors.
That takes a while the first time, and is cached for later fits.
All three uncertainties matter, since for typical values [Fe/H] moves the coefficients about as much as Teff does.

Your own fixed coefficients (for the quadratic law, $u_1$ and $u_2$):

```yaml
stellar:
  ld_coefficients: [0.12, 0.25]
flags:
  ld_prior: fixed
```

For a secondary eclipse, the star isn't being occulted, so its limb darkening doesn't matter and can be fixed at `[0.0, 0.0]`.
