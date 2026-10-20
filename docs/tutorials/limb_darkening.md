# Choosing limb darkening

The star is usually dimmer near its limb than at its center.
That brightness profile affects the shape and depth of the transit, so our choice of limb darkening can change the inferred planet radius.
Let's start with koala's default and then try a stellar-model prior.
You can use the bundled SOSS configuration for both fits.

## Start with the default

The default is a quadratic law with independent uniform priors on $u_1+u_2$ in $(-1,2)$ and $u_1-u_2$ in $(-2,2)$.
The bundled SOSS example already uses these settings.
To write them explicitly, use:

```yaml
flags:
  ld_profile: quadratic
  ld_prior: uniform
  ld_uniform_basis: uplus_uminus
```

These broad priors let the light curve constrain the coefficients without restricting them to the physical quadratic triangle.
The stellar models still provide starting values.
They do not set the coefficient priors for this fit.
The [validation results below](#default-selection-and-bias-evidence) explain why we chose this default.

```{figure} ../_static/limb_darkening.png
:alt: Quadratic and power-2 specific intensity profiles from stellar limb to center

Two illustrative brightness profiles, normalized at the stellar center.
The quadratic coefficients are (0.34, 0.20) and the power-2 coefficients are (0.62, 0.72).
These are not fitted SOSS coefficients.
```

## Try a stellar-model prior

Now suppose we want to use the uncertainty in the star's temperature, gravity, and metallicity to inform the limb darkening.
Copy the SOSS configuration, choose a new output directory, and merge in these settings:

```yaml
stellar:
  teff: 5509
  logg: 4.22
  feh: 0.04
  teff_sigma: 28
  logg_sigma: 0.07
  feh_sigma: 0.02
  ld_model: stagger
  ld_data_path: exotic_ld_data
flags:
  ld_profile: power2
  ld_prior: stellarprior
```

This selects a power-2 law and propagates the stellar-parameter uncertainties through the atmosphere grid.
You can also use `stellarprior` with the quadratic law.
All three stellar parameters matter.
For uncertainties of 100 K, 0.1 dex and 0.1 dex, [Fe/H] moves the coefficients about as much as Teff does, and log g about a third as much.
For quadratic limb darkening, koala fits quadratic coefficients at each stellar-grid point.
It then uses their weighted scatter to set independent coefficient-prior widths.

Run the new configuration and compare the spectrum, white-light geometry, and residuals with the default fit.
Changing limb darkening can change the orbit inferred from white light, so rerun the complete fit.
The [combining models tutorial](combining_models.md) shows how to compare several such choices.

## Fixing a coefficient

If you trust the stellar model, `ld_prior: fixed` holds both coefficients at their atmosphere values.
You can also supply them directly with `stellar.ld_coefficients`.
For a quadratic law, `ld_prior: fixed_u1` holds $u_1$ at the stellar-model value and fits $u_2$ uniformly within `ld_uniform_coefficient_bounds`.
Both transit engines support this option.

Fixing a coefficient does not propagate its stellar-model uncertainty.
In the tests below, fixing the wrong $u_1$ biased the depths even though the chains converged and the reported errors were smaller.
A good-looking fit alone cannot establish that the stellar model is accurate enough.

## Reference

### Intensity laws

With $\mu=\cos\theta$, the supported laws are:

| `ld_profile` | $I(\mu)/I(1)$ |
|---|---|
| `quadratic` | $1-u_1(1-\mu)-u_2(1-\mu)^2$ |
| `power2` | $1-c(1-\mu^\alpha)$ |
| `kipping` | Quadratic, sampled as $u_1=2\sqrt{q_1}q_2$ and $u_2=\sqrt{q_1}(1-2q_2)$ |

For `kipping` with `ld_prior: uniform`, $q_1,q_2\sim U(0,1)$ gives a uniform prior over the physical triangle $u_1>0$, $u_1+u_2<1$, $u_1+2u_2>0$.
This prior needs no stellar grid.
The `gaussian` option instead puts a normal prior in $(q_1,q_2)$ around the converted atmosphere prediction, with linearly propagated widths and truncation to $(0,1)$.
For `fixed`, `stellar.ld_coefficients` contains $[q_1,q_2]$.
Posterior tables report both $q_1,q_2$ and $u_1,u_2$.
The `sing`, `stellarprior`, and `fixed_u1` treatments are unavailable with `kipping`.

### Coefficient priors

| `ld_prior` | Treatment |
|---|---|
| `fixed` | Hold atmosphere-grid coefficients or explicit `stellar.ld_coefficients` fixed. |
| `gaussian` | Gaussian priors around the atmosphere prediction. |
| `uniform` | Broad priors without stellar-model information. Power-2 uses the Maxted coordinates below. |
| `stellarprior` | Propagate `teff_sigma`, `logg_sigma`, and `feh_sigma` through the selected quadratic or power-2 law. The coefficients are computed on a grid of `stellar.ld_prior_n_grid` (default 9) values per parameter over ±`stellar.ld_prior_nsigma` (default 5) standard deviations, skipping points with negligible weight, and their Gaussian-weighted mean and scatter set the prior. |
| `sing` | Quadratic coefficients with a shared gray calibration offset. |
| `fixed_u1` | Fix quadratic $u_1$ to the stellar-model value and sample $u_2$ uniformly. |

The Gaussian coefficient-space width is `flags.ld_gaussian_sigma`, which defaults to 0.2 and must be positive.
It applies in white light and spectroscopy where the engine supports that treatment.
The Kipping Gaussian uses propagated widths in $q$ coordinates, so it differs from a direct Gaussian density in quadratic coefficients.

For quadratic `stellarprior`, the priors are independent coefficient-space normals truncated to $(0,1)$.
The width floor is `stellar.ld_prior_min_sigma`.
This propagates marginal scatter, without constructing a correlated stellar-mixture density.
Cached quadratic grids use `u1_*` and `u2_*` columns and include the law and optional `ld_mu_min` in their cache identity.
For `sing`, `stellar.ld_mu_min` excludes the extreme limb from the atmosphere-profile fit.

To sample independent quadratic coefficients over a box, use `ld_uniform_basis: coefficients` and `ld_uniform_coefficient_bounds: [-3, 3]`.
The bounds default to `[0, 1]` when omitted.
The same bounds apply to the free $u_2$ in `fixed_u1`, without a coordinate transformation.
This box can extend outside the physical triangle.
For a pure eclipse, use `ld_profile: quadratic`, `ld_prior: fixed`, and `stellar.ld_coefficients: [0.0, 0.0]`.

### Power-2 priors

The power-2 uniform prior is flat in the Maxted (2018) coordinates

$$h_1=I(1/2)=1-c(1-2^{-\alpha}),\qquad h_2=I(1/2)-I(0)=c\,2^{-\alpha},$$

on $0<h_1<1$ and $0<h_2<\min(1,h_1+0.5)$.
This permits any $\alpha>0$ and $c<1.5$, including a limb intensity as low as $-0.5$.
The disk-integrated stellar flux stays positive throughout the support.
The Gaussian prior is centered on the atmosphere prediction and truncated to $c\in(0,1)$ and $\alpha\in(0.001,1)$.

We tested a flat $c,\alpha\sim U(0,1)$ box with HATS-6 b PRISM injections using the real timestamps, measured per-channel noise, and 64 noise draws per case.
It shifted depths by up to $+0.6\sigma$ near 2 µm, where the stellar limb is nearly black.
For a star with $\alpha=1.6$, it shifted depths by $-2.9\sigma$ because the box cuts off at $\alpha=1$.
The Maxted prior gave the smallest shifts among the uninformative priors tested, at the same sampling cost:

| Truth (channel) | Box $U(0,1)^2$ | Maxted uniform |
|---|---|---|
| $c=0.73$, $\alpha=1.07$ (0.62 µm) | $-0.27\sigma$ | $-0.10\sigma$ |
| $c=0.72$, $\alpha=0.48$ (1.0 µm) | $+0.20\sigma$ | $-0.04\sigma$ |
| $c=0.93$, $\alpha=0.24$ (2.0 µm) | $+0.61\sigma$ | $+0.41\sigma$ |
| $c=0.50$, $\alpha=0.44$ (3.0 µm) | $+0.13\sigma$ | $+0.13\sigma$ |
| $c=0.41$, $\alpha=0.26$ (4.4 µm) | $+0.16\sigma$ | $+0.04\sigma$ |
| $c=0.60$, $\alpha=1.6$ (0.9 µm noise) | $-2.86\sigma$ | $-0.30\sigma$ |
| $c=0$ (no limb darkening, 4.4 µm noise) | $-0.40\sigma$ | $-0.51\sigma$ |

We also tested a box widened to $c\in[-1,1.5]$, $\alpha\in[0,3]$, a box extending only $\alpha$ to 2, and a prior flat in quadratic-equivalent coefficients.
All three gave larger shifts for realistic stellar profiles.
The wide boxes needed 1.5–2 times more leapfrog steps.

No uninformative per-channel prior is unbiased for every star.
A noisy channel may barely constrain the extreme-limb intensity, even though it sets the stellar flux normalization and therefore the inferred radius.
The resulting shifts can share a sign across neighboring channels and matter after spectral binning.
If you trust the stellar model in these regions, compare a `stellarprior` or `fixed` fit.

### Power-2 sampling coordinates

The Gaussian and uniform modes use smooth unconstrained coordinates by default (`ld_parameterization: auto`, which selects `maxted_smooth`).
For the uniform prior these are $(\mathrm{logit}\,h_1,\mathrm{logit}[h_2/\min(1,h_1+0.5)])$.
For the Gaussian prior they are a smooth Maxted-like map of the truncated box.
Both preserve the intended density with an exact change of variables.

Use `ld_parameterization: decorrelated` to sample $(h_1,h_2)$ with hard boundaries while preserving the full prior.
The direct `coefficients` option uses hard boundaries in $(c,\alpha)$.
Its uniform mode retains the older $\alpha<40$ cap, which removes about $5.2\times10^{-13}$ of the Maxted prior mass.
The default smooth uniform option has no finite upper cap on $\alpha$.

### Power-2 transit evaluation

Ordinary unspotted power-2 transits integrate the intensity law directly in float64.
With fixed spectroscopic geometry, koala can use a Chebyshev table checked against the integral in both flux and derivatives.
Values outside the table domain use direct integration, so the table does not restrict the coefficient or radius priors.

Long white-light series use a 4,097-node contact grid of the direct integral, including derivatives as the geometry moves.
Coalescing inner contacts and unusually small or large radii fall back to integration at each cadence.
Short series and Keplerian fits also use direct cadence integration.
The fallback uses small quadrature slabs to limit memory use when fitting many channels.
For a single planet, it avoids an extra orbit-batching layer so independent GPU chains only evaluate the fallback when needed.

Older fits used a degree-12 polynomial projection.
At small exponents, its flux errors could range from several to hundreds of ppm.
Changing the atmosphere-profile `mu_min` does not correct that error.
Old likelihood exports without an evaluation-method field retain their polynomial model when replayed for scoring.
New fits record the direct method and invalidate old posterior checkpoints.
Spot, eclipse, and interpolated-profile calculations retain their polynomial evaluation.

### Default selection and bias evidence

The default uses wide uniform priors on the sum and difference of the quadratic coefficients.
[Coulombe, Roy & Benneke (2024)](https://arxiv.org/abs/2409.03812) discuss wavelength-dependent biases from physically constrained quadratic parameterizations and recommend wide uninformative priors.
The following numbers are koala validation results, rather than results from that paper.
They give the maximum absolute depth bias, in posterior standard deviations, for noise-free HATS-6 injections in eight channels:

| Treatment | Near-edge truths | Interior stellar-model truths |
|---|---:|---:|
| Kipping uniform | 0.27 | 0.21 |
| $u_1,u_2\sim U(-2,2)$ | 0.092 | 0.137 |
| $u_1,u_2\sim U(-3,3)$ | 0.145 | 0.126 |
| Uniform sum and difference (default) | 0.092 | 0.084 |
| Fixed $u_1$, $u_2\sim U(-3,3)$ | 1.29 | 0.049 |

We then tested the default through the normal command using noise-free PRISM injections in eight channels with all 17,850 integrations.
For constant, linear, quadratic, cubic, and exponential-plus-linear trends, every recovered depth was within $0.1\sigma$ of the truth (the worst channel was $0.09\sigma$).
Over 20 noisy realizations, the mean pull was $-0.05\pm0.09$, and 72% of the 68% intervals contained the true depth.
In the same test, power-2 limb darkening reached $0.11\sigma$ with Gaussian priors, $0.18\sigma$ with free priors, and $0.03\sigma$ with fixed coefficients.
The quadratic stellar-model prior reached $0.04\sigma$.
The Kipping parameterization reached $0.29\sigma$ in this comparison.

These tests support the default for these injections.
They do not establish that it is unbiased for every observation.
Fixing $u_1$ gave a maximum bias of $0.049\sigma$ and about 10% narrower errors when the model coefficient matched the truth.
With an intentionally mismatched $u_1$, the fit still converged and reported smaller errors, but the bias reached $1.29\sigma$.
