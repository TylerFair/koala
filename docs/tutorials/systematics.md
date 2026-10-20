# Trends and instrument systematics

A transit is rarely the only structure in a light curve.
We also need a model for the instrument trend, and that choice can affect the inferred depth.
Here we'll start with the linear trend in the bundled SOSS example and see how to try an alternative.

## Start with a line

The trend multiplies the astrophysical light curve, so $F=(1+\mathrm{transit})\times\mathrm{trend}$.
The SOSS configuration selects a linear trend:

```yaml
flags:
  detrending_type: linear
```

First, fit that configuration and inspect the white-light residuals.
Then, in a copy with a separate output directory, try `detrending_type: quadratic`.
Keep the data, masks, and other priors the same so we can see what the trend choice changes.
Compare the depths as well as the residuals.

```{figure} ../_static/trend_gallery.png
:alt: Gallery of polynomial, ramp, step, spot, combined, and GP trend shapes

Examples of the supported trend shapes, without a transit or eclipse.
The vertical scales are arbitrary.
The GP curve illustrates one possible realization.
```

An exponential ramp can trade its timescale against a shallow eclipse.
If the depth changes when you add a ramp, compare a simpler baseline after trimming the initial settling.
The [rocky-planet eclipse](../case_studies/rocky_eclipse.md) gives an example of this ambiguity.

## Gaussian processes

If the residuals have time-correlated structure, we can try a Gaussian process:

```yaml
flags:
  detrending_type: linear+gp
gp_solver: auto
```

The white-light fit uses a Matérn-3/2 GP from tinygp.
The wavelength channels then use the saved white-light trend as a fixed curve with a fitted amplitude.
Their fits are therefore conditional on the inferred white-light shape.

A GP with a timescale near ingress or eclipse duration can absorb part of the astrophysical signal.
Compare its depths with a simpler trend using the same data and priors.
The [combining models tutorial](combining_models.md) explains how predictive comparisons work and where correlated noise limits them.

## Exclude time ranges

To remove initial settling or a known bad interval, use explicit ranges:

```yaml
flags:
  exclude_times: [[null, "min(t) + 10 / 1440"]]
  exclude_integrations: [[400, 405], [-5, null]]
```

This removes the first 10 minutes, integrations 400 through 405, and the last five integrations.
Bounds are inclusive.
Integration indices start at zero, and negative indices count from the end.
Use `null` to leave a bound open.
Time expressions use days and the loaded BMJD_TDB array.
Both settings also work at the top level of the YAML.

The older `mask_start` and `mask_end` settings are merged with these exclusions.
So are `outlier_clip.mask_integrations_start` and `mask_integrations_end`.
Every retained integration contributes separately to the fit.

## Reference

### Trend names

In elapsed days, write $x=t-\min(t)$.
A polynomial trend is $T=c+vx+v_2x^2+\cdots$, and an exponential adds $A\exp(-x/\tau)$.

| `detrending_type` | Shape |
|---|---|
| `none` | Unity, with no fitted normalization. |
| `constant` | Fitted normalization `c`. |
| `linear`, `quadratic`, `cubic`, `quartic` | Polynomial through the named order. |
| `explinear` | Exponential ramp plus linear baseline. |
| `linear_discontinuity` | Linear baseline with a smooth step. |
| `spot`, `2spot` | Linear baseline with one or two spot-crossing templates. |
| `quadratic+spot` | Quadratic baseline and one spot template. |
| `spot+linear_discontinuity` | Spot template, linear baseline, and step. |
| `spot+explinear`, `2spot+explinear` | One or two spot templates with a ramp and linear baseline. |
| `gp` | GP about a constant baseline. |
| `linear+gp`, `quadratic+gp`, `cubic+gp`, `quartic+gp`, `explinear+gp` | Named mean trend plus a GP. |

Set a spot template's initial position, width, and amplitude with `spot_center`, `spot_width`, and `spot_amp`.
A second template uses the suffix `2`.
Step guesses use `t_jump_guess` and `jump_guess`.
Times and widths are in days, and amplitudes are in normalized flux.
Spectroscopy reuses the white-light spot and step shapes with an amplitude for each channel.

For an exponential ramp, `flags.spectro_fixed_timescale_trends` defaults to `true`.
This holds the white-light ramp shape fixed in spectroscopy.
Set it to `false` to sample a timescale in each channel with a log-uniform prior.
The fixed ramp can be combined with GP, spot, and two-spot templates.
Harmonica requires a sampled timescale and raises an error for a fixed ramp.

### White-light trend priors

The optional top-level `trend_priors` block changes the white-light priors.
Each entry uses the [planet-parameter syntax](planet_parameters.md#prior-syntax), with an optional `value` for the starting point.
Omitted entries keep these defaults:

| Key | Default prior | Default start |
|---|---|---|
| `c` | Uniform(0.9, 1.1) | 1 |
| `v` | Uniform(−0.1, 0.1) d⁻¹ | 0 |
| `v2`, `v3`, `v4` | Uniform(−0.1, 0.1), in elapsed-day units for each degree | 0 |
| `A` | Uniform(−0.1, 0.1) | 0.001 |
| `tau` | Log-uniform(1e-3, 1e-1) d, sampled as `log_tau` | 0.01 d |
| `jitter` | Log-uniform(1e-5, 1e-2), sampled as `log_jitter` | 1e-4 |

The jitter is added in quadrature to the reported white-light errors.
Use `jitter: {value: 0.0, prior: fixed}` to use those errors alone.
If a paper fitted the same family of curves, for example $F_*[1+m(t-\bar t)+A e^{-(t-t_0)/\tau}]$ with $t_0$ the first timestamp, you can set comparable priors on koala's coefficients like this (map the paper's parameters onto these before reusing its numbers):

```yaml
trend_priors:
  c:   {prior: uniform, low: 0.9, high: 1.1}
  v:   {prior: gaussian, value: 0.0, sigma: 0.01}
  A:   {prior: uniform, low: -0.01, high: 0.01}
  tau: {prior: uniform, low: 0.0, high: 0.55}
```

A uniform or Gaussian timescale is sampled directly as `tau`.
Inside the exponential it is floored at 1e-6 d, so a lower bound of zero is allowed.
If the default start falls outside a configured range, koala uses its midpoint (the geometric midpoint for a log-uniform prior).
The eclipse-time scan searches the configured timescale range.
These settings change only the white-light fit.
Spectroscopic channels have their own coefficient priors.

### Spectroscopic trend priors

For sampled constant through quartic trends, the command-line fit uses

$$x=\frac{t-\mathrm{mean}(t_{\rm observed})}{\mathrm{span}(t_{\rm observed})}$$

with independent $U(-10,10)$ priors on every polynomial coefficient, including the intercept.
Each channel uses its own observed-time mask.
The mean defines a time origin without averaging integrations.
koala records the effective basis and bounds before checking whether a saved fit can be reused.

Use `flags.spectro_trend_basis: elapsed_days` to select elapsed-day priors explicitly.
The low-level model factory also accepts an explicit basis.
Non-polynomial templates and Gaussian-integrated trends retain their own coordinate defaults.
Output coefficients and reconstructed conditional draws retain elapsed-day units.

`flags.spectro_trend_bounds` changes individual sampled bounds in the selected basis.
It accepts finite ordered pairs for `c`, `v`, `v2`, `v3`, `v4`, ramp amplitude `A`, `log_tau`, and template amplitudes `A_spot`, `A_spot2`, `A_jump`, and `A_gp`.
For example, `A_jump: [-10, 10]` widens the jump-amplitude prior.
Normalized time changes only the polynomial terms.
A `log_tau` bound requires a sampled-timescale ramp and is in natural-log days.
The broader template bounds remain opt-in while their scientific width and convergence tests are pending.

The default `flags.trend_inference: sampled_uniform` samples the coefficients.
The `gaussian_marginalized` option integrates conditionally linear coefficients under Gaussian priors on the jaxoplanet path.
It needs at least one linear coefficient and does not support GP trends.
Use `trend_prior_scale` for those Gaussian widths (the default is 0.1 per coefficient).
Uniform bounds cannot specify a Gaussian density.
The two treatments have different priors, so compare their scientific assumptions as well as their run times.

Normalized polynomials also work with mixed fixed templates and free ramps.
Gaussian integration transforms the polynomial columns and leaves template amplitudes unchanged.
Exact out-of-transit statistics preserve that model and include every integration.
Spectral spot, two-spot, and step models include a linear baseline.
Old saved templates without a slope reconstruct with zero slope, and their cached fits are distinct from current fits.

Batched model grids support normalized linear, quadratic, and step fits with `c`, `v`, `v2`, and `A_jump` bounds.
Other coefficient controls or custom white-light `trend_priors` require ordinary grid fitting.
Harmonica requires sampled trend coefficients and a sampled ramp timescale.
Use jaxoplanet for Gaussian-integrated coefficients or a fixed ramp template.

### Two fixed spot templates

The spectroscopic model `quadratic+2spot_spectroscopic` combines a quadratic baseline with two fixed Gaussian templates in observer time:

$$F=(1+\mathrm{transit})[c+v\Delta t+v_2\Delta t^2+A_{\rm spot}T_1+A_{\rm spot2}T_2].$$

Set signed template-amplitude bounds with `flags.spectro_trend_bounds`.
With `spectro_free_t0`, the channel epoch shifts the transit while the templates remain at the supplied observer times.
This model also supports normalized polynomials and Gaussian trend integration.
An additive-template injection needs a separate equivalence check before using it to validate this multiplicative model.

### GP solvers

`gp_solver` accepts `auto`, `serial`, or `parallel`.
You can also set it through `KOALA_GP_SOLVER`.
Automatic selection uses the parallel solver on supported accelerators and the serial solver on CPU.
The pinned tinygp dependency supplies the parallel implementation.
Changing the solver changes the computation, without changing the model.
