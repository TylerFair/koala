# Choose a systematics trend

The trend describes flux changes that are not part of the transit. Start with the smallest model that explains the out-of-transit data, then inspect the residuals. A more flexible baseline can absorb transit depth when the visit does not constrain it.

Set the model with one line:

```yaml
flags:
  detrending_type: linear
```

## Which trend should I try?

| What the white-light curve shows | First model to try | YAML value |
|---|---|---|
| A flat or gently sloped baseline | Linear | `linear` |
| Smooth curvature across the whole visit | Quadratic | `quadratic` |
| Strong settling at the start | Exponential plus linear | `explinear` |
| A sudden common-mode level change | Linear plus step | `linear_discontinuity` |
| A localized in-transit bump | Spot template | `spot` |
| Residual correlated structure after a justified mean model | Gaussian process | `linear+gp` |

`none`, `cubic`, and `quartic` are also supported. Higher polynomial orders are sensitivity tests, not automatic improvements. Combined names such as `spot+explinear` should be used only when both structures are visible in white light.

```{image} ../_static/tutorial_trend_types.svg
:alt: Schematic comparison of linear, quadratic, exponential-ramp, step, and spot trend shapes
:width: 820px
:align: center
```

The curves are schematics of trend shape, not fits to observed data.

## Try linear and quadratic on one visit

Make two copies of the complete first-transit configuration. The snippets below are edits to those full YAML files; keep the other dataset, stellar, and path fields. In the first, run only white light with a linear baseline:

```yaml
output_dir: results/WASP-39_LINEAR
flags:
  analysis_stage: whitelight
  detrending_type: linear
  ld_profile: power2
  ld_prior: stellarprior
```

In the second, change only the output directory and trend:

```yaml
output_dir: results/WASP-39_QUADRATIC
flags:
  analysis_stage: whitelight
  detrending_type: quadratic
  ld_profile: power2
  ld_prior: stellarprior
```

Run both complete YAML files:

```bash
python fit_jwst.py -c wasp39_linear.yaml
python fit_jwst.py -c wasp39_quadratic.yaml
```

Compare `*_whitelight_summary.png` and the transit parameters in `*_bestfit_params.csv`. Keep quadratic only when the baseline supports curvature and the residuals improve without distorting ingress or egress. Restore `analysis_stage: all` for the chosen production fit.

## A practical comparison

Fit white light with the candidate trends that have a physical or instrumental reason. For each fit:

1. Check that the model follows the out-of-transit baseline without bending through ingress or egress.
2. Check the residual plot for remaining time structure.
3. Compare the inferred transit depth and geometry. Large movement means trend choice is part of the scientific uncertainty.
4. Reject fits with poor sampling diagnostics, regardless of residual RMS.

Keep the same masks, limb darkening, and sampling policy during this comparison. If several trends remain credible and change the spectrum, propagate the choice with [model stacking](model_stacking.md).

## Common models

With $x=t-\min(t)$ in days, polynomial trends are additive to the transit model:

$$
S(t)=c+vx+v_2x^2+\cdots.
$$

`linear`, `quadratic`, `cubic`, and `quartic` stop after the corresponding term. Linear is a sensible starting point for many SOSS and NIRSpec visits.

An exponential ramp adds early-time settling:

$$
S(t)=c+vx+A\exp(-x/\tau).
$$

```yaml
flags:
  detrending_type: explinear
```

White light fits both $A$ and the decay time $\tau$. By default the spectroscopic fits reuse the white-light decay time and fit an amplitude in each channel. This shares the well-measured shape without forcing the ramp to be achromatic.

A discontinuity uses a smooth step centered on `t_jump`:

```yaml
flags:
  detrending_type: linear_discontinuity
  t_jump_guess: 59867.20
  jump_guess: 0.0
```

The channel fits reuse the white-light step time and width. Use this for a real common-mode transition, not a single deviant cadence.

A spot template describes a localized Gaussian-shaped feature:

```yaml
flags:
  detrending_type: spot
  spot_amp: 0.001
  spot_center: 60115.47
  spot_width: 0.005
```

Use `2spot` only when two distinct features are supported. A spot-shaped residual may also come from timing, limb darkening, or extraction systematics, so inspect ingress and egress before giving it a stellar interpretation.

## Sample or marginalize the coefficients?

The usual mode samples trend coefficients with every channel posterior:

```yaml
flags:
  trend_inference: sampled_uniform
```

Conditionally linear coefficients can instead be integrated out under Gaussian priors:

```yaml
flags:
  trend_inference: gaussian_marginalized
```

Marginalization can make the nonlinear sampler smaller while preserving draws of the trend coefficients in saved products. It supports the jaxoplanet path and requires a trend with at least one linear coefficient. It does not support GP trends. Treat the two settings as different prior specifications, not merely two computational routes.

## When is a GP warranted?

Add `+gp` only after choosing an adequate mean trend, for example:

```yaml
flags:
  detrending_type: linear+gp
```

The GP describes correlated residuals. It needs enough baseline to constrain its timescale and cannot be combined with Gaussian trend marginalization. Always compare its inferred depth with a simpler accepted model; a GP can trade against transit shape when its timescale overlaps ingress or egress.

## What to inspect

The white-light summary shows the fit and residuals. `*_whitelight_timeseries.csv` contains `trend_model` and `detrended_flux`; `*_bestfit_params.csv` contains active trend coefficients. Noise-binning plots show whether residual RMS approaches the white-noise expectation as points are averaged.

Jitter accounts for extra *uncorrelated* scatter. It does not repair a misspecified baseline or correlated residuals.
