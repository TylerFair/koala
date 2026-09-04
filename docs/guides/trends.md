# Systematics trends

Choose the white-light trend with `flags.detrending_type`; the spectroscopic builder uses the corresponding channel model.

```yaml
flags: {detrending_type: linear}
```

| Model | YAML | Parameters |
|---|---|---|
| None | `detrending_type: none` | Unit baseline |
| Linear | `detrending_type: linear` | `c`, `v` |
| Quadratic | `detrending_type: quadratic` | `c`, `v`, `v2` |
| Cubic | `detrending_type: cubic` | Adds `v3` |
| Quartic | `detrending_type: quartic` | Adds `v4` |
| Spot | `detrending_type: spot` | Linear baseline plus a fixed-center/width spot template; set `spot_amp`, `spot_center`, `spot_width` |
| Two spots | `detrending_type: 2spot` | Adds the `spot_amp2`, `spot_center2`, `spot_width2` template |
| Step | `detrending_type: linear_discontinuity` | Linear baseline plus `t_jump`, `jump`, and sigmoid `width`; initialize with `t_jump_guess`, `jump_guess` |
| Exponential + linear | `detrending_type: explinear` | `c + vt + A exp(-t/tau)` |

For an exponential-linear ramp use:

```yaml
flags:
  detrending_type: explinear
```

The white-light stage fits `A` and `tau`. The spectroscopic stages fix `tau` to
the white-light posterior median and fit a channel amplitude `A`. NIRSpec PRISM
frequently needs `explinear`; SOSS and G395H examples use linear, quadratic,
spot, or explinear based on visit behavior. Select the simplest trend supported
by residuals. GP variants are selected by including `gp` in the detrending
type, such as `linear_gp` or `explinear_gp`.

## Equations and sampled priors

Define $x=t-\min(t)$ in days and let $T(t)$ be the transit contribution. All trends are additive: $f(t)=T(t)+S(t)$. For `none`, $S(t)=1$.

The polynomial family is

$$S_p(t)=c+vx+v_2x^2+v_3x^3+v_4x^4,$$

truncated at the selected order. The sampled prior is $c\sim\mathcal U(0.9,1.1)$. Every active $v,v_2,v_3,v_4\sim\mathcal U(-0.1,0.1)$.

```yaml
flags: {detrending_type: cubic}
```

Start with linear for SOSS and many G395H visits. Increase order only for smooth residual structure supported by out-of-transit baseline.

## Exponential ramp

The white-light formula is

$$S(t)=c+vx+A\exp(-x/\tau).$$

$A\sim\mathcal U(-0.1,0.1)$. `log_tau` is uniform between $\log(10^{-3})$ and $\log(10^{-1})$, so $\tau$ spans 0.001--0.1 day. The white-light stage fits both $A$ and $\tau$.

The production spectroscopic stage fixes $\tau$ to the white-light posterior median. Each channel fits `c`, `v`, and its own `A` multiplying the fixed exponential shape. PRISM frequently needs this ramp because visit settling can be strong.

Some SOSS visits also show ramps; decide from the broadband baseline.

## Spot templates

One spot component is

$$G(t)=a\exp[-(t-\mu_s)^2/(2\sigma_s^2)].$$

The white-light `spot` model is $S(t)=c+vx+G(t)$. $a\sim\mathcal U(0,0.1)$. $\mu_s\sim\mathcal N(\mu_\mathrm{guess},0.01)$ day.

$\sigma_s\sim\mathcal U(10^{-4},0.1)$ day.

```yaml
flags:
  detrending_type: spot
  spot_amp: 0.001
  spot_center: 60115.47
  spot_width: 0.005
```

`2spot` adds the analogous second component. Spectroscopic fits reuse the white-light spot shape and fit `A_spot` or `A_spot2` scales. Their sampled bounds are 0.5--2.

Use this model for a localized in-transit feature coherent across wavelength.

## Discontinuities

The default step template is a smooth sigmoid,

$$H(t;t_j,w)=\left[1+\exp(-(t-t_j)/w)\right]^{-1}.$$

The white-light model is $S(t)=c+vx+jH(t;t_j,w)$. $t_j\sim\mathcal N(t_{j,\mathrm{guess}},0.01)$ day and $j\sim\mathcal N(j_\mathrm{guess},0.01)$. The width has a log-uniform prior from half the median cadence to 30 minutes and is reported in days and minutes.

```yaml
flags:
  detrending_type: linear_discontinuity
  t_jump_guess: 59867.20
  jump_guess: 0.0
```

The channel stage fixes both `t_jump` and `width` to their white-light posterior medians, then fits `A_jump` on 0.5--2. Use this for a G395H detector tilt event or another common-mode transition.

## Combined and GP models

Spot-plus-step and spot-plus-explinear names sum the corresponding terms. Use a combined model only when both structures are present in white light. Including `gp` selects a tinygp likelihood around the requested mean trend.

`GP_log_sigma` is uniform from $\log(10^{-5})$ to $\log(10^3)$. `GP_log_rho` is uniform from $\log(0.007)$ to $\log(0.3)$ day.

```yaml
flags: {detrending_type: linear_gp}
```

GP trends cannot use Gaussian coefficient marginalization.

## Jitter

The channel uncertainty is

$$\sigma_{j,i,\mathrm{total}}^2=\sigma_{j,i}^2+s_j^2.$$

For the default prior,

$$\log s_j\sim\mathcal N[\log(0.5\,\mathrm{median}_i\sigma_{j,i}),2^2].$$

The `log_uniform` alternative spans jitter from $10^{-6}$ to 1. White light uses a log-uniform jitter from $10^{-5}$ to $10^{-2}$. Jitter describes extra uncorrelated scatter, not time-correlated systematics.

## Outputs and checks

Active coefficients appear in `*_bestfit_params.csv` with central, standard-deviation, and asymmetric-error columns. Possible fields include `c`, `v`, `v2`, `v3`, `v4`, `A`, `tau`, `t_jump`, `jump`, spot parameters, and template scales. The white-light time-series table includes `trend_model` and `detrended_flux` when available.

Inspect both before interpreting wavelength-dependent depths. Use the noise-binning products to check whether residual RMS approaches white-noise scaling.

## White-light trend coordinates

The fitter chooses numerically stable white-light coordinates for each trend
family while continuing to report physical centers, widths, jump times, and
amplitudes. These coordinate choices preserve the configured physical priors
and are intentionally internal.
