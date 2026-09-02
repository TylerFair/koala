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
| Step | `detrending_type: linear_discontinuity` | Linear baseline plus `t_jump` and `jump`; initialize with `t_jump_guess`, `jump_guess` |
| Exponential + linear | `detrending_type: explinear` | `c + vt + A exp(-t/tau)` |

For fixed spectroscopic ramp timescales use:

```yaml
flags:
  detrending_type: explinear
  spectro_fixed_timescale_trends: true
```

The white-light stage fits `A` and `tau`. The spectroscopic stages fix `tau` to the white-light posterior median and fit a channel amplitude `A`.

The default channel jitter prior is `spectro_jitter_prior: lognormal`; `log_uniform` remains available. NIRSpec PRISM frequently needs `explinear`; SOSS and G395H examples use linear, quadratic, spot, or explinear based on visit behavior. Select the simplest trend supported by residuals.

Conditionally linear coefficients can be integrated out:

```yaml
flags:
  trend_inference: gaussian_marginalized
  trend_prior_means: {c: 1.0, v: 0.0}
  trend_prior_scales: {c: 0.1, v: 0.1}
```

This is available for jaxoplanet spectroscopic fits and changes the coefficient prior from bounded uniform to Gaussian. It is unavailable for `none` and GP trends. GP variants are selected by including `gp` in the detrending type, such as `linear_gp` or `explinear_gp`.

