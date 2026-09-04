# Model stacking for transmission spectra

Transmission spectra inherit choices made while fitting each light curve. Reasonable limb-darkening priors or baseline trends can yield different depths even when every fit passes its sampler checks. Model stacking carries that uncertainty into the spectrum instead of selecting one assumption and treating it as known.

This tutorial uses the HAT-P-18 b NIRSpec/G395M NRS1 visit. Its production reference grid has 208 wavelength channels. The scientific question is deliberately plain: how much does the transmission spectrum move when we change the limb-darkening treatment, and what spectrum results when we marginalize over that choice? Three full pipeline fits compare fixed power-2, uniform quadratic, and Sing quadratic limb darkening while holding the linear systematics trend fixed.

## The model matrix

Each variant is a full pipeline run. It therefore gets its own white-light fit and geometry handoff, low-resolution bridge, and reference-grid spectroscopic fit. The matrix file uses small overrides on the checked stellar-informed configuration:

```yaml
dataset: hatp18_nrs1_g395m_ref
analysis_stage: all
variants:
  - name: fixed_power2_linear
    overrides:
      input_dir: /scratch/midway3/tfairnington/FITS
      flags: {ld_prior: fixed, ld_profile: power2, detrending_type: linear}
  - name: uniform_quadratic_linear
    overrides:
      input_dir: /scratch/midway3/tfairnington/FITS
      flags: {ld_prior: uniform, ld_profile: quadratic,
              detrending_type: linear}
  - name: sing_quadratic_linear
    overrides:
      input_dir: /scratch/midway3/tfairnington/FITS
      stellar: {ld_mu_min: 0.2}
      flags: {ld_prior: sing, ld_profile: quadratic,
              detrending_type: linear}
```

Keep the sampling policy identical: independent NUTS, a Laplace metric, lognormal jitter, and a minimum depth ESS of 400. Changing both the scientific assumption and sampler policy would make the comparison difficult to interpret.

## Run the fits and stack them

`run_matrix.py` materializes one isolated configuration and one GPU queue script per variant. On a system without the project dispatcher, the generated configurations can instead be passed to `fit_jwst.py` individually.

```bash
python tools/stacking/run_matrix.py \
  configs_fiducial_stellarinformed/HAT-P-18_nrs1_g395m_config.yaml \
  configs_stacking/hatp18_nrs1_g395m_reference_matrix.yaml \
  --queue-start 410
```

After every fit has a successful exit marker, make the likelihood archives and combined products on CPU:

```bash
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
python tools/stacking/stack_spectra.py \
  configs_stacking/hatp18_nrs1_g395m_reference_analysis.yaml \
  --stage high_resolution \
  --output acceleration_reports/stacking \
  --label hatp18_nrs1_g395m_reference \
  --n-out 20000
```

The analysis replays the exact saved NumPyro stage for every posterior draw and stores the Normal log density of every cadence. Computation is float64; the reusable likelihood archives are float32.

## What stacking optimizes

For model $m$, channel $c$, and cadence $i$, PSIS-LOO estimates the predictive density $p_m(y_{ci}\mid y_{c,-i})$. Stacking chooses non-negative channel-specific weights that sum to one and maximize

\[
\sum_i \log\left[\sum_m w_{mc}p_m(y_{ci}\mid y_{c,-i})\right].
\]

The target is prediction of another light-curve point in the same wavelength channel. This is more direct than asking which complete model is true. The posterior depth is a draw-level mixture: select a model using its channel weight, then select one of that model's depth draws.

Pseudo-BMA+ is provided as a cheaper comparison. It exponentiates summed LOO scores and averages over 1,000 Bayesian-bootstrap reweightings of the cadences. It is usually smoother but does not optimize the predictive mixture directly. White-light Laplace BMA instead approximates a marginal likelihood and is sensitive to prior volume. It is only available when the pipeline saves the white-light MAP log joint and full Hessian.

## Achromatic offset alignment

A constant depth displacement is degenerate with the planet's reference radius or reference pressure. It should not masquerade as wavelength-dependent model uncertainty. For each model, the analysis first computes

\[
\Delta_m =
\frac{\sum_c \left(\tilde d_{mc}-\bar d_c\right)/s_{mc}^2}
     {\sum_c 1/s_{mc}^2},
\]

where $\tilde d_{mc}$ is the model median, $s_{mc}$ its posterior standard deviation, and $\bar d_c$ the across-model mean median. Mixture draws use $d_{mcs}-\Delta_m$, then add back the model-weighted average offset. Thus the headline spectrum keeps the model-average absolute level while its extra width measures spectral-shape disagreement.

The CSV also retains every absolute model spectrum and an unaligned stacked spectrum. Use those columns when the reference-radius level is scientifically relevant or when comparing to an atmosphere model with an explicit reference pressure.

## Read the diagnostic figure

```{image} ../_static/model_stacking_hatp18_fitted_sing_uplus_v2.png
:alt: HAT-P-18 b NIRSpec G395M model-stacking diagnostics
:width: 900px
:align: center
```

The first panel shows the absolute candidate spectra and the headline offset-aligned stack. The second compares aligned and absolute stacked intervals. If their widths differ strongly while their shapes agree, the candidate models mainly disagree about reference radius.

The third panel shows channel-specific stacking weights and pseudo-BMA+ weights. A zero stacking weight is not a failed model: it means that model does not improve the optimal predictive mixture in that channel. Rapid wavelength-to-wavelength changes can be real, but they also motivate checking whether cadence-level noise correlations are being ignored.

The final panel shows the maximum Pareto $\hat k$ and the disagreement ratio. Values below 0.7 support ordinary PSIS-LOO; a channel with points above 0.7 needs exact refits, moment matching, or a more appropriate grouped predictive unit. `disagreement` is the aligned 16--84 percent half-width divided by the smallest single-model half-width. Values near one mean shape robustness; values above one identify assumption-sensitive channels.

For the HAT-P-18 run shown here, all 428,064 pointwise diagnostics satisfy $\hat k<0.7$; the global maximum is 0.537. Mean stacking weights are 0.673 for fixed power-2, 0.120 for wide-uniform quadratic, and 0.208 for Sing quadratic. Fixed is dominant in 140 of 208 channels, while uniform and Sing dominate 24 and 44 channels. Their fitted achromatic depth offsets are -11.4, +5.7, and +6.1 ppm. Once those gray shifts are removed, median disagreement is 1.010 and the largest channel reaches 2.084. The aligned spectrum differs from the staged stellar-informed production spectrum by -9.8 +/- 17.7 ppm, with a residual slope of 9.99 ppm/micron and a median error-bar ratio of 0.991.

Here `uniform` uses the current wide flat priors $u_+\in[-1,2]$ and
$u_-\in[-2,2]$, following Sing et al. (2026). This is a broader prior than the
historical independent $u_1,u_2\in[0,1]$ box, so the uniform model was rerun
rather than reusing its older posterior.

The revised Sing calibration follows Section 3.3 of Sing et al. (2026): it fits independent broad $u_+$ and $u_-$ coordinates, transforms the posterior to $(l,\delta)$, and uses ESS-inflated channel uncertainties for pooling. It measured $\Delta l=+0.00725\pm0.01745$ and $\Delta\delta=+0.00349\pm0.00366$. All 11 channels exceeded bulk ESS 235, with zero calibration divergences, so the fitted correction—not the tabulated fallback—was used. Both measurements agree with the Table 3 population values within their quoted star-to-star scatter.

## Second worked example: WASP-39 b

The same machinery was also validated on four WASP-39 b G395H models over its 68-channel production grid. There the informed linear-plus-discontinuity model had mean stacking weight 0.942, global maximum $\hat k=0.547$, and the aligned stack agreed with production at -5.43 +/- 15.93 ppm. Its figure remains available as `docs/_static/model_stacking_wasp39.png`; full details, including two excluded initialization failures, are in `acceleration_reports/stacking/stacking.md`.

## Outputs

- `*_stacked.csv` is the portable table. Headline columns are aligned; `absolute_*` columns retain the unaligned mixture.

- `*_diagnostics.json` records offsets, Pareto summaries, and model-average weights.

- `*_arrays.npz` retains weights, LOO results, mixture draws, and summaries for downstream plots.

- `*_loglik_*.npz` stores one reusable pointwise likelihood cube per model.

## Assumptions and next steps

Pointwise LOO assumes conditionally independent residuals. If residuals remain time-correlated, use blocked LOO or leave-future-out prediction rather than interpreting cadence-level weights literally. The candidate list also matters: omitting a plausible trend creates model-expansion bias, while adding many nearly duplicate variants can change pseudo-BMA-style probabilities.

Useful extensions are to serialize the white-light MAP Hessian and log joint for genuine Laplace BMA, compare channel-wise trend selection with white-light-only screening, test quadratic and physically justified exponential-linear trends, and propagate detector-level offsets when joining NRS1 and NRS2. Read [Systematics trends](trends.md), [Samplers](samplers.md), and [Outputs](outputs.md) before expanding the matrix.
