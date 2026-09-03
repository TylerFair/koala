# Per-channel Bayesian model stacking prototype

Date: 2026-09-02

## Method

For model `m`, posterior draw `s`, wavelength channel `c`, and cadence `i`, the code replays the exact dumped NumPyro stage and records the Normal log density `log p(y_ci | theta_mcs)`. PSIS smooths the raw leave-one-out importance ratios `1 / p(y_ci | theta_mcs)` and returns the pointwise leave-one-out log predictive density and Pareto k.

Stacking chooses one simplex vector per channel:

`w* = argmax_w sum_i log(sum_m w_m p_m(y_ci | y_c,-i))`.

This targets prediction of another light-curve point in the same channel. The stacked depth posterior is a draw-level mixture of the model posteriors, not an average of summary rows. Consequently it preserves multimodality and widens when supported assumptions give different depths. `disagreement` is the stacked 16--84 percent half-width divided by the smallest corresponding single-model half-width.

Pseudo-BMA+ instead exponentiates LOO ELPDs after 1,000 Bayesian-bootstrap reweightings of the cadences. It is cheap and less brittle than plain pseudo-BMA, but it does not directly optimize the predictive mixture and can give more concentrated model probabilities.

White-light BMA uses `log Z = log p(y, theta_MAP) + d/2 log(2 pi) - 1/2 log det(H)`. This is a model-probability calculation and is sensitive to prior volume; it is not a per-channel predictive score. `models/stacking.py` implements and tests the calculation. The completed production pipeline did not serialize the full white-light MAP Hessian or log joint, so numerical white-light BMA weights cannot honestly be reconstructed from these artifacts. `stack_spectra.py` records that status rather than substituting LOO scores and calling them evidence.

Pointwise LOO is appropriate only under the fitted conditionally independent Normal residual model. If time-correlated residuals matter, leave-future-out or blocked LOO is the relevant predictive experiment.

## Files added

| File | Purpose |
|---|---|
| `models/stacking.py` | Pointwise replay likelihood, PSIS-LOO, stacking, pseudo-BMA+, Laplace evidence, posterior mixtures, CSV writer |
| `tools/stacking/run_matrix.py` | Deep-merge matrix materializer; creates isolated configs and queue scripts in the assigned range |
| `tools/stacking/stack_spectra.py` | Checkpoint combination, likelihood archives, all available weights, mixture CSV, arrays, diagnostics, and figure |
| `tests/test_stacking.py` | Five synthetic numerical tests, including offset alignment |
| `tests/test_ld_initialization.py` | Coordinate-aware power-2 LD initialization regression tests |
| `configs_stacking/*.yaml` | R20 proposal matrices, generated variant configs, retry records, and analysis manifest |
| `docs/guides/model_stacking.md` | Astronomer-facing guide |
| `acceleration_reports/stacking/*` | Four float32 likelihood archives, stacked CSV/NPZ/JSON, figure, and this report |

## WASP-39 b NRS1 R20 result

The completed comparison uses five wavelength channels and four power-2 LD assumptions with the same linear trend: stellar-informed, uniform, wide Gaussian, and fixed. Every final run used a full independent white-light fit, Laplace white-light metric, independent spectroscopic NUTS, Laplace spectroscopic metric, lognormal jitter, and `spectro_min_depth_ess: 400`. The uniform run automatically extended to 2,000 retained draws to satisfy the ESS policy; the other runs retained 1,000.

Walls were 307, 296, 324, and 255 seconds respectively (5.1, 4.9, 5.4, 4.2 minutes).

Stacking weights by increasing wavelength (2.900, 3.086, 3.284, 3.494, 3.718 micron) were:

| Model | 2.900 | 3.086 | 3.284 | 3.494 | 3.718 |
|---|---:|---:|---:|---:|---:|
| informed | 0.000 | 0.000 | 0.000 | 0.199 | 0.269 |
| uniform | 0.000 | 0.000 | 0.000 | 0.172 | 0.058 |
| wide Gaussian | 0.651 | 0.755 | 0.781 | 0.629 | 0.673 |
| fixed | 0.349 | 0.245 | 0.219 | 0.000 | 0.000 |

Pseudo-BMA+ weights were respectively `[0.115,0.351,0.380,0.155]`, `[0.037,0.385,0.525,0.053]`, `[0.041,0.375,0.535,0.049]`, `[0.047,0.457,0.458,0.039]`, and `[0.112,0.394,0.388,0.106]` in model order informed/uniform/wide/fixed.

All 9,100 Pareto diagnostics were below 0.7. Maximum k by model was 0.265, 0.329, 0.339, and 0.453; there are no flagged R20 channels. Stacked medians are 21706.7, 21456.2, 21220.8, 21204.4, and 21020.5 ppm. Symmetric 16--84 half-widths are 141.0, 126.0, 119.0, 108.9, and 113.3 ppm. Disagreement ratios are 2.94, 3.22, 2.59, 2.19, and 2.25: the requested uncertainty inflation is clearly active.

White-light geometry was intentionally refit for every assumption. Across the four final variants, `b` spans 0.4467--0.4763, duration 0.119591--0.119635 d, and white-light radius ratio 0.145250--0.146021. This geometry variation contributes to the model uncertainty and is not artificially removed by sharing a handoff.

The staged production reference contains 68 native/reference-grid channels, not these five R20 bins. Interpolating its median spectrum to the R20 centers gives stacked-minus-production differences `[-355,-99,-145,+33,-113]` ppm, mean -136 ppm and RMS 185 ppm. This is only a grid-mismatched diagnostic, not a parity test: a defensible comparison must integrate the production light curves or posterior spectrum through the exact R20 bin response.

## Exact commands

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false JAX_ENABLE_X64=1 /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest tests/test_stacking.py -rA
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/stacking/run_matrix.py configs_fiducial_stellarinformed/WASP-39_nrs1_g395h_config.yaml configs_stacking/wasp39_nrs1_r20_matrix_retry2.yaml --queue-start 308
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/stacking/run_matrix.py configs_fiducial_stellarinformed/WASP-39_nrs1_g395h_config.yaml configs_stacking/wasp39_nrs1_r20_matrix_ld_extra.yaml --queue-start 312
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/stacking/run_matrix.py configs_fiducial_stellarinformed/WASP-39_nrs1_g395h_config.yaml configs_stacking/wasp39_nrs1_r20_matrix_fixed_retry3.yaml --queue-start 314
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false JAX_ENABLE_X64=1 /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/stacking/stack_spectra.py configs_stacking/wasp39_nrs1_r20_matrix_analysis.yaml --output acceleration_reports/stacking --n-out 20000
```

Queue scripts set `JAX_ENABLE_X64=1`, run from the absolute repository path, and write sampler inputs/results under the required scratch roots. The final test result was 4 passed in 2.88 s.

## Failures and open risks

The first jobs 300--303 failed before sampling because changing the common config `path` also relocated the input FITS directory. Jobs 304--307 used an absolute input directory; 304--306 completed, while 307 was killed at 90 minutes after resource contention. Its clean replacement 311 completed in 296 seconds. Job 313 failed before Python started because its Slurm allocation ended during step creation; replacement 314 completed in 255 seconds. No failed artifact was removed or overwritten.

The candidate set is chosen by the analyst, so omitted plausible trends or LD laws cause model-expansion bias. Current weights should not be generalized beyond this four-LD linear-trend prototype. Correlated residuals violate pointwise LOO independence. A future comparison should include the production discontinuity, quadratic, and scientifically justified exponential-linear trends, serialize the white-light Hessian/log joint for actual BMA weights, and decide whether trend screening on white light is scientifically acceptable or hides channel-dependent systematics. Production-resolution fitting was not attempted because the R20 work plus queue recovery consumed the available validation window.

Postscript: the redundant uniform+step replacement 310 reached 90% of its white-light chain but exited 124 at the queue wall; retry 315 exited 143 when its parent allocation ended. Recovery script 316 validates and explicitly reuses the already successful equivalent job 306. At 19:09 CDT it remained pending because the dispatcher stopped consuming its otherwise empty queue. It is not a pending science result and no claim above depends on it.

## Production resolution

The production analysis uses the same 68-channel `prism_template.csv` reference grid as the staged production spectrum. The included models are `informed_linear` (320), `uniform_linear` (321), `fixed_linear` (323), and `informed_linear_step` (324). Their full-pipeline walls were 561, 737, 347, and 5041 seconds (9.4, 12.3, 5.8, and 84.0 minutes). The step run approached the dispatcher limit because its white-light chain repeatedly reached maximum tree depth, but it exited zero and produced both high-resolution checkpoint chunks.

All four spectroscopic outputs cover 68 channels. Informed-linear, fixed-linear, and informed-step have zero divergences and minimum reported depth ESS of 806, 744, and 829. Uniform-linear recorded 12 divergences in its high-resolution diagnostics; its final gate-attempt record lists the four selectively rerun channels with minimum depth ESS 733. This sampler caveat is retained even though its PSIS diagnostics are stable.

The two other planned variants are excluded from the numerical stack. `wide_linear` failed before NUTS with `Normal distribution got invalid loc parameter`; `uniform_linear_step` failed with `Cannot find valid initial parameters`. Input fluxes and errors are finite on the same edge channels successfully fit by the four included models. The offending point is the white-light optimized initializer, before a spectroscopic channel exists: wide linear produced `(c1,c2)=(-2232.5,-0.895)`, while uniform plus step produced `(c1,c2)=(-2373.7,14.7)`, `t_jump=108110.9`, and `jump=506.2`. The first cause found was a coordinate mismatch: free/wide/uniform power-2 models default to the sampled `ld_decorrelated` site, but the staged optimizer initialized and selected deterministic `c1,c2` sites. `fit_jwst.py` now transforms the physical LD start into the actual latent coordinate and selects that latent site for optimization. Two regression tests cover the offending stellar coefficients `(0.37111993, 0.33026019)`. Post-fix run 380 then exposed the remaining issue: the decorrelated transform advertises an unconstrained real codomain, allowing optimization to reach `ld_decorrelated=(316.7,0.449)`, whose physical inverse is `(c1,c2)=(-315.3,NaN)`; geometry also reached `b=2` and `rprs=0.707`. A safe repair requires constrained optimization or validation-and-fallback for the complete staged solution. That broader change was not attempted after the 45-minute diagnosis time box. In contrast, post-fix uniform-step 381 reached a physical optimized point (`c1=0.928`, `c2=0.099`, `b=0.455`, `rprs=0.1465`, `jump=-0.00148`) and passed its gradient diagnostic at `4.76e-5`. Its long white-light chain was stopped by cancelling only Slurm step `57474887.33` when the diagnosis time box expired; dispatcher exit was 137. It was not requeued. The four-model result below does not depend on either job.

### Achromatic alignment and weights

Offsets are inverse-variance weighted means of each model median minus the across-model median-depth mean. They are:

| Model | Delta (ppm) | Uncertainty (ppm) |
|---|---:|---:|
| informed linear | -117.14 | 11.66 |
| uniform linear | +122.32 | 13.45 |
| fixed linear | -114.16 | 11.68 |
| informed linear + step | +110.13 | 11.27 |

The headline mixture subtracts each Delta from its model draws, mixes by the unchanged light-curve LOO weights, and adds back the model-weighted average Delta. This preserves the model-average absolute level while allowing only spectral-shape disagreement to widen the interval. The CSV also contains every absolute model spectrum and an unaligned `absolute_*` stack.

Mean stacking weights are 0.0112, 0.0296, 0.0176, and 0.9416 in the table order above; informed plus step is the largest weight in all 68 channels. All pointwise Pareto diagnostics pass: global maximum `k=0.547`, with zero channels at or above 0.7. Median aligned disagreement is 1.020 (range 0.981--1.181); the absolute value is 1.032 (range 0.989--1.811). Ratios slightly below one are finite-mixture quantile Monte Carlo variation, not interval shrinkage by construction.

### Same-grid production comparison

The stacked and staged production wavelength arrays match exactly. Using inverse combined variance, aligned stacked minus staged production is `-5.43 +/- 15.93 ppm`. The weighted residual slope is `-0.29 ppm/um`. The median ratio of aligned stacked to production 68-percent half-width is 1.016. These are now direct same-channel comparisons, unlike the superseded R20 interpolation diagnostic.

Products are `wasp39_nrs1_reference_four_stacked.csv`, `wasp39_nrs1_reference_four_diagnostics.json`, `wasp39_nrs1_reference_four_stacking_arrays.npz`, four float32 pointwise-likelihood archives, and the four-panel `wasp39_nrs1_reference_four_stacking.png`. The figure was copied to `docs/_static/model_stacking_wasp39.png`.

### Production commands and failures

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false JAX_ENABLE_X64=1 /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest tests/test_ld_initialization.py tests/test_stacking.py -q
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/stacking/run_matrix.py configs_fiducial_stellarinformed/WASP-39_nrs1_g395h_config.yaml configs_stacking/wasp39_nrs1_reference_initfix.yaml --queue-start 380
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false JAX_ENABLE_X64=1 /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/stacking/stack_spectra.py configs_stacking/wasp39_nrs1_reference_analysis_four.yaml --output acceleration_reports/stacking --stage high_resolution --label wasp39_nrs1_reference_four --n-out 20000
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m sphinx -E -W -b html docs docs/_build/html
```

Initialization retries 322 and 326 failed because the former lost its Slurm step and the latter hit the wide-LD invalid initializer. Retry 333 reproduced the initializer failure. Runs 325 and 334 reproduced the uniform-step invalid initializer. Copied-output experiments 335 and 336 were correctly rejected by science-artifact fingerprint checks and were not used. No failed output was removed or overwritten. White-light Laplace BMA remains unavailable because these pipeline artifacts do not serialize the full Hessian and MAP log joint; the analytic implementation and test are present, but no LOO value is mislabeled as evidence.

The literal `sphinx-build` command first failed with exit 127 because that console script is not on the login-node `PATH`. Running the same Sphinx warning-as-error build through the specified environment's Python succeeded with Sphinx 8.1.3. The combined LD-initialization and stacking test run passed 7 tests in 44.16 seconds.

## HAT-P-18 G395M

The documentation headline example is now HAT-P-18 b NIRSpec/G395M NRS1 on the exact 208-channel production reference grid. All three requested linear-trend variants completed: fixed power-2 (queue 410), uniform quadratic in physical coefficients (411), and Sing quadratic in physical coefficients with `mu_min=0.2` (412). Full-pipeline walls were 876, 735, and 933 seconds (14.6, 12.2, and 15.6 minutes).

The spectroscopic gate/swap outputs contain all 208 channels for every model. Final `sampler_used` counts were:

| Model | independent NUTS | independent HMC | joint NUTS | divergences recorded |
|---|---:|---:|---:|---:|
| fixed power-2 | 203 | 4 | 1 | 5 |
| uniform quadratic | 203 | 5 | 0 | 15 |
| Sing quadratic | 202 | 6 | 0 | 8 |

Thus the automatic gate did route the difficult lanes instead of silently accepting the first sampler everywhere. The nonzero final diagnostic divergence totals remain a caveat even though all depth ESS routing completed and PSIS is stable. The Sing broad gray-offset calibration executed but failed its calibration ESS gate (`79.91 < 100`, zero divergences), so the pipeline explicitly fell back to the tabulated Stagger offsets (`delta_l=+0.020`, `delta_delta=-0.003`). No initialization failure or GPU retry occurred.

### Offsets, weights, and PSIS

The inverse-variance achromatic offsets relative to the across-model channel reference are:

| Model | Delta (ppm) | Uncertainty (ppm) | mean stacking weight | mean pseudo-BMA+ weight | dominant channels |
|---|---:|---:|---:|---:|---:|
| fixed power-2 | -11.01 | 11.64 | 0.6687 | 0.4010 | 135 |
| uniform quadratic | -14.52 | 14.69 | 0.1271 | 0.2719 | 29 |
| Sing quadratic | +26.99 | 12.96 | 0.2042 | 0.3271 | 44 |

LOO weights are computed from light-curve prediction and are unchanged by depth alignment. The headline mixture subtracts each Delta, mixes draws channel by channel, and restores the model-weighted average offset. The CSV retains the unaligned `absolute_*` mixture and each absolute candidate spectrum. Median aligned disagreement is 1.008 and its maximum is 1.526.

All 428,064 Pareto diagnostics (3 models x 208 channels x 686 cadences) are below 0.7. The global maximum is 0.541; maxima for fixed, uniform, and Sing are 0.525, 0.541, and 0.457. There are no flagged channels.

### Same-grid production comparison

The staged production and stacked wavelength arrays match exactly, including all 208 channels. With inverse combined-variance weighting, aligned stack minus production is `-11.39 +/- 17.50 ppm`; the weighted residual slope is `+9.12 ppm/um`, and the median ratio of aligned-stack to production 68-percent half-width is 0.983. This directly measures the LD-marginalized example against the saved stellar-informed production spectrum; it is not an interpolated comparison.

Products are `hatp18_nrs1_g395m_reference_stacked.csv`, `hatp18_nrs1_g395m_reference_diagnostics.json`, `hatp18_nrs1_g395m_reference_stacking_arrays.npz`, three float32 pointwise likelihood archives, and `hatp18_nrs1_g395m_reference_stacking.png`. The figure is copied to `docs/_static/model_stacking_hatp18.png`.

### HAT-P-18 files and exact commands

| File | Change |
|---|---|
| `configs_stacking/hatp18_nrs1_g395m_reference_matrix.yaml` | Three-model scientific matrix |
| `configs_stacking/hatp18_nrs1_g395m_reference_analysis.yaml` | Completed-run analysis manifest |
| `configs_stacking/hatp18_nrs1_g395m_ref_*.yaml` | Generated isolated pipeline configurations |
| `tools/stacking/stack_spectra.py` | Added bounded-memory draw batching and streamed NPZ likelihood caches |
| `docs/guides/model_stacking.md` | Recast tutorial around HAT-P-18 and retained WASP-39 as a second example |
| `docs/_static/model_stacking_hatp18.png` | HAT-P-18 four-panel diagnostic figure |
| `docs/conf.py`, `docs/index.md` | Removed the temporary guide exclusion and added it to the Guides navigation |
| `acceleration_reports/stacking/hatp18_nrs1_g395m_reference_*` | Likelihood caches, stack table, arrays, diagnostics, and figure |

```bash
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/stacking/run_matrix.py configs_fiducial_stellarinformed/HAT-P-18_nrs1_g395m_config.yaml configs_stacking/hatp18_nrs1_g395m_reference_matrix.yaml --queue-start 410
while [ ! -f acceleration_reports/gpu_queue/done/410_stacking_fixed_power2_linear.exit ] || [ ! -f acceleration_reports/gpu_queue/done/411_stacking_uniform_quadratic_linear.exit ] || [ ! -f acceleration_reports/gpu_queue/done/412_stacking_sing_quadratic_linear.exit ]; do sleep 60; done
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/stacking/stack_spectra.py configs_stacking/hatp18_nrs1_g395m_reference_analysis.yaml --output acceleration_reports/stacking --stage high_resolution --label hatp18_nrs1_g395m_reference --n-out 20000
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false JAX_ENABLE_X64=1 /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest tests/test_stacking.py tests/test_ld_initialization.py -q
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m sphinx -E -W -b html docs docs/_build/html
```

The first CPU stacker attempt was killed with exit 137 because vectorizing all 1,000 draws across 208 channels materialized a very large JAX likelihood evaluation. No completed output was overwritten. The stacker now evaluates 25 draws at a time into one multi-member float32 NPZ and scores one channel at a time; the rerun completed with all draws and cadences. White-light Laplace BMA remains unavailable because the fit artifacts do not serialize the full white-light Hessian and MAP log joint. Open scientific risks remain model-list expansion bias, cadence-correlated residuals (which would require blocked LOO), the Sing calibration fallback, and whether trend selection should occur on white light or per channel.

Final verification: `tests/test_stacking.py tests/test_ld_initialization.py` passed 10 tests in 25.74 seconds (three dependency deprecation warnings), and a clean `python -m sphinx -E -W -b html docs docs/_build/html` rendered all 18 sources, including `guides/model_stacking`, successfully with Sphinx 8.1.3.
