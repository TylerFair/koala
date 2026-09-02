# PLATON Laplace-metric NUTS study

## Bottom line

A single-mode Laplace metric is **not a safe general replacement for 1000-live-point MultiNest** for these retrievals. Even the nominally simpler HAT-P-30 case is strongly non-Gaussian (cloud-pressure skew 2.61 and a resolved minor component), and the WASP-121 TLS-2D posterior contains boundary-skewed, curved, strongly correlated atmospheric/TLS coordinates. A fixed local metric may still reduce NUTS trajectory length inside one selected mode, but it does not provide evidence and cannot establish mode coverage.

I built a read-only chain diagnostic and a float64 exact-JAX prototype. Chain diagnostics completed. On the login node the full exact float64 prototype was terminated during construction/JIT of the HAT-P-30 forward/gradient program, before the first timing record or MAP; no NUTS result exists and I therefore report **no measured speedup**. The same heavier WASP-121 run was not attempted after this failure. This is an important engineering result: the fork's production JAX likelihood is deliberately float32, and switching its global forward-model dtype to float64 is not currently a cheap drop-in operation on CPU.

## Files built

| file | purpose |
|---|---|
| `tools/platon_study/analyze_chains.py` | Read saved `RetrievalResult.equal_samples`; report skewness, kurtosis, correlations, covariance conditioning, and one- versus two-Gaussian marginal BIC diagnostics. |
| `tools/platon_study/laplace_prototype.py` | Standalone, non-mutating float64 JAX likelihood; unconstrained prior transform; multi-start L-BFGS MAP; exact JAX Hessian with spectral repair; fixed dense-metric NumPyro NUTS with 200-step step-size-only adaptation. |
| `acceleration_reports/platon_study/chain_diagnostics.json` | Machine-readable diagnostics for HAT-P-30 and WASP-121. |

Nothing under `retrievals/platon` was changed. No Slurm command was issued.

## 1. Inventory

### Likelihood/API path

`tyler_scripts/run_chemeq_retrieval.py` builds a `FitInfo`, spectral bins/depths/errors, and calls `platon.experimental.combined_retriever.CombinedRetriever.run_multinest`. For transit-only data, `run_multinest` constructs a `TransitDepthCalculator`, calls `prepare_jax_data`, and creates a jitted per-point likelihood with `make_per_point_lnlike`. The likelihood is the unchanged independent Gaussian spectrum likelihood, including the fitted `error_multiple` if enabled. Invalid runtime states return NaNs and are mapped to a negligible/invalid nested-sampling likelihood.

The important precision finding is explicit in the fork. `experimental/_jax_forward_model.py` converts measured values, errors, model values and the Gaussian calculation to the module-wide `_FM_DTYPE`; the experimental NumPyro entry point explicitly runs `jax.config.update("jax_enable_x64", False)` and samples float32 distributions. Its comment describes float32 as an intentional GPU optimization. The current experimental NumPyro implementation therefore does **not** meet the study's float64 constraint. The prototype changes `_FM_DTYPE` only in its own process before tracing and enables JAX x64; it does not edit PLATON.

Production YAMLs select `sampler: pymultinest`, `nlive: 1000`, `resume: true`, `importance_nested_sampling: false` in the TLS-2D case, and `nsteps: 4` (irrelevant to PyMultiNest). MultiNest outputs include weighted/equal-weight chains, live points, evidence files and resume state.

### Representative cases and priors

| case | data points | fitted dimensions | notable fitted groups |
|---|---:|---:|---|
| HAT-P-30, no TLS, 1-D Guillot/parametric saved result | 354 | 15 | Gaussian `Mp`; uniform `Rp`, thermal/cloud/haze parameters, two detector offsets, `log_SO2`, `logZ`, C/O. |
| WASP-121, TLS, 2-D Guillot identifiable-limb clamp | 346 | 25 | Gaussian `Mp`,`T_star`; uniform radius, Guillot center/contrast variables, cloud/haze center/contrast variables, `T_spot`, spot coverage, limb fraction, offsets, TiO/VO/SO2, metallicity and C/O. |

Exact bounds and posterior summaries for every coordinate are retained in `chain_diagnostics.json`; the bounds are also serialized in each result's `FitInfo`. The WASP-121 opacity list in its YAML is CO2, CH4, HCN, CO, H2O, K, Na, SO2, TiO, VO and SiO.

### Evaluation cost

| measurement | value | qualification |
|---|---:|---|
| HAT-P-30 saved run | 1,651,884 calls / 5,223.55 s = **3.162 ms/call** | Instrumented JAX likelihood time saved in the result; device identity is not serialized, so this is not labeled CPU or GPU. |
| Related production log (`pop1dg_pool_54571140_0.out`) | examples **2.873--4.208 ms/call** | Direct log observations; again, not a controlled device benchmark. |
| HAT-P-30 non-JAX CPU `_ln_like` | **1.537 s** best point; **3.079 s** marginal-median point | One call each, 16 pinned cores, both returned `-inf` because reconstruction without the complete original opacity/runtime options did not reproduce valid support; useful only as an initialization/cost warning, not a valid likelihood benchmark. |
| Exact float64 JAX value/gradient | **not measured** | Process terminated during preparation/JIT before first completed call. |

No GPU was requested because the queue permits one pending script and the exact float64 program had not first passed a CPU smoke test.

### Existing posterior geometry

Raw physical-unit covariance condition numbers are meaningless but requested: **6.93e60** (HAT-P-30) and **2.02e61** (WASP-121), dominated by kg/metre versus dimensionless units. After standardizing each coordinate to unit marginal variance, the correlation-matrix condition numbers are **122.4** and **96.5**. The largest absolute correlations are 0.835 (`offset_nrs1_niriss`, `offset_nrs2_niriss`) and 0.926 (`log_scatt_factor_center`, `log_scatt_factor_contrast_frac`). These are sample-covariance conditions, not MAP Hessian conditions.

| case/parameter | skew | excess kurtosis | two-Gaussian separation | component weights | interpretation |
|---|---:|---:|---:|---:|---|
| HAT-P-30 `log_cloudtop_P` | 2.61 | 7.15 | 3.00 sigma | 0.897 / 0.103 | Strong minor component/tail; clearest marginal multimodality candidate. |
| HAT-P-30 `log_SO2` | -0.80 | -0.64 | 3.07 sigma | 0.398 / 0.602 | Two-component marginal candidate. |
| HAT-P-30 `log_gamma` | -1.45 | 4.30 | 1.25 sigma | 0.296 / 0.704 | Skewed, not resolved bimodality. |
| WASP-121 `spot_cov_frac` | 1.47 | 2.68 | 2.00 sigma | 0.722 / 0.278 | Strong skew/shoulder. |
| WASP-121 `log_gamma_contrast_frac` | -1.59 | 7.40 | 1.75 sigma | 0.356 / 0.644 | Boundary-skewed/long tail. |
| WASP-121 `T_irr_contrast_frac` | -1.12 | 1.42 | 1.90 sigma | 0.371 / 0.629 | Boundary-skewed. |
| WASP-121 `log_TiO` | -0.41 | -0.64 | 2.73 sigma | 0.360 / 0.640 | Broad two-component candidate, not cleanly separated. |

I classify marginal multimodality conservatively: separation about 3 sigma plus non-negligible weights. Thus HAT-P-30 `log_cloudtop_P` and `log_SO2` are the clearest flagged parameters. **No WASP-121 single marginal crosses that conservative threshold**, despite widespread large two-Gaussian BIC improvements. A Gaussian mixture will split bounded, skewed or nearly uniform marginals (notably `T_int`), so BIC alone is not proof of a physical mode. Multivariate/curved modes can also project to unimodal marginals. The chain evidence supports “strongly non-Gaussian and mode-risky,” not an overconfident list of distinct physical modes.

## 2. Prototype and outcome

The prototype works in unconstrained coordinates. Uniform priors use a logit transform and the exact logistic Jacobian; Gaussian priors use standardized normal coordinates. Its target is the JAX spectral likelihood plus transformed prior. It performs three L-BFGS starts, constructs `jax.hessian(potential)` at the best MAP, symmetrizes and floors eigenvalues at `max(1e-8 lambda_max, 1e-8)`, then gives the repaired precision to NumPyro NUTS with mass adaptation off, target acceptance 0.9, depth 8 and 200 warmup steps. It records divergences, leapfrog counts, ESS, approximate gradient evaluations/ESS, and marginal differences from MultiNest.

The HAT-P-30 run initialized opacities and emitted the stellar-coverage and abundance-table warnings, then was terminated without a Python traceback or output JSON during forward/gradient preparation. Because the output is atomic-at-end, there are no partial MAP or timing numbers to misinterpret. Likely causes are memory/resource pressure from tracing the full float64 opacity program on the shared CPU node; this is an inference, not a measured peak-memory diagnosis. The WASP-121 TLS-2D case has a larger state and was deliberately not launched after this failure.

Consequently the requested comparisons (MAP agreement across starts, NUTS medians/sigmas, mode coverage, wall time, gradient evaluations per ESS) are **not available**. Reporting sample-covariance preconditioning as if it were a successful Hessian/NUTS experiment would be misleading.

## 3. Assessment

### Where it can apply

Laplace-metric NUTS remains plausible for a deliberately selected posterior that is unimodal after sensible transforms, has a reproducible multi-start MAP, a positive/reparable local Hessian, and no important curved ridge. It can reduce warmup from full dense-mass adaptation to step-size-only adaptation and could reduce leapfrogs substantially. The right validation gate is several dispersed starts with R-hat/ESS/divergence checks and agreement with the same-input nested posterior, including joint plots—not only medians.

The two examined production posteriors do not establish that regime. HAT-P-30 is simpler than TLS-2D but still has cloud/SO2 non-Gaussianity; WASP-121 is plainly a poor single-Laplace target. A chain initialized in one local basin cannot be credited with covering another basin merely because NUTS is exact within its explored connected region.

### What to use for multimodality

Use multi-start MAPs followed by one chain per distinct basin, with tempered/parallel chains if posterior mass exchange is required. Mode weights cannot be recovered from ordinary independent NUTS counts. If evidence is a science product, retain nested sampling. “Laplace-initialized live points” must be mixed with broad prior live points and validated carefully; concentrating all live points around found MAPs can miss an unknown mode and invalidates the prior-volume exploration that evidence requires.

### Honest wall-time expectation

There is no identical-input successful NUTS timing here, so no measured speedup. The saved HAT-P-30 likelihood arithmetic alone is 5,223 s (1.45 h) across 1.65 million calls. Other logs show runs reaching millions to tens of millions of likelihood evaluations. A successful local NUTS chain requiring, illustratively, 20k--200k gradient calls could reduce call count by one to two orders of magnitude, but a gradient is more expensive than a value and the present float64 compile failed. I would budget **0.5--3 h per well-behaved single-mode retrieval on a suitable GPU only as a planning range**, after a valid benchmark, versus multi-hour/roughly-day PyMultiNest campaigns; this is an engineering estimate, not a measured speedup. For TLS-2D, dispersed/tempered chains and mode validation plausibly return the cost to several hours or more, while still not producing evidence. The user's prior ~10 h NUTS observation is therefore more credible than extrapolating the light-curve 20--60x result.

### Other likely wins

1. Keep the already-jitted opacity/static data resident and reuse compiled executables across likelihood calls and runs; initialization/JIT is material and was the prototype blocker.
2. Batch likelihoods for nested samplers over proposed/live points. The fork already has batched/Pallas plumbing; benchmark batching at the actual proposal batch sizes. This preserves nested-sampling mode coverage and evidence.
3. Do not globally demand float64 inside every opacity interpolation unless fidelity tests show it is necessary. The project constraint here required float64, but the fork intentionally uses float32 forward arithmetic with a higher-accuracy reduction. A scientifically approved mixed-precision boundary could be a much larger win; it is outside this study's allowed prototype.
4. Cache/reuse stellar spectra, opacity bundles and chemistry interpolants, and avoid Python reconstruction in callbacks. The non-JAX CPU calls measured seconds versus millisecond jitted calls.
5. Use posterior-informed reparameterizations (ordered center/contrast, bounded logits, standardized Gaussian physical quantities) before attributing all difficulty to the metric.

## Exact reproduction commands

```bash
cd /project/ekempton/tfairnington/JWST

JAX_PLATFORMS=cpu \
PYTHONPATH=/project/ekempton/tfairnington/retrievals/platon \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  tools/platon_study/analyze_chains.py \
  /project/ekempton/tfairnington/retrievals/platon/tyler_scripts/HAT-P-30_FINAL_NOTLS_1D_GUILLOT/retrieval_result_HAT-P-30_full_run_chemeq_1d_guillot_notls_retrieval.pkl \
  /project/ekempton/tfairnington/retrievals/platon/tyler_scripts/WASP-121_FINAL_TLS_2D_GUILLOT_CLAMP/retrieval_result_WASP-121_full_run_chemeq_2d_guillot_ident_frac_retrieval.pkl \
  --output acceleration_reports/platon_study/chain_diagnostics.json

JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
PYTHONPATH=/project/ekempton/tfairnington/retrievals/platon \
taskset -c 0-15 /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  tools/platon_study/laplace_prototype.py \
  /project/ekempton/tfairnington/retrievals/platon/tyler_scripts/HAT-P-30_FINAL_NOTLS_1D_GUILLOT/retrieval_result_HAT-P-30_full_run_chemeq_1d_guillot_notls_retrieval.pkl \
  --startag HAT-P-30_stellar_spectra_newera.pkl \
  --warmup 200 --samples 300 --starts 3 \
  --output acceleration_reports/platon_study/hatp30_prototype.json

JAX_PLATFORMS=cpu PYTHONPATH=/project/ekempton/tfairnington/retrievals/platon \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m py_compile \
  tools/platon_study/analyze_chains.py tools/platon_study/laplace_prototype.py
```

The second command is the failed attempt described above; absence of `hatp30_prototype.json` is expected for that observed run.

## Open risks

- The saved equal-weight samples can contain duplicated resamples; marginal mixture BIC values should not be treated as independent-sample hypothesis tests.
- The serialized result does not preserve enough provenance to label its instrumented millisecond timings CPU versus GPU.
- The standalone reconstruction needs explicit parity tests for every runtime option (`include_opacities`, T-grid validation/clamping, TLS/limb canonicalization) before any posterior difference is scientifically interpretable.
- Direct x64 promotion may expose dtype assumptions in the forward model and exceed CPU memory; a GPU smoke test should be queued only after adding phase checkpoints/peak-memory telemetry.
- Neither MultiNest evidence nor relative mode masses can be reproduced by ordinary NUTS.
