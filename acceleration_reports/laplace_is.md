# Laplace-centred adaptive importance sampling / IMH backend

## Bottom line

I implemented and integrated a float64, per-channel Laplace/adaptive-importance-sampling backend with PSIS diagnostics, an exact independence-Metropolis-Hastings (IMH) output mode, systematic-resampling mode, and automatic independent-NUTS fallback.

The raw backend was fast on CPU: 17.9x faster than independent NUTS on the real HAT-P-12 SOSS low-resolution dump and 9.1x faster on the real GJ-3470 G395H low-resolution dump with equal 100-warmup/100-returned-draw workloads. These are **not production speedups**: fallback was disabled to isolate the new backend, and the candidate failed the literature-fidelity gate. With production proposal settings, only 9/24 SOSS lanes and 4/5 G395H lanes passed the backend k-hat/IS-ESS/IMH gate, while only 55/216 and 13/40 calibrated site/channel fidelity checks passed, respectively. The default fallback is therefore necessary, and the present method does not meet the project's end-to-end 10x target.

The main scientific failure is proposal mismatch in weakly identified/bounded directions, especially the lower tail of `log_jitter` and the SOSS limb-darkening parameters. The real SOSS optimizations also did not meet the requested gradient tolerance in any lane.

## Files

- `models/laplace_is.py` (new): backend, reusable runner, JAX PSIS, MAP/Hessian proposal, adaptive moment matching, resampling, IMH, gates, JSON diagnostics, and NUTS fallback.
- `tests/test_laplace_is.py` (new): PSIS cross-check, analytic Gaussian recovery, deterministic-site/padding checks, real windowed Power-2 regression against independent NUTS, forced fallback splice, fit dispatch/runner reuse, and option parsing.
- `tools/benchmark_laplace_is.py` (new): identical-input synthetic or dumped-stage benchmark with per-site/per-channel summaries and CPU wall time.
- `tools/compare_laplace_reference.py` (new): per-site/per-channel comparison against pooled joint-NUTS pickle references or another benchmark JSON, including calibrated noise-floor gates.
- `fit_jwst.py` (targeted edits): `spectro_sampler: laplace_is`, low/high-resolution option resolution, chunked and unchunked dispatch, runner caching, checkpoint signatures, validation, and workload source fingerprinting.
- `acceleration_reports/data/laplace_*.json` (new): raw benchmark, diagnostic, and fidelity artifacts. These JSON files contain every per-site/per-channel median, width, q16/q84, ESS, k-hat, acceptance, and rejection-run value; the tables below aggregate them without discarding the channel-level records.

No NUTS backend was removed or weakened. No Slurm job was submitted.

## Implementation

The public entry point is:

```python
get_samples_laplace_is(
    model, key, t, yerr, y, init_params,
    nuts_kwargs=None, mcmc_kwargs=None, diagnostics_path=None,
    lane_width=None, channel_varying_kwargs=(), _runner=None,
    **model_kwargs,
)
```

It returns the same `[draw, channel, ...]` dictionary layout as the existing backends, including latent and deterministic sites. `build_laplace_is_runner(...)` builds a reusable, shape-specialized runner.

The backend reuses the independent-NUTS kwarg partitioning, padding, initialization, unconstrained-transform, static/dynamic-argument, and postprocessing helpers. Per lane it uses JAXopt L-BFGS in NumPyro unconstrained coordinates, an exact `jax.hessian`, eigenvalue-clipped inverse Hessian, and a multivariate Student-t proposal. The production defaults are 80 MAP iterations, Student-t nu=5, scale inflation 1.2, 4,096 importance draws, two moment-matching rounds, draw chunks of 16, and IMH output.

Importance draws are evaluated with the potential function only. The draw axis is statically chunked with `jax.lax.map` to control the `draw x lane x cadence` working set. MAP/Hessian lanes use `lax.map` with static batches of at most four lanes because compiling a full-width 40-lane optimizer/Hessian graph exhausted login-node memory in the first attempt.

PSIS is implemented in JAX using the same tail length and Zhang-Stephens empirical-Bayes GPD fit used by ArviZ. Moment matching uses Pareto-smoothed normalized weights. The IMH kernel draws from the final proposal and reports acceptance plus the longest rejection run; its post-warmup stationary distribution is the exact target. Systematic resampling is available as an explicitly approximate alternative.

Default per-lane acceptance is:

```text
k-hat < 0.7
IS-ESS >= max(400, 0.2 * importance_draws)
IMH acceptance >= 0.2
all returned unconstrained values finite
```

Failed lanes are rerun by `models.independent_nuts.get_samples_independent` at the same padded lane width, and all sites are spliced back at their original channel indices. Fallback is enabled by default and switchable for benchmarking. Diagnostics include convergence, k-hat, IS-ESS, IMH acceptance, maximum rejection run, gradient norm, Hessian condition number, MAP iterations, gate/fallback masks, and synchronized stage timings.

`fit_jwst.py` resolves generic `laplace_is_*`, `spectro_laplace_is_*`, and stage-specific `lowres_laplace_is_*`/`highres_laplace_is_*` options. The stage-specific value wins. Supported controls are `output`, `num_draws`, `rounds`, `draw_chunk_size`, `student_df`, `scale_inflation`, `map_maxiter`, `map_tol`, `khat_threshold`, `min_ess`, `min_ess_fraction`, `min_imh_acceptance`, and `fallback`.

## Tests

Exact command:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 0-15 \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest \
tests/test_laplace_is.py tests/test_fit_independent_routing.py \
tests/test_spectro_safety_guards.py -x -q
```

Final result for the complete new test file: `6 passed, 2 warnings in 131.35s`. A broader compatibility run of `tests/test_laplace_is.py`, `tests/test_fit_independent_routing.py`, and `tests/test_spectro_safety_guards.py` before adding the last two lightweight assertions produced `13 passed, 2 warnings in 115.47s`. The added dispatch/runner-cache, string-boolean parsing, and PSIS tests were also isolated with:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 0-15 \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest \
tests/test_laplace_is.py::test_fit_chunked_laplace_dispatch_reuses_runner \
tests/test_laplace_is.py::test_laplace_stage_options_parse_string_boolean \
tests/test_laplace_is.py::test_psis_matches_arviz_random_weights -q
```

Result: `3 passed, 2 warnings in 9.91s`.

The PSIS test agrees with `arviz.psislw` at `rtol=atol=1e-12`. The analytic Gaussian test covers a padded 3-active/4-resident-lane call and checks the deterministic `twice_x` site. The real-model test constructs the actual windowed, informed Power-2, linear-trend model and compares `rors`, `c1`, `c2`, and `total_error` against independent NUTS. The fallback test forces every gate to fail and verifies that all latent/deterministic values are replaced at the correct channel positions.

All changed Python files also passed `py_compile`:

```bash
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m py_compile \
models/laplace_is.py tests/test_laplace_is.py \
tools/benchmark_laplace_is.py tools/compare_laplace_reference.py fit_jwst.py
```

## CPU study design

All wall times include tracing/compilation, optimization, sampling, and postprocessing in a fresh Python process. Runs used float64 and the requested 16-core/8-thread-limited CPU environment. Fallback was disabled only in timing/fidelity runs so that the new algorithm could be measured separately; this is not the production default.

The synthetic case is the production-shaped problem from `tools.benchmark_independent_nuts_gpu.make_problem`: 40 channels, 230 cadences, informed Power-2 limb darkening, linear trend, approximately 1% transit depths, and realistic channel noise. Real tests use the replayable HAT-P-12 SOSS R20 low-resolution dump (24 channels, 226 cadences, 95 active-window cadences) and GJ-3470 G395H R20 low-resolution dump (5 channels, 2,058 cadences, 751 active-window cadences). Fidelity references are the supplied pooled joint-NUTS files with 3,000 draws and their calibrated between-seed noise-floor gates.

The requested 500/500 all-method synthetic command was attempted first, but the original full-width MAP/Hessian compilation was killed with exit 137 after about 31 minutes and produced no timing artifact. That failure led to the four-lane static optimizer/Hessian batching. A later independent-NUTS 500/500 attempt exceeded the approximately 40-minute limit and was stopped without claiming a timing. The complete equal-work comparisons below therefore use 100 warmup and 100 returned draws. A 500-returned-draw Laplace-only synthetic run is also recorded, but it is not used for a cross-method speed claim.

## Timing and sampler diagnostics

### Identical-input wall time

| Workload | Sampler/configuration | Warmup / returned | Wall (s) | Raw speed vs independent NUTS | Median chain ESS | Min chain ESS | Sum ESS/s |
|---|---|---:|---:|---:|---:|---:|---:|
| synthetic 40 x 230 | Laplace-IS, 1,024/1 round | 100 / 100 | 113.135 | 11.31x | NaN | NaN | NaN |
| synthetic 40 x 230 | independent dense NUTS | 100 / 100 | 1,279.495 | 1.00x | 60.98 | 5.26 | 17.62 |
| synthetic 40 x 230 | joint diagonal NUTS | 100 / 100 | 1,455.077 | 0.88x | 111.31 | 3.50 | 25.27 |
| SOSS R20 real | Laplace-IS, **4,096/2 default** | 100 / 100 | 111.904 | **17.92x raw** | 20.89 | 5.50 | 47.59 |
| SOSS R20 real | independent dense NUTS | 100 / 100 | 2,004.742 | 1.00x | 56.93 | 5.50 | 7.16 |
| G395H R20 real | Laplace-IS, **4,096/2 default** | 100 / 100 | 188.314 | **9.09x raw** | 31.59 | 10.52 | 7.22 |
| G395H R20 real | independent dense NUTS | 100 / 100 | 1,711.823 | 1.00x | 75.94 | 11.55 | 1.67 |

The synthetic Laplace ESS aggregate is NaN because at least one 100-draw IMH lane was completely stuck (acceptance 0, rejection run 100), which propagates through NumPyro's ESS summary. This is a sampler failure, not a reporting omission.

Raw speed divides identical-input independent-NUTS wall time by Laplace-only wall time. It excludes the NUTS fallback demanded by the gates, so it must not be interpreted as an end-to-end pipeline speedup. Joint-NUTS wall time is available for the synthetic case. The real pooled joint-NUTS artifacts did not include a reproducible wall-time field, so I use them only for fidelity and do not invent a real-data joint speed comparison.

### Production-default proposal diagnostics on real dumps

| Workload | k-hat min / med / max | IS-ESS min / med / max | IMH accept min / med / max | Max rejection run min / med / max | MAP grad norm min / med / max | Hessian cond. min / med / max | Backend gate |
|---|---|---|---|---|---|---|---:|
| SOSS R20 | 0.276 / 0.643 / 0.990 | 153 / 707 / 1,893 | 0.05 / 0.345 / 0.45 | 5 / 14.5 / 79 | 8.01e-4 / 11.24 / 171.75 | 1.42e6 / 2.14e7 / 3.94e7 | 9/24 |
| G395H R20 | 0.333 / 0.473 / 0.866 | 371 / 2,173 / 2,499 | 0.37 / 0.54 / 0.67 | 5 / 8 / 11 | 5.23e-3 / 2.19e-2 / 7.80e-2 | 9.27e5 / 1.22e7 / 2.59e7 | 4/5 |

No real-data lane met the strict `1e-5` MAP gradient-norm convergence flag within 80 iterations. This is especially serious for SOSS, where the median final norm was 11.24. The PSIS/IMH gate intentionally follows the requested k-hat/ESS/acceptance definition rather than silently adding convergence, but the disagreement shows that convergence should be treated as an additional safety signal before deployment.

The production-default importance settings improved G395H relative to the reduced benchmark: median k-hat changed from 0.599 to 0.473 and median IS-ESS from 482 to 2,173. They did not make the result literature-grade.

## Fidelity against pooled joint NUTS

The entries below are aggregates across all channels/components. “Shift” is `|candidate median - reference median| / sigma_reference`; q16/q84 columns are the p95 absolute standardized quantile shifts. Full arrays, including every channel, are in:

- `acceleration_reports/data/laplace_real_soss_lowres_default_vs_pooled_joint.json`
- `acceleration_reports/data/laplace_real_g395h_lowres_default_vs_pooled_joint.json`

### HAT-P-12 SOSS R20, production proposal defaults

| site | median shift | p95 shift | max shift | sigma ratio median [min,max] | abs(q16 shift) p95 | abs(q84 shift) p95 |
|---|---:|---:|---:|---:|---:|---:|
| A_spot | 0.267 | 0.689 | 0.939 | 0.959 [0.596,1.318] | 0.726 | 0.790 |
| c | 0.278 | 0.601 | 0.642 | 1.018 [0.109,1.412] | 1.140 | 0.627 |
| c1 | 0.176 | 0.459 | 1.237 | 0.729 [0.297,1.072] | 0.530 | 0.858 |
| c2 | 0.139 | 0.604 | 1.619 | 0.735 [0.370,1.409] | 0.794 | 0.785 |
| depths | 0.202 | 0.667 | 0.815 | 0.978 [0.368,1.304] | 0.578 | 0.822 |
| log_jitter | 0.164 | 0.346 | 0.888 | 0.925 [0.261,1.359] | 1.276 | 0.533 |
| rors | 0.202 | 0.667 | 0.816 | 0.978 [0.368,1.303] | 0.579 | 0.821 |
| total_error | 0.207 | 0.582 | 0.818 | 0.911 [0.323,1.215] | 0.963 | 0.667 |
| v | 0.314 | 0.811 | 1.070 | 0.962 [0.391,1.096] | 0.441 | 0.794 |

Only **55/216** supplied calibrated site/channel gates passed. Several lanes severely underestimated widths: minimum sigma ratios were 0.297 for `c1`, 0.370 for `c2`, 0.368 for `rors`, and 0.261 for `log_jitter`. This fails the requested 0.1-sigma-level fidelity decisively.

### GJ-3470 G395H R20, production proposal defaults

| site | median shift | p95 shift | max shift | sigma ratio median [min,max] | abs(q16 shift) p95 | abs(q84 shift) p95 |
|---|---:|---:|---:|---:|---:|---:|
| c | 0.168 | 0.253 | 0.272 | 0.985 [0.874,1.057] | 0.299 | 0.603 |
| c1 | 0.169 | 0.227 | 0.238 | 1.105 [0.858,1.161] | 0.550 | 0.216 |
| c2 | 0.205 | 0.493 | 0.534 | 1.072 [0.862,1.100] | 0.240 | 0.474 |
| depths | 0.169 | 0.405 | 0.438 | 0.963 [0.816,1.074] | 0.494 | 0.282 |
| log_jitter | 0.170 | 0.291 | 0.307 | 0.927 [0.694,1.094] | 0.608 | 0.342 |
| rors | 0.169 | 0.405 | 0.438 | 0.963 [0.816,1.074] | 0.494 | 0.282 |
| total_error | 0.034 | 0.546 | 0.654 | 1.025 [0.553,1.236] | 0.028 | 1.020 |
| v | 0.286 | 0.450 | 0.487 | 0.958 [0.709,1.029] | 0.479 | 0.604 |

Only **13/40** supplied calibrated site/channel gates passed. G395H is substantially better behaved than SOSS and four of five backend diagnostic gates passed, but typical median shifts remain 0.17-0.29 reference sigma rather than below 0.1 sigma.

### Synthetic 40-channel comparison

The completed synthetic comparison used the reduced 1,024-draw/one-round proposal and 100/100 independent NUTS. Per-channel arrays are in `laplace_synth40_vs_independent_nuts_100.json`. Aggregate median shifts were 0.163 (`c`), 0.235 (`c1`), 0.298 (`c2`), 0.306 (`log_jitter`), 0.348 (`rors`), and 0.257 (`v`) reference sigma. The `log_jitter` median width ratio was only 0.613. One channel had zero IMH acceptance, so this run also failed fidelity/chain-quality requirements. The Laplace-only 500/500 run took 125.407 s, but no completed identical 500/500 reference exists and no speedup is claimed from it.

## Explicit `log_jitter` assessment

The Student-t plus two adaptive moment-matching rounds is not sufficient for the prior-bound plateau in all channels.

- SOSS: median standardized median shift 0.164; p95 0.346; median sigma ratio 0.925 but range 0.261-1.359; p95 lower-quantile shift 1.276 sigma.
- G395H: median standardized median shift 0.170; p95 0.291; median sigma ratio 0.927 with range 0.694-1.094; p95 lower-quantile shift 0.608 sigma.
- Reduced synthetic run: median standardized shift 0.306 and median sigma ratio 0.613.

The asymmetric q16 error and occasional severe width collapse are consistent with a broad, bounded lower-tail plateau that a single local Hessian-centred elliptical proposal does not represent. A dedicated one-dimensional treatment is needed before this can replace NUTS: either a mixture component spanning the unconstrained prior plateau, or a profile/conditional marginal proposal for `log_jitter`, followed by the same PSIS and IMH checks. Merely increasing the Student-t scale would also degrade the well-identified directions and is not a sufficient fix.

## Exact benchmark commands

Common prefix used for every long run:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 0-15 \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python
```

Synthetic reduced Laplace, independent NUTS, and joint NUTS:

```bash
PREFIX tools/benchmark_laplace_is.py --channels 40 --cadences 230 \
  --warmup 100 --samples 100 --methods laplace_is \
  --laplace-draws 1024 --laplace-rounds 1 --draw-chunk-size 16 \
  --map-maxiter 80 --disable-fallback \
  --json acceleration_reports/data/laplace_synth40_laplace_100_reduced.json

PREFIX tools/benchmark_laplace_is.py --channels 40 --cadences 230 \
  --warmup 100 --samples 100 --methods independent_nuts \
  --json acceleration_reports/data/laplace_synth40_independent_nuts_100.json

PREFIX tools/benchmark_laplace_is.py --channels 40 --cadences 230 \
  --warmup 100 --samples 100 --methods joint_nuts \
  --json acceleration_reports/data/laplace_synth40_joint_nuts_100.json
```

Here `PREFIX` denotes the literal common prefix above; it is shown only to keep the commands readable.

Real production-default Laplace runs:

```bash
PREFIX tools/benchmark_laplace_is.py \
  --dump /scratch/midway3/tfairnington/accel_stage_inputs/HAT-P-12_NIRISS_SOSS_order1_R20_low_resolution_inputs.pkl \
  --warmup 100 --samples 100 --methods laplace_is \
  --laplace-draws 4096 --laplace-rounds 2 --draw-chunk-size 16 \
  --map-maxiter 80 --disable-fallback \
  --json acceleration_reports/data/laplace_real_soss_lowres_laplace_100_default.json

PREFIX tools/benchmark_laplace_is.py \
  --dump /scratch/midway3/tfairnington/accel_stage_inputs/GJ-3470_NIRSPEC_G395H_nrs1_R20_low_resolution_inputs.pkl \
  --warmup 100 --samples 100 --methods laplace_is \
  --laplace-draws 4096 --laplace-rounds 2 --draw-chunk-size 8 \
  --map-maxiter 80 --disable-fallback \
  --json acceleration_reports/data/laplace_real_g395h_lowres_laplace_100_default.json
```

Real independent-NUTS timing runs:

```bash
PREFIX tools/benchmark_laplace_is.py \
  --dump /scratch/midway3/tfairnington/accel_stage_inputs/HAT-P-12_NIRISS_SOSS_order1_R20_low_resolution_inputs.pkl \
  --warmup 100 --samples 100 --methods independent_nuts \
  --json acceleration_reports/data/laplace_real_soss_lowres_independent_nuts_100.json

PREFIX tools/benchmark_laplace_is.py \
  --dump /scratch/midway3/tfairnington/accel_stage_inputs/GJ-3470_NIRSPEC_G395H_nrs1_R20_low_resolution_inputs.pkl \
  --warmup 100 --samples 100 --methods independent_nuts \
  --json acceleration_reports/data/laplace_real_g395h_lowres_independent_nuts_100.json
```

Reference comparisons:

```bash
JAX_PLATFORMS=cpu /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
tools/compare_laplace_reference.py \
  --candidate-json acceleration_reports/data/laplace_real_soss_lowres_laplace_100_default.json \
  --reference-pkl /scratch/midway3/tfairnington/accel_stage_inputs/references/HAT-P-12_NIRISS_SOSS_order1_R20_low_resolution_inputs_ch0_24_pooled_joint_nuts.pkl \
  --noise-floor-json /scratch/midway3/tfairnington/accel_stage_inputs/references/HAT-P-12_NIRISS_SOSS_order1_R20_low_resolution_inputs_ch0_24_pooled_joint_nuts_noise_floor.json \
  --output acceleration_reports/data/laplace_real_soss_lowres_default_vs_pooled_joint.json

JAX_PLATFORMS=cpu /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
tools/compare_laplace_reference.py \
  --candidate-json acceleration_reports/data/laplace_real_g395h_lowres_laplace_100_default.json \
  --reference-pkl /scratch/midway3/tfairnington/accel_stage_inputs/references/GJ-3470_NIRSPEC_G395H_nrs1_R20_low_resolution_inputs_ch0_5_pooled_joint_nuts.pkl \
  --noise-floor-json /scratch/midway3/tfairnington/accel_stage_inputs/references/GJ-3470_NIRSPEC_G395H_nrs1_R20_low_resolution_inputs_ch0_5_pooled_joint_nuts_noise_floor.json \
  --output acceleration_reports/data/laplace_real_g395h_lowres_default_vs_pooled_joint.json
```

## Failures and open risks

1. **The scientific gate failed.** Backend diagnostics accepted 9/24 SOSS and 4/5 G395H lanes, but the independent calibrated fidelity checks accepted only 55/216 and 13/40 site/channel values. k-hat/ESS/acceptance alone do not certify the required 0.1-sigma fidelity at these short chain lengths.
2. **Production fallback speed is unmeasured and likely far below the raw speed.** The default path would rerun 15 SOSS lanes and one G395H lane through independent NUTS. The reported 17.9x/9.1x values have fallback disabled and cannot be advertised as pipeline gains.
3. **MAP convergence is poor on real inputs.** All 29 real lanes hit the 80-iteration budget without satisfying `||grad|| <= 1e-5`; SOSS norms were sometimes O(100). A more robust box-aware/preconditioned optimizer, better initialization, or a higher budget is needed. Increasing iterations trades away speed.
4. **`log_jitter` needs a nonlocal proposal.** Its bounded plateau is not well described by a local Hessian and causes lower-tail/width errors. A mixture or profile-based one-dimensional component is the clearest next experiment.
5. **SOSS limb darkening is underdispersed.** Median width ratios for `c1`/`c2` are 0.729/0.735, with minima 0.297/0.370. A single elliptical proposal does not capture those correlations/boundaries reliably.
6. **Short-chain ESS is noisy.** Timing comparisons use only 100 retained draws because 500/500 CPU NUTS exceeded the practical limit. ESS/s is included where finite but should not be treated as a production estimate.
7. **GPU performance and memory are unmeasured.** No Slurm submission was permitted. Draw-axis chunking and four-lane optimizer/Hessian batching bound static shapes, but the 40-lane/16,515-active-cadence PRISM case still needs an orchestrator-submitted GPU memory test.
8. **Exactness differs by output mode.** IMH is the production default and is exact after convergence. `resample` is approximate and must retain PSIS gating; it should not be used as an exact MCMC substitute.
9. **Timing fields are batch timings.** JSON keeps arrays channel-aligned by repeating synchronized batch-stage wall time per active lane. They must not be summed across lanes.
10. **JAXopt maintenance risk.** JAXopt 0.8-compatible APIs work in the supplied environment, but importing it emits its upstream “no longer maintained” deprecation warning.

## Recommendation

Keep `laplace_is` available as an experimental, guarded backend with fallback enabled, but do not make it the production default. The next technically justified iteration is a mixture proposal that treats `log_jitter` nonlocally and a stronger MAP optimizer/convergence gate, followed by the same pooled-reference tests. Until those pass, independent/joint NUTS remains the fidelity reference and no end-to-end acceleration claim is warranted.
