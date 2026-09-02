# Laplace-preconditioned independent NUTS: production follow-up

## Outcome

I implemented the requested opt-in Laplace-mass path without changing any
existing default.  It runs a fixed-budget, JAX-only, vmapped per-lane Newton
search, evaluates the exact unconstrained Hessian, repairs its spectrum, and
uses the inverse Hessian as NumPyro's dense `inverse_mass_matrix` while adapting
only step size.  The adaptive independent-NUTS path is unchanged.

The production measurements do **not** support enabling this configuration for
all spectroscopic fits:

- Both real G395H cases passed every calibrated science coordinate: 240/240
  for high resolution and 30/30 for low resolution.  High-resolution NUTS used
  8.86 lane steps/draw on average, had zero divergences, and took 106.61 s for
  the compile-inclusive first V100 chunk versus 238.00 s for same-prior joint
  NUTS (2.23x).  Its measured compile-excluded residual was 7.95x smaller.
- Both SOSS cases failed the fidelity/divergence gate.  High resolution had 26
  divergences, limb-darkening ESS minima of 5, and only 267/280 calibrated
  science coordinates passed.  Low resolution had 18 divergences and 162/168
  science coordinates passed.  These speed measurements are diagnostic, not
  deployable speedups.
- Compilation grew from the existing adaptive independent runner's 45.47 s to
  about 77--80 s because differentiating the transit potential through the
  exact Hessian is expensive.  I fused the new path into two top-level JITs
  (MAP/Hessian/init/warmup and sample/postprocess), but the first-order compile
  cost remains.

The practical recommendation is therefore an opt-in G395H configuration, with
SOSS staying on joint NUTS (or the existing adaptive independent path) until a
nonlocal metric or longer effective chain passes its calibrated gates.  This
does not reach the project's 10x end-to-end target.

## Files built or changed

| File | Change |
|---|---|
| `models/independent_nuts.py` | Added `mass_matrix=adaptive|laplace`, fixed-budget vmapped MAP, exact Hessian, eigenvalue repair, fixed dense mass, step-only adaptation, fused two-program execution, MAP diagnostics, and reusable-runner support. |
| `fit_jwst.py` | Minimally added spectroscopic mass-matrix and Laplace warmup/target/depth/start flag resolution and validation. |
| `tests/test_independent_nuts.py` | Added option, MAP-diagnostic, pipeline-resolution, and no-stale-data compiled-runner reuse tests. |
| `tools/run_sampler_on_stage_inputs.py` | Added repeatable `--nuts-override`, MAP diagnostic capture, effective-warmup metadata, and ESS sites `A_spot`/`total_error`. |
| `tools/diag_nuts/run_precond_gpu.sh` | Reproducible four-stage candidate driver (and optional PRISM case). |
| `tools/diag_nuts/run_joint_lognormal_gpu.sh` | Same-prior joint baseline driver. |
| `tools/diag_nuts/summarize_precond.py` | Calibrated pooled-reference gates, per-site ESS, lockstep, divergence, and MAP summaries. |
| `acceleration_reports/diag_nuts/*precond*.json` | Machine-readable calibrated comparisons and sampler summaries. |

I did not edit `models/laplace_is.py`, `models/jaxoplanet/builder.py`, or any
transit/likelihood implementation.

## Implementation details

The Laplace path uses the model's NumPyro unconstrained PyTree ordering.  For
each lane it:

1. ravels the initialized unconstrained state;
2. performs 16 fixed Newton/trust iterations with a unit trust radius, eight
   backtracking scales, and a normalized-gradient fallback;
3. computes `jax.hessian(potential)` at the final accepted point;
4. symmetrizes the Hessian and floors eigenvalues at
   `1e-8 * max(max(abs(eigenvalue)), 1)`;
5. passes the repaired inverse Hessian (posterior covariance) as NumPyro's
   dense `inverse_mass_matrix`, with `adapt_mass_matrix=False`;
6. adapts only step size for `laplace_warmup` iterations, then samples.

All Newton iterations and line-search candidates have static shapes and are
vmapped across lanes.  The diagnostics contain per-lane gradient norm, Newton
decrement, iteration count, repaired condition number, and raw minimum Hessian
eigenvalue.  `laplace_start_at_map=False` retains the recorded physical initial
state for HMC; the MAP is still used for the metric.

The production candidate was:

```yaml
spectro_sampler: independent_nuts
spectro_mass_matrix: laplace
spectro_jitter_prior: lognormal
spectro_laplace_warmup: 150
spectro_laplace_target_accept: 0.95
spectro_laplace_max_tree_depth: 10
spectro_laplace_start_at_map: false
```

The global defaults remain `spectro_sampler=joint_nuts`,
`spectro_mass_matrix=adaptive`, and `spectro_jitter_prior=log_uniform`.

## CPU validation

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 0-15 \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest \
  tests/test_independent_nuts.py tests/test_independent_runner_reuse.py -x -q
```

Result: **14 passed** in 165.69 s.  A second focused run after the final edits
gave **9 passed** in 88.36 s for `tests/test_independent_nuts.py`.

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 0-15 \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest \
  tests/test_stage_inputs_dump.py -x -q
```

Result: **3 passed** in 14.57 s.  The new reuse test runs two different data,
prior, and initialization payloads through one Laplace runner, compares the
second result bit-for-bit with a fresh runner under the same key, and confirms
`program_build_count == 1`.

The real one-channel CPU smoke converged in 13/16 iterations with gradient norm
`5.61e-5`, Newton decrement `1.01e-6`, Hessian minimum eigenvalue 0.5, condition
number `1.50e6`, and no divergence.

## GPU protocol and exact commands

Jobs were submitted only through the documented file queue.  A missing `.gpu`
marker identifies the V100; RTX 6000 runs were used only as accuracy checks and
are excluded from timing comparisons.  The V100 candidate queue jobs were
`50_soss_high_precond`, `53_g395h_low_precond`,
`58_g395h_high_precond_v100_retry`, and `59_soss_low_precond_v100`.

The durable driver command for each required stage is:

```bash
bash tools/diag_nuts/run_precond_gpu.sh soss_high
bash tools/diag_nuts/run_precond_gpu.sh soss_low
bash tools/diag_nuts/run_precond_gpu.sh g395h_high
bash tools/diag_nuts/run_precond_gpu.sh g395h_low
```

The script expands to `tools/run_sampler_on_stage_inputs.py` with 1000 returned
draws, `--builder-override jitter_prior=lognormal`,
`--nuts-override mass_matrix=laplace`, warmup 150, target 0.95, depth 10, and
MAP start false.  Although `--warmup 1000` preserves the recorded MCMC payload,
the timing JSON records `effective_warmup: 150` for this path.

The same-prior joint baselines use:

```bash
bash tools/diag_nuts/run_joint_lognormal_gpu.sh g395h_low
bash tools/diag_nuts/run_joint_lognormal_gpu.sh soss_high
bash tools/diag_nuts/run_joint_lognormal_gpu.sh g395h_high
```

An exact comparison command (substitute the stage stems for the other cases)
is:

```bash
OMP_NUM_THREADS=8 taskset -c 0-15 \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  tools/diag_nuts/summarize_precond.py \
  --candidate /scratch/midway3/tfairnington/accel_gpu_results/precond/g395h_high_laplace_map16_w150_d10_ta095_v100.pkl \
  --reference /scratch/midway3/tfairnington/accel_stage_inputs/references/GJ-3470_NIRSPEC_G395H_nrs1_R300_high_resolution_inputs_ch0_40_pooled_joint_nuts.pkl \
  --noise-floor /scratch/midway3/tfairnington/accel_stage_inputs/references/GJ-3470_NIRSPEC_G395H_nrs1_R300_high_resolution_inputs_ch0_40_pooled_joint_nuts_noise_floor.json \
  --diagnostics /scratch/midway3/tfairnington/accel_gpu_results/precond/g395h_high_laplace_map16_w150_d10_ta095_v100.diagnostics.npz \
  --end 40 \
  --output acceleration_reports/diag_nuts/g395h_high_precond_v100_vs_pooled.json
```

## Same-GPU timing

All rows return 1000 draws and use the lognormal jitter prior.  `Compile` is
JAX's recorded compilation duration; `residual` is `chunk wall - compile`, not
a separately instrumented kernel timer.  First-chunk speedups are direct V100
wall ratios on identical channel ranges.  Residual ratios quantify the
compile-amortized ceiling and are shown separately.

| Stage/range | Joint first wall | Joint compile | Laplace first wall | Laplace compile | First-chunk speedup | Residual ratio |
|---|---:|---:|---:|---:|---:|---:|
| SOSS high 0:40 | 200.80 s | 22.94 s | 117.50 s | 80.22 s | **1.71x** | 4.77x |
| SOSS low 0:24 | 183.35 s | 17.03 s | 109.54 s | 76.62 s | **1.67x** | 5.05x |
| G395H high 0:40 | 238.00 s | 22.10 s | 106.61 s | 79.44 s | **2.23x** | 7.95x |
| G395H low 0:5 | 187.22 s | 16.81 s | 90.06 s | 77.77 s | **2.08x** | 13.86x |

These equal-input lognormal baselines differ from the supplied older V100
numbers (217, 200, 269, and 212 s), which used the recorded prior.  I do not use
cross-prior or RTX/V100 ratios as speedups.

The high-resolution candidate runs covered the full stage to exercise cache
reuse:

| Stage | Candidate chunks (wall; compile) | Candidate full-stage wall | Same-prior joint full-stage wall | Full-stage ratio |
|---|---|---:|---:|---:|
| SOSS high, 118 ch | 0:40 117.50;80.22, 40:80 42.38;0.00, 80:118 14.06;0.74 s | **174.10 s measured** | **483.57 s measured** | **2.78x measured** |
| G395H high, 74 ch | 0:40 106.61;79.44, 40:74 24.21;0.79 s | **130.96 s measured** | 476.00 s projected as two measured 0:40 first-chunk costs | 3.63x projected |

The second equal-width SOSS chunk gives a direct cached-width comparison on the
same channels: joint NUTS took 171.03 s (8.00 s recorded compile) and the
Laplace runner took 42.38 s with no compilation, a **4.04x compile-inclusive
steady-chunk speedup**.  Removing the recorded joint compile component gives a
3.85x residual ratio.  The final padded candidate chunks triggered only
0.74--0.79 s of small helper compilation; the runner itself stayed at one
build.  For G395H, comparing the measured joint 0:40 residual with its observed
34-real-lane padded subsequent chunk gives a 9.22x stage-specific residual
ratio (`215.90/(24.21-0.79)`), not a generic equal-width claim.

The candidate has only two explicit JIT boundaries rather than the adaptive
runner's four, but the transit/Hessian trace still reported 329 compile events.
On SOSS high the existing adaptive lognormal run was 185.50 s including 45.47 s
compile; the Laplace candidate was 117.50 s including 80.22 s compile.  Thus the
new sampler is faster in total but did **not** reduce first-use compile time.

## Sampler accounting

`Lane max` is the mean, over draws, of the maximum `num_steps` across lanes.

| Stage/range | Lane mean steps | Lane-max steps | Lockstep | Accept | Divergences |
|---|---:|---:|---:|---:|---:|
| SOSS high 0:40 | 25.64 | 65.77 | 2.56x | 0.953 | **26** |
| SOSS low 0:24 | 23.72 | 63.67 | 2.68x | 0.942 | **18** |
| G395H high 0:40 | 8.86 | 17.98 | 2.03x | 0.957 | 0 |
| G395H low 0:5 | 9.81 | 15.10 | 1.54x | 0.957 | **1** |

### ESS bulk, minimum / median across channels and components

| Site | SOSS high | SOSS low | G395H high | G395H low |
|---|---:|---:|---:|---:|
| `A_spot` | 394 / 643 | 285 / 559 | -- | -- |
| `c` | 609 / 1054 | 618 / 1028 | 881 / 1240 | 848 / 1153 |
| `v` | 457 / 812 | 484 / 790 | 867 / 1209 | 997 / 1172 |
| `c1` | **5 / 88** | **15 / 108** | 950 / 1273 | 988 / 1280 |
| `c2` | **5 / 86** | **16 / 107** | 729 / 1258 | 808 / 1070 |
| `rors` / `depths` | 248 / 934 | 612 / 987 | 735 / 1285 | 833 / 1204 |
| `log_jitter` / `total_error` | 55 / 361 | 27 / 883 | 265 / 607 | 164 / 470 |

The SOSS failure is not a small calibrated-gate fluctuation: the wide-Gaussian
limb-darkening ridge has single-digit ESS in the high-resolution run.

## MAP diagnostics

| Stage/range | Gradient norm med / max | Newton decrement med / max | Condition med / max |
|---|---:|---:|---:|
| SOSS high 0:40 | 0.00109 / 0.593 | `3.17e-5` / 0.195 | `9.62e6` / `1.42e7` |
| SOSS low 0:24 | 0.00920 / 52.84 | `1.93e-5` / 0.0454 | `2.15e7` / `3.92e7` |
| G395H high 0:40 | 0.145 / 0.202 | 0.111 / 0.153 | `3.30e6` / `4.27e6` |
| G395H low 0:5 | 0.00898 / 0.0369 | 0.00136 / 0.00173 | `3.44e7` / `8.10e7` |

The fixed budget is deliberately exposed by these diagnostics.  Several lanes
do not meet a strict optimizer convergence criterion even though G395H's metric
produces excellent ESS and exact science agreement.  Production automation
should retain and inspect these diagnostics rather than assuming every local
metric is reliable.

## Calibrated fidelity gates

The gate is applied coordinate by coordinate using each stage's pooled
three-seed joint-NUTS reference and its calibrated site-class limits.

| Stage/range | Science passed | All sites passed | Result |
|---|---:|---:|---|
| SOSS high 0:40 | 267/280 | 324/360 | **fail** |
| SOSS low 0:24 | 162/168 | 208/216 | **fail** |
| G395H high 0:40 | **240/240** | 244/320 | science pass; noise sites changed by prior |
| G395H low 0:5 | **30/30** | 30/40 | science pass; noise sites changed by prior |

For the requested separation of prior and sampler effects, the plain SOSS-high
joint-NUTS lognormal run at
`/scratch/midway3/tfairnington/accel_gpu_results/40_soss_high_lognormal_joint`
passes **280/280 science coordinates** and 339/360 overall.  Its maximum
science median shifts are 0.110 (`A_spot`), 0.096 (`c`), 0.087 (`c1`), 0.123
(`c2`), 0.123 (`depths/rors`), and 0.097 (`v`) reference sigma.  The only
failures are 8/40 `log_jitter` and 13/40 `total_error` coordinates.  Therefore
the science insensitivity to the lognormal prior is confirmed; the additional
SOSS failures in the Laplace run come from sampler efficiency/divergences, not
from the prior.

Machine-readable results:

- `acceleration_reports/diag_nuts/soss_high_lognormal_joint_vs_pooled.json`
- `acceleration_reports/diag_nuts/soss_high_precond_vs_pooled.json`
- `acceleration_reports/diag_nuts/soss_low_precond_v100_vs_pooled.json`
- `acceleration_reports/diag_nuts/g395h_high_precond_v100_vs_pooled.json`
- `acceleration_reports/diag_nuts/g395h_low_precond_vs_pooled.json`

## Failed variants and open risks

- On identical SOSS-high 0:40 inputs, increasing the MAP budget to 32 while
  retaining the recorded start took 118.82 s, produced 41 divergences, and
  passed only 262/280 science coordinates.  Starting at that better-converged
  MAP took 137.16 s on the V100, produced 66 divergences, and passed 260/280.
  These are worse than the 16-iteration/recorded-start candidate, so neither is
  the production default.
- Local Hessian preconditioning is reliable for the informed-LD linear G395H
  model but not for the SOSS wide-Gaussian-LD plus spot-amplitude posterior.
  A local covariance does not straighten that nonlocal ridge.  This is the main
  unresolved algorithmic risk.
- The G395H-low run has one divergence in 5000 lane-draws.  Its science gate and
  ESS are strong, but a strict zero-divergence deployment policy should rerun a
  second seed or raise target acceptance slightly.
- The calibrated references use the old log-uniform jitter prior.  Science
  coordinates are directly tested and pass for G395H; `log_jitter` and
  `total_error` are expected to differ and are reported rather than hidden.
- Candidate accuracy is one seed against a pooled three-seed reference.  The
  gates account for the measured reference noise floor, but more candidate
  seeds would still be useful before a default change.
- PRISM low 0:21 was not queued.  The required four measurements and the failed
  SOSS fidelity investigation took priority; no PRISM speedup is claimed.
- The compile-duration listener sums reported JAX phases.  `wall - compile` is
  useful accounting but is not an independently timed sampling region.

## Recommended production use

Enable the configuration above only for G395H after retaining the emitted MAP
and sampler diagnostics.  Keep SOSS on joint NUTS for literature-grade output.
For G395H high, the measured full-stage candidate is 130.96 s and the calibrated
science gate is 240/240; this is the strongest deployable result.  Do not use a
depth cap below 10 or MAP start: the accepted diagnosis already showed more
divergences for those settings, and the real SOSS MAP-start test worsened again.

The result is a useful 2.08--2.23x compile-inclusive and 7.95--13.86x
compile-amortized improvement on G395H, not a universal 10x pipeline speedup.
