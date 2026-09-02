# PRISM Laplace preconditioning

## Outcome

The PRISM failure was primarily an optimizer-range failure, not a failure of
the local Hessian.  The original unit-radius Newton solver moved too slowly
from the dumped initialization: after 16 iterations its median gradient norm
was `4.06e4`, median Newton decrement was 2322, and every repaired metric was
pinned at condition number `1e8`.

I added an opt-in trust radius, a Newton-decrement stopping criterion, and a
batched convergence-aware loop to the shared Laplace preparation used by both
independent NUTS and HMC.  With radius 5 and at most 200 FD-Newton iterations,
the real 21-channel PRISM low-resolution problem reached median gradient norm
`2.49e-3`, median decrement `7.49e-5`, and median condition number `1.17e7`.
The median lane stopped after 71 iterations.  Defaults remain radius 1, 16
iterations, and the original gradient stopping rule.

The best requested candidate is FD-Laplace independent NUTS, depth 5, target
accept 0.99, initialized at the MAP.  On the V100 it returned 1000 draws for
all 21 low-resolution channels in **246.22 s**, including **64.56 s** compile,
with zero divergences and mean 20.94 leapfrogs/draw.  Matched production joint
NUTS took 1718.45 s, so the same-V100-model, equal-returned-draw speedup is
**6.98x**.
Science medians agree within 0.25 reference sigma, but `A` channel 13 has a
candidate/reference sigma ratio of 0.633.  Depth 5 is therefore not fully
fidelity-qualified despite the large speed gain.

HMC-8 remained faster (123.07 s at target 0.99), but its minimum bulk ESS was
only 15.4.  At the less conservative target it had 14 divergences.  It is not
the selected PRISM sampler.

The selected NUTS configuration completed the 106-channel PRISM high-resolution
stage in **1080.50 s (18.0 min)**, including **71.24 s** recorded compilation.
The first 80 channels had zero divergences; the final 26-channel chunk had four
and minimum ESS 15.5, so this high-resolution result is diagnostic rather than
an unconditional deployment recommendation.

## Files built or changed

| File | Change |
|---|---|
| `models/independent_nuts.py` | Added `laplace_trust_radius`, optional Newton-decrement convergence, and a vmapped `while_loop` that stops the optimizer once all lanes have converged.  Existing defaults are unchanged. |
| `models/independent_hmc.py` | Accepted and validated the shared PRISM MAP controls. |
| `fit_jwst.py` | Minimally exposed generic/stage-specific trust-radius and decrement-tolerance flags. |
| `tests/test_independent_nuts.py` | Added a distant-mode CPU test that verifies the larger radius reaches the mode and exits before its iteration cap; extended resolver coverage. |
| `tools/diag_nuts/run_prism_map_pilot.sh` | Radius-5/radius-20 real-input pilots. |
| `tools/diag_nuts/run_prism_fd_step_pilot.sh` | Smaller-FD-step diagnostic. |
| `tools/diag_nuts/run_prism_low_candidates.sh` | 1000-draw HMC/NUTS low-resolution runs. |
| `tools/diag_nuts/run_prism_low_high_accept.sh` | Target-0.99 qualification runs. |
| `tools/diag_nuts/run_prism_high.sh` | Full 106-channel high-resolution driver. |
| `tools/diag_nuts/run_prism_joint_reference.sh` | Production-prior joint NUTS 1000/1000 reference driver. |
| `tools/diag_nuts/run_prism_low_depth6.sh` | Prepared depth-6 follow-up; not run within this task's GPU timing window. |
| `tools/diag_nuts/summarize_prism.py` | Per-site medians, sigmas, ESS, and per-lane MAP summaries against finite-draw references. |
| `acceleration_reports/diag_nuts/prism_*.json` | Machine-readable comparisons and MAP diagnostics. |

No transit, likelihood, prior, data, mask, precision, or limb-darkening code
was changed.

## Robust MAP implementation

The existing solver already supplied the necessary safeguards: exact
float64 gradients, central-FD Hessians, spectral repair, trust clipping,
backtracking, and a normalized-gradient fallback.  The blocking choice was a
hard-coded one-unit trust radius.  I made that radius configurable and changed
the fixed `fori_loop` to a batched `while_loop`.  A lane stops on either the
existing gradient tolerance or an optional Newton-decrement tolerance; JAX's
batched loop terminates when all lanes are done.  Thus raising PRISM's maximum
to 200 does not force SOSS/G395H to execute 200 iterations, and their default
maximum remains 16.

The selected PRISM controls are:

```yaml
spectro_sampler: independent_nuts
spectro_mass_matrix: laplace
spectro_jitter_prior: lognormal
spectro_laplace_hessian_method: finite_difference
spectro_laplace_fd_relative_step: 0.0002
spectro_laplace_trust_radius: 5.0
spectro_laplace_map_decrement_tolerance: 0.0001
spectro_laplace_warmup: 200
spectro_laplace_target_accept: 0.99
spectro_laplace_start_at_map: true
spectro_laplace_fuse_program: true
spectro_laplace_max_tree_depth: 5
```

The offline measurements additionally set `laplace_map_iterations=200`; that
expert option is accepted by the backend and is intentionally not a new global
default.

### Optimizer experiments on PRISM low 0:21

| MAP variant | Iteration cap | Gradient norm median / max | Decrement median / max | Condition median / max | Interpretation |
|---|---:|---:|---:|---:|---|
| Original HMC, radius 1 | 16 | `4.06e4 / 4.07e4` | `2322 / 3344` | `1e8 / 1e8` | failed |
| Original NUTS, radius 1 | 32 | `3.73e4 / 3.90e4` | `461 / 669` | `1e8 / 1e8` | failed |
| Radius 5, FD `2e-4` | 96 | `1.81e-3 / 0.113` | `2.80e-5 / 1.97e-3` | `1.17e7 / 3.06e7` | entered Newton basin |
| Radius 20, FD `2e-4` | 96 | `1.98e-3 / 0.279` | `1.24e-4 / 8.92e-4` | `1.17e7 / 3.06e7` | worse median; rejected |
| **Radius 5, FD `2e-4`, early stop** | **200** | **`2.49e-3 / 0.114`** | **`7.49e-5 / 1.97e-3`** | **`1.17e7 / 3.06e7`** | selected metric |
| Radius 5, FD `2e-5` | 200 | `1.39e-3 / 0.181` | `9.64e-6 / 9.48e-4` | `1.16e7 / 3.09e7` | lower median decrement, but every lane hit cap and max gradient worsened |

The larger radius alone moves the optimizer into the correct basin; no
model-specific linear solve or SciPy optimizer was necessary.  Seven selected
lanes hit the 200-iteration cap.  Their remaining decrements are small enough
to yield an effective metric, but the per-lane values below are retained for
production auditing.

### Selected MAP diagnostics by low-resolution channel

| Channel | Gradient norm | Newton decrement | Iterations | Condition number |
|---:|---:|---:|---:|---:|
| 0 | `1.82e-4` | `1.48e-6` | 55 | `4.06e6` |
| 1 | `2.73e-3` | `1.63e-5` | 11 | `2.41e6` |
| 2 | `1.14e-1` | `6.19e-5` | 10 | `1.45e7` |
| 3 | `4.74e-2` | `1.81e-4` | 200 | `1.69e7` |
| 4 | `2.21e-3` | `2.25e-5` | 10 | `1.27e7` |
| 5 | `3.00e-3` | `7.61e-5` | 13 | `2.02e7` |
| 6 | `1.50e-3` | `7.55e-4` | 200 | `1.37e7` |
| 7 | `2.49e-3` | `1.20e-3` | 200 | `1.94e7` |
| 8 | `8.77e-3` | `3.79e-6` | 15 | `3.06e7` |
| 9 | `1.46e-2` | `8.18e-4` | 200 | `8.33e6` |
| 10 | `6.18e-3` | `7.49e-5` | 17 | `1.56e7` |
| 11 | `2.79e-2` | `2.60e-4` | 200 | `1.58e7` |
| 12 | `3.70e-3` | `1.97e-3` | 200 | `1.54e7` |
| 13 | `6.74e-3` | `5.41e-5` | 17 | `1.17e7` |
| 14 | `1.48e-3` | `9.75e-4` | 200 | `4.75e6` |
| 15 | `1.73e-3` | `8.93e-4` | 200 | `7.81e6` |
| 16 | `1.81e-3` | `1.14e-6` | 72 | `5.57e6` |
| 17 | `6.18e-4` | `1.73e-5` | 71 | `2.97e6` |
| 18 | `3.09e-4` | `8.33e-5` | 70 | `3.23e6` |
| 19 | `7.18e-5` | `1.41e-5` | 64 | `1.53e6` |
| 20 | `4.98e-5` | `2.80e-5` | 84 | `1.25e6` |

## Low-resolution V100 sampler measurements

All candidate rows use the same dump, lognormal jitter prior, radius-5
200-iteration MAP, 1000 returned draws, and the third Tesla V100.  Compile is
included in wall.

GPU provenance matters here.  Queue markers place jobs 90--94 on
`Tesla V100-PCIE-16GB` job 57150049 (the "third V100").  The long joint
reference was picked up by the newly added, identical-model V100 job 57157918
(the "fourth V100").  The queue README explicitly states that V100-to-V100
timings are comparable, but these are same-model rather than same-physical-GPU
measurements.  No RTX result is used in a ratio.

| Sampler | Target / warmup | Wall | Compile | Mean / max steps | Accept | Divergences | Minimum ESS |
|---|---|---:|---:|---:|---:|---:|---:|
| HMC-8 ±25% | 0.85 / 150 | 120.59 s | 54.19 s | 8.00 / 10 | 0.953 | 14 | 47.4 science; 44.0 including `log_tau` |
| NUTS depth 5 | 0.95 / 150 | 219.28 s | 64.54 s | 12.59 / 31 | 0.952 | 9 | 34.8 science; 36.5 `log_tau` |
| HMC-8 ±25% | 0.99 / 200 | **123.07 s** | 54.54 s | 8.00 / 10 | 0.988 | **0** | **15.4** |
| **NUTS depth 5** | **0.99 / 200** | **246.22 s** | **64.56 s** | **20.94 / 31** | **0.988** | **0** | **83.1 science; 48.1 `log_tau`** |

The target-0.99 NUTS row is selected because it eliminates divergences without
the HMC row's very low trend ESS.

### Bulk ESS, selected NUTS (minimum / median across 21 channels)

| Site | ESS min / median |
|---|---:|
| `A` | 101.7 / 326.4 |
| `c` | 83.1 / 350.9 |
| `v` | 86.2 / 404.9 |
| `depths` / `rors` | 134.5 / 763.2 |
| `log_tau` | 48.1 / 163.5 |
| `log_jitter` / `total_error` | 343.5 / 634.3 |

### Comparison with the available 200-draw joint reference

The reference has only 200 draws and minimum bulk ESS 28.7, while candidates
have 1000 draws.  Consequently its sample median and sigma are noisy; this is
a descriptive comparison, not a calibrated 0.1-sigma gate.

| Site | Median absolute median shift | Maximum absolute shift | Sigma ratio min / median / max |
|---|---:|---:|---:|
| `A` | 0.052 sigma | 0.209 sigma | 0.721 / 1.023 / 1.160 |
| `c` | 0.040 | 0.205 | 0.796 / 1.016 / 1.168 |
| `v` | 0.047 | 0.242 | 0.821 / 1.020 / 1.172 |
| `depths` | 0.063 | 0.243 | 0.827 / 1.001 / 1.206 |
| `rors` | 0.063 | 0.243 | 0.826 / 1.001 / 1.206 |
| `log_tau` | 0.062 | 0.453 | 0.854 / 1.001 / 1.191 |

Fixed `c1/c2` sites have zero reference variance and are excluded from
sigma-normalized comparisons.  Noise-site changes from the opt-in lognormal
jitter prior are also not evidence of sampler error.

## Production-prior joint 1000/1000 reference

The new reference was deliberately run after the requested candidates and
high-resolution stage.  It ran on the replacement fourth V100, whereas the
candidate ran on the identical-model third V100; see the provenance note
above.  Both rows return 1000 draws per channel.  The candidate uses the
requested lognormal jitter prior; the production reference retains its dumped
log-uniform prior.

| Sampler | Wall | Compile | Mean / median / max steps | Accept | Divergences | Minimum bulk ESS |
|---|---:|---:|---:|---:|---:|---:|
| Production joint NUTS | **1718.45 s** | 18.29 s | 116.82 / 127 / 255 | 0.868 | 0 | 274.5 |
| FD-Laplace NUTS depth 5 | **246.22 s** | 64.56 s | 20.94 / -- / 31 | 0.988 | 0 | 83.1 science |

The compile-inclusive ratio is **6.98x**.  Removing recorded compile time gives
a 9.36x sampling-residual ratio.  A conservative adjustment that forces the
candidate's worst science ESS (83.1) to equal the reference minimum (274.5)
gives an estimated 813.2 s candidate wall and **2.11x equal-minimum-ESS
speedup**.  That normalization is deliberately pessimistic and is not an
additional wall measurement.

### Candidate versus the 1000-draw reference

| Site | Median absolute median shift | Maximum absolute shift | Sigma ratio min / median / max |
|---|---:|---:|---:|
| `A` | 0.058 sigma | 0.138 sigma | **0.633** / 1.017 / 1.070 |
| `c` | 0.036 | 0.145 | 0.807 / 1.011 / 1.101 |
| `v` | 0.049 | 0.152 | 0.833 / 1.026 / 1.084 |
| `depths` | 0.027 | 0.143 | 0.920 / 1.010 / 1.070 |
| `rors` | 0.027 | 0.143 | 0.920 / 1.010 / 1.070 |
| `log_tau` | 0.055 | **0.250** | 0.903 / 1.005 / 1.110 |

All science-site median shifts are <=0.251 reference sigma and the median
sigma ratio for every site is within 2.6% of unity.  The important exception
is posterior width for `A`, channel 13: its sigma ratio is 0.633.  This is not
readily dismissed as reference noise: joint-reference bulk ESS for `A` is
420--1178 (median 744), while candidate `A` ESS is 102--766 (median 326).
Depth-5 tree truncation is the leading hypothesis.  A depth-6 driver was
prepared after discovering this result, but the shared dispatchers had moved
to other campaign jobs; no depth-6 claim is made.

As expected from changing the jitter prior, `log_jitter` has a median shift of
1.23 reference sigma and median sigma ratio 0.780.  It is separated from the
science comparison rather than attributed to the inference algorithm.

For completeness, HMC-8 at target 0.99 compares as follows.  Its medians are
also broadly consistent, but its `A/c/v/log_tau` widths and low ESS are worse
than NUTS:

| Site | Median absolute median shift | Maximum absolute shift | Sigma ratio min / median / max |
|---|---:|---:|---:|
| `A` | 0.070 sigma | 0.283 sigma | 0.811 / 0.996 / 1.084 |
| `c` | 0.083 | 0.162 | 0.848 / 1.002 / 1.278 |
| `v` | 0.068 | 0.144 | 0.879 / 1.008 / 1.250 |
| `depths` | 0.053 | 0.182 | 0.923 / 1.007 / 1.073 |
| `rors` | 0.053 | 0.183 | 0.923 / 1.007 / 1.073 |
| `log_tau` | 0.093 | **0.525** | **0.587** / 0.993 / 1.149 |

Machine-readable final comparisons are
`acceleration_reports/diag_nuts/prism_low_nuts5_ta99_vs_joint1000.json` and
`acceleration_reports/diag_nuts/prism_low_hmc8_j25_ta99_vs_joint1000.json`.

## High-resolution PRISM, 106 channels

There is no external production reference for this dump.  The measurements
below are internal diagnostics only.

| Chunk | Channels | Wall | Compile | Mean / max steps | Accept | Divergences | MAP decrement median / max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0:40 | 408.99 s | 70.47 s | 19.77 / 31 | 0.9881 | 0 | `2.05e-5 / 4.12e-4` |
| 1 | 40:80 | 334.96 s | 0 s | 20.86 / 31 | 0.9881 | 0 | `4.65e-5 / 4.73e-4` |
| 2 | 80:106 | 336.23 s | 0.77 s | 20.80 / 31 | 0.9845 | **4** | `2.11e-5 / 4.01e-4` |
| **Full stage** | **0:106** | **1080.50 s** | **71.24 s** | -- | -- | **4** | -- |

Bulk ESS by site across all 106 channels:

| Site | Minimum / median |
|---|---:|
| `c` | 15.5 / 385.9 |
| `v` | 17.4 / 434.5 |
| `depths` / `rors` | 63.9 / 759.5 |
| `log_jitter` / `total_error` | 104.2 / 590.0 |

The cached 40-channel sampling wall is 334.96 s.  The 26-channel tail needs a
second static width and, despite only 0.77 s of recorded helper compilation,
takes 336.23 s because its slow lanes reach the depth cap and include all four
divergences.  This tail should be investigated before production rollout.

## Exact reproduction commands

GPU jobs were run only by the orchestrator through the file queue.  Inside the
V100 allocation, the durable commands are:

```bash
bash tools/diag_nuts/run_prism_map_pilot.sh
bash tools/diag_nuts/run_prism_fd_step_pilot.sh
bash tools/diag_nuts/run_prism_low_candidates.sh
bash tools/diag_nuts/run_prism_low_high_accept.sh
bash tools/diag_nuts/run_prism_high.sh nuts
bash tools/diag_nuts/run_prism_joint_reference.sh
```

The 200-draw comparison is reproduced with:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 0-15 \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  tools/diag_nuts/summarize_prism.py \
  --candidate /scratch/midway3/tfairnington/accel_gpu_results/prism/prism_low_map200_r5_nuts5_ta99.pkl \
  --reference /scratch/midway3/tfairnington/accel_gpu_results/20_prism_low_joint_200/prism_low_0_21_joint_200_200.pkl \
  --diagnostics /scratch/midway3/tfairnington/accel_gpu_results/prism/prism_low_map200_r5_nuts5_ta99.diagnostics.npz \
  --end 21 \
  --output acceleration_reports/diag_nuts/prism_low_nuts_ta99_vs_joint200.json
```

For the final 1000-draw comparison, replace `--reference` with
`/scratch/midway3/tfairnington/accel_gpu_results/prism/prism_low_joint_1000_1000.pkl`
and write to `acceleration_reports/diag_nuts/prism_low_nuts5_ta99_vs_joint1000.json`.

CPU regression command:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 0-15 \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest \
  tests/test_independent_nuts.py tests/test_independent_hmc.py \
  tests/test_independent_runner_reuse.py -x -q
```

Result: **20 passed**, with two dependency warnings, in 122.60 s.

## Failures and open risks

- The MAP is now useful, but seven low-resolution lanes still hit 200
  iterations.  Their decrements, not their raw gradient norms, are the more
  meaningful scale-aware diagnostic.
- Radius 20 did not improve the median solution.  A ten-times-smaller FD step
  improved decrement statistics but forced every lane to the cap and worsened
  the maximum raw gradient; neither variant was promoted.
- HMC-8 is not qualified for PRISM.  Target 0.85 gives 14 divergences; target
  0.99 removes them but creates minimum ESS 15.4 through short-trajectory
  autocorrelation/resonance.
- NUTS target 0.95 still gives nine divergences.  Target 0.99 is required on
  this geometry and increases work from 12.6 to 20.9 leapfrogs/draw.
- The high-resolution tail has four divergences and minimum ESS 15.5.  There is
  no reference posterior for that stage, so its 18-minute wall is a performance
  measurement, not a fidelity qualification.
- The candidate deliberately uses the accepted lognormal jitter prior, whereas
  the production reference uses the production log-uniform prior.  Science
  sites are compared separately from `log_jitter`/`total_error`.
- Compilation remains 55--71 s and is a first-order low-resolution cost, but
  the width-40 cache is reused across the first two high-resolution chunks.
- The new 1000-draw reference reveals one material depth-5 width failure:
  `A` channel 13 has sigma ratio 0.633.  A depth-6 follow-up is the next
  experiment; it was not measured in the available dispatcher window.
