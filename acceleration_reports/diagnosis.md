# Why spectroscopic NUTS needs 63--255 leapfrogs

## Executive result

The dominant problem is poorly scaled, correlated posterior geometry in NumPyro's
unconstrained coordinates, compounded by synchronized per-channel execution. It
is not primarily the transit forward cost, and the flat jitter direction is not
by itself the tree-length driver.

- The 40-channel, 230-cadence synthetic Hessians have median condition number
  `7.42e4`. The production dumps are worse: median `9.60e6` for SOSS, `1.74e6`
  for G395H, and `1.27e6` for the first eight PRISM lanes. The 2000-cadence
  white-light Hessian has condition number `2.42e7`.
- The synthetic posterior has median `corr(c,v)=-0.843` and `|corr|>0.5` in all
  40 lanes. The informed power-2 coefficients have median
  `corr(c1,c2)=-0.540` and exceed 0.5 in 38/40 lanes. Real SOSS has median
  `corr(c1,c2)=-0.977`; real G395H has `corr(c,v)=-0.800`; real PRISM has
  `corr(c,v)=-0.939`.
- Joint dense adaptation did not fix this. On identical eight-channel input it
  increased mean steps from 78.5 to 107.2 and wall time from 408.1 to 502.5 s.
  One dense matrix is a poor match to independent blocks with lane-specific
  scales.
- Independent dense adaptation lowered mean lane steps to 27.8, but synchronized
  execution waited for 52.5 steps per draw and ESS/draw fell. Wall was 419.3 s.
- Fixing each lane's inverse mass matrix to its Laplace covariance and adapting
  only step size reduced synchronized steps to 14.0 and wall to 62.8 s at the
  same 300 warmup + 300 draws, target 0.8, and depth 10: **6.67x wall speedup
  over independent dense adaptation** and **6.50x over production joint
  diagonal**. Its ESS/s gain over joint diagonal was 7.13x. This run had 6
  divergences in 2400 lane-draws, so it is diagnostic.
- Raising the preconditioned path to target acceptance 0.95 removed divergences
  in the eight-lane run. It still gave 5.80x wall and 6.16x summed ESS/s over
  joint diagonal, but used 150 rather than 300 warmup steps. Its strict
  short-chain posterior-agreement gate was not passed.
- In white light, Laplace preconditioning reduced mean steps from 312.2 to 14.2,
  wall from 1951.8 to 71.1 s, and increased summed ESS/s by 58.4x. Both runs had
  zero divergences. The candidate used 100 rather than 300 warmup and its largest
  median/SD shifts were 0.163/0.220 reference sigma, so it needs a longer check.

The best next configuration to validate on full real chunks is per-channel
Laplace inverse mass, step-size-only adaptation, 150 warmup,
`target_accept_prob=0.95`, and `max_tree_depth=10`. Do not cap depth at 5 or 6,
and do not use 50 warmup, until the full posterior gate passes.

## Files built

No production source was modified. In particular, I did not edit `fit_jwst.py`
or anything under `models/`.

| File | Purpose |
|---|---|
| `tools/diag_nuts/common.py` | Synthetic and dumped-input loaders, ESS and agreement helpers |
| `tools/diag_nuts/spectro_geometry.py` | Per-lane MAP, exact AD Hessian, correlations, physical sigma/prior ratios, jitter profiles |
| `tools/diag_nuts/spectro_samplers.py` | Joint, existing independent, fixed-Laplace-mass NUTS, non-Gaussianity, and held-jitter runs |
| `tools/diag_nuts/white_light.py` | White-light geometry and matched NUTS comparisons |
| `tools/diag_nuts/potential_timing.py` | Full-potential and value-plus-gradient timings |
| `acceleration_reports/diag_nuts/*.json` | Machine-readable measurements |
| `acceleration_reports/diag_nuts/*.npz` | MAP/Hessian matrices and retained draws |

## Scope and reference caveat

All work used float64 and the real model builders, transformations, priors,
likelihood, transit-window machinery, and streamed power-2 transit calculation.
All timings are CPU timings from the login node, restricted to CPUs 0--15.
There are no GPU speed claims here.

The synthetic spectroscopic problem has 40 channels, 230 cadences, informed
TruncatedNormal power-2 limb darkening, free linear `c,v`, log jitter, and 46
transit-window cadences. The matched sampler sweep uses the first eight channels
of exactly that 40-channel realization because the unmodified 40-lane independent
dense reference failed operationally (documented below).

The provisional reference is exact NUTS, not a Gaussian posterior: independent
lane NUTS with fixed Laplace mass, 150 warmup, and 1000 draws on all 40 lanes. It
had 95 divergences among 40,000 lane-draws (0.238%), concentrated in difficult
jitter tails. The comparison numbers are useful diagnostics but are **not a
literature-fidelity certification**.

## 1. Spectroscopic posterior geometry

### Synthetic 40-channel Hessians

All 40 exact Hessians were positive definite. Condition numbers span
`5.60e4`--`7.51e5`, with median `7.42e4`.

The following ratios transform local covariance back to physical coordinates
and divide posterior sigma by the full bounded support width.

| Site | Median sigma / prior width | Min | Max |
|---|---:|---:|---:|
| `c` | 0.000393 | 0.000344 | 0.000449 |
| `v` | 0.00113 | 0.000993 | 0.00130 |
| `rors` | 0.000586 | 0.000520 | 0.000662 |
| `c1` | 0.0391 | 0.0356 | 0.0422 |
| `c2` | 0.0404 | 0.0375 | 0.0433 |
| `log_jitter` | 0.0288 | 0.0128 | 0.107 |

| Pair | Median correlation | Lanes with `abs(corr)>=0.5` |
|---|---:|---:|
| `c--v` | -0.843 | 40/40 |
| `c1--c2` | -0.540 | 38/40 |
| `c--rors` | +0.220 | 0/40 |
| `c1--rors` | -0.254 | 0/40 |
| `c2--rors` | -0.158 | 0/40 |
| jitter with another latent | median absolute values about 0.01 | 0/40 |

The main local rotations are trend intercept/slope and the two power-2
coefficients. `rors` is coupled more weakly in this realization, and jitter is
almost locally orthogonal despite being non-Gaussian marginally.

### Real stage dumps

I analyzed the first production chunk for SOSS and G395H, and the first eight
PRISM lanes. PRISM was limited to eight because one exact 40,738-cadence
MAP/profile pass took about 30 seconds per lane on this CPU; the requested
synthetic audit remains all 40 lanes.

| Dump | Lanes | Cadences (active) | Hessian condition min / median / max | Dominant correlations |
|---|---:|---:|---:|---|
| HAT-P-12 SOSS high | 40 | 225 (94) | `1.50e6 / 9.60e6 / 1.44e7` | `c1-c2=-0.977`, `c-rors=+0.574`, both 40/40 |
| GJ-3470 G395H high | 40 | 2058 (751) | `1.15e4 / 1.74e6 / 2.25e6` | `c-v=-0.800` in 40/40 |
| HAT-P-65 PRISM high | 8 | 40,738 (16,440) | `4.54e5 / 1.27e6 / 3.35e6` | `c-v=-0.939` in 8/8; `c-log_tau=-0.783`, `log_tau-v=+0.708` in 5/8 |

Real SOSS also exposes a practically prior-dominated `v`: local
sigma/prior-width is 0.353553 in every lane. Median local `log_jitter`
sigma/prior-width is 0.0153 (SOSS), 0.0996 (G395H), and 0.1025 (PRISM).

There is a numerical qualification. Transit contact branches make SciPy's line
search report abnormal termination even when it cannot find a lower point; an
exact-Hessian trust-region retry was added. All retained Hessians are positive
definite, but strict absolute-gradient success was reached for only 15/40
synthetic, 16/40 SOSS, 9/40 G395H, and 4/8 PRISM modes. Median/max unconstrained
gradient magnitudes were `0.00266/0.264`, `0.00753/0.471`, `0.0328/2.15`, and
`0.00983/0.108`. The matrices are effective metrics, but real-data condition
numbers are local estimates rather than perfectly converged global-MAP invariants.

### Non-Gaussianity and jitter

The exact 1000-draw lane chains were compared with 20,000 transformed Laplace
draws per channel. QQ deviation is the maximum difference among the 1, 5, 16,
50, 84, 95, and 99 percent quantiles in NUTS sigma units.

| Site | Median / max abs skew | Median / max abs excess kurtosis | Median / max QQ deviation | NUTS median sigma/prior width |
|---|---:|---:|---:|---:|
| `c` | 0.034 / 0.195 | 0.155 / 0.515 | 0.186 / 0.518 | 0.000381 |
| `v` | 0.036 / 0.163 | 0.148 / 0.432 | 0.153 / 0.408 | 0.00110 |
| `rors` | 0.046 / 0.195 | 0.130 / 0.681 | 0.183 / 0.322 | 0.000568 |
| `c1` | 0.100 / 0.250 | 0.106 / 0.454 | 0.217 / 0.379 | 0.0386 |
| `c2` | 0.121 / 0.422 | 0.156 / 1.17 | 0.152 / 0.330 | 0.0404 |
| `log_jitter` | **0.474 / 2.27** | **1.24 / 7.99** | **2.44 / 3.72** | **0.117** |

Every site except jitter is locally close to Gaussian. Jitter's marginal chain
width is about four times its median local Laplace width (`0.117` versus `0.0288`
of prior width).

The conditional jitter profile has median width 1.95 in unconstrained
coordinates for `Delta U <= 2` (range 0.1--3.1). In a deliberately
negligible-jitter realization it is 3.0 (`Delta U <= 2`) and 1.2
(`Delta U <= 0.5`), with condition number `6.43e5`. Real G395H and PRISM have
median `Delta U <= 2` widths 3.0 and 3.1; real SOSS is more identified at 0.3.

This plateau does **not** drive long trees after dense adaptation:

| Diagnostic model | Mean / median / max steps | Accept | Div. | Wall |
|---|---:|---:|---:|---:|
| Jitter free | 17.58 / 15 / 47 | 0.889 | 1 | 32.74 s |
| Jitter fixed at MAP | 18.92 / 15 / 31 | 0.932 | 0 | 30.58 s |

Holding jitter fixed made mean steps 7.7% worse (`17.58/18.92=0.929`) and wall
only 1.07x better. Jitter explains the Laplace mismatch and some divergences,
not the baseline 63--255-step trees.

## 2. Matched sampler accounting

All rows use the same first eight lanes and 300 retained draws. `Lane max` is
the average maximum across lanes per draw: the work seen by synchronized
execution. Divergences for independent rows are among 2400 lane-draws.

| Configuration | Warmup | Step size | Mean lane steps | Lane max | Lockstep | Accept | Div. | ESS/draw | ESS/s | Wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| joint diagonal, production | 300 | 0.0517 | 78.47 | 78.47 | 1.00 | 0.925 | 0 | 1.063 | 37.52 | 408.09 s |
| joint dense | 300 | 0.0466 | 107.16 | 107.16 | 1.00 | 0.939 | 0 | 1.029 | 29.49 | 502.54 s |
| independent dense adaptation | 300 | 0.0698--0.117 | 27.75 | 52.51 | 1.89 | 0.898 | 0 | 0.629 | 21.59 | 419.25 s |
| Laplace mass, depth 10 | 300 | 0.549--0.719 | 7.52 | 13.97 | 1.86 | 0.897 | 6 | 1.166 | 267.32 | 62.83 s |
| Laplace mass, depth 10 | 150 | 0.538--0.791 | 8.02 | 17.05 | 2.13 | 0.892 | 2 | 1.273 | 337.81 | 54.25 s |
| Laplace mass, depth 5 | 150 | 0.482--0.756 | 7.68 | 14.65 | 1.91 | 0.885 | 8 | 1.211 | 322.94 | 53.99 s |
| Laplace mass, depth 6 | 150 | 0.441--0.847 | 8.50 | 19.69 | 2.32 | 0.874 | 3 | 1.294 | 328.08 | 56.78 s |
| Laplace mass, MAP start | 50 | 0.318--1.16 | 7.84 | 18.97 | 2.42 | 0.829 | 6 | 1.223 | 343.39 | 51.28 s |
| Laplace, target 0.90 | 150 | 0.414--0.660 | 9.21 | 20.86 | 2.26 | 0.923 | 2 | 1.229 | 294.54 | 60.07 s |
| **Laplace, target 0.95** | **150** | **0.409--0.547** | **10.63** | **25.76** | **2.42** | **0.942** | **0** | **1.129** | **231.11** | **70.35 s** |

The full-40-lane preconditioned 150/1000 reference took 713.59 s. Lane mean was
7.89, lane max 28.10, and lockstep 3.56; accept was 0.886, ESS/draw 1.193, and
ESS/s 401.1. Growth from 1.9--2.4 lockstep at eight lanes to 3.56 at 40 lanes
shows synchronization is a first-order production cost.

### Posterior agreement

No 300-draw candidate passed the all-sites 0.1-reference-sigma requirement. This
threshold is below the expected worst-coordinate Monte Carlo error of 300 versus
1000 draws, and the reference jitter divergences weaken it further. These
results neither prove bias nor certify fidelity.

| Configuration | Max median shift | Max SD shift | Fraction medians / SDs within 0.1 sigma |
|---|---:|---:|---:|
| joint diagonal | 1.346 | 7.859 | 0.646 / 0.771 |
| joint dense | 0.306 | 0.381 | 0.771 / 0.833 |
| independent dense | 0.375 | 0.249 | 0.604 / 0.854 |
| Laplace w300, target 0.8 | 0.634 | 0.614 | 0.792 / 0.729 |
| Laplace w150, target 0.8 | 0.752 | 0.183 | 0.813 / 0.854 |
| Laplace depth 5 | 0.512 | 0.305 | 0.771 / 0.750 |
| Laplace depth 6 | 1.648 | 0.852 | 0.750 / 0.708 |
| Laplace MAP/w50 | 1.178 | 0.273 | 0.708 / 0.813 |
| Laplace target 0.90 | 0.776 | 0.400 | 0.813 / 0.792 |
| Laplace target 0.95 | 1.443 | 0.249 | 0.750 / 0.771 |

Large maxima are mainly `log_jitter`; most transit/trend coordinates are closer.
A deployment decision needs clean, pooled, higher-ESS real-input references.

Interpretation:

- Joint dense is 1.23x slower than joint diagonal and has 21% lower ESS/s.
- Independent adaptive dense shortens individual trees, but adaptation and
  lockstep erase the gain; ESS/s is only 0.58x joint diagonal.
- Comparing independent dense with Laplace w300 holds input, schedule, target,
  and depth fixed. Preconditioning improves wall 6.67x, synchronized steps
  3.76x, and ESS/s 12.38x, but gives 6 divergences at target 0.8.
- Halving fixed-metric warmup from 300 to 150 adds only 1.16x wall speed.
- Depth 5 saves 0.5% over depth-10/w150 but raises divergences from 2 to 8.
  Depth 6 is slower and has 3 divergences. MAP plus 50 warmup is only 1.06x
  faster and has 6 divergences.
- Target 0.95 is divergence-free and remains 5.80x faster in wall and 6.16x in
  ESS/s than joint diagonal, with shorter warmup contributing to this ratio.

## 3. White-light stage

The pipeline already supplies a MAP-like start: `fit_jwst.py` lines 5705--5734
optimize geometry, then transit/LD, then all sites; lines 5790--5819 pass that
solution through `init_to_value`. Starting closer to the mode therefore cannot
explain the remaining 255-step median. The missing piece is local curvature.

The 2000-cadence synthetic model has nine unconstrained coordinates. Its Hessian
is positive definite, spans eigenvalues 28.6 to `6.91e8`, and has condition
number `2.42e7`.

| Pair | Correlation |
|---|---:|
| `c1--c2` | -0.992 |
| `c--v` | -0.842 |
| `c1--rors` | -0.747 |
| `b--rors` | +0.740 |
| `c2--rors` | +0.678 |
| `c1--logD` | +0.489 |
| `c2--logD` | -0.483 |
| `b--logD` | +0.430 |

The exact chain is mildly non-Gaussian, not a strong funnel. `b` has skew 0.198,
excess kurtosis -0.596, and Laplace QQ error 0.456 sigma. `logD` has skew -0.274,
kurtosis -0.244, and QQ error 0.446. The largest QQ error is 0.719 for `c2`
(skew 0.397). White jitter has skew -0.125, kurtosis 0.617, and QQ error 0.358.

| Configuration | Warmup/draws | Step size | Mean / median / p95 / max steps | Accept | Div. | ESS/draw | ESS/s | Wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Production diagonal | 300/300 | 0.00619 | 312.2 / 255 / 831 / 1023 | 0.903 | 0 | 0.442 | 0.612 | 1951.77 s |
| Laplace dense, step adaptation | 100/300 | 0.2839 | 14.2 / 15 / 15 / 23 | 0.958 | 0 | 0.941 | 35.71 | 71.12 s |

This is a 21.98x step reduction, 27.44x equal-draw wall speedup, 2.13x ESS/draw
gain, and 58.39x ESS/s gain. Maximum median and SD shifts are 0.163 and 0.220
sigma; 6/9 medians and 5/9 SDs are within 0.1 sigma. This is the largest measured
opportunity, but the strict gate still needs a longer equal-ESS run.

## 4. Bottleneck accounting and ceilings

For a retained effective sample, model compute is approximately

```text
T / ESS = (mean lane leapfrogs / ESS_per_draw)
          * (value+gradient time)
          * (lane-max / lane-mean lockstep)
          * ((warmup + draws) / draws).
```

Joint NUTS has `78.47/1.063 = 73.8` gradients per retained effective draw and a
production warmup multiplier near 2. The divergence-free target-0.95 path has
`10.63/1.129 = 9.41`, multiplied by 2.424 lockstep to 22.81, and a 1.5 warmup
multiplier.

### Measured complete-potential cost

| Cadences | Channels | Forward | Value + gradient | Gradient/forward |
|---:|---:|---:|---:|---:|
| 230 (SOSS proxy) | 1 | 0.123 ms | 0.480 ms | 3.90 |
| 230 | 40 | 3.395 ms | 6.621 ms | 1.95 |
| 2058 (G395H) | 1 | 0.547 ms | 2.007 ms | 3.67 |
| 2058 | 40 | 9.573 ms | 88.933 ms | 9.29 |
| 40,787 (PRISM proxy) | 1 | 3.276 ms | 8.581 ms | 2.62 |
| 40,787 | 40 | 520.66 ms | 1198.38 ms | 2.30 |

These medians follow JIT compilation and cover transforms, priors, streamed
transit, trend, jitter, and likelihood. The high 40-channel G395H reverse-mode
ratio is repeatable on this CPU and should not be extrapolated to GPU.

Applying the 40-channel costs to measured synthetic tree/ESS factors gives this
accounting projection, not measured end-to-end real-data wall time:

| Shape | Production retained ESS | Production incl. 50% warmup | Preconditioned retained ESS | Preconditioned incl. 33% warmup | Preconditioned no lockstep |
|---|---:|---:|---:|---:|---:|
| SOSS-like | 0.489 s | 0.977 s | 0.151 s | 0.227 s | 0.0623 s |
| G395H-like | 6.56 s | 13.13 s | 2.03 s | 3.04 s | 0.837 s |
| PRISM-like | 88.44 s | 176.88 s | 27.34 s | 41.01 s | 11.28 s |

The bottleneck is many gradient calls times cadence-dependent gradient cost;
independent lanes then add 1.9--3.6x synchronization. Warmup matters less after
a fixed metric is available.

| Lever | Ceiling/result | Qualification |
|---|---:|---|
| Per-lane Laplace preconditioning alone | 6.67x wall, 12.38x ESS/s | Same independent implementation and 300/300; target-0.8 run had 6 divergences |
| Remove warmup | about 2x arithmetic maximum for 1000/1000 | Halving 300 to 150 measured 1.16x; deleting the full recorded w300 phase is an optimistic 2.13x |
| Remove lockstep | 1.89x adaptive dense; 1.86--2.42x preconditioned at 8 lanes; **3.56x at 40 lanes** | Compute ceiling requiring asynchronous lanes or grouping |
| Gradient-free importance, 4096/8192 proposals | **115x / 57.5x raw compute ceiling** | Requested forward = one-third gradient assumption; excludes proposal, memory, and importance ESS |

Using measured one-channel forward/gradient ratios gives raw 4096/8192 ceilings
of 149/75x (SOSS), 141/70x (G395H), and 100/50x (PRISM). At 40-channel batching
they are 75/37x, 356/178x, and 88/44x. These are cost ceilings, not posterior
speedups; without measured importance ESS and tail coverage they fail the
fidelity requirement by construction. The individual ceilings also must not be
multiplied because synchronized steps already include lockstep and measured wall
already includes warmup.

## Recommendations

1. Validate per-channel Hessian metrics on full real chunks. Pass the inverse
   Hessian as `inverse_mass_matrix`, set `adapt_mass_matrix=False`, retain
   step-size adaptation, use 150 warmup, target 0.95, and depth 10. This was
   divergence-free and 5.80x faster / 6.16x higher ESS/s on matched input.
2. Do not use one joint dense matrix; it was slower than the current diagonal
   path. The block geometry is lane-specific.
3. Do not cap depth at 5/6 or use 50 warmup yet; savings were negligible and
   divergences rose.
4. Reduce warmup only after fixing the metric. Preconditioning is the large
   lever; halving warmup added 1.16x.
5. Reduce lockstep through asynchronous execution or curvature/step-size lane
   buckets. The measured full-40 penalty is 3.56x.
6. Apply the Hessian metric to white light first. The pipeline already has a
   MAP-like start and the measured gains are 22x in steps and 58x in ESS/s.
   Confirm with at least 1000 clean retained draws before changing defaults.
7. Do not fix jitter in production. It is non-Gaussian but did not drive trees;
   fixing it would change the model and has no measured performance rationale.

## Failures and open risks

- The unmodified 40-lane independent dense reference (500 warmup, 1000 draws)
  compiled into a monolithic CPU scan, produced no output/checkpoint for 90
  minutes, and was interrupted. There is no speed number from it.
- The provisional reference has 0.238% divergences. No candidate passed the
  requested all-sites 0.1-sigma gate. High-ESS real-input fidelity is the main
  unresolved risk.
- Contact branches impede optimizer line searches. Hessians are positive
  definite and useful as metrics, but some absolute MAP gradients remain high.
- Real dumps were used for geometry only. Full real PRISM NUTS belongs on the
  orchestrator's GPU run, not the login CPU.
- Wall timings include JIT unless phases are separated. GPU compilation,
  vectorization, and memory ratios can differ.
- Importance numbers are ceilings, not an implemented inference method.
- The shell emits harmless module `/dev/log` and CUDA-plugin discovery errors
  even with `JAX_PLATFORMS=cpu`; all saved execution metadata confirmed CPU.

## Exact reproduction commands

Run from `/project/ekempton/tfairnington/JWST`:

```bash
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=true
export OMP_NUM_THREADS=8
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
PY=/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python
```

Geometry:

```bash
taskset -c 0-15 "$PY" tools/diag_nuts/spectro_geometry.py \
  --channels 40 --cadences 230 --reference-channels 40 \
  --output acceleration_reports/diag_nuts/spectro_geometry.json \
  --npz acceleration_reports/diag_nuts/spectro_geometry.npz

taskset -c 0-15 "$PY" tools/diag_nuts/spectro_geometry.py \
  --channels 1 --cadences 230 --negligible-jitter \
  --output acceleration_reports/diag_nuts/spectro_geometry_negligible_jitter.json \
  --npz acceleration_reports/diag_nuts/spectro_geometry_negligible_jitter.npz

taskset -c 0-15 "$PY" tools/diag_nuts/spectro_geometry.py \
  --stage-input /scratch/midway3/tfairnington/accel_stage_inputs/HAT-P-12_NIRISS_SOSS_order1_Rreference_high_resolution_inputs.pkl \
  --channels 40 --maxiter 700 \
  --output acceleration_reports/diag_nuts/real_soss_geometry.json \
  --npz acceleration_reports/diag_nuts/real_soss_geometry.npz

taskset -c 0-15 "$PY" tools/diag_nuts/spectro_geometry.py \
  --stage-input /scratch/midway3/tfairnington/accel_stage_inputs/GJ-3470_NIRSPEC_G395H_nrs1_R300_high_resolution_inputs.pkl \
  --channels 40 --maxiter 700 \
  --output acceleration_reports/diag_nuts/real_g395h_geometry.json \
  --npz acceleration_reports/diag_nuts/real_g395h_geometry.npz

taskset -c 0-15 "$PY" tools/diag_nuts/spectro_geometry.py \
  --stage-input /scratch/midway3/tfairnington/accel_stage_inputs/HAT-P-65_NIRSPEC_PRISM_nrs1_R50_high_resolution_inputs.pkl \
  --channels 8 --maxiter 700 \
  --output acceleration_reports/diag_nuts/real_prism_geometry.json \
  --npz acceleration_reports/diag_nuts/real_prism_geometry.npz
```

Provisional reference, shape comparison, and jitter ablation:

```bash
taskset -c 0-15 "$PY" tools/diag_nuts/spectro_samplers.py \
  --config laplace_d10_w150 --channels 40 --reference-channels 40 \
  --samples 1000 --geometry acceleration_reports/diag_nuts/spectro_geometry.npz \
  --reference /tmp/diag_nuts_no_reference.npz \
  --output-dir acceleration_reports/diag_nuts
cp acceleration_reports/diag_nuts/laplace_d10_w150_draws.npz \
  acceleration_reports/diag_nuts/reference_draws.npz

taskset -c 0-15 "$PY" tools/diag_nuts/spectro_samplers.py \
  --config non_gaussian --channels 40 --reference-channels 40 \
  --geometry acceleration_reports/diag_nuts/spectro_geometry.npz \
  --reference acceleration_reports/diag_nuts/reference_draws.npz \
  --output-dir acceleration_reports/diag_nuts

taskset -c 0-15 "$PY" tools/diag_nuts/spectro_samplers.py \
  --config jitter_ablation --channels 1 --negligible-jitter \
  --geometry acceleration_reports/diag_nuts/spectro_geometry_negligible_jitter.npz \
  --reference /tmp/diag_nuts_no_reference.npz \
  --output-dir acceleration_reports/diag_nuts
```

Eight-channel views and matched sweep:

```bash
taskset -c 0-15 "$PY" tools/diag_nuts/spectro_geometry.py \
  --channels 8 --cadences 230 --reference-channels 40 \
  --output /tmp/spectro_geometry_8of40.json \
  --npz acceleration_reports/diag_nuts/spectro_geometry_8of40.npz

"$PY" -c 'import numpy as np; p="acceleration_reports/diag_nuts/reference_draws.npz"; q="acceleration_reports/diag_nuts/reference_draws_8of40.npz"; a=np.load(p); np.savez_compressed(q, **{k:(a[k][:,:8] if a[k].ndim>1 and a[k].shape[1]==40 else a[k]) for k in a.files})'

for CFG in joint_diag joint_dense independent_dense laplace_d10_w300 \
  laplace_d10_w150 laplace_d5_w150 laplace_d6_w150 \
  laplace_map_d10_w50 laplace_d10_w150_ta90 laplace_d10_w150_ta95; do
  taskset -c 0-15 "$PY" tools/diag_nuts/spectro_samplers.py \
    --config "$CFG" --channels 8 --reference-channels 40 \
    --warmup 300 --samples 300 \
    --geometry acceleration_reports/diag_nuts/spectro_geometry_8of40.npz \
    --reference acceleration_reports/diag_nuts/reference_draws_8of40.npz \
    --output-dir acceleration_reports/diag_nuts/sweep8
done
```

The failed standard reference command was:

```bash
taskset -c 0-15 "$PY" tools/diag_nuts/spectro_samplers.py \
  --config reference --channels 40 --reference-channels 40 \
  --warmup 500 --samples 1000 \
  --reference acceleration_reports/diag_nuts/standard_reference_draws.npz
# Interrupted manually after 90 minutes with no completed scan/checkpoint.
```

White light and timing:

```bash
taskset -c 0-15 "$PY" tools/diag_nuts/white_light.py --mode geometry \
  --cadences 2000 --output-dir acceleration_reports/diag_nuts
taskset -c 0-15 "$PY" tools/diag_nuts/white_light.py --mode production \
  --cadences 2000 --warmup 300 --samples 300 \
  --output-dir acceleration_reports/diag_nuts
taskset -c 0-15 "$PY" tools/diag_nuts/white_light.py --mode laplace \
  --cadences 2000 --warmup 100 --samples 300 \
  --output-dir acceleration_reports/diag_nuts
taskset -c 0-15 "$PY" tools/diag_nuts/white_light.py --mode non_gaussian \
  --cadences 2000 --output-dir acceleration_reports/diag_nuts

taskset -c 0-15 "$PY" tools/diag_nuts/potential_timing.py \
  --cadences 230,2058,40787 --channels 1,40 --repeats 10 \
  --output acceleration_reports/diag_nuts/potential_timing.json
```

Verification:

```bash
"$PY" -m py_compile tools/diag_nuts/common.py \
  tools/diag_nuts/spectro_geometry.py tools/diag_nuts/spectro_samplers.py \
  tools/diag_nuts/white_light.py tools/diag_nuts/potential_timing.py

taskset -c 0-15 "$PY" -m pytest tests/test_independent_nuts.py -x -q
# 6 passed in 75.70s
```
