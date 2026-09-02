# Finite-difference Laplace metrics and fixed-trajectory HMC

## Outcome

I added two opt-in improvements while preserving every existing default:

1. a central-finite-difference (FD) Hessian of the exact float64 gradient,
   including the existing Newton/trust-region MAP search, spectral repair, and
   step-size-only warmup; and
2. the same fixed Laplace metric in the independent HMC backend, with a fixed
   number of leapfrog steps and optional per-draw/per-lane trajectory jitter.

The most efficient measured configuration is eight-step HMC with trajectory
length uniformly jittered over 6--10 steps.  On one V100 allocation, with
1000 returned draws per channel and one cached runner build per width, it took:

| Full high-resolution stage | Production joint NUTS | FD Laplace HMC | Measured full-stage speedup |
|---|---:|---:|---:|
| Stellar-prior SOSS, 118 channels | 623.75 s | 77.76 s | **8.02x** |
| G395H, 74 channels | 544.69 s | 87.48 s | **6.23x** |

Those are same-GPU, equal-returned-draw ratios.  They compare the complete
requested candidate (including the lognormal jitter prior) with the production
log-uniform-prior baseline, so they contain both a prior and an inference
effect.  On stellar SOSS channel 0:40, the separately measured prior-only
effect was 198.47/148.66 = **1.34x**.  Against same-prior joint NUTS, the
compile-inclusive first HMC chunk was **2.12x** faster.

The sampler itself is now cheap after compilation: cached 40-channel HMC chunks
took 3.37 s for SOSS and 9.92 s for the padded 34-channel G395H chunk.  However,
the requested <=25 s compile target was **not met**.  First-width compilation
was 63.69 s for SOSS HMC and 65.55 s for G395H HMC.  A cached-width runner is
therefore essential.

Fidelity is strong but not sufficient to make HMC the silent default yet:

- G395H channel 0:40 passed **240/240** calibrated science coordinates.
- The full SOSS HMC run passed **279/280** under my literal reconstruction of
  the supplied class thresholds.  The sole miss was `c`, channel 16, at
  0.1687 reference sigma versus a 0.164 limit (1.03 times the threshold).
  The orchestrator's accepted exact-Hessian NUTS run also scores 279/280 under
  this reconstruction despite its recorded official verdict that every
  science coordinate passes.  I therefore report the discrepancy rather than
  reinterpret the threshold.
- Full G395H HMC had one divergence among 74,000 retained lane draws; SOSS had
  zero.  A multi-seed qualification run remains advisable before changing the
  production default.

The conservative production-safe opt-in remains FD Laplace independent NUTS
with depth 5.  Its measured full-stage ratios were 5.53x (SOSS) and 4.53x
(G395H), with 279/280 and 240/240 science coordinates passing, respectively.

## Files built or changed

| File | Change |
|---|---|
| `models/independent_nuts.py` | Added FD Hessian construction, scaled FD steps, optional exact-metric comparison, diagonal MAP diagnostic variant, fused execution, and sampler-independent Laplace preparation.  Added jittered fixed-work HMC execution to the reusable runner. |
| `models/independent_hmc.py` | Added opt-in `mass_matrix=laplace`, FD/exact metric options, fixed-metric step-size warmup, trajectory jitter, diagnostics, and fused runner support.  Adaptive defaults are unchanged. |
| `fit_jwst.py` | Minimally wired Laplace HMC, Hessian method/step, trajectory jitter, and fused-program flags through the existing resolver and validation. |
| `tests/test_independent_nuts.py` | Added exact-versus-FD metric accuracy, fused/diagonal-path, and cached-runner reuse coverage. |
| `tests/test_independent_hmc.py` | Added fixed-metric FD HMC, jittered work bounds, diagnostics, and pipeline flag tests. |
| `tools/run_sampler_on_stage_inputs.py` | Added the new MAP/Hessian diagnostic to durable output. |
| `tools/diag_nuts/summarize_precond.py` | Added FD metric-error summaries. |
| `tools/diag_nuts/run_precond2_pilots.sh` | Reproducible NUTS-depth and HMC 8/16/32-step matrix. |
| `tools/diag_nuts/run_precond2_metric_check.sh` | Real-input exact-versus-FD Hessian comparison. |
| `tools/diag_nuts/run_precond2_diagonal.sh` | Failed diagonal-MAP experiment. |
| `tools/diag_nuts/run_precond2_full_baseline.sh` | Full-stage production joint-NUTS measurement. |
| `tools/diag_nuts/run_precond2_full_candidate.sh` | Full-stage fused FD Laplace NUTS measurement. |
| `tools/diag_nuts/run_precond2_full_hmc.sh` | Full-stage fused FD Laplace HMC measurement. |
| `tools/diag_nuts/build_precond2_matrix.py` | Machine-readable gate/ESS/timing matrix builder. |
| `acceleration_reports/diag_nuts/precond2/*.json` | Machine-readable comparisons and summaries. |

I did not edit `models/laplace_is.py`, the model builder, likelihood, priors,
transit physics, or any data/mask code.

## Implementation

### Finite-difference metric

For unconstrained coordinate `i`, the implementation evaluates

```text
H[:, i] = (gradient(x + h_i e_i) - gradient(x - h_i e_i)) / (2 h_i)
```

and symmetrizes the result before applying the existing eigenvalue floor.  A
first diagonal curvature estimate scales `h_i`; the scale is clipped to keep
the perturbations finite in weakly identified directions.  All `2*d`
perturbations and all lanes are vmapped with static shapes.  No SciPy, reduced
precision, model approximation, or transit grid is used.

On the real stellar-prior SOSS 0:40 MAP, using relative step `2e-4`, the FD
metric's relative Frobenius error against `jax.hessian` was:

| Statistic over 40 lanes | Relative error |
|---|---:|
| Median | **2.874e-6** |
| Maximum | **1.143e-5** |

The corresponding MAP gradient norm was 0.00167 median / 0.0679 maximum and
the repaired Hessian condition number was `9.61e6` median.

### Fixed-metric HMC

Laplace HMC uses the repaired inverse Hessian as NumPyro's dense
`inverse_mass_matrix`, sets `adapt_mass_matrix=False`, and adapts only step size
for 150 iterations.  With `num_steps=8` and `trajectory_jitter=0.25`, each lane
independently draws 6--10 leapfrog steps.  Thus there is no data-dependent NUTS
tree stopping or tree-building overhead.  The observed lane-max/mean work ratio
is about 1.25, but this is intentional randomized trajectory length rather than
slow-lane synchronization waste.  Setting jitter to zero gives exactly 1.0.

### New production flags

The measured HMC configuration is expressible as:

```yaml
spectro_sampler: independent_hmc
spectro_mass_matrix: laplace
spectro_jitter_prior: lognormal
spectro_laplace_hessian_method: finite_difference
spectro_laplace_fd_relative_step: 0.0002
spectro_laplace_warmup: 150
spectro_laplace_target_accept: 0.85
spectro_laplace_start_at_map: false
spectro_laplace_fuse_program: true
spectro_hmc_num_steps: 8
spectro_hmc_trajectory_jitter: 0.25
```

All defaults remain adaptive mass, no trajectory jitter, exact Hessian when an
opt-in Laplace NUTS path is selected, and an unfused program.  The global
pipeline sampler default is unchanged.

## Compile-time experiments

The compile target was not reached.

| Stellar SOSS 0:40 configuration | GPU | Backend / JAXPR / MLIR events | Recorded compile | Wall |
|---|---|---:|---:|---:|
| Exact Hessian NUTS, orchestrator `C` | V100 | 329 / 360 / 329 | 77.56 s | 94.34 s |
| FD Hessian NUTS, unfused | RTX 6000 | 329 / 374 / 329 | 76.29 s | 106.54 s |
| FD Hessian NUTS, fused | V100 | 328 / 373 / 328 | 75.73 s first chunk | 91.42 s first chunk |
| FD Hessian HMC-8+jitter, fused | V100 | 328 / 351 / 328 | **63.69 s** first chunk | **70.06 s** first chunk |
| Production joint NUTS | V100 | 341 / 330 / 341 | 21.87 s first chunk | 197.62 s first chunk |

The unfused FD timing is on RTX and is not used for a speedup.  The event
counts answer the structural question: changing exact Hessians to `2*d` exact
gradients did not remove the hundreds of compiled transit-potential
subcomputations.  Fusing preparation, warmup, sampling, and postprocessing
reduced two explicit top-level JIT programs to one, but only reduced backend
events from 329 to 328.  It does remove redundant Python/JIT boundaries and is
retained as an opt-in.

I also replaced Newton curvature with a diagonal approximation as a diagnostic.
It compiled in 71.65 s, still far above target, and failed badly: median MAP
gradient norm 655, median Newton decrement 77.9, mean NUTS work 28.0 steps,
minimum science ESS 1.31, and only 247/280 science coordinates passed.  It is
not a candidate.

## NUTS depth cap

All rows use FD curvature, 150 warmup iterations, 1000 returned draws, and the
lognormal jitter prior.

| Dataset, 0:40 | Depth | GPU | Wall | Compile | Lane mean | Lane max mean | Divergences | Science gate |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Stellar SOSS | 10 | RTX 6000 | 106.39 s | 76.29 s | 9.05 | 24.93 | 0 | 279/280 |
| Stellar SOSS | 5 | RTX 6000 | 104.42 s | 76.29 s | 9.14 | 24.69 | 0 | 279/280 |
| G395H | 10 | V100 | 107.68 s | 80.60 s | 8.97 | 17.74 | 0 | 240/240 |
| G395H | 5 | V100 | 102.68 s | 79.75 s | 9.00 | 16.87 | 0 | 240/240 |

Depth 5 leaves typical work unchanged but caps rare long trees.  It reduced the
G395H post-compile residual from 27.08 to 22.93 s and preserved every science
gate, so it is the recommended NUTS cap for this preconditioned path.

## Fixed-trajectory matrix

`Median ESS` below is the median of per-site median bulk ESS over science
sites.  `ESS/s` includes compile.  `Eq.-1000 wall` is `wall * 1000 / median
ESS`; ESS above 1000 is possible for antithetic chains.  The SOSS pilot timing
is RTX-only and is presented for configuration selection, not as a V100
speedup.

### Stellar-prior SOSS high 0:40 (RTX 6000)

| Sampler | Wall / compile | Science gate | ESS min / median | Median ESS/s | Eq.-1000 wall | Div. |
|---|---:|---:|---:|---:|---:|---:|
| NUTS depth 5 | 104.42 / 76.29 s | 279/280 | 499.9 / 1454 | 13.93 | 71.80 s | 0 |
| HMC 8, no jitter | 71.45 / 61.89 s | 268/280 | 439.8 / 1489 | 20.84 | 47.99 s | 0 |
| **HMC 8, ±25%** | **75.00 / 63.54 s** | **276/280** | 308.5 / 1673 | **22.31** | **44.82 s** | 0 |
| HMC 16, no jitter | 77.99 / 62.08 s | 248/280 | 24.2 / 2180 | 27.95 | 35.78 s | 0 |
| HMC 16, ±25% | 83.05 / 63.60 s | 273/280 | 225.5 / 1008 | 12.13 | 82.43 s | 0 |
| HMC 32, no jitter | 90.37 / 61.91 s | 241/280 | 1.52 / 765 | 8.47 | 118.11 s | 0 |
| HMC 32, ±25% | 98.73 / 63.31 s | 276/280 | 264.1 / 906 | 9.18 | 108.95 s | 0 |

### G395H high 0:40 (V100)

| Sampler | Wall / compile | Science gate | ESS min / median | Median ESS/s | Eq.-1000 wall | Div. |
|---|---:|---:|---:|---:|---:|---:|
| NUTS depth 5 | 102.68 / 79.75 s | 240/240 | 728.3 / 1250 | 12.18 | 82.11 s | 0 |
| HMC 8, no jitter | 76.70 / 66.48 s | 214/240 | 680.8 / 3000 | 39.11 | 25.57 s | 0 |
| **HMC 8, ±25%** | **79.00 / 67.08 s** | **240/240** | 341.2 / 3000 | **37.97** | **26.33 s** | 0 |
| HMC 16, no jitter | 83.44 / 66.15 s | 142/240 | 1.33 / 110 | 1.32 | 757.84 s | 1 |
| HMC 16, ±25% | 87.90 / 67.75 s | 231/240 | 195.4 / 476 | 5.42 | 184.59 s | 1 |
| HMC 32, no jitter | 97.77 / 66.58 s | 190/240 | 1.44 / 509 | 5.20 | 192.26 s | 9 |
| HMC 32, ±25% | 104.65 / 67.52 s | 239/240 | 378.1 / 977 | 9.33 | 107.15 s | 4 |

The non-jittered fixed lengths have strong resonance failures: large median ESS
can coexist with failed coordinates and near-unit ESS in particular sites.
Jitter is necessary.  Eight steps is clearly better than 16 or 32 on these
metrics; longer trajectories reintroduce resonances and G395H divergences.

## Full-stage V100 measurements

All six full stages below ran sequentially on the same Tesla V100-PCIE-16GB
allocation (`57128921`, second V100).  Each returns 1000 draws per channel.

| Dataset / sampler | Chunk walls (compile) | Full wall | Full compile | Ratio vs production |
|---|---|---:|---:|---:|
| SOSS production joint, log-uniform | 197.62 (21.87), 243.45 (8.74), 182.53 (21.34) s | **623.75 s** | 51.94 s | 1.00x |
| SOSS FD Laplace NUTS d5, lognormal | 91.42 (75.73), 11.39 (0), 9.89 (0.75) s | **112.86 s** | 76.49 s | **5.53x** |
| SOSS FD Laplace HMC-8±25%, lognormal | 70.06 (63.69), 3.37 (0), 4.17 (0.74) s | **77.76 s** | 64.42 s | **8.02x** |
| G395H production joint, log-uniform | 267.49 (22.46), 277.06 (21.62) s | **544.69 s** | 44.08 s | 1.00x |
| G395H FD Laplace NUTS d5, lognormal | 98.98 (76.01), 20.98 (0.83) s | **120.12 s** | 76.84 s | **4.53x** |
| G395H FD Laplace HMC-8±25%, lognormal | 77.40 (65.55), 9.92 (0.82) s | **87.48 s** | 66.37 s | **6.23x** |

Same-GPU first-chunk and cached-chunk ratios are:

| Dataset / candidate | First chunk vs production | Cached subsequent chunk vs production |
|---|---:|---:|
| SOSS NUTS d5 | 2.16x | 21.37x (equal-width channel 40:80) |
| SOSS HMC-8±25% | **2.82x** | **72.15x** (equal-width channel 40:80) |
| G395H NUTS d5 | 2.70x | 13.20x (padded 34-lane second chunk) |
| G395H HMC-8±25% | **3.46x** | **27.92x** (padded 34-lane second chunk) |

The G395H cached ratio is not an equal-width microbenchmark; it compares the
actual second production chunk with the actual padded candidate chunk.  No
cross-GPU ratios are reported as speedups.

The full HMC science-site ESS minima/medians were:

| Site | SOSS, min / median | G395H, min / median |
|---|---:|---:|
| `A_spot` | 682 / 3000 | -- |
| `c` | 312 / 1575 | 314 / 3000 |
| `v` | 964 / 3000 | 308 / 3000 |
| `c1` | 326 / 1747 | 464 / 3000 |
| `c2` | 319 / 1655 | 365 / 3000 |
| `depths` / `rors` | 311 / 1615 | 445 / 3000 |
| `log_jitter` / `total_error` | 128 / 516 | 335 / 1898 |

Using the median science-site ESS, the HMC full-stage wall adjusted to 1000
effective draws is 39.18 s for SOSS and 29.16 s for G395H.  The corresponding
G395H NUTS figure is 96.06 s.  These are ESS normalizations, not additional
wall measurements.

Full-stage HMC sampler accounting:

| Dataset | Mean steps | Maximum allowed | Mean accept | Divergences |
|---|---:|---:|---:|---:|
| SOSS, 118 channels | 8.00 | 10 | 0.948 across chunks | 0 |
| G395H, 74 channels | 8.00 | 10 | 0.948 across chunks | **1** |

## Prior effect separated from sampler effect

The orchestrator's same-input SOSS 0:40 V100 runs were:

| Run | Prior / sampler | Wall | Compile | Lane work | Science verdict |
|---|---|---:|---:|---:|---|
| A | log-uniform / production joint NUTS | 198.47 s | 21.76 s | median 127 | reference |
| B | lognormal / production joint NUTS | 148.66 s | 22.25 s | median 63 | official science pass |
| C | lognormal / exact Laplace independent NUTS | 94.34 s | 77.56 s | lane mean 9.1, max 27.0 | official science pass |
| This work, full-run first chunk | lognormal / FD Laplace HMC-8±25% | 70.06 s | 63.69 s | lane mean 8.00, max 10 | 279/280 literal reconstructed science gates |

Thus A/B = 1.34x is attributable to the prior on this case.  Against B, the
first-chunk ratios are 1.63x for fused FD NUTS and 2.12x for fused FD HMC.  The
larger 8.02x full-stage HMC ratio additionally benefits from compile-cache
amortization across three chunks.

## Exact reproduction commands

GPU commands were executed only by the orchestrator through the documented
file queue.  Inside a GPU allocation, the durable drivers are:

```bash
bash tools/diag_nuts/run_precond2_metric_check.sh
bash tools/diag_nuts/run_precond2_pilots.sh soss
bash tools/diag_nuts/run_precond2_pilots.sh g395h
bash tools/diag_nuts/run_precond2_diagonal.sh

bash tools/diag_nuts/run_precond2_full_baseline.sh soss
bash tools/diag_nuts/run_precond2_full_candidate.sh soss
bash tools/diag_nuts/run_precond2_full_hmc.sh soss

bash tools/diag_nuts/run_precond2_full_baseline.sh g395h
bash tools/diag_nuts/run_precond2_full_candidate.sh g395h
bash tools/diag_nuts/run_precond2_full_hmc.sh g395h
```

Build the consolidated calibrated summary with:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 0-15 \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  tools/diag_nuts/build_precond2_matrix.py
```

CPU validation command:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 0-15 \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest \
  tests/test_independent_nuts.py tests/test_independent_hmc.py \
  tests/test_independent_runner_reuse.py -x -q
```

Result after the implementation: **19 passed**, with two dependency warnings.
The final resolver-only flag addition also passed its focused test.

## Failures and open risks

- **Compile target failed.**  FD curvature is numerically excellent but does
  not reduce the transit potential's many backend compilation events.  The
  remaining realistic engineering direction is a persistent JAX compilation
  cache across processes or reducing the model's internal compilation
  fragmentation, neither of which was implemented here.
- **Diagonal MAP failed.**  It neither compiled fast enough nor found a usable
  local metric.
- **Fixed-length resonance is real.**  No-jitter HMC and 16/32-step HMC can
  show attractive aggregate ESS while failing many coordinates.  Eight steps
  with jitter is the only tested HMC setting that passed all 240 G395H science
  gates.
- **One SOSS threshold miss remains.**  The full HMC run's single science miss
  is only 0.0047 reference sigma beyond the reconstructed limit, but it must
  not be called a complete literal gate pass.
- **One G395H divergence remains in the full 74-channel run.**  It did not
  occur in the 0:40 qualification chunk, and 0:40 passed every science gate,
  but production rollout should require a multi-seed/full-range check.
- **Single-chain ESS can exceed draw count.**  ArviZ reports antithetic ESS as
  high as 3000 for 1000 draws.  I report both minimum ESS and coordinate-level
  gates so the aggregate median cannot hide resonant failures.
- **Speedup decomposition matters.**  The full production ratios include the
  accepted lognormal prior change.  Same-prior first-chunk ratios are shown
  separately; no unmeasured same-prior full-stage ratio is claimed.

Recommended rollout: enable fused FD Laplace NUTS depth 5 first; retain HMC-8
with ±25% jitter as an explicit experimental backend until it passes a
multi-seed SOSS/G395H qualification with zero or an agreed negligible
divergence rate.  In both cases, process all same-width chunks in one process
so the 64--77 s runner build is paid once.
