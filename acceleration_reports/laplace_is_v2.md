# Laplace-IS v2: exact-Hessian MAP, lognormal jitter, GPU measurements

## Executive result

This follow-up fixed the engineering bottlenecks in the first implementation,
but it did **not** establish a production-safe 10x speedup.

- G395H is the favorable regime.  All 5 low-resolution and all 40
  high-resolution lanes passed the configured k-hat/IS-ESS/IMH gate without
  fallback.  Compile-inclusive V100 speedups over like-for-like lognormal-prior
  joint NUTS were 2.55x (low) and 3.10x (high).  Removing recorded compilation
  time gives derived, not separately warmed, speedups of 40.8x and 31.0x.
- SOSS is not adequately represented by one elliptical Student-t proposal.
  Only 6/24 low-resolution and 6/40 first high-resolution lanes passed the
  operational gate.  Exact independent-NUTS fallback made those stages 1.38x
  and 1.28x *slower* than joint NUTS, respectively.
- The full 118-channel SOSS stage took 453.47 s versus 483.57 s for joint NUTS,
  only 1.07x faster.  Runner reuse reduced the non-fallback Laplace work after
  the first chunk to about 1.1 s/chunk, but 70 NUTS fallbacks consumed 370.77 s.
- PRISM returned 1000 draws in 1488.67 s.  Its MAP + three IS rounds + IMH
  path was only about 98.8 s, but two fallback lanes consumed 1389.82 s.  The
  supplied 200/200 joint reference took 1011.60 s; because draw counts differ,
  those walls are reported separately rather than called a speedup.
- None of the SOSS/G395H lanes reached the requested Newton decrement below
  `1e-8`.  After 200 exact-Hessian iterations the median decrement was
  `2.48e-7` to `3.48e-6`, depending on the stage.  PRISM behaved differently:
  14/21 lanes converged, at a median 173 iterations, while seven reached a
  similar numerical floor.  Every lane reports its status honestly.
- The lognormal jitter prior removed the old lower-prior plateau.  There is no
  evidence here for a special one-dimensional jitter proposal.  SOSS still
  has poor k-hat, so its remaining mismatch is in the correlated science/trend
  posterior rather than the removed jitter plateau.
- The fallback result is exact MCMC for failed lanes, but the default gate had
  false positives: the retained IMH lanes sometimes missed the calibrated
  fidelity gate.  I therefore do not recommend enabling this backend as the
  production default.

## What changed

### `models/laplace_is.py`

- Replaced L-BFGS with a fully vectorized trust-region/damped Newton optimizer
  in NumPyro unconstrained coordinates.  It uses `jax.hessian`, direct Newton
  solves, eigenvalue repair, backtracking, a Newton-decrement stopping rule,
  and per-lane iteration/decrement diagnostics.
- Reworked proposal evaluation into four reusable JIT programs: MAP,
  importance round, IMH/resampling, and postprocessing.
- Evaluates a large static draw block with `vmap` (256 draws for SOSS/G395H),
  rather than the old 16-draw `lax.map` path.  A conservative 4 GiB cadence
  budget automatically selects 64 draws for 21 x 40,780 PRISM data.
- Added an exact same-centre Student-t mixture option and covariance
  normalization.  Measurements below show that neither the naive nor the
  covariance-matched mixture fixed SOSS.
- Added optional exact-IMH thinning (`laplace_is_imh_thin`), detailed timing
  fields, Hessian diagnostics, Newton decrement/iteration diagnostics, and
  effective draw-chunk diagnostics.
- Reuses the independent-NUTS fallback runner across full-stage chunks, so the
  fallback executable is not rebuilt from scratch for every 40-channel block.
- Tuned defaults to the best tested single proposal: `nu=3`, scale inflation
  1.5, no wide component, 200 MAP iterations.  Fallback remains on by default.

### Other files

- `fit_jwst.py`: parses/validates the new mixture and IMH-thinning options and
  passes the updated defaults through the existing `laplace_is` integration.
- `tools/run_sampler_on_stage_inputs.py`: records all new per-stage diagnostics,
  accepts repeatable `--backend-override`, and uses the actual channel count as
  the resident lane width for small stages (24 for SOSS low, 5 for G395H low).
- `tests/test_laplace_is.py`: tests the Newton decrement, dynamic draw-chunk
  sizing, PSIS against ArviZ, deterministic/padded output, fallback splicing,
  runner reuse, option parsing, and thinned-IMH output shape.
- `tools/benchmark_laplace_is.py`: uses the 256-draw GPU-oriented default.
- Exact queue scripts are preserved under
  `acceleration_reports/gpu_queue/done/`.

I did not edit `models/independent_nuts.py`.  The concurrently added Newton
code there is embedded in its runner rather than exposed as a reusable helper,
so this backend retains its own implementation as allowed by the task.

## Final measured configuration

Unless explicitly marked as a sensitivity run:

| option | value |
|---|---:|
| jitter prior | `lognormal` |
| returned draws / IMH warmup | 1000 / 1000 |
| importance draws per round | 4096 |
| moment-matching rounds + final round | 2 + 1 |
| Student-t degrees of freedom | 3 |
| scale inflation | 1.5 |
| wide-mixture fraction | 0 |
| requested / SOSS effective draw block | 256 / 256 |
| PRISM effective draw block | 64 |
| MAP iteration budget / tolerance | 200 / `1e-8` |
| operational gate | k-hat < 0.7, IS-ESS >= 819.2, IMH acceptance >= 0.2 |
| preferred diagnostic | k-hat < 0.5 |
| fallback | independent NUTS, enabled |

All reported arrays and model evaluations remained float64.

## GPU provenance and timing

The rows below used the original, untagged Tesla V100 queue worker.  The joint
NUTS references were also untagged V100 runs with the lognormal prior.  These
are equal-returned-draw (1000), identical-input comparisons.  `steady-derived`
subtracts JAX's recorded compilation seconds from both walls; it is useful but
is not a separately warmed measurement.  The full-stage chunk rows below are
the stronger direct evidence for reuse.

| stage | joint NUTS wall (s) | Laplace-IS + fallback wall (s) | fallback lanes | compile-inclusive speedup | steady-derived speedup |
|---|---:|---:|---:|---:|---:|
| SOSS low 0:24 | 183.49 | 253.88 | 18/24 | 0.72x | 1.09x |
| G395H low 0:5 | 187.37 | 73.41 | 0/5 | 2.55x | 40.8x |
| SOSS high 0:40 | 200.95 | 257.20 | 34/40 | 0.78x | 1.19x |
| G395H high 0:40 | 238.15 | 76.85 | 0/40 | 3.10x | 31.0x |
| SOSS high 0:118 | 483.57 | 453.47 | 70/118 | 1.07x | 1.26x |

For reference, the older baseline numbers in the task (217, 200, 212, 269 s)
used a different run set.  I use the timing JSON beside each new-prior
reference above so prior, inputs, GPU, and returned draw count are all matched.

### Wall-time split

The MAP, IS, and IMH entries include their own first-call compilation.  The
separate `recorded compile` column overlaps those entries and must not be added
to them.

| stage | recorded compile (s) | MAP (s) | IS rounds (s) | IMH (s) | postprocess (s) | fallback (s) |
|---|---:|---:|---|---:|---:|---:|
| SOSS low 0:24 | 101.22 | 39.85 | 7.367, 0.039, 0.039 | 9.23 | 0.29 | 181.80 |
| G395H low 0:5 | 69.22 | 41.44 | 6.870, 0.037, 0.036 | 9.50 | 0.27 | 0 |
| SOSS high 0:40 | 107.10 | 43.97 | 7.864, 0.061, 0.061 | 9.45 | 0.29 | 179.64 |
| G395H high 0:40 | 69.89 | 42.59 | 7.909, 0.232, 0.232 | 9.98 | 0.28 | 0 |

The steady IS round is 0.039--0.061 s for SOSS and 0.232 s for the 2,058-cadence
G395H high block.  This verifies that batching removed the earlier
160--620 s draw-evaluation pathology; compilation, the MAP executable, IMH
compilation, and especially fallback now dominate.

### Full SOSS runner reuse

| channels | wall (s) | recorded compile (s) | MAP (s) | IS total (s) | IMH (s) | fallback lanes | fallback (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0:40 | 256.31 | 107.34 | 44.19 | 7.950 | 9.35 | 34 | 178.94 |
| 40:80 | 123.06 | 1.36 | 0.385 | 0.178 | 0.514 | 31 | 121.20 |
| 80:118 (padded to 40) | 73.95 | 2.76 | 0.382 | 0.178 | 0.512 | 5 | 70.63 |

The compiled Laplace runner is genuinely reused: its post-first-chunk core is
about 1.08 s.  The end-to-end post-compile chunk speedups over like-for-like
joint NUTS were only 1.39x and 1.51x because the exact fallback dominates.

## Proposal and MAP diagnostics

| stage | k-hat min/median/max | k<0.5 | k<0.7 | IS-ESS min/median/max | IMH accept min/median/max | MAP decrement median/max | converged |
|---|---|---:|---:|---|---|---|---:|
| SOSS low 0:24 | 0.325/0.602/0.843 | 8/24 | 18/24 | 234/657/1625 | 0.197/0.279/0.430 | 9.50e-7/5.84e-6 | 0/24 |
| G395H low 0:5 | 0.046/0.183/0.371 | 5/5 | 5/5 | 1796/2228/2599 | 0.444/0.506/0.587 | 3.48e-6/8.40e-6 | 0/5 |
| SOSS high 0:40 | 0.276/0.591/0.939 | 11/40 | 27/40 | 186/464/1176 | 0.000/0.252/0.342 | 2.48e-7/1.39e-6 | 0/40 |
| G395H high 0:40 | -0.120/0.119/0.343 | 40/40 | 40/40 | 2519/2589/2656 | 0.542/0.572/0.612 | 8.29e-7/3.32e-6 | 0/40 |
| SOSS high 0:118 | 0.200/0.567/0.939 | 40/118 | 96/118 | 77/697/2268 | 0.000/0.303/0.568 | 6.15e-7/4.13e-6 | 0/118 |

All SOSS/G395H lanes used all 200 requested Newton iterations.  Hessian condition numbers
were of order `1e6`--`8e7`, consistent with the independent diagnosis.  The
smallest achieved real-lane decrement was `4.03e-8`, still above the requested
threshold.  Continuing full Newton steps alternates at the reduction-noise
floor rather than decreasing to `1e-8`; relabeling those lanes converged would
be misleading.

## Fidelity against like-for-like joint NUTS

The pass counts use the calibrated, site-specific gates in
`acceleration_reports/reference.md`, not the driver's older flat 0.1-sigma /
10% diagnostic.  Noise sites use the lognormal-prior reference.  Every row in
each `*.comparison.json` gives the requested per-site, per-channel median,
sigma, p16, and p84 shifts.

| stage | calibrated science passes | science values | result |
|---|---:|---:|---|
| SOSS low 0:24 | 151 | 168 | fail |
| G395H low 0:5 | 22 | 30 | fail |
| SOSS high 0:40 | 256 | 280 | fail |
| G395H high 0:40 | 224 | 240 | fail |
| SOSS high 0:118 (global gates) | 736 | 826 | fail |

The medians of the per-channel science-site sigma ratios are close to one
(roughly 0.96--1.04), but isolated retained IMH lanes cause unacceptable
outliers.  Examples are SOSS-low c1 (maximum median shift 0.757 sigma,
sigma-ratio range 0.751--1.732), SOSS-high c2 (0.646 sigma maximum shift), and
G395H-high depth/rors (0.226 sigma maximum shift).  Full per-site summaries:

| stage/site | calibrated pass | median-shift min/median/max (sigma) | sigma-ratio min/median/max | max abs p16 / p84 shift (sigma) |
|---|---:|---|---|---|
| SOSS low A_spot | 22/24 | .011/.056/.220 | .801/1.000/1.060 | .185/.468 |
| SOSS low c,c1,c2 | 66/72 | max .405/.757/.718 | med .995/.978/.982 | max .206/2.417 |
| SOSS low depth,rors,v | 63/72 | max .409/.409/.753 | med 1.003/1.003/1.006 | max .223/.599 |
| G395H low c,c1,c2 | 13/15 | max .077/.089/.209 | med .958/1.000/1.024 | max .645/.112 |
| G395H low depth,rors,v | 9/15 | max .189/.189/.154 | med 1.043/1.043/1.006 | max .149/.259 |
| SOSS high A_spot | 37/40 | .001/.052/.207 | .912/1.008/1.153 | .402/.284 |
| SOSS high c,c1,c2 | 112/120 | max .347/.556/.646 | med 1.001/.983/.987 | max .368/.484 |
| SOSS high depth,rors,v | 107/120 | max .257/.257/.181 | med .996/.996/.996 | max .346/.285 |
| G395H high c,c1,c2 | 113/120 | max .141/.204/.149 | med .995/1.001/1.005 | max .264/.219 |
| G395H high depth,rors,v | 111/120 | max .226/.226/.185 | med 1.002/1.002/.988 | max .231/.278 |

The full-stage comparison used the global gate class thresholds because the
calibrated reference covers 0:40.  Its worst shifts were 0.853 sigma for
`A_spot`, 0.646 for c2, and 0.593 for v.

### ESS per second

These are median ArviZ bulk ESS over every stored scalar site/channel divided
by compile-inclusive wall.  The single-chain values are meaningful as a
relative efficiency diagnostic; R-hat is unavailable.

| stage | Laplace-IS median ESS / ESS s-1 | joint NUTS median ESS / ESS s-1 |
|---|---|---|
| SOSS low | 490 / 1.93 | 1051 / 5.73 |
| G395H low | 298 / 4.06 | 1160 / 6.19 |
| SOSS high 0:40 | 482 / 1.87 | 1127 / 5.61 |
| G395H high 0:40 | 360 / 4.69 | 2203 / 9.25 |
| SOSS high 0:118 | 328 / 0.72 | 1247 / 2.58 |

Thus even the wall-time-positive G395H cases did not improve ESS/second at
1000 consecutive IMH states.

## Tuning studies that did not fix SOSS

All timing claims in the main tables use the identical final configuration;
the following are diagnostic only.

1. **Heavier single t:** changing from `nu=5`, inflation 1.2 to `nu=3`,
   inflation 1.5 improved SOSS-low median k-hat from about 0.70 to 0.60 and
   made every G395H-low lane preferred (`k<0.5`).
2. **Naive 10% 3x mixture:** 8192 draws and three adaptation rounds yielded
   SOSS-low median/max k-hat 0.663/0.882, median ESS 1100/8192, and only 1/24
   operational passes.
3. **Covariance-matched 5% 3x mixture:** median/max k-hat 0.632/0.891 and
   median ESS 767/8192; again only 1/24 passed.  It confirms that simply adding
   radial tails does not model the SOSS posterior geometry.
4. **Four IMH transitions per retained draw:** G395H-low calibrated science
   passes improved from 22/30 to 28/30 and scalar-site median bulk ESS rose to
   roughly 669--819 for only 0.63 s more wall.  SOSS without fallback improved
   to 123/168 science passes but remained unacceptable.  The option is kept,
   default 1, for future tests rather than declared a fix.

## PRISM low 0:21

The required fallback-enabled run completed on the original untagged V100.
The RTX6000 duplicate is excluded.  The comparison reference used the old
jitter prior and only 200 warmup + 200 returned draws, so there is deliberately
no equal-draw speedup claim.  Science sites are still compared because the
jitter-prior intervention does not alter their model; `log_jitter`,
`total_error`, and the reference's zero-variance limb-darkening rows are not
treated as a reliable prior comparison.

| quantity | Laplace-IS, lognormal prior | joint NUTS reference, old prior |
|---|---:|---:|
| warmup / returned draws | 1000 / 1000 | 200 / 200 |
| V100 wall | 1488.67 s | 1011.60 s |
| recorded compile | 84.38 s | 18.55 s |
| median bulk ESS / ESS s-1 | 218 / 0.146 | 200 / 0.198 |

The Laplace-IS split was MAP 52.29 s, importance rounds 11.49/5.55/5.55 s,
IMH 9.38 s, postprocessing 0.28 s, and independent-NUTS fallback **1389.82 s**.
Thus the warmed three-round forward arithmetic is about 16.65 s, close to the
15 s prediction.  Only two failed lanes made the end-to-end method slower than
the much shorter 200/200 joint reference.  The pre-fallback candidate path was
about 98.8 s, but it is neither an equal-draw NUTS comparison nor a valid final
answer for those two lanes.

| k-hat / sampler diagnostic | result |
|---|---|
| k-hat min/median/max | 0.245 / 0.434 / 0.835 |
| k-hat <0.5 / <0.7 | 15/21 / 20/21 |
| operational passes / fallbacks | 19/21 / 2/21 |
| IS-ESS min/median/max | 334 / 1351 / 2049 |
| IMH acceptance min/median/max | 0.159 / 0.416 / 0.528 |
| Newton converged | 14/21 |
| Newton iterations median/max | 173 / 200 |
| decrement min/median/max | 3.74e-9 / 8.31e-9 / 3.93e-6 |

Using the global calibrated science gates gave 92/147 passes.  Per-site counts
were A 17/21, c 15/21, c1 8/21, c2 12/21, depth 12/21, rors 12/21, and v 16/21.
Depth/rors had median shifts of 0.051 sigma, median sigma ratios 0.987, and
maximum shifts 0.381 sigma.  A, c, and v had maximum shifts 0.323, 0.265, and
0.395 sigma.  The 200-draw reference has zero or undefined c1/c2 variance in
seven rows, making those seven sigma-ratio gates undefined; they are counted
as failures rather than silently discarded.  Full per-channel median, width,
p16, and p84 rows are in `prism_low_0_21.comparison.json`.

The 32-iteration, fallback-off sensitivity completed in 82.87 s on the second
V100, but it is invalid inference: 0/21 MAP lanes converged, median decrement
was 2242, median k-hat was 6.52, median IS-ESS was approximately 1, and median
IMH acceptance was zero.  PRISM genuinely needs the longer optimizer; simply
cutting MAP iterations is not an acceptable acceleration.

## Tests

Final command:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
taskset -c 0-15 \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  -m pytest tests/test_laplace_is.py -x -q
```

Result: **7 passed, 2 warnings in 245.31 s**.  The warnings are the existing
`pkg_resources` and JAXopt deprecations.

## Exact reproduction

GPU commands are preserved verbatim in these queue scripts.  To reproduce,
copy the desired script into `acceleration_reports/gpu_queue/pending/` and let
the orchestrator dispatcher execute it; do not invoke Slurm directly.

| purpose | script |
|---|---|
| final low stages, fallback on | `acceleration_reports/gpu_queue/done/75_laplace_v2_final_lows.sh` |
| final high 0:40 stages, fallback on | `acceleration_reports/gpu_queue/done/76_laplace_v2_final_highs.sh` |
| full SOSS 0:118, fallback on | `acceleration_reports/gpu_queue/done/77_laplace_v2_final_soss_full.sh` |
| IMH thin=4 sensitivity | `acceleration_reports/gpu_queue/done/79_laplace_v2_low_thin4.sh` |
| PRISM 200-iteration V100 | `acceleration_reports/gpu_queue/done/80_laplace_v2_prism_low_v100.sh` |
| PRISM 32-iteration, fallback-off sensitivity | `acceleration_reports/gpu_queue/done/81_laplace_v2_prism_map32_v100.sh` |

Calibrated comparisons were reproduced with, for example:

```bash
JAX_PLATFORMS=cpu \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  tools/diag_nuts/summarize_precond.py \
  --candidate /scratch/midway3/tfairnington/accel_gpu_results/76_laplace_v2_final_highs/g395h_high_0_40.pkl \
  --reference /scratch/midway3/tfairnington/accel_gpu_results/42_g395_high_lognormal_joint/lognormal_joint_1000_1000.pkl \
  --noise-floor /scratch/midway3/tfairnington/accel_stage_inputs/references/GJ-3470_NIRSPEC_G395H_nrs1_R300_high_resolution_inputs_ch0_40_pooled_joint_nuts_noise_floor.json \
  --start 0 --end 40 \
  --output acceleration_reports/data/laplace_v2_76_g395h_high_calibrated.json
```

Equivalent calibrated JSON files for all four low/high blocks are in
`acceleration_reports/data/laplace_v2_75_*` and
`acceleration_reports/data/laplace_v2_76_*`.  Raw samples, timing, diagnostics,
ArviZ ESS, and per-channel comparison rows are under
`/scratch/midway3/tfairnington/accel_gpu_results/{75,76,77,79,80,81}_*/`.

## Failures and open risks

- **The requested MAP convergence was not achieved across all lanes.**  The algorithm reaches
  a float64 reduction/gradient floor around `1e-7`--`1e-6`.  Fourteen PRISM
  lanes did reach the strict tolerance, but seven did not.  A robust production
  implementation needs a numerically scaled objective or a demonstrably
  justified decrement criterion, not a larger iteration budget.
- **SOSS is non-elliptical at the required accuracy.**  More t tails, more
  importance draws, and radial mixtures do not help.  A skewed/multi-centre
  proposal or a transport method would be needed, with fresh PSIS validation.
- **The configured gate has false positives.**  In SOSS low, every calibrated
  science failure after fallback was in one of the six retained IMH lanes.
  Tightening the gate would restore fidelity only by falling back on nearly all
  SOSS lanes, eliminating acceleration.
- **IMH Monte Carlo efficiency is low.**  Thin=4 helps G395H, but its final
  calibrated pass count is still 28/30 and ESS/second remains below joint NUTS.
- **Strict MAP convergence is diagnostic-only in the current operational
  gate.**  Adding `converged` to the gate would make every measured SOSS/G395H
  lane and seven PRISM lanes fall back, which is honest but reduces this
  backend to a slower wrapper around NUTS in most measured stages.
- **No 10x compile-inclusive result was measured.**  The arithmetic after
  compilation can exceed 10x in the no-fallback G395H regime, but compilation
  and IMH reduce equal-draw end-to-end speedups to 2.55--3.10x.
- **PRISM fallback is prohibitive.**  Two lanes consumed 1389.82 s of NUTS,
  while all MAP + IS + IMH work consumed under 100 s.  Its science comparison
  also failed 55/147 global calibrated rows against the limited 200-draw
  reference.

The safe disposition is to keep `laplace_is` opt-in with fallback enabled and
not promote it as the spectroscopic production default.
