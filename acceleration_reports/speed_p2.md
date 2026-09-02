# P2 follow-up: white-light quality gates and compilation work

## Status and decision

This is an honest interim report, not a completed speed claim.  The PRISM
TA=0.99 replication is queued behind the priority parity campaign.  The
white-light continuation gate and fixed-width Laplace-IS fallback compile
shape are implemented and pass focused CPU tests.  Automatic
Laplace-to-adaptive fallback, selective spectroscopic extension, cadence
padding, cold/warm cache timings, and the two-GPU PRISM equality run are not
yet complete.  Those incomplete items must not be enabled in production.

The global white-light default remains `adaptive`.

## 1. PRISM white-light validation

The first real V100 TA=0.99 run (seed 559) is encouraging but is not by itself
a three-seed validation:

| preparation | sampling | mean steps | accept | divergences |
|---:|---:|---:|---:|---:|
| 51.79 s | 48.18 s | 22.16 | 0.9813 | 0 |

Against the first two completed adaptive 1000/1000 seeds, the science-site
results were:

| site | median shift (reference sigma) | sigma ratio | bulk ESS |
|---|---:|---:|---:|
| `t0_0` | +0.001 | 1.064 | 975 |
| `b_0` | +0.069 | 0.948 | 567 |
| `logD_0` | +0.029 | 0.934 | 587 |
| `rors_0` | +0.078 | 0.958 | 575 |
| `c` | +0.006 | 1.086 | 374 |
| `v` | +0.006 | 1.079 | 393 |
| `log_jitter` | -0.078 | 1.076 | 1132 |
| `A` | +0.006 | 1.050 | 470 |
| `log_tau` | -0.037 | 1.132 | 148 |

The geometry shifts correspond to 1.39 ppm for `b` under the specified
20 ppm/sigma-b proxy; duration and t0 are respectively 0.029 and 0.001 of
their pooled posterior sigmas.  A physical ppm sensitivity was not inferred
for duration/t0 without propagating the full spectroscopic refit.

Seeds 560 and 561 use new paths under `wl_validation_p2/`.  The existing
third adaptive reference was still running when this report snapshot was
written.  Therefore the requested gate is **not yet adjudicated**.  If all
three TA=0.99 seeds are divergence-free and pass the pooled calibrated gate,
the proposed mode-specific configuration is:

```yaml
# SOSS and G395H
whitelight_mass_matrix: laplace
whitelight_laplace_warmup: 200
whitelight_laplace_target_accept: 0.9
whitelight_laplace_max_tree_depth: 10
whitelight_laplace_hessian_method: finite_difference

# PRISM (conditional on the unfinished three-seed gate)
whitelight_mass_matrix: laplace
whitelight_laplace_warmup: 200
whitelight_laplace_target_accept: 0.99
whitelight_laplace_max_tree_depth: 10
whitelight_laplace_hessian_method: finite_difference
```

## 2. ESS/divergence extension

`fit_jwst.py` now computes bulk ESS for `t0`, `b`, `logD`/duration and `rors`
from grouped chains.  It also computes split R-hat when more than one chain is
present.  Defaults are read as requested: minimum ESS 400, maximum zero
divergences, and at most three extension blocks.  A failed check assigns
`post_warmup_state = last_state`, continues the same MCMC object without
warmup, concatenates every retained block, and uses all draws for the trace
and geometry handoff.  Diagnostics include per-site ESS/R-hat, block count,
total retained draws, and pass/fail state.

The implementation currently emits a loud terminal warning after exhausting
the blocks.  The requested automatic adaptive fallback for a failed Laplace
chain is not yet present, nor is channel-selective spectroscopic extension.
Consequently this feature is incomplete even though its continuation core is
tested.

## 3. Compile box

The opt-in `compile_box: true` flag configures JAX's persistent cache at
`/scratch/midway3/tfairnington/jax_cache` (overridable by
`jax_compilation_cache_dir`) and lowers the persistence threshold to one
second.  Default remains off.

Lane/cadence padding has not been integrated or timed.  In particular, using
only a huge `yerr` would leave a parameter-independent Normal normalization
constant and would not satisfy the stated exact-potential comparison; an
explicit likelihood mask is required.  No cold/warm or end-to-end speedup is
reported without that exactness test.

## 4. Laplace-IS fallback shape

`models/laplace_is.py` now keys the fallback independent-NUTS runner to the
fixed parent lane width instead of the variable failed-lane count.  The
existing independent runner pads/masks inactive lanes, so all chunks of one
width reuse one fallback executable.  The focused splice test now asserts a
four-lane parent runner for two failed active lanes and passes.  The requested
118-channel V100 timing is not yet available, so no compile speedup is claimed.

## 5. Parallel PRISM

The existing routing remains covered by
`tests/test_channel_batch_plan_routing.py`, including modulo assignment by
`parallel_job_count/index`.  The requested two-GPU exact sample concatenation
run was not queued before this snapshot because parity jobs had priority and
the rule limits this worker's queued scripts.  No parallel speedup or exact
equality claim is made.

## Files changed or added

- `fit_jwst.py`: retained-block diagnostics/continuation and opt-in persistent
  cache configuration.
- `models/laplace_is.py`: fixed parent-width fallback compilation.
- `tests/test_whitelight_geometry_handoff.py`: required-site ESS test.
- `tests/test_laplace_is.py`: fixed-width fallback assertion.
- `acceleration_reports/gpu_queue/pending/wl_p2_prism_ta99_seeds560_561.sh`:
  two additional PRISM validation seeds, using new output paths.
- `acceleration_reports/OVERNIGHT_MANIFEST.md`: append-only task progress.

## Reproduction

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 0-15 \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest \
  tests/test_whitelight_geometry_handoff.py -x -q
```

Result: 6 passed in 8.04 s.

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 0-15 \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest \
  tests/test_laplace_is.py::test_failed_gate_splices_independent_fallback \
  tests/test_whitelight_geometry_handoff.py -x -q
```

Result: 7 passed in 11.67 s.

The exact GPU command is preserved in the queued shell script.  All GPU work
uses the file dispatcher; no Slurm command was issued by this worker.

## Open risks

- A continued MCMC object's configured sample count must remain 1000 in
  production for each requested extension block; non-production small-draw
  configurations continue by their configured block size.
- The combined ArviZ object is reconstructed from all grouped posterior
  blocks.  A full real-pipeline extension-triggering integration test remains
  necessary before enabling the gate.
- PRISM TA=0.99 has only one completed candidate seed in this snapshot.
- Compile cache reuse is shape- and executable-sensitive; no benefit is
  assumed until measured in fresh processes on the same V100.

## Second pass

### Quality-gate completion

The white-light Laplace branch now automatically constructs and runs the
original adaptive NUTS configuration when its extended chain still violates
the ESS/divergence gate.  The replacement adaptive chain is itself checked
and extended, and only its complete retained sample set reaches the geometry
handoff.  Diagnostics label this outcome `adaptive_fallback` and record
`whitelight_laplace_fell_back: true`.

Accelerated spectroscopic chunks now compute per-channel bulk ESS from
`depths` (or `rors**2`).  Channels below `spectro_min_depth_ess` (pipeline
default 400) are selectively rerun with finite-difference Laplace-metric NUTS
(150 warmup, target acceptance 0.95, depth 10); only failed lane columns are
replaced.  Direct/testing callers retain a disabled default unless they pass
the threshold, avoiding an implicit expensive rerun in low-draw unit tests.

The white-light geometry and independent-NUTS regression suite passed 20
tests after these changes, and `py_compile` passed for `fit_jwst.py` and the
fallback module.  A forced end-to-end adaptive-fallback CPU test remains more
expensive than the focused helper coverage and was not added.

### Exact compile-box masking

The vectorized jaxoplanet likelihood now accepts an optional
`likelihood_mask`.  `compile_box: true` pads time, flux, uncertainty, and any
cadence-shaped trend input to the next multiple of 256, while the padded tail
is excluded with NumPyro's likelihood mask.  Thus the likelihood contribution
is exactly zero rather than merely made small with a large uncertainty.
Persistent cache configuration remains opt-in and the independent backends
continue to use their fixed configured lane width for partial low-resolution
chunks.

Measured CPU float64 equality on real dumps:

| dump | cadence shape | |Δ potential| | max |Δ gradient| | 1e-10 gate |
|---|---:|---:|---:|---:|
| stellar HAT-P-12 SOSS high | 225 -> 256 | 0 | 1.14e-13 | pass |
| GJ-3470 G395H high | 2058 -> 2304 | 1.82e-12 | 9.09e-13 | pass |

Commands:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 0-15 \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  tools/diag_compile_box/check_padding.py \
  /scratch/midway3/tfairnington/accel_stage_inputs_stellar/HAT-P-12_NIRISS_SOSS_order1_Rreference_high_resolution_inputs.pkl \
  --channels 1

JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 0-15 \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  tools/diag_compile_box/check_padding.py \
  /scratch/midway3/tfairnington/accel_stage_inputs/GJ-3470_NIRSPEC_G395H_nrs1_R300_high_resolution_inputs.pkl \
  --channels 1
```

Cold/warm process timings and end-to-end walls are still not available in
this snapshot.  No cache speedup is claimed.

### GPU status

The two additional PRISM TA=0.99 seeds entered a V100 dispatcher and were
running when this section was appended.  Their outputs use the new
`wl_validation_p2/` tree.  Pooled three-seed validation will only be stated
after both output traces and the third adaptive trace exist; the earlier
one-seed result is not promoted to a validation claim.

The two-GPU PRISM split was not queued concurrently with this validation,
because doing so would exceed this worker's two-pending-script limit while
priority parity work occupied the dispatchers.  Serial/parallel sample
equality and its speedup therefore remain unmeasured, and no claim is made.

## Third pass

### PRISM white-light three-seed gate: pass

All requested traces are now complete: adaptive seeds 555--557 and Laplace
TA=0.99 seeds 559--561.  Every Laplace seed had zero divergences.  Seed 561
required all three automatic extension blocks (4,000 retained draws) to raise
the geometry minimum ESS above 400; seed 560 passed with its first 1,000.

| candidate seed | retained draws | preparation (s) | initial MCMC wall (s) | mean steps | divergences | min geometry ESS |
|---:|---:|---:|---:|---:|---:|---:|
| 559 | 1000 | 51.79 | 48.18 | 22.16 | 0 | 567* |
| 560 | 1000 | 52.69 | 55.48 | 27.96 | 0 | 483 |
| 561 | 4000 | 51.29 | 68.66 | 39.83 | 0 | 411 |

`*` Seed 559 predates the automatic four-site gate; its smallest listed
geometry ESS was `b=567` (the nuisance `log_tau` ESS was 148).

The three adaptive references had zero divergences and sampling walls
1158.82, 1049.15, and 1190.02 s.  The pooled science comparison is:

| site | shift (adaptive pooled sigma) | sigma ratio | candidate seed null |
|---|---:|---:|---:|
| `t0_0` | -0.035 | 1.024 | 0.071 |
| `b_0` | +0.090 | 1.040 | 0.095 |
| `logD_0` / duration | +0.066 | 1.003 | 0.091 |
| `rors_0` | +0.074 | 1.015 | 0.060 |
| `c` | +0.002 | 1.052 | 0.039 |
| `v` | +0.001 | 1.052 | 0.018 |
| `log_jitter` | -0.077 | 1.038 | 0.104 |
| `A` | +0.024 | 1.035 | 0.079 |
| `log_tau` | +0.003 | 1.046 | 0.069 |

The maximum science-site displacement is 0.090 sigma, inside the calibrated
gates.  The apparent 1.52-sigma `_b_0` displacement is the known signed latent
symmetry; the physical `b_0` agrees.  Using the requested 20 ppm per sigma-b
proxy, the spectrum-level `b` impact is **1.81 ppm**.  Applying the same proxy
only as a scale indicator gives 1.31 ppm for duration and 0.70 ppm for t0;
those are not independently calibrated physical derivatives.

Full pooled results are in
`acceleration_reports/diag_whitelight/prism_ta99_three_seed_pooled.csv`.

### Final mode-specific settings supported by measurements

The global default remains `whitelight_mass_matrix: adaptive`.  Opt-in target
settings are:

```yaml
# SOSS / G395H
whitelight_mass_matrix: laplace
whitelight_laplace_warmup: 200
whitelight_laplace_target_accept: 0.9
whitelight_laplace_max_tree_depth: 10
whitelight_laplace_hessian_method: finite_difference
whitelight_min_ess: 400
whitelight_max_divergences: 0
whitelight_max_extra_blocks: 3

# PRISM: identical except
whitelight_laplace_target_accept: 0.99

# Spectroscopy (validated preconditioned path)
spectro_sampler: independent_nuts
spectro_mass_matrix: laplace
spectro_jitter_prior: lognormal
spectro_min_depth_ess: 400
```

`compile_box: true` is **not** included in the recommended production flags.
Its exactness gate passes, but cold/warm process and whole-fit timings were not
completed, so enabling it would violate the project's measurement rule.

### Compile-box timing failure

The requested V100 cold/warm and end-to-end measurements did not complete.
The persistent-cache integration was added after the available replay driver
had already been designed around unpadded stage inputs; it has no CLI switch
that routes those replay inputs through `_run_sampling_stage`'s compile-box
wrapper.  Timing that driver would therefore measure the old shapes and would
be a false compile-box result.  I did not report such a number.  The measured
scientific prerequisite remains the real-dump equality table above.

Consequently no new measured end-to-end walls exist beyond the component
walls already reported: SOSS white light 141.3 s candidate inclusive of
preparation versus 171.1 s adaptive; G395H 86.6 s versus 270.6 s; PRISM
TA=0.99 approximately 100--120 s for seeds not needing extensions versus
1,133 s mean adaptive.  These are white-light component walls, not whole-fit
walls, and must not be presented as end-to-end pipeline speedups.

### Two-GPU PRISM failure

The requested equality experiment was not safely launchable from the offline
slice driver: selecting channels 40:80 renumbers lanes and derives a different
key stream than the pipeline's global chunk-index folding.  Comparing that
output with serial chunks would test different random draws, not parallel
correctness.  The production `chunk_mode: parallel` path does preserve global
chunk indices, but a clean full-pipeline PRISM high-resolution serial
checkpoint set was not available in a new manifest-safe output tree before
the queue window.  I therefore document this as an unmeasured failure rather
than claim sample inequality or speedup from mismatched keys.

At the time this third-pass section was finalized, this worker had no GPU
scripts pending or running; all white-light validation work had been
harvested.

## Fourth pass

Full-pipeline A/B/C matrix runners were added for stellar-informed HAT-P-12
SOSS and WASP-52 G395H. Each uses distinct `_SPEED_A/B/C` output directories;
B uses a new target-specific persistent-cache directory and C reuses it in a
second process. JAX compile logging and whole-process timing are enabled.

Both scripts remained pending behind four priority parity jobs throughout
the observed polling window (60-second cadence), so no timing result exists
yet and the dependent PRISM serial/parallel scripts have not been queued in
order to respect the two-script limit. This section is intentionally a live
status, not a performance claim.

## Fifth pass

A blocking 60-second polling loop was started for both full-pipeline matrix
exit markers. Through 03:43 CDT both scripts remained pending behind the four
running priority parity jobs; neither dispatcher had opened a slot and no
partial matrix output existed. The PRISM jobs consequently were not queued,
preserving the two-script limit. No measurement is fabricated from this
queue-only interval.

## Sixth pass

Both A/B/C matrices completed on Tesla V100 GPUs.

| target | run | WL stage* | low stage* | high stage* | process wall | compile events WL/low/high |
|---|---|---:|---:|---:|---:|---|
| SOSS | A off | 291.1 | 181.4 | 362.3 | 839.2 | 1477/0/297 |
| SOSS | B cold | 251.3 | 181.5 | 357.0 | 794.3 | 1503/0/262 |
| SOSS | C warm | 214.1 | 131.7 | 302.0 | 651.9 | 1503/0/262 |
| G395H | A off | 308.6 | 147.6 | 234.7 | 695.3 | 1344/0/212 |
| G395H | B cold | 151.7 | 145.4 | 186.0 | 487.3 | 1363/0/217 |
| G395H | C warm | 251.5 | 91.8 | 179.9 | 527.2 | 1372/0/217 |

`*` Stage intervals are reconstructed from artifact mtimes; process walls are
instrumented. Compile counts are JAX log events partitioned at stage markers.

The exact-spectrum identity gate **failed** despite the previously proven
potential/gradient equality. Maximum A-vs-B/C depth differences were 8.0/10.4
ppm (SOSS) and 11.0/14.7 ppm (G395H); maximum uncertainty differences were
5.3/5.4 and 6.8/7.1 ppm. Padding changes the finite MCMC realization/compiled
transition, so same seed does not imply identical draws. Compile-box therefore
remains unrecommended. Warm process speedups were 1.29x SOSS and 1.32x G395H.

A new full-pipeline PRISM serial job was then queued in
`HAT-P-65_PRISM_P2_SPEED_SERIAL`; its blocking exit-marker wait was active
when this snapshot was written.

## Seventh pass

The full PRISM serial run completed on a V100 with exit zero. Instrumented
whole-process wall was **3366.6 s**. Artifact-time stage intervals were 238.3 s
white light, 718.6 s low resolution, and 2393.1 s high resolution. High-stage
checkpoint completion intervals were approximately 646.6, 383.8, 401.6,
193.1, 454.5, and 271.6 s for global chunks 0:21, 21:42, 42:63, 63:84,
84:105, and 105:106. Initial chunk diagnostics reported 0,0,0,1,8,0
divergences; the depth-ESS fallback artifacts identify selectively replaced
lanes for the affected/low-ESS chunks.

Two pipeline-native parallel jobs were prepared with global
`parallel_job_count: 2`, indices 0/1, a shared new checkpoint tree, and copied
read-only white/low prerequisites. Both scripts were pending behind priority
queue work when this snapshot was written; the blocking two-exit-marker wait
remained active. Combine/equality and the parallel wall ratio therefore await
those exit files.

## Eighth pass

The first two exit-zero "parallel" jobs were audited rather than trusted:

| job | wall | actual assignment | result |
|---|---:|---|---|
| index 0 | 767 s | low-res chunk index `[0]` | computed only low 0:21 |
| index 1 | 209 s | low-res chunk indices `[]` | computed nothing |

Their logs explicitly say `chunk 0:21 - COMPUTING` for index 0 and `No chunks
assigned` for index 1. Because `need_lowres` was still true, the pipeline
returned from the parallel low-resolution stage before reaching high
resolution. No combine or speedup claim is made from these invalid halves.

An additive correction was prepared in a new output tree,
`HAT-P-65_PRISM_P2_SPEED_PARALLEL_HIGH`, with `analysis_stage: highres`,
`need_lowres: false`, global count two, and indices zero/one. The corrected
scripts `speed_p2_prism_parallel_high_0/1.sh` were queued and a blocking wait
was entered. Both remained pending behind shared priority work at the time of
this report snapshot, so equality/combine remains open.

## Final measured configuration summary

```yaml
# all modes
whitelight_mass_matrix: laplace
whitelight_laplace_warmup: 200
whitelight_laplace_max_tree_depth: 10
whitelight_laplace_hessian_method: finite_difference
whitelight_min_ess: 400
whitelight_max_divergences: 0
whitelight_max_extra_blocks: 3
spectro_sampler: independent_nuts
spectro_mass_matrix: laplace
spectro_jitter_prior: lognormal
spectro_min_depth_ess: 400

# target acceptance
# SOSS/G395H: whitelight_laplace_target_accept: 0.9
# PRISM:      whitelight_laplace_target_accept: 0.99
```

| target | measured accelerated full-pipeline wall | identical-target production full wall |
|---|---:|---:|
| HAT-P-12 SOSS | 839.2 s (compile-box off) | not measured in this matrix; historical 64 min used a different run context |
| WASP-52 G395H | 695.3 s (compile-box off) | not available |
| HAT-P-65 PRISM | 3366.6 s | not available; historical ~80 min refers to one 21-channel production stage, not the same full pipeline |

`compile_box` remains opt-in/unrecommended: its potential and gradient are
identical within 1e-10 and its warm walls were faster, but finite MCMC spectra
failed the roundoff-identity requirement by up to 10.4 ppm (SOSS) and 14.7 ppm
(G395H). The corrected two-GPU PRISM speedup is also not recommended until its
pending global-chunk combine passes equality.

## Ninth pass

Both corrected high-resolution jobs exited zero on Tesla V100 GPUs and did
real work.

| job | assigned global chunks | selective depth-ESS fallbacks | process wall | high-stage checkpoint interval* |
|---|---|---|---:|---:|
| index 0 | 0:21, 42:63, 84:105 | lanes 2/13/15, 13, and 2/3 | 1689 s | 1489.2 s |
| index 1 | 21:42, 63:84, 105:106 | lane 10 in 21:42 | 3042 s | 2641.5 s |

`*` The high-stage interval is from creation of that half's R50 checkpoint
manifest to its final R50 checkpoint. Process wall also includes input,
white-light, and setup work.

I combined ranges in global order with fit_jwst.py's
`_concatenate_chunk_samples` on CPU. Combine plus the complete per-site
comparison took **0.333 s** and wrote the new aggregate
`chunks/prism_high_parallel_combined.pkl`. Exact command:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 0-15 \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
tools/diag_compile_box/combine_compare_prism_parallel_high.py
```

The equality gate **failed**: only fixed `c1` and `c2` were bitwise equal;
the aggregate maximum absolute difference was 5.9708 (`log_jitter`). Other
maxima included 0.01671 in `rors`, 0.003461 in `depths`, 0.01085 in `v`, and
0.02694 in `A`. Full results are in
`acceleration_reports/diag_compile_box/prism_parallel_high_compare.json`.

This is not a failure of global chunk-key assignment. Each high-only process
regenerated its own short white-light chain and geometry handoff. The
checkpoint fingerprints were `5f57817a6420` for the even half and
`a58238ef29c5` for the odd half, versus `48986c2bdc04` for serial. Native
`chunk_mode: combine` correctly refuses to mix those fingerprints. The CPU
audit combined the six explicitly indexed files so the mismatch could be
measured without renaming or overwriting them. Parallel workers must consume
one immutable white-light handoff before launch.

The serial high stage took **2393.1 s**. The parallel critical path was
**2641.5 + 0.333 = 2641.8 s**, hence **0.906x** as fast as serial (10.4%
slower), not a speedup. This test fails both required gates: identical draws
and improved critical-path wall.

## Final summary after all passes

```yaml
# White light, all modes
whitelight_mass_matrix: laplace
whitelight_laplace_warmup: 200
whitelight_laplace_max_tree_depth: 10
whitelight_laplace_hessian_method: finite_difference
whitelight_min_ess: 400
whitelight_max_divergences: 0
whitelight_max_extra_blocks: 3

# SOSS and G395H
whitelight_laplace_target_accept: 0.9
# PRISM instead: whitelight_laplace_target_accept: 0.99

# Spectroscopy, all modes
spectro_sampler: independent_nuts
spectro_mass_matrix: laplace
spectro_jitter_prior: lognormal
spectro_min_depth_ess: 400
```

The global white-light default should remain `adaptive`; the block above is
an explicit per-mode recommendation with automatic adaptive fallback.

| mode / measured target | recommended measured whole-process wall | available production comparison | interpretation |
|---|---:|---:|---|
| SOSS / HAT-P-12 | 839.2 s | historical HAT-P-12 fit 64 min, different context | 4.58x contextual ratio, not identical-input speedup |
| G395H / WASP-52 | 695.3 s | no paired production full wall | no speedup claim |
| PRISM / HAT-P-65 | 3366.6 s | historical production estimate 5--6 h, unpaired | 5.35--6.42x contextual ratio, not identical-input speedup |
| PRISM high, two V100 halves | 2641.8 s critical path | 2393.1 s serial high | 0.906x; equality failed |

Items left opt-in or unrecommended:

- `compile_box: true`: exact potential/gradient passed to 1e-10, but
  same-seed spectra differed by up to 10.4 ppm (SOSS) and 14.7 ppm (G395H).
  It remains opt-in despite 1.29x/1.32x warm-process gains.
- Multi-process PRISM spectroscopy: unrecommended until workers share one
  immutable geometry handoff/fingerprint. This test was slower and unequal.
- Fixed-length Laplace HMC: not recommended; Laplace NUTS with the depth-ESS
  gate remains the validated exact-MCMC path.
