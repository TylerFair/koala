# P3 exact opt-in efficiency study

Date: 2026-09-02. All defaults were left unchanged. “Exact” below means the
NumPyro target density is unchanged; it does not mean floating-point-identical
MCMC trajectories after an XLA shape or accelerator change.

## Outcome

| item | measured result | saving / decision |
|---|---|---|
| `compile_box` | padded vs unpadded potential differed by 0 (SOSS) and 1.82e-12 (G395H); gradients by at most 1.14e-13 and 9.09e-13 | exact-potential gate passes; recommend as opt-in |
| calibrated SOSS spectrum | worst median displacement 0.100 sigma; sigma ratios 0.951--1.049; offset -0.40 ppm, slope +0.47 ppm/um, detrended RMS 1.47 ppm | calibrated gate passes |
| calibrated G395H spectrum | worst median displacement 0.126 sigma; sigma ratios 0.941--1.051; offset -0.90 ppm, slope +2.44 ppm/um, detrended RMS 4.10 ppm | calibrated gate passes |
| warm compile-box process | SOSS 839.2 -> 651.9 s; G395H 695.3 -> 527.2 s | 187.3 s (1.29x) and 168.1 s (1.32x) |
| LD construction | 3 stages x 125 stellar grid points; exact atomic cache already keyed by stellar/grid/data/code inputs | a cache hit avoids all 375 ExoTiC object constructions and all per-bin coefficient calls |
| white light | validated Laplace metric uses 200 preparation warmup; measured SOSS/G395H/PRISM sampling walls are seconds to about 2 min rather than adaptive PRISM's about 18 min | retain opt-in Laplace flags and ESS continuation; fewer than 1000 draws not validated |
| PRISM 8-channel subset, 2 workers | serial 133 s; workers 107 and 102 s; critical path 107 s | 26 s, 1.24x, but strict draw equality failed, so do not promote parallel mode |
| plotting/CSV | full-run logs show many small JAX compilations in post-fit summaries; no exact posterior-summary rewrite was made | instrumentation is insufficient for a defensible saving claim |

## 1. Compile box

`compile_box: true` pads cadence arrays to a multiple of 256, passes an
explicit NumPyro likelihood mask for padded cadences, and enables the JAX
persistent compilation cache. The unmasked data and every prior are unchanged.
The earlier bitwise-sample rejection is not a posterior-bias test: changing an
XLA shape changes floating-point reduction order and therefore the chaotic
MCMC trajectory.

The full-pipeline A/B/C matrix used A=unboxed, B=boxed cold cache, C=boxed warm
cache. Warm C versus A is the clean measured operational comparison. These
runs used distinct targets within each instrument matrix, so they demonstrate
warm reuse after a fresh process; the present matrix did not isolate two
different datasets with otherwise identical shapes. Persistent-cache reuse is
shape and executable dependent, and that cross-target qualification remains a
risk.

Calibrated results are in `diag_efficiency_p3/compile_box_calibrated.json`.
Potential/gradient checks and process walls are in `speed_p2.md` and
`diag_compile_box/*_matrix.json`.

## 2. ExoTiC-LD construction

The informed power-2 prior uses a 5-point axis for each of Teff, logg and
[Fe/H], hence 5^3 = 125 weighted stellar combinations. It repeats this for
white light, low resolution and high resolution: 375 `StellarLimbDarkening`
constructions in the measured G395H log. Each construction then evaluates all
bins for that stage, which explains why counting coefficient calls gives a
number different from counting stellar-grid objects (the quoted 342 is not a
universal constant; it depends on the wavelength grids).

The current `get_or_build_power2_ld_prior` implementation already writes an
atomic CSV cache and reloads it exactly. Its digest includes stellar values and
uncertainties, wavelength centers/errors, instrument/order/mode bounds,
ExoTiC model and interpolation mode, ExoTiC data-directory metadata, ExoTiC
implementation contents, grid controls, and the uncertainty floor. Loading
returns the same serialized coefficient means and stellar-propagation sigmas;
tests cover malformed-cache rejection and cache-key controls. Per-stage grids
cannot generally be shared because their wavelength grids differ, but each
dataset/stage result can be reused across reruns and parallel consumers.

The P2 G395H matrices used output-local caches, so every A/B/C process rebuilt
three grids. A shared `stellar.ld_prior_cache_dir` is therefore the important
operational setting. No vectorized ExoTiC replacement was introduced: proving
it returned exactly the library's scalar result would require upstream support.

## 3. White light

The existing opt-in finite-difference Laplace metric uses 200 warmup steps and
an ESS/divergence continuation gate. Measurements already recorded in
`speed_p2.md` include PRISM preparation 51--53 s and initial sampling 48--69 s,
versus adaptive sampling 1049--1190 s. One of three PRISM seeds required 4000
retained draws to reach minimum geometry ESS 400; another passed with 1000.
Consequently 200 metric warmup is supported, but fewer than 1000 default draws
is not supported by the current evidence. Optimizer/preparation and sampling
are timed separately; compile, warmup, retained draws, and plotting are not yet
separately instrumented, so no invented sub-split is reported.

## 4. PRISM parallel chunks

The sampler runner already provides opt-in `chunk_mode: parallel` and
`chunk_mode: combine`. It derives each stream by folding the immutable stage
key with the global chunk index, atomically checkpoints disjoint ranges, binds
checkpoints to the complete sampling-workload and white-light handoff
fingerprints, and refuses a mismatched combine.

Jobs 460--462 replayed HAT-P-65 PRISM R50 channels 0:8 at native 40,738
cadences, width 4, 150 Laplace warmup and 300 retained draws. Serial wall was
133 s; disjoint 0:4 and 4:8 processes were 107 and 102 s. Comparison found
fixed `c1`/`c2` exactly equal, but dynamic sites differed (maximum `rors`
0.00293 and `depths` 6.06e-4). Job 460 ran on a V100 with a 32 GB allocator
limit while jobs 461/462 ran on 16 GB V100s. Thus this is a valid warning that
draw-for-draw identity is not guaranteed across accelerator variants even with
identical keys and target density. The requested strict proof did not pass;
parallel mode remains opt-in and unrecommended under that criterion. Full
comparison: `diag_efficiency_p3/prism_subset_parallel_compare.json`.

## 5. Plotting and I/O

The full-pipeline logs show post-fit work performs posterior quantiles,
per-channel model evaluation, light-curve CSV writes, noise-binning products,
and figures. Model curves are already evaluated from posterior summaries in
the principal high-resolution plotting path rather than storing a
draw-by-cadence tensor. Changing scientific products or reducing posterior
draws would not be an exact I/O optimization. No plotting flag or default was
changed. Dedicated phase timers should be added before claiming a saving.

## Recommended few-minute flag set

```yaml
flags:
  compile_box: true
  jax_compilation_cache_dir: /scratch/midway3/tfairnington/jax_cache/shared
  whitelight_mass_matrix: laplace
  whitelight_laplace_warmup: 200
  whitelight_laplace_target_accept: 0.9   # use 0.99 for PRISM
  whitelight_laplace_max_tree_depth: 10
  whitelight_min_ess: 400
  whitelight_max_divergences: 0
  whitelight_max_extra_blocks: 3
  spectro_sampler: independent_nuts
  spectro_mass_matrix: laplace
  spectro_jitter_prior: lognormal
  spectro_min_depth_ess: 400
  vmap_chunk: 4                          # native-cadence PRISM only
stellar:
  ld_prior_cache_dir: /scratch/midway3/tfairnington/ld_prior_cache/shared
```

Do not add `chunk_mode: parallel` to the recommended set: its target is exact
and its combine is fingerprint-safe, but the requested cross-process
draw-for-draw proof failed. SOSS/G395H should use the measured wider lane
settings documented in `memory_chunk_study.md`, not PRISM width 4.

## Files added in this pass

| file | purpose |
|---|---|
| `tools/diag_efficiency_p3/compare_parallel_chunks.py` | strict site-wise serial/parallel draw comparator |
| `acceleration_reports/gpu_queue/done/460_p3_prism_subset_serial.sh` | immutable serial replay command (moved by dispatcher) |
| `acceleration_reports/gpu_queue/done/461_p3_prism_subset_part0.sh` | first disjoint replay command |
| `acceleration_reports/gpu_queue/done/462_p3_prism_subset_part1.sh` | second disjoint replay command |
| `acceleration_reports/diag_efficiency_p3/*.json` | calibrated compile-box and PRISM comparison measurements |
| `acceleration_reports/efficiency_p3.md` | this report |

No production source file or default was changed in this pass.

## Exact commands and tests

The complete GPU commands are preserved verbatim in the three dispatcher
`done/*.sh` files. They wrote only below
`/scratch/midway3/tfairnington/accel_gpu_results/46*_p3_*` and all exited 0.
The strict comparator command was:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
tools/diag_efficiency_p3/compare_parallel_chunks.py \
  --serial /scratch/midway3/tfairnington/accel_gpu_results/460_p3_prism_subset_serial/serial.pkl \
  --part /scratch/midway3/tfairnington/accel_gpu_results/461_p3_prism_subset_part0/part0.pkl \
  --part /scratch/midway3/tfairnington/accel_gpu_results/462_p3_prism_subset_part1/part1.pkl \
  --output acceleration_reports/diag_efficiency_p3/prism_subset_parallel_compare.json
```

It exited 1 by design because strict equality failed. Focused regression tests:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest \
 tests/test_mcmc_runner_reuse.py tests/test_channel_batch_plan_routing.py \
 tests/test_science_artifact_cache.py tests/test_whitelight_geometry_handoff.py -q
```

Result: 18 tests passed. `py_compile` also passed for the new comparator.

## Open risks

- Cross-dataset persistent-cache reuse was not isolated from other full-pipeline
  costs; the measured warm savings are whole-process paired values.
- No cold/warm wall timer was added around LD alone; the exact number of seconds
  saved by a cache hit remains unmeasured.
- White-light warmup and draw phases are not independently timed, and one PRISM
  seed needed continuation, so fewer draws cannot be recommended.
- The PRISM subset used 300 draws and heterogeneous V100 memory variants; strict
  cross-process equality failed and the full 106-channel 2--3 GPU critical path
  remains unproven with a single immutable executable environment.
- Plotting/CSV lacks phase-level timing, so no optimization claim is made.

## Second pass: phase instrumentation and same-model parallelism

Date: 2026-09-03.

### Instrumentation

`FIT_JWST_PHASE_TIMERS=1` or `flags.phase_timers: true` now enables an
observational timer. It atomically rewrites `phase_timings.json` after every
completed event and at process exit. The JSON contains event metadata,
aggregated wall time, and JAX compilation-event counts/durations attributed to
the active phase. Default behavior is unchanged.

The implementation intentionally leaves every `MCMC.run` intact. NumPyro does
not expose a non-perturbing timer boundary between warmup and retained draws in
this execution path. Splitting it into `warmup()` and a second call solely for
timing can alter state/key handling and violate the observational requirement.
Accordingly the field is honestly named `whitelight_warmup_and_draws`;
extension blocks remain visible in the existing white-light diagnostics. JAX
compile-event duration is a summed event metric and can exceed enclosing wall
time because compilations overlap; it must not be subtracted from wall time.

Measured wall split (seconds):

| dataset/run | process | data | LD | WL optimizer | WL atomic MCMC | low-res chunks | high-res chunks | other/unattributed* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HAT-P-12 default, 484 | 731.6 | 6.6 | 229.9 | 21.2 | 78.9 | 88.8 | 110.9 | 195.2 |
| WASP-52 default, 485 | 729.8 | 5.1 | 116.0 | 25.1 | 30.2 | 108.4 | 101.9 | 343.1 |
| HAT-P-12 fast cold, 486 | 702.8 | 6.7 | 231.0 | 21.6 | 71.2 | 91.2 | 112.0 | 169.1 |
| WASP-52 fast cold, 487 | 530.9 | 4.2 | 122.6 | 19.3 | 23.3 | 95.9 | 54.9 | 210.8 |
| HAT-P-12 fast warm, 488 | 1037.6 | 6.6 | 0.025 | 19.3 | 629.7 | 42.3 | 58.0 | 281.6 |
| WASP-52 fast warm, 489 | **281.7** | 4.1 | **0.026** | 17.8 | 15.7 | 47.5 | 51.7 | 144.8 |

`*` Includes geometry handoff, posterior summaries, plotting, CSV/NPY/NPZ I/O,
and JAX work outside an active instrumented context. It is computed as process
wall minus the non-overlapping recorded phase walls. This pass did not pretend
that those components were individually measured: the requested plot/CSV
subdivision remains an instrumentation gap.

The largest removable fixed cost is unambiguously the LD grid: shared warm
cache lookup is 25--26 ms versus 116--231 s cold, saving 116 s for G395H and
230 s for SOSS. The warm G395H whole pipeline is **281.7 s (4.69 min)** versus
729.8 s (12.16 min), a 2.59x speedup. The warm SOSS repetition is not a speed
measurement: its white-light atomic MCMC took 629.7 s, indicating a difficult
chain/continuation realization, and the whole wall regressed to 1037.6 s.
The comparable cold fast SOSS run was 702.8 s versus 731.6 s default. More SOSS
seeds are required before quoting a stable fast wall.

The current production defaults already select the Laplace white-light metric
with `whitelight_laplace_warmup=200`; there is no current 1000-warmup default
to reduce in these configs. Both default and fast runs therefore used the
validated 200-step metric preparation plus 1000 retained draws. This corrects
the premise in the task rather than manufacturing a 1000-to-200 saving.

### PRISM parallel validation

Job 490 provided a new serial eight-channel reference on
Tesla V100-PCIE-16GB. Jobs 461 and 462 were the two disjoint halves on the same
GPU model. The combined result is bitwise identical for every value of all 11
sites across 300 draws. The prior failure was caused by comparing against the
32-GB V100 executable environment, not by chunk-key routing. Result:
`diag_efficiency_p3/prism_subset_same_model_compare.json`.

Full native-cadence HAT-P-65 PRISM R50, 106 channels, width 4, 150 metric
warmup and 1000 draws:

| execution | worker walls (s) | critical path (s) | versus established serial 2393.1 s |
|---|---|---:|---:|
| 2 workers (491--492) | 641, 737 | **737** | **3.25x** |
| 3 workers (493--495) | 480, 489, 487 | **489** | **4.89x** |

The super-linear ratio relative to the old serial measurement is plausible
because each independent process compiles once and handles fewer sequential
chunks, while the old full-pipeline stage also encountered selective fallback
work. Combine overhead was previously measured as 0.333 s and is negligible.
These range replays consume the immutable stage dump (which already contains
the fixed white-light geometry/model arguments); production `chunk_mode`
additionally binds every checkpoint to the white-light handoff and complete
sampling-workload fingerprints and refuses mismatches.

Recommendation: enable `chunk_mode: parallel` for PRISM only when all workers
have the same GPU model and software environment, then run `combine`. Use three
workers when available; the measured high-resolution critical path is 8.15
minutes, versus 12.28 minutes for two workers and 39.89 minutes serial.

### Updated fast flags

```yaml
flags:
  phase_timers: true                    # measurement only
  compile_box: true
  jax_compilation_cache_dir: /scratch/midway3/tfairnington/jax_cache/shared
  whitelight_mass_matrix: laplace
  whitelight_laplace_warmup: 200
  whitelight_num_samples: 1000
  spectro_sampler: independent_nuts
  spectro_mass_matrix: laplace
stellar:
  ld_prior_cache_dir: /scratch/midway3/tfairnington/ld_prior_cache/shared
```

For PRISM add `vmap_chunk: 4`, `chunk_mode: parallel`, and fixed worker
count/index values. No flag or default was changed globally.

### Files and commands

| file | change |
|---|---|
| `fit_jwst.py` | opt-in atomic phase timer; data, LD hit/miss, optimizer, white-light atomic MCMC, and per-chunk timing |
| `tests/test_phase_timers.py` | opt-in/atomic JSON regression test |
| `tools/diag_efficiency_p3/run_timed_pipeline.py` | immutable default/fast timing driver |
| `tools/diag_efficiency_p3/run_prism_range.sh` | full PRISM range replay helper |
| `acceleration_reports/gpu_queue/done/480*`--`495*` | exact dispatched commands and exit status |

Jobs 480--483 failed before analysis because the dispatcher captured the file
while the JAX callback signature was being corrected; each exited 1 with
`unexpected keyword argument 'fun_name'`. The callback was fixed to accept JAX
metadata, tests passed, and jobs 484--495 all exited 0. Failed outputs were
preserved.

Focused test command:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest \
 tests/test_phase_timers.py tests/test_science_artifact_cache.py \
 tests/test_mcmc_runner_reuse.py -q
```

Result: **15 passed**. All GPU commands are preserved verbatim in the numbered
`done/*.sh` files; all results are under the corresponding numbered
`/scratch/midway3/tfairnington/accel_gpu_results/` directories.

## Third pass: output costs and spot white-light routing

Date: 2026-09-03.

### What changed

`flags.plots: minimal` is a new opt-in output-only mode. It suppresses
diagnostic noise-binning, polynomial, and wavelength/light-curve grid plots.
It retains sampling, posterior arrays, masks, science spectrum CSVs, detailed
fit CSVs, and the transmission-spectrum plot. The phase timer now wraps
geometry-handoff writes, atomic checkpoint reads/writes, named plot helpers,
noise-binning, spectrum plots, light-curve-grid/CSV helpers, and CSV/NPY/NPZ
writes. No sampler input, prior, potential, draw, or default changed.

The timing driver also accepts `--adaptive-white`, which simply emits the
existing exact opt-in `whitelight_mass_matrix: adaptive` setting. It does not
introduce a new sampler or automatic default route.

### Job 488 diagnosis

The 629.7-second white-light interval was not extension blocks. The first
Laplace chain had geometry ESS 5.65--10.66 with zero divergences and triggered
the existing ESS<50 fail-fast immediately. It then ran one adaptive fallback,
which passed with ESS 435--1096 and zero divergences. Diagnostics show why the
Laplace metric was unsuitable for this spot-trend realization:

| metric | value |
|---|---:|
| finite-difference Hessian minimum eigenvalue | -5991.91 |
| regularized condition number | 1.0e12 |
| MAP gradient norm | 6.55 |
| Newton decrement | 108.21 |
| adaptive mean / maximum NUTS steps | 69.3 / 151 |

The fitted spot amplitude/center/width are strongly coupled to transit
geometry, producing the same poorly conditioned local-Hessian pattern seen in
step-like trends. Fail-fast already prevented wasting extension blocks, but it
cannot recover the initial Laplace preparation/chain cost. Direct adaptive
white light is therefore the exact opt-in remedy measured for this SOSS spot
case.

### Warm minimal-output measurements

| phase (seconds) | WASP-52 G395H job 500 | HAT-P-12 SOSS job 501 |
|---|---:|---:|
| process wall | **271.45** | **512.64** |
| data load | 4.10 | 7.66 |
| LD cache | 0.026 | 0.071 |
| white-light optimizer | 17.85 | 22.31 |
| white-light MCMC | 16.32 Laplace | 170.35 adaptive |
| low-resolution chunk sampling | 52.48 | 99.88 |
| high-resolution chunk sampling | 50.18 | 120.90 |
| geometry handoff | <0.001 | <0.001 |
| checkpoint I/O | 0.001 | 0.003 |
| science CSV + detailed light-curve CSV | 1.03 | 1.67 |
| retained spectrum plot | 0.29 | 0.28 |
| skipped diagnostic plots | 0 | 0 |
| still unattributed | **120.52** | **78.88** |

The requested under-10-second unattributed gate **did not pass**. The wrappers
prove that named plotting and file writes are not the large cost: minimal mode
saved only 10.2 seconds versus the prior warm G395H job 489 (281.67 -> 271.45
s). The residual is inline main-body work—posterior medians/quantiles, JITted
per-channel model evaluation used to construct residuals and detailed science
tables, low-to-high preparation/transitions, plus import/startup—not calls to
the named plotting helpers. Relabeling this residual as “pipeline overhead”
would not constitute attribution, so it remains explicit. A fourth pass would
need structural extraction of those inline blocks into timed functions or
timeline intervals; this pass does not claim complete attribution.

No posterior thinning was implemented. ArviZ summaries and model curves based
on a thinned draw set can change reported diagnostics/uncertainties and are not
exact output transformations. The production curve path already evaluates at
posterior summaries rather than materializing every draw by cadence.

### End-to-end comparison and recommendation

| dataset | default job | fast warm job | wall saving | status versus “few minutes” |
|---|---:|---:|---:|---|
| WASP-52 G395H | 729.77 s | **271.45 s (4.52 min)** | 458.32 s, 2.69x | target met at the upper edge |
| HAT-P-12 SOSS spot | 731.61 s | **512.64 s (8.54 min)** | 218.97 s, 1.43x | target not met |

Recommended opt-in fast settings:

```yaml
flags:
  compile_box: true
  jax_compilation_cache_dir: /scratch/midway3/tfairnington/jax_cache/shared
  plots: minimal
  whitelight_mass_matrix: laplace       # ordinary linear G395H/SOSS
  whitelight_laplace_warmup: 200
  whitelight_num_samples: 1000
  spectro_sampler: independent_nuts
  spectro_mass_matrix: laplace
stellar:
  ld_prior_cache_dir: /scratch/midway3/tfairnington/ld_prior_cache/shared
```

For empirically ill-conditioned spot/step white-light datasets, override
`whitelight_mass_matrix: adaptive`. This avoids a doomed Laplace attempt but
adaptive sampling itself remains the dominant 170-second SOSS phase. It should
not be made an automatic default from one target.

### Files, tests, and failures

| file | change |
|---|---|
| `fit_jwst.py` | timed output-helper wrappers and opt-in minimal plots |
| `tools/diag_efficiency_p3/run_timed_pipeline.py` | minimal/adaptive measurement switches |
| `tests/test_phase_timers.py` | verifies diagnostic plots are skipped while science CSV helper still runs |
| `gpu_queue/done/500*`, `501*` | exact GPU commands; both exit 0 |

Focused CPU command was the same as the second pass and now reports **16
passed**. Jobs 500 and 501 both ran on Tesla V100-PCIE-16GB, exited zero, and
no numbered P3 job remains pending. Their complete products are under
`/scratch/midway3/tfairnington/accel_gpu_results/500_p3_g395h_fast_minimal/`
and `501_p3_soss_fast_minimal_adaptive/`.

Open risk: `plots: minimal` currently suppresses named diagnostic helpers but
does not bypass the inline per-channel model construction needed by residual
and detailed-fit products. That is why it is exact and safe, but also why its
speed benefit is small.
