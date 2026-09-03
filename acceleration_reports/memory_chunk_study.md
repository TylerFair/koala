# GPU memory and chunk-width study

Date: 2026-09-02. Device: Tesla V100-PCIE-16GB; JAX reported a usable `bytes_limit` of 12,701,761,536 B. All dynamic rows use float64, exact dumped stage inputs, FD-Laplace NUTS (150 metric warmup, target 0.95, depth 5), and 1000 retained draws. The allocator was the platform allocator used by the dispatcher (`bytes_reserved=0`); no preallocation workaround was needed.

## Outcome

The flat width 40 is not a memory limit for SOSS or G395H. Width 160 completed for both. Peak live memory was only 195 MiB for SOSS (225 cadences) and 1,188 MiB for G395H (2,058 cadences). The native-cadence PRISM dump is the genuinely large case: width 4 used 880 MiB at 40,738 cadences.

The dominant removable waste in G395H is the FD Hessian's all-coordinate `vmap`: bounding it to one coordinate at a time reduced width-160 peak from 1,188 to 730 MiB (39%) and total wall from 153.7 to 147.7 s. It did not help PRISM width 4: both variants were 880 MiB and 105 s, so PRISM's peak is elsewhere in the cadence-sized transit/NUTS program. The bounded FD path is now the default; `laplace_fd_batch_size=0` restores the historical all-coordinate program.

## Exact workload dimensions

| dump | channels | cadences | active transit cadences | latent/science coordinates represented in initialization |
|---|---:|---:|---:|---|
| SOSS order 1 high | 118 | 225 | 95 | rors, c1/c2, c, v, log-jitter, A_spot |
| G395H NRS1 high | 74 | 2,058 | 751 | rors, c1/c2, c, v, log-jitter |
| PRISM high | 106 | 40,738 | 16,440 | rors, c, v, A, log-tau, log-jitter |

This corrects an important premise: the available high-resolution SOSS replay has 225, not 1,500–5,000, cadences. G395H has 2,058; PRISM has 40,738.

## Dynamic V100 measurements

Widths larger than the dump's real channel count use the production padding path and retain only real-channel output.

| mode | resident width | real lanes | peak MiB | total wall s | recorded compile s | mean leapfrogs | draw-wise lane-max mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| SOSS | 20 | 20 | 160.4 | 83.0 | 70.1 | 7.97 | 16.94 |
| SOSS | 40 | 40 | 163.7 | 85.3 | 69.2 | 9.03 | 24.06 |
| SOSS | 80 | 80 | 166.5 | 117.5 | 79.2 | 9.18 | 25.89 |
| SOSS | 120 | 118 | 169.7 | 105.2 | 81.6 | 8.81 | 26.47 |
| SOSS | 160 | 118 | 194.8 | 105.6 | 81.2 | 8.81 | 26.47 |
| G395H | 20 | 20 | 177.7 | 97.5 | 77.8 | 8.86 | 16.91 |
| G395H | 40 | 40 | 303.5 | 103.3 | 78.2 | 9.00 | 17.12 |
| G395H | 80 | 74 | 595.1 | 119.2 | 78.9 | 8.76 | 18.72 |
| G395H | 120 | 74 | 895.9 | 136.7 | 77.3 | 8.76 | 18.72 |
| G395H | 160 | 74 | 1,187.9 | 153.7 | 77.8 | 8.76 | 18.72 |

No tested width OOMed. SOSS 120/160 and G395H 80/120/160 contain duplicated padding lanes, so their wall is useful for resident-shape cost but not an ESS comparison across additional real channels.

The lockstep penalty is visible: at SOSS width 40 the average lane uses 9.03 leapfrogs while each vmapped draw executes to an average maximum of 24.06. It reaches 26.47 at the wider shape. Nevertheless, throughput per real channel continues improving through the largest all-real width: for SOSS the 118-lane sampling remainder is 24 s versus 16 s for 40 lanes; for G395H 74 lanes take a 40 s remainder versus 25 s for 40. Thus 40 is conservative, not a measured speed optimum.

## Memory model and rule

A linear fit `peak = a + b * width` at fixed cadence gives:

| mode | a (MiB) | b (MiB/lane) | b/cadence (B/lane/cadence) | maximum relative residual |
|---|---:|---:|---:|---:|
| SOSS | 154.3 | 0.210 | 980 | 5.8% |
| G395H | 21.0 | 7.274 | 3,706 | 6.3% |

The implemented cross-mode safety envelope is

`peak_bytes = 160,000,000 + 6,000 * width * cadences + 128 * width * retained_draws`.

It uses the worst observed cadence slope with margin. The a-priori width is the largest integer whose prediction is no more than 75% of `memory_stats()['bytes_limit']`, then capped at the measured speed range: 160 for stages below 10,000 cadences and 4 for PRISM-like stages. `spectro_chunk_size: auto` logs the limit, cadence count, headroom, speed cap, and selected width. With this V100 it resolves to 160 for the measured SOSS/G395H shapes and 4 for native PRISM. The default remains 40 unless `auto` is requested.

Recommendations: SOSS orders 1/2: 118/all available lanes here, with 160 as the compiled cap for larger stages; G395H/M: 80/all available lanes here, with 160 safe but unvalidated for additional real channels; PRISM: remain at 4 until widths above 4 are timed, even though the simple memory extrapolation is less restrictive.

## Static analysis and buffer ownership

Code tracing found no per-cadence deterministic posterior site. `y_model` is used directly by the observed likelihood and is not registered with `numpyro.deterministic`. Retained output is compact posterior sites plus three draw diagnostics. The model also accepts precomputed per-channel median uncertainty, so `nanmedian`/sort is not in production replay execution.

The large live arrays are cadence-shaped transit/error/likelihood intermediates inside each vmapped lane, multiplied again by the FD Hessian coordinate axis in the historical implementation. Inputs are padded as `[resident_width, 1, cadence]`; shared time, phase offsets/mask, and fixed trends remain shared rather than copied per lane at Python level. Draw storage is `[draw, width, site-components]`, not `[draw, width, cadence]`.

A requested `compiled.memory_analysis()` table was not successfully captured in this pass; the public jitted runner does not retain the lowering arguments after execution. The allocator peak measurements above are complete, but generated-code/argument/output/temp byte subdivision and HLO buffer-assignment names remain an open instrumentation item. This is the main incomplete deliverable.

## Fix verification

| case | historical peak/wall | FD batch 1 peak/wall | result |
|---|---:|---:|---|
| SOSS width 160 | 194.8 MiB / 105.6 s | 172.2 MiB / 97.6 s | -12% peak, no slowdown |
| G395H width 160 | 1,187.9 MiB / 153.7 s | 729.8 MiB / 147.7 s | -39% peak, no slowdown |
| PRISM width 4 | 880.0 MiB / 105.1 s | 880.0 MiB / 104.9 s | neutral |

The finite-difference batching changes only how gradients at the same perturbation coordinates are scheduled. It does not change the potential or gradient function. Same-seed NUTS draws are not bit-identical, as expected when Hessian evaluation scheduling changes floating-point reduction order; this candidate therefore uses posterior/parity tests, not bitwise draw equality. No precision was reduced and no potential term changed.

## Files

| file | change |
|---|---|
| `models/independent_nuts.py` | bounded FD-Hessian coordinate batches; batch size 1 default, 0 legacy |
| `models/channel_batching.py` | affine memory model, fitter, and 25%-headroom resolver |
| `fit_jwst.py` | `spectro_chunk_size: auto`, logging, conservative fitted envelope, FD batch flag |
| `tools/run_sampler_on_stage_inputs.py` | explicit resident lane width and before/after allocator statistics |
| `tests/test_spectro_memory_model.py` | fit, cap/headroom, and impossible-budget tests |
| `acceleration_reports/gpu_queue/pending/350...367*.sh` | immutable queue scripts for baseline, fix, PRISM, and retries |

All result artifacts are under `/scratch/midway3/tfairnington/accel_gpu_results/350_memory_...` through `367_memory_...`.

## Commands

CPU/test environment prefix used throughout:

`JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python`

Focused tests:

`python -m pytest -q tests/test_stage_inputs_dump.py tests/test_independent_runner_reuse.py`

`python -m pytest -q tests/test_spectro_memory_model.py`

Syntax check:

`python -m py_compile fit_jwst.py models/independent_nuts.py models/channel_batching.py tools/run_sampler_on_stage_inputs.py`

Each GPU row used `tools/run_sampler_on_stage_inputs.py ... --platform gpu --backend independent_nuts --warmup 1000 --samples 1000 --builder-override jitter_prior=lognormal --nuts-override mass_matrix=laplace --nuts-override laplace_hessian_method=finite_difference --nuts-override laplace_warmup=150 --nuts-override laplace_target_accept=0.95 --nuts-override laplace_max_tree_depth=5 --nuts-override laplace_fuse_program=false`, with the row-specific range, chunk size, resident width, and FD batch size recorded verbatim in queue scripts 350–367.

Measured test results: 8/8 stage-input/runner tests passed; 3/3 new memory-model tests passed. A broader combined pytest invocation produced no captured summary in this shell, so it is not claimed as passing.

## Failures and risks

- Job 352 is an invalid baseline pilot because it loaded a transient sequential-map default during development. It completed and was retained. Replacement job 367 explicitly requested historical batch size 0 and exited 0; job 366 measured the new default and also exited 0.
- The exact SOSS dump is unusually short (225 cadences); do not extrapolate its tiny peak to an unmeasured 5,000-cadence SOSS reduction without applying the cadence-aware rule.
- PRISM was measured only at width 4. Its dominant peak was not the FD coordinate batching, so 4 remains the speed cap.
- The rule is calibrated for this model family, float64, 1000 draws, and the V100 allocator. It intentionally refuses auto mode if the backend supplies no `bytes_limit`.
- CPU/GPU compiled-memory subdivisions and named HLO buffer assignments are not yet captured, as noted above.

## Follow-up: historical OOM forensics and native PRISM sweep

Date: 2026-09-02. The search covered `logs/`, `queue_runs/`, and top-level `*.err`/`*.out` with the exact expression `RESOURCE_EXHAUSTED|Out of memory|OOM|CUDA_ERROR_OUT_OF_MEMORY|failed to allocate|MemoryError|oom-kill|Killed`. It found 34 affected files: 33 repeated PRISM high-resolution device OOMs and one older Harmonica PRISM low-resolution device OOM. There were no Slurm `oom-kill` or host-RAM failures in the searched files.

### What actually OOMed

| dates / log IDs | dataset, stage, shape | sampler and metric | allocator evidence | preallocation / sharing evidence |
|---|---|---|---|---|
| 2026-05-10--11; `49449072, 49450987, 49452482, 49453165, 49453220, 49453590, 49485236` | HAT-P-65 NIRSpec/PRISM, stellar-informed LD, native high resolution; 369 channels, 39,901 cadences, first chunk width 100 | one joint adaptive NUTS chain over the whole chunk; NumPyro default diagonal mass (not dense); 1,000 warmup + 1,000 draws | five V100 logs report 16.34 GiB scheduled input/output and failure requesting 304.02 MiB; two 24-GB RTX logs report 16.04 GiB after rematerialization and failure requesting 8.66 GiB | launch environment is not preserved, so preallocation is unknown; each Slurm job requested one GPU and there is no log evidence of simultaneous processes |
| 2026-07-10--13; `51698502, 51747400, 51762456, 51770525, 51786640, 51797125, 51802450, 51812329, 51817788, 51824234, 51832733, 51836173, 51843282, 51846830, 51860225, 51872301, 51884258, 51897025, 51913124, 51922152, 51946322, 51970120, 51996116, 52021875, 52051437, 52070999` | same PRISM native high-resolution shape, uniform LD; first chunk width 100 | same joint adaptive NUTS, diagonal mass | 19 V100 failures request 304.04 MiB after XLA reports 16,526,892,400 B of arguments; seven RTX failures request 2.23 GiB after a 16.04-GiB schedule | preserved `mw_all.sh` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`, memory fraction 0.8, and `cuda_malloc_async`; thus preallocation was off for this campaign. One GPU was requested; no evidence of sharing in the logs |
| 2026-07-31; top-level `log_hatp65.err` | asymmetric HAT-P-65 PRISM, Harmonica low resolution; 21 channels of a 40,787-cadence series, configured block width 40 but real width 21 | joint adaptive NUTS; low-resolution Harmonica configuration used dense mass, but failure occurred during initial potential evaluation before adaptation | GPU allocator failed requesting 1.28 GiB inside the old Harmonica occulted-moment kernel | launch environment and process occupancy are not recorded, so both are unknown |

The repeated historical spectroscopic failure was therefore real GPU memory exhaustion in the PRISM width-100 joint program, not host RAM and not a dense-mass covariance. The strongest direct evidence is XLA's 16.5-GB input/output footprint before another allocation. The old model also returned a cadence-sized `total_error` deterministic, so NumPyro retained a draw-by-channel-by-cadence output; that output shape explains why changing allocator policy could not make width 100 safe. The current builder instead makes `total_error` compact per channel and uses the cadence-sized array only as the likelihood scale. The current default independent Laplace-NUTS runner collects compact latent/deterministic sites and three scalar-per-lane diagnostics. It does not retain warmup states, gradients, or a cadence curve per draw. Thus the dominant historical failure is no longer present in the default path.

The separate July 31 Harmonica event occurred at initialization in the older occulted-moment implementation, not in plotting, the LD-prior host construction, white-light GP work, or draw collection. Later Harmonica acceleration work replaced that production spectroscopic route with the independent backend, but very long-cadence Harmonica forward kernels remain a distinct mode-specific memory risk.

### Legacy-shaped replay

Queue job 373 replayed the current `joint_nuts` branch at the historical width 100, 40,738 cadences, and 1,000/1,000 schedule. Its NUTS configuration was adaptive and diagonal (`dense_mass=false`). It did not reproduce the old immediate OOM because the cadence-sized deterministic has already been compacted. Instead, XLA compilation consumed one CPU continuously and reached 3,054,384 KiB maximum host RSS without reaching sampling after 42 minutes. To respect the requested sub-60-minute job envelope, only Slurm step `57474887.32` was cancelled; the dispatcher recorded exit 137. Consequently there is no honest current-code `memory_stats()` GPU peak for this replay. The old executable's measured failure envelope remains 16.04--16.34 GiB plus the failed 304 MiB--8.66 GiB request.

This replay also rules against the proposed dense-metric explanation for the common failure: both the historical/common and replay configurations were diagonal. NumPyro did collect four extra fields in the present joint runner (`diverging`, `accept_prob`, `potential_energy`, `num_steps`), but all are scalar per retained draw and cannot explain the 16.5-GB output. Warmup states are scan carry, not retained output.

### Native PRISM width sweep

All rows use the same 106-channel, 40,738-cadence dump, FD-Laplace NUTS, 150 effective warmup transitions, depth 5, target acceptance 0.95, and 1,000 retained draws. Width 4 is queue job 365; widths 8, 16, and 32 are jobs 370--372. Width 4 ran on a V100 with a 12,701,761,536-B limit. Jobs 370--372 reported a 19,043,401,728-B limit, so they landed on a 24-GB device even though the dispatcher's `.gpu` sidecar said V100; the allocator limit, not the stale label, is used below.

| width | peak MiB | wall s | recorded compile s | post-compile s/channel | mean leapfrogs | draw-wise lane-max mean | minimum bulk ESS | divergences |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 880.0 | 104.9 | 65.1 | 9.94 | 16.22 | 25.44 | 160.3 | 0 |
| 8 | 906.6 | 233.1 | 62.7 | 21.30 | 14.08 | 26.69 | 113.3 | 0 |
| 16 | 1,673.1 | 413.9 | 61.9 | 22.00 | 13.25 | 28.43 | 106.4 | 0 |
| 32 | 3,344.5 | 762.1 | 61.4 | 21.90 | 12.35 | 29.43 | 106.4 | 1 |

The fitted safety envelope predicts 7.99 GB at width 32, comfortably above the observed 3.51 GB and therefore conservative. On the measured V100 limit, 25% headroom permits width 38 but rejects 53 (predicted 13.12 GB versus a 9.53-GB budget); widths 53 and 106 were therefore not queued. The prediction holds at width 32.

Width 4 is the speed winner. Although per-lane mean leapfrogs falls with width, each vmapped transition waits for the slowest lane: the draw-wise maximum rises from 25.44 to 29.43, and the measured post-compile cost per channel more than doubles by width 8. The PRISM auto-resolver speed cap remains 4. No source change to that cap was warranted.

### Compiled-memory / HLO request

The optional `compile().memory_analysis()` capture was not feasible in under 30 minutes. Reconstructing the private prepared-state arguments requires instrumenting the runner, and the simpler current joint width-100 lowering itself remained in XLA compilation for 42 minutes. No helper or speculative buffer table was added. The dynamic allocator measurements and historical XLA schedule sizes above are complete; the requested five named buffers for G395H width 40 and PRISM width 4 remain open.

### Files and exact commands

| file | follow-up change |
|---|---|
| `acceleration_reports/gpu_queue/done/370_prism_w8.sh` | additive native-PRISM width-8 replay |
| `acceleration_reports/gpu_queue/done/371_prism_w16.sh` | additive native-PRISM width-16 replay |
| `acceleration_reports/gpu_queue/done/372_prism_w32.sh` | additive native-PRISM width-32 replay |
| `acceleration_reports/gpu_queue/done/373_prism_joint_w100.sh` | additive current joint-NUTS legacy-shaped probe; cancelled after 42 minutes of compilation |
| `acceleration_reports/memory_chunk_study.md` | this appended follow-up |
| `acceleration_reports/OVERNIGHT_MANIFEST.md` | appended completion line |

Forensics command:

`rg -n -i 'RESOURCE_EXHAUSTED|Out of memory|OOM|CUDA_ERROR_OUT_OF_MEMORY|failed to allocate|MemoryError|oom-kill|Killed' logs queue_runs . --glob '*.err' --glob '*.out' --glob '!acceleration_reports/gpu_queue/**'`

The exact GPU commands are preserved verbatim in queue scripts 370--373. Jobs 370--372 invoke `tools/run_sampler_on_stage_inputs.py` with `--platform gpu --backend independent_nuts`, their respective start/end/chunk/resident widths, `--warmup 1000 --samples 1000`, lognormal jitter, finite-difference Laplace metric batch size 1, 150 metric warmup steps, target 0.95, depth 5, and an unfused program. Job 373 invokes the same dump with `--backend joint_nuts --start 0 --end 100 --chunk-size 100 --warmup 1000 --samples 1000`.

Focused test command:

`JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest -q tests/test_spectro_memory_model.py`

Result: 3 passed in 1.32 s. Queue exits were 0, 0, 0, and 137 for jobs 370--373 respectively. Result artifacts are under `/scratch/midway3/tfairnington/accel_gpu_results/370_prism_w8`, `371_prism_w16`, `372_prism_w32`, and `373_prism_joint_w100` (the last contains no sampler result because compilation was cancelled).
