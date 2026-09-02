# V100 production-input sampler benchmark (`gpu_bench`)

Date: 2026-09-01  
GPU: Tesla V100-PCIE-16GB, one device  
Precision: JAX float64 enabled throughout  
Software: JAX 0.6.2, NumPyro 0.19.0, jaxoplanet 0.1.0, ArviZ 0.22

## Bottom line

None of the tested inference changes reached the required 10x speedup on an
identical production input, and none passed the strict requested-site fidelity
gate against the saved joint-NUTS reference.

- Dense per-lane independent NUTS was the only faster candidate: `1.114x`
  total-wall speedup on SOSS high resolution (217.97 s to 195.70 s). It had two
  divergences, substantially lower ESS, and failed 67/280 requested-site
  channel rows.
- Default Laplace-IS was slower than joint NUTS on every equal-draw comparison:
  speedups were `0.758x` (SOSS high), `0.701x` (SOSS low), `0.307x` (G395H
  low), and `0.708x` (G395H high). It failed the strict fidelity gate in every
  case.
- The SOSS Laplace runs fell back heavily (34/40 and 15/24 lanes). G395H had
  only one fallback lane in each stage, but compilation, padded-width work in
  the five-channel low-resolution stage, and IMH autocorrelation still made it
  slower.
- The 21-channel PRISM low-resolution stage was run at the allowed 200/200
  setting only. It already took 1011.60 s (16.86 min), with a mean/median of
  456/511 leapfrogs per retained draw. A 1000/1000 timing was not measured and
  no speedup is claimed for it.

## Scope and measurement definitions

All runs replayed exact dumps under
`/scratch/midway3/tfairnington/accel_stage_inputs/`. The model, data, masks,
priors, likelihood, float64 precision, and transit/limb-darkening physics were
unchanged. Joint and candidate comparisons returned 1000 draws from the same
channel slice. PRISM is the sole 200-draw exception.

The timing listener records JAX trace, lowering, and backend-compilation event
durations. `Sampling wall` below is `total sampler wall - recorded compile`.
It includes warmup, retained sampling, model postprocessing, diagnostics
returned by the backend, and synchronization/device transfer. It is not a
second precompiled run, so it should be treated as a useful decomposition, not
an independently measured steady-state replay. Output serialization and ArviZ
ESS calculation occur after the timed sampler interval.

ESS is one-chain ArviZ bulk ESS, reported as the minimum/median across the
requested sites and channels. R-hat is unavailable because these replays
produce one chain. `num_steps` describes retained draws, not warmup. The
fidelity gate is exactly `|median shift| < 0.1 sigma_ref` and sigma ratio in
`[0.9, 1.1]` for every row.

## Main timing and quality tables

### HAT-P-12 SOSS high resolution, channels 0:40, 1000/1000

| Backend | Compile s | Sampling wall s | Total wall s | Mean/max steps | Divergences | ESS min/median | Fidelity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| joint NUTS | 23.337 | 194.631 | 217.968 | 127.128 / 255 | 0/1000 | 123.8 / 1054.4 | reference |
| independent NUTS, dense/lane | 47.141 | 148.555 | 195.696 | 30.125 / 255 | 2/40000 | 62.4 / 561.8 | **FAIL**, 67/280 requested; 86/360 all |
| Laplace-IS default | 127.442 | 160.195 | 287.637 | -- | not returned | 5.3 / 497.3 | **FAIL**, 73/280 requested; 95/360 all |

The independent backend reduced mean leapfrogs by 4.22x but achieved only
1.114x total-wall speedup because its vmapped transitions, compilation, and
postprocessing are more expensive per reported leapfrog. Its requested-site
worst median shift was `log_jitter`, channel 31, at 0.443 sigma; its worst
sigma ratio was `log_jitter`, channel 4, at 2.171.

### HAT-P-12 SOSS R20 low resolution, all 24 channels, 1000/1000

| Backend | Compile s | Sampling wall s | Total wall s | Mean/max steps | Divergences | ESS min/median | Fidelity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| joint NUTS | 17.336 | 182.437 | 199.773 | 127.128 / 255 | 0/1000 | 71.3 / 1070.8 | reference |
| Laplace-IS default | 125.433 | 159.367 | 284.800 | -- | not returned | 10.9 / 343.0 | **FAIL**, 61/168 requested; 74/216 all |

### GJ-3470 G395H R20 low resolution, all 5 channels, 1000/1000

| Backend | Compile s | Sampling wall s | Total wall s | Mean/max steps | Divergences | ESS min/median | Fidelity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| joint NUTS | 17.041 | 211.911 | 228.951 | 125.304 / 255 | 0/1000 | 161.1 / 1041.9 | reference |
| Laplace-IS default | 125.648 | 619.121 | 744.769 | -- | not returned | 43.7 / 308.2 | **FAIL**, 13/35 requested; 15/40 all |

The Laplace runner used lane width 40 for only five real channels. It had just
one fallback lane, but the proposal/evaluation programs still carried 40
lanes; this is a major engineering cost for small low-resolution stages.

### GJ-3470 G395H R300 high resolution, channels 0:40, 1000/1000

| Backend | Compile s | Sampling wall s | Total wall s | Mean/max steps | Divergences | ESS min/median | Fidelity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| joint NUTS | 22.233 | 248.781 | 271.014 | 63.000 / 63 | 0/1000 | 6.9 / 1771.6 | reference |
| Laplace-IS default | 125.963 | 256.885 | 382.847 | -- | not returned | 33.1 / 326.5 | **FAIL**, 61/280 requested; 83/320 all |

The reference minimum ESS of 6.9 is `log_jitter`, channel 9. That reference row
is weak and makes its exact fidelity ratios especially uncertain; the other
site medians are strong.

### HAT-P-65 PRISM R10 low resolution, all 21 channels, 200/200 only

| Backend | Compile s | Sampling wall s | Total wall s | Mean/max steps | Divergences | ESS min/median | Fidelity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| joint NUTS | 18.548 | 993.049 | 1011.597 | 455.960 / 1023 | 0/200 | 28.7 / 200.0 | not tested; reduced-draw baseline |

The retained-step distribution was p05/median/p95 = 255/511/767, with the
maximum tree-depth value 1023. A 200/200 run already consumed 16.86 minutes;
the requested production stage was therefore not queued. The `c1` and `c2`
values are fixed deterministics for this dump, so their apparent ESS of 200 is
not a sampling-efficiency measurement.

## Equal-draw total-wall speedups

Every ratio uses identical input, channel slice, and returned draw count. A
value below one means the candidate was slower.

| Stage | Candidate | Reference s | Candidate s | Measured speedup | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| SOSS high 0:40 | independent NUTS | 217.968 | 195.696 | **1.114x** | FAIL |
| SOSS high 0:40 | Laplace-IS | 217.968 | 287.637 | **0.758x** | FAIL |
| SOSS low 0:24 | Laplace-IS | 199.773 | 284.800 | **0.701x** | FAIL |
| G395H low 0:5 | Laplace-IS | 228.951 | 744.769 | **0.307x** | FAIL |
| G395H high 0:40 | Laplace-IS | 271.014 | 382.847 | **0.708x** | FAIL |

## Raw potential cost

Each row is a synchronized float64 NumPyro potential at the dump's initial
state. Steady values are medians of 20 calls after the cold call and two extra
warm calls. The cold columns include first JIT compilation.

| Dump / slice | Cadences | Potential median ms | Value+gradient median ms | Cold potential s | Cold value+gradient s |
| --- | ---: | ---: | ---: | ---: | ---: |
| SOSS high 0:40 | 225 | 0.172 | 0.392 | 1.697 | 5.147 |
| SOSS low 0:24 | 226 | 0.196 | 0.348 | 1.461 | 4.177 |
| G395H low 0:5 | 2058 | 0.137 | 0.395 | 1.364 | 4.142 |
| G395H high 0:40 | 2058 | 0.296 | 0.887 | 1.801 | 5.120 |
| PRISM low 0:21 | 40780 | 1.198 | 4.437 | 1.766 | 4.442 |

## ESS by requested site

Cells are minimum/median bulk ESS across channels. The full per-channel rows
are in each run's `.arviz.json` artifact.

| Run | rors | depths | c | v | log_jitter | c1 | c2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SOSS high joint | 896.2/1349.4 | 896.2/1349.4 | 846.2/1067.1 | 902.9/1542.1 | 123.8/389.7 | 404.3/628.6 | 421.3/611.6 |
| SOSS high independent | 293.2/784.1 | 293.2/784.1 | 629.1/963.7 | 192.3/549.2 | 62.4/256.6 | 64.7/150.8 | 65.9/145.5 |
| SOSS high Laplace | 41.8/816.6 | 41.8/816.6 | 24.3/995.5 | 53.3/526.0 | 8.2/197.1 | 6.0/291.6 | 5.3/283.0 |
| SOSS low joint | 829.9/1104.7 | 829.9/1104.7 | 828.3/1010.4 | 1358.6/2138.5 | 71.3/1465.4 | 270.5/754.9 | 278.1/764.2 |
| SOSS low Laplace | 116.8/919.4 | 116.8/919.4 | 33.2/907.8 | 63.3/598.3 | 48.8/456.6 | 12.1/277.0 | 10.9/273.3 |
| G395H low joint | 950.0/1205.5 | 950.0/1205.5 | 1037.6/1447.2 | 1084.3/1535.2 | 161.1/412.6 | 694.1/982.2 | 788.1/841.7 |
| G395H low Laplace | 128.9/321.8 | 128.9/321.8 | 95.1/357.6 | 78.9/336.4 | 43.7/208.7 | 245.3/288.3 | 146.7/298.8 |
| G395H high joint | 1164.9/2043.7 | 1164.9/2043.7 | 923.0/1663.3 | 982.3/1769.1 | 6.9/643.2 | 1458.8/1814.7 | 984.7/1558.4 |
| G395H high Laplace | 163.2/352.4 | 163.2/352.4 | 205.1/326.2 | 154.6/321.2 | 33.1/249.9 | 146.0/328.6 | 231.9/333.9 |
| PRISM joint 200/200 | 126.1/203.1 | 126.1/203.1 | 108.8/204.6 | 108.4/209.4 | 28.7/94.5 | fixed | fixed |

`rors` and `depths` are deterministically related here and therefore have
nearly identical diagnostics.

## Fidelity details

The tables below aggregate the exact per-channel rows requested in the task.
`p16` and `p84` are the signed shifts with the largest absolute magnitude for
that site, in reference-sigma units. The complete unaggregated rows are saved
in each `.comparison.json`.

### SOSS high independent NUTS versus joint NUTS

| Site | Failed | Worst median shift (ch) | Sigma-ratio range | Largest-|p16| shift (ch) | Largest-|p84| shift (ch) |
| --- | ---: | ---: | ---: | ---: | ---: |
| c | 5/40 | 0.141 (28) | 0.920--1.105 | -0.238 (39) | +0.194 (30) |
| c1 | 19/40 | 0.198 (26) | 0.825--1.257 | -0.333 (13) | +0.475 (39) |
| c2 | 16/40 | 0.192 (18) | 0.882--1.137 | -0.306 (25) | +0.479 (13) |
| depths | 2/40 | 0.109 (39) | 0.902--1.071 | -0.188 (39) | +0.294 (4) |
| log_jitter | 16/40 | 0.443 (31) | 0.716--2.171 | -1.527 (36) | +0.159 (19) |
| rors | 2/40 | 0.109 (39) | 0.902--1.071 | -0.188 (39) | +0.293 (4) |
| v | 7/40 | 0.265 (31) | 0.925--1.086 | -0.193 (4) | -0.205 (31) |

### SOSS high Laplace-IS versus joint NUTS

| Site | Failed | Worst median shift (ch) | Sigma-ratio range | Largest-|p16| shift (ch) | Largest-|p84| shift (ch) |
| --- | ---: | ---: | ---: | ---: | ---: |
| c | 6/40 | 0.447 (18) | 0.893--1.130 | +0.160 (6) | +0.205 (8) |
| c1 | 15/40 | 0.235 (18) | 0.643--1.361 | -0.207 (18) | +1.324 (18) |
| c2 | 12/40 | 0.166 (0) | 0.695--1.376 | -1.121 (18) | -0.231 (38) |
| depths | 7/40 | 0.761 (18) | 0.912--1.073 | +0.410 (18) | +0.348 (18) |
| log_jitter | 17/40 | 0.337 (5) | 0.682--3.450 | -0.958 (35) | -0.127 (21) |
| rors | 7/40 | 0.760 (18) | 0.912--1.073 | +0.410 (18) | +0.347 (18) |
| v | 9/40 | 0.400 (18) | 0.889--1.081 | +0.160 (18) | -0.357 (18) |

### SOSS low Laplace-IS versus joint NUTS

| Site | Failed | Worst median shift (ch) | Sigma-ratio range | Largest-|p16| shift (ch) | Largest-|p84| shift (ch) |
| --- | ---: | ---: | ---: | ---: | ---: |
| c | 6/24 | 1.035 (23) | 0.908--1.107 | -0.190 (7) | -0.202 (23) |
| c1 | 12/24 | 0.904 (23) | 0.789--1.204 | -0.767 (23) | -0.370 (17) |
| c2 | 10/24 | 1.023 (23) | 0.816--2.064 | +0.227 (8) | +2.942 (23) |
| depths | 7/24 | 0.256 (23) | 0.851--1.133 | +0.406 (23) | -0.208 (3) |
| log_jitter | 11/24 | 0.567 (23) | 0.279--1.273 | +0.504 (23) | -0.232 (23) |
| rors | 7/24 | 0.256 (23) | 0.851--1.133 | +0.408 (23) | -0.208 (3) |
| v | 8/24 | 0.339 (23) | 0.740--1.096 | +0.448 (23) | -0.477 (23) |

### G395H low Laplace-IS versus joint NUTS

| Site | Failed | Worst median shift (ch) | Sigma-ratio range | Largest-|p16| shift (ch) | Largest-|p84| shift (ch) |
| --- | ---: | ---: | ---: | ---: | ---: |
| c | 2/5 | 0.125 (0) | 0.950--1.108 | -0.364 (1) | +0.061 (4) |
| c1 | 1/5 | 0.113 (2) | 0.910--1.049 | +0.158 (3) | -0.135 (2) |
| c2 | 2/5 | 0.156 (2) | 0.883--1.068 | -0.169 (3) | -0.159 (0) |
| depths | 2/5 | 0.134 (1) | 0.991--1.066 | -0.244 (1) | -0.087 (2) |
| log_jitter | 3/5 | 0.166 (3) | 0.793--1.033 | +0.342 (3) | -0.100 (0) |
| rors | 2/5 | 0.134 (1) | 0.990--1.066 | -0.245 (1) | -0.087 (2) |
| v | 1/5 | 0.173 (1) | 0.933--1.117 | -0.101 (3) | +0.526 (1) |

### G395H high Laplace-IS versus joint NUTS

| Site | Failed | Worst median shift (ch) | Sigma-ratio range | Largest-|p16| shift (ch) | Largest-|p84| shift (ch) |
| --- | ---: | ---: | ---: | ---: | ---: |
| c | 7/40 | 0.197 (7) | 0.909--1.127 | -0.237 (38) | -0.261 (22) |
| c1 | 8/40 | 0.188 (5) | 0.920--1.094 | +0.200 (7) | -0.225 (3) |
| c2 | 9/40 | 0.175 (36) | 0.899--1.136 | -0.321 (28) | +0.296 (30) |
| depths | 6/40 | 0.156 (36) | 0.920--1.143 | -0.273 (15) | +0.229 (3) |
| log_jitter | 18/40 | 0.327 (3) | 0.162--1.090 | +1.957 (9) | +0.316 (7) |
| rors | 6/40 | 0.156 (36) | 0.920--1.143 | -0.274 (15) | +0.228 (3) |
| v | 7/40 | 0.191 (16) | 0.909--1.098 | -0.293 (4) | +0.315 (6) |

## Laplace-IS internal diagnostics

These diagnostics are calculated before replacing failed lanes with the
independent-NUTS fallback. IMH acceptance is shown as p05/median/p95. No MAP
lane met the backend's `gradient_norm <= 1e-5` convergence flag, although the
backend's gate currently does not include that `converged` boolean.

| Stage | Internal gate pass | Fallback | MAP converged | k-hat median/max | IMH p05/median/p95 | IS ESS median | MAP grad norm median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SOSS high | 6/40 | 34/40 | 0/40 | 0.715 / 1.091 | 0.109 / 0.285 / 0.403 | 518.0 | 2.400 |
| SOSS low | 9/24 | 15/24 | 0/24 | 0.672 / 0.923 | 0.145 / 0.322 / 0.400 | 757.1 | 14.282 |
| G395H low | 4/5 | 1/5 | 0/5 | 0.449 / 0.675 | 0.339 / 0.551 / 0.555 | 2020.3 | 0.0125 |
| G395H high | 39/40 | 1/40 | 0/40 | 0.312 / 0.716 | 0.526 / 0.567 / 0.600 | 2449.1 | 0.00130 |

Fallback channel indices are saved in diagnostics and were:

- SOSS high: 1--4, 6--8, 10--16, 19--37, and 39
  (gate-passing channels were 0, 5, 9, 17, 18, 38).
- SOSS low: 0, 1, 2, 4, 9, 11--16, 18--21.
- G395H low: channel 1.
- G395H high: channel 18.

## Files built

No edits were made to `fit_jwst.py` or anything under `models/`.

- `tools/gpu_bench/raw_potential_cost.py`: reconstructs a dump, validates its
  initial potential, and times jitted potential/value-gradient calls.
- `tools/gpu_bench/run_sampler_stage_gpu.py`: thin replay wrapper. It retains
  initial-potential validation with a 1e-4 cross-GPU absolute tolerance,
  preserves all dataclass diagnostics, and removes the dump's joint-only
  `dense_mass=False` override when `independent_nuts` is selected so that the
  requested dense per-lane default is actually used.
- `tools/gpu_bench/summarize_run.py`: reduces timing, raw diagnostics, ArviZ
  rows, and fidelity rows into `summary.json` without changing samples.
- `acceleration_reports/gpu_bench.md`: this report.

Saved result roots are `/scratch/midway3/tfairnington/accel_gpu_results/13_*`
through `/scratch/midway3/tfairnington/accel_gpu_results/22_*`. Every root
contains the sampler `.pkl` and `.npz`, raw diagnostics, timing JSON, ArviZ
per-channel ESS JSON, comparison JSON where applicable, and a compact
`summary.json`.

Reference posterior pickles are:

- `/scratch/midway3/tfairnington/accel_gpu_results/13_soss_high_joint_retry/soss_high_0_40_joint_1000_1000.pkl`
- `/scratch/midway3/tfairnington/accel_gpu_results/15_soss_low_joint_retry/soss_low_0_24_joint_1000_1000.pkl`
- `/scratch/midway3/tfairnington/accel_gpu_results/16_g395_low_joint/g395_low_0_5_joint_1000_1000.pkl`
- `/scratch/midway3/tfairnington/accel_gpu_results/17_g395_high_joint/g395_high_0_40_joint_1000_1000.pkl`

## Exact reproduction

CPU test used before the GPU Laplace runs:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
taskset -c 0-15 \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  -m pytest tests/test_laplace_is.py -q
```

It exited 0. Helper syntax checks used:

```bash
python -m py_compile \
  tools/gpu_bench/raw_potential_cost.py \
  tools/gpu_bench/run_sampler_stage_gpu.py \
  tools/gpu_bench/summarize_run.py
```

The exact successful queued scripts, including every argument and output path,
are preserved at:

```text
acceleration_reports/gpu_queue/done/13_soss_high_joint_retry.sh
acceleration_reports/gpu_queue/done/14_soss_high_independent_retry.sh
acceleration_reports/gpu_queue/done/15_soss_low_joint_retry.sh
acceleration_reports/gpu_queue/done/16_g395_low_joint.sh
acceleration_reports/gpu_queue/done/17_g395_high_joint.sh
acceleration_reports/gpu_queue/done/18_soss_high_laplace_is.sh
acceleration_reports/gpu_queue/done/19_soss_low_laplace_is.sh
acceleration_reports/gpu_queue/done/20_prism_low_joint_200.sh
acceleration_reports/gpu_queue/done/21_g395_low_laplace_is.sh
acceleration_reports/gpu_queue/done/22_g395_high_laplace_is.sh
```

To reproduce through the same dispatcher, copy at most three scripts at a time
to unique names in `pending/`; for example:

```bash
cp acceleration_reports/gpu_queue/done/13_soss_high_joint_retry.sh \
  acceleration_reports/gpu_queue/pending/repro_13_soss_high_joint.sh
```

Then poll the matching `.exit` no faster than every 20 seconds and inspect the
tail of `.out` before trusting the result. To regenerate a compact summary:

```bash
JAX_PLATFORMS=cpu \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  tools/gpu_bench/summarize_run.py \
  /scratch/midway3/tfairnington/accel_gpu_results/13_soss_high_joint_retry/soss_high_0_40_joint_1000_1000 \
  --output /scratch/midway3/tfairnington/accel_gpu_results/13_soss_high_joint_retry/summary.json
```

## Failed attempts and open risks

### Failed attempts

The initial scripts `10`, `11`, and `12` exited before sampling because the
generic loader used an absolute `1e-8` initial-potential equality check across
different accelerators. On this V100, the SOSS high scalar potential differed
from the stored value by `1.715e-5` (about `1.4e-10` relative); SOSS low differed
by `2.346e-5`. The failures are preserved at:

```text
acceleration_reports/gpu_queue/done/10_soss_high_joint.{sh,out,exit}
acceleration_reports/gpu_queue/done/11_soss_high_independent.{sh,out,exit}
acceleration_reports/gpu_queue/done/12_soss_low_joint.{sh,out,exit}
```

All have `.exit=1`; no timing from them was used. The GPU-only wrapper kept
validation enabled with `atol=1e-4`. All retry and subsequent scripts exited 0,
and their `.out` files were checked for traceback, OOM, `RESOURCE_EXHAUSTED`,
`Killed`, and nonzero exit markers.

### Open risks

1. The strict comparisons use a single 1000-draw joint-NUTS reference. Sampling
   variation alone can fail a 0.1-sigma/10%-width all-row gate, especially for
   weak-reference rows. This does not excuse the failures, but a multi-seed
   reference would better distinguish bias from Monte Carlo error.
2. The G395H high reference has bulk ESS 6.9 for `log_jitter`, channel 9. Its
   very narrow candidate sigma ratio (0.162) is not trustworthy as a
   literature-level statement without a stronger reference.
3. Every Laplace MAP run exhausted almost all of the 80-iteration budget and
   none met the stated gradient convergence flag. Yet `converged` is not part
   of the current internal fallback gate. That mismatch should be resolved
   before considering the backend production-safe.
4. Laplace compilation was consistently about 125--127 s per new shape, much
   larger than the 17--23 s joint-NUTS compilation. A persistent executable
   cache or precompilation could improve repeated jobs, but it cannot repair
   the observed fidelity failures or SOSS fallback rates.
5. Low-resolution runs were padded to lane width 40. This is catastrophic for
   the five-channel G395H Laplace stage and should be replaced by a smaller
   resident width if the algorithm is pursued.
6. PRISM used only 200/200, as allowed. It is a runtime/tree-depth baseline,
   not a literature posterior; its ESS and any posterior summaries must not be
   compared directly with the 1000-draw runs.
7. `Sampling wall` is a cold-run subtraction using recorded compiler-event
   duration, not a second warm executable measurement. Only `total wall` is
   used for the headline speedups.

## Recommendation from these measurements

Keep production on joint NUTS for these dumps. Dense independent NUTS remains
an interesting engineering direction because it cut leapfrogs and achieved a
small wall-time gain, but this exact run had two divergences, lower ESS, and
failed the fidelity requirement. Default Laplace-IS should not be enabled: it
was slower on all four stages and failed the fidelity gate even when nearly
every G395H lane passed its own internal gate.
