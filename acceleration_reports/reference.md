# Exact-NUTS reference ensembles and Monte-Carlo noise floor

Worker: `reference`  
Date: 2026-09-01

## Outcome

I generated twelve float64, exact `joint_nuts` reference chains on the shared
V100: three RNG seeds for each of SOSS high resolution channels 0:40, SOSS
low resolution channels 0:24, G395H high resolution channels 0:40, and G395H
low resolution channels 0:5. Every chain used the stage dump's production
NUTS options and `1000` warmup + `1000` retained draws in one chunk. All four
queue scripts exited 0 and all twelve chains had zero divergences.

For each stage, the three chains were concatenated on the draw axis into a
3,000-draw reference with the standard `[draw, channel, ...]` layout. The
companion noise-floor JSON contains every channel/seed-pair row, per-site
maximum and 95th-percentile summaries, and calibrated fidelity gates.

The main result is that a flat 0.1-sigma median gate is inside ordinary
1000-draw NUTS Monte-Carlo noise. Conservative global limits inferred from
these four stage ensembles are 0.133 sigma for depth/radius, 0.164 sigma for
trends, 0.202 sigma for limb darkening, and 0.358 sigma for noise terms. These
are fidelity tolerances, not measured sampler errors or speedups.

## Files built or changed

| File | Purpose |
|---|---|
| `tools/run_sampler_on_stage_inputs.py` | Added `--seed`, production-equivalent stage-key derivation, production-equivalent global chunk folding, and configurable initial-potential tolerance. Seed 0/default preserves the dumped pipeline key exactly. |
| `tools/reference_noise_floor.py` | Loads seed sample sets, calculates every requested channel/site/pair metric, writes full JSON, proposes gates, and atomically writes pooled references. |
| `tests/test_stage_inputs_dump.py` | Added seed-key semantics coverage. |
| `tests/test_reference_noise_floor.py` | Covers component-aware pairwise metrics, gate construction, and pooled CLI output. |
| `acceleration_reports/gpu_queue/done/30_reference_soss_high_3seed.sh` | Exact queued SOSS-high three-seed run. |
| `acceleration_reports/gpu_queue/done/31_reference_soss_low_3seed.sh` | Exact queued SOSS-low three-seed run. |
| `acceleration_reports/gpu_queue/done/32_reference_g395h_low_3seed.sh` | Exact queued G395H-low three-seed run. |
| `acceleration_reports/gpu_queue/done/33_reference_g395h_high_3seed.sh` | Exact queued G395H-high three-seed run. |
| `acceleration_reports/reference.md` | This report. |

No Slurm command was called. The four executable scripts were placed in
`gpu_queue/pending/` and the orchestrator dispatcher moved them through
`running/` to `done/`, oldest first. I kept no more than two of my scripts
pending at once.

## RNG and metric definitions

The pipeline master seed is 555 and it splits seven stage keys. Low-resolution
spectroscopic sampling uses split key 3; high resolution uses split key 5.
The driver now uses:

- omitted `--seed` or `--seed 0`: the exact RNG key stored in the dump;
- positive replicate `N`: `PRNGKey(555 + N)`, then the same seven-way split
  and stage-key selection as the pipeline;
- chunk key: `fold_in(stage_key, (start + local_start) // chunk_size)`, the
  same global chunk index as `fit_jwst.get_samples_chunked`.

Seed 0 for all four stages reproduced the pre-existing baseline pickle
bit-for-bit across every site (`max_abs = 0`). This directly validates both
the default key and chunk-fold path.

For seed pair A/B and a scalar site component/channel, the noise tool uses

`pooled_sigma = sqrt((sample_sigma_A^2 + sample_sigma_B^2) / 2)`.

It records absolute median shift divided by pooled sigma, raw sigma ratio,
and signed 16th/84th-percentile shifts divided by pooled sigma. Array-valued
sites such as `depths[0]` are component-expanded. The JSON retains all raw
rows; the tables below use `p95 / max` notation for absolute shifts and
`p05-p95 (min-max)` for sigma ratios.

The stage/class median gate is

`max(0.1 sigma, 1.5 * max(per-site p95 seed-to-seed median shift))`.

The sigma-ratio interval is the full observed class spread, including
reciprocal ratios so it is order-independent, widened multiplicatively by 5%.

## Inputs and pooled references

The captured stages use the exact dumped model, priors, data, masks, init,
float64 setting, and production NUTS options. Only RNG replication changed.

| Stage | Selected / dump channels | Cadences | Active-window cadences |
|---|---:|---:|---:|
| SOSS high | 40 / 118 | 225 | 94 |
| SOSS low | 24 / 24 | 226 | 95 |
| G395H high | 40 / 74 | 2,058 | 751 |
| G395H low | 5 / 5 | 2,058 | 751 |

Pooled reference root:
`/scratch/midway3/tfairnington/accel_stage_inputs/references/`

| Reference basename | Shape of scalar sites | Sites | Bytes |
|---|---:|---:|---:|
| `HAT-P-12_NIRISS_SOSS_order1_Rreference_high_resolution_inputs_ch0_40_pooled_joint_nuts.pkl` | `[3000, 40]` | 9 | 8,640,513 |
| `HAT-P-12_NIRISS_SOSS_order1_R20_low_resolution_inputs_ch0_24_pooled_joint_nuts.pkl` | `[3000, 24]` | 9 | 5,184,513 |
| `GJ-3470_NIRSPEC_G395H_nrs1_R300_high_resolution_inputs_ch0_40_pooled_joint_nuts.pkl` | `[3000, 40]` | 8 | 7,680,467 |
| `GJ-3470_NIRSPEC_G395H_nrs1_R20_low_resolution_inputs_ch0_5_pooled_joint_nuts.pkl` | `[3000, 5]` | 8 | 960,467 |

Each has a same-stem `_noise_floor.json`. SOSS sites are `A_spot`, `c`, `c1`,
`c2`, `depths`, `log_jitter`, `rors`, `total_error`, and `v`; G395H has the
same set without `A_spot`.

## V100 production-baseline timing and diagnostics

“Sampler wall” is the driver's per-chunk wall interval and includes warmup,
draws, and compilation. “Process real” is `/usr/bin/time -p` around the
complete driver process. Recorded compile time is instrumentation, not a
subtraction-based steady benchmark. Each run contained only one compiling
chunk, so no honest compile-free steady interval exists.

| Stage | Seed | Sampler wall (s) | Recorded compile (s) | Process real (s) | Divergences | Steps median/max | Min bulk ESS |
|---|---:|---:|---:|---:|---:|---:|---:|
| SOSS high 0:40 | 0 | 217.467 | 22.872 | 230.28 | 0 | 127 / 255 | 123.8 |
| SOSS high 0:40 | 1 | 216.096 | 23.132 | 229.13 | 0 | 127 / 255 | 48.7 |
| SOSS high 0:40 | 2 | 246.529 | 23.018 | 259.87 | 0 | 127 / 511 | 61.6 |
| SOSS low 0:24 | 0 | 197.103 | 17.189 | 209.44 | 0 | 127 / 255 | 71.3 |
| SOSS low 0:24 | 1 | 204.715 | 17.271 | 217.34 | 0 | 127 / 255 | 74.2 |
| SOSS low 0:24 | 2 | 199.682 | 17.153 | 212.25 | 0 | 127 / 127 | 104.9 |
| G395H low 0:5 | 0 | 228.617 | 16.944 | 240.78 | 0 | 127 / 255 | 161.1 |
| G395H low 0:5 | 1 | 194.310 | 16.838 | 206.89 | 0 | 127 / 255 | 181.1 |
| G395H low 0:5 | 2 | 212.178 | 16.992 | 224.82 | 0 | 127 / 383 | 195.3 |
| G395H high 0:40 | 0 | 268.984 | 22.333 | 281.95 | 0 | 63 / 63 | 6.9 |
| G395H high 0:40 | 1 | 275.112 | 22.386 | 288.29 | 0 | 63 / 63 | 118.4 |
| G395H high 0:40 | 2 | 254.806 | 22.380 | 268.50 | 0 | 63 / 63 | 324.1 |

The median sampler baselines are 217.467 s (SOSS high), 199.682 s (SOSS
low), 212.178 s (G395H low), and 268.984 s (G395H high). They are measured
like-for-like production settings, but no speedup is claimed because this
task did not run a candidate accelerated sampler.

The G395H-high seed-0 minimum ESS of 6.91 is one `log_jitter` value in channel
9; the next-lowest values in that chain are hundreds. This is a real mixing
warning and explains why `log_jitter` needs a much wider seed-calibrated gate.
The single-chain driver correctly records R-hat as unavailable rather than
fabricating a multi-chain statistic.

## Per-site seed-to-seed noise

All shifts below are in pooled-sigma units. Full-precision values and all
individual channel/pair rows are in the corresponding JSON files.

### SOSS high, channels 0:40

| Site | Median p95 / max | Sigma ratio p05-p95 (min-max) | abs q16 p95 / max | abs q84 p95 / max |
|---|---:|---:|---:|---:|
| `A_spot` | 0.100 / 0.176 | 0.924-1.073 (0.882-1.150) | 0.190 / 0.248 | 0.148 / 0.249 |
| `c` | 0.096 / 0.144 | 0.937-1.081 (0.920-1.095) | 0.142 / 0.224 | 0.156 / 0.230 |
| `c1` | 0.100 / 0.141 | 0.883-1.111 (0.791-1.175) | 0.117 / 0.191 | 0.216 / 0.347 |
| `c2` | 0.104 / 0.149 | 0.925-1.086 (0.847-1.123) | 0.178 / 0.248 | 0.163 / 0.248 |
| `depths[0]` | 0.085 / 0.113 | 0.916-1.080 (0.867-1.116) | 0.132 / 0.239 | 0.135 / 0.233 |
| `log_jitter` | 0.140 / 0.209 | 0.773-1.281 (0.245-2.555) | 0.447 / 0.862 | 0.100 / 0.141 |
| `rors[0]` | 0.085 / 0.113 | 0.917-1.080 (0.867-1.116) | 0.132 / 0.240 | 0.134 / 0.233 |
| `total_error` | 0.238 / 0.487 | 0.918-1.101 (0.785-1.135) | 0.281 / 0.707 | 0.180 / 0.269 |
| `v` | 0.109 / 0.165 | 0.941-1.054 (0.923-1.105) | 0.134 / 0.181 | 0.127 / 0.169 |

### SOSS low, channels 0:24

| Site | Median p95 / max | Sigma ratio p05-p95 (min-max) | abs q16 p95 / max | abs q84 p95 / max |
|---|---:|---:|---:|---:|
| `A_spot` | 0.100 / 0.125 | 0.932-1.084 (0.916-1.132) | 0.156 / 0.231 | 0.161 / 0.199 |
| `c` | 0.094 / 0.197 | 0.925-1.069 (0.887-1.082) | 0.120 / 0.159 | 0.164 / 0.194 |
| `c1` | 0.132 / 0.213 | 0.937-1.118 (0.903-1.183) | 0.135 / 0.177 | 0.212 / 0.316 |
| `c2` | 0.135 / 0.186 | 0.941-1.081 (0.929-1.138) | 0.161 / 0.262 | 0.167 / 0.266 |
| `depths[0]` | 0.085 / 0.100 | 0.921-1.070 (0.871-1.105) | 0.130 / 0.175 | 0.132 / 0.199 |
| `log_jitter` | 0.082 / 0.107 | 0.787-1.510 (0.593-2.150) | 0.444 / 1.229 | 0.097 / 0.134 |
| `rors[0]` | 0.085 / 0.100 | 0.921-1.070 (0.871-1.105) | 0.130 / 0.175 | 0.132 / 0.199 |
| `total_error` | 0.141 / 0.215 | 0.910-1.091 (0.874-1.148) | 0.324 / 0.574 | 0.121 / 0.152 |
| `v` | 0.087 / 0.113 | 0.937-1.052 (0.918-1.137) | 0.148 / 0.197 | 0.147 / 0.221 |

### G395H low, channels 0:5

| Site | Median p95 / max | Sigma ratio p05-p95 (min-max) | abs q16 p95 / max | abs q84 p95 / max |
|---|---:|---:|---:|---:|
| `c` | 0.085 / 0.106 | 0.967-1.024 (0.959-1.027) | 0.108 / 0.120 | 0.069 / 0.071 |
| `c1` | 0.072 / 0.097 | 0.962-1.123 (0.935-1.124) | 0.161 / 0.182 | 0.144 / 0.149 |
| `c2` | 0.089 / 0.102 | 0.958-1.065 (0.948-1.087) | 0.127 / 0.155 | 0.137 / 0.167 |
| `depths[0]` | 0.073 / 0.074 | 0.950-1.025 (0.943-1.051) | 0.191 / 0.203 | 0.067 / 0.073 |
| `log_jitter` | 0.162 / 0.163 | 0.902-1.041 (0.876-1.044) | 0.431 / 0.543 | 0.083 / 0.090 |
| `rors[0]` | 0.073 / 0.074 | 0.950-1.025 (0.943-1.051) | 0.191 / 0.204 | 0.067 / 0.073 |
| `total_error` | 0.209 / 0.223 | 0.879-1.041 (0.877-1.060) | 0.010 / 0.014 | 0.179 / 0.231 |
| `v` | 0.055 / 0.067 | 0.941-1.040 (0.926-1.059) | 0.122 / 0.156 | 0.165 / 0.175 |

### G395H high, channels 0:40

| Site | Median p95 / max | Sigma ratio p05-p95 (min-max) | abs q16 p95 / max | abs q84 p95 / max |
|---|---:|---:|---:|---:|
| `c` | 0.090 / 0.137 | 0.925-1.081 (0.889-1.147) | 0.138 / 0.238 | 0.115 / 0.214 |
| `c1` | 0.080 / 0.095 | 0.927-1.094 (0.876-1.139) | 0.144 / 0.214 | 0.126 / 0.171 |
| `c2` | 0.077 / 0.121 | 0.927-1.078 (0.890-1.101) | 0.142 / 0.205 | 0.157 / 0.177 |
| `depths[0]` | 0.089 / 0.119 | 0.922-1.080 (0.875-1.167) | 0.161 / 0.224 | 0.131 / 0.200 |
| `log_jitter` | 0.218 / 0.277 | 0.964-1.048 (0.762-3.476) | 0.185 / 2.640 | 0.110 / 0.146 |
| `rors[0]` | 0.089 / 0.119 | 0.922-1.080 (0.875-1.167) | 0.162 / 0.225 | 0.131 / 0.200 |
| `total_error` | 0.015 / 0.231 | 0.775-1.243 (0.684-1.570) | 0.001 / 1.176 | 0.149 / 0.219 |
| `v` | 0.079 / 0.116 | 0.927-1.070 (0.868-1.152) | 0.152 / 0.237 | 0.137 / 0.179 |

## Calibrated gates

The stage-specific gates produced mechanically by the requested formula are:

| Stage | Site class | Median-shift limit (sigma) | Sigma-ratio interval |
|---|---|---:|---:|
| SOSS high | depth/rors | 0.128 | [0.826, 1.211] |
| SOSS high | trend `c,v` | 0.164 | [0.862, 1.161] |
| SOSS high | LD `c1,c2` | 0.156 | [0.754, 1.327] |
| SOSS high | `log_jitter/total_error` | 0.358 | [0.233, 4.292] |
| SOSS high | other (`A_spot`) | 0.150 | [0.829, 1.207] |
| SOSS low | depth/rors | 0.128 | [0.830, 1.205] |
| SOSS low | trend `c,v` | 0.141 | [0.838, 1.193] |
| SOSS low | LD `c1,c2` | 0.202 | [0.805, 1.242] |
| SOSS low | `log_jitter/total_error` | 0.211 | [0.443, 2.257] |
| SOSS low | other (`A_spot`) | 0.150 | [0.842, 1.188] |
| G395H low | depth/rors | 0.110 | [0.898, 1.114] |
| G395H low | trend `c,v` | 0.128 | [0.882, 1.134] |
| G395H low | LD `c1,c2` | 0.134 | [0.847, 1.181] |
| G395H low | `log_jitter/total_error` | 0.314 | [0.834, 1.199] |
| G395H high | depth/rors | 0.133 | [0.816, 1.226] |
| G395H high | trend `c,v` | 0.134 | [0.826, 1.210] |
| G395H high | LD `c1,c2` | 0.121 | [0.835, 1.198] |
| G395H high | `log_jitter/total_error` | 0.328 | [0.274, 3.650] |

For one instrument-independent conservative gate, take the envelope across
the measured stages:

| Site class | Median-shift limit (sigma) | Sigma-ratio interval |
|---|---:|---:|
| depth/rors | 0.133 | [0.816, 1.226] |
| trend `c,v` | 0.164 | [0.826, 1.210] |
| LD `c1,c2` | 0.202 | [0.754, 1.327] |
| `log_jitter/total_error` | 0.358 | [0.233, 4.292] |
| other (`A_spot`) | 0.150 | [0.829, 1.207] |

The mode-specific table is preferable when the matching reference exists.
In particular, the broad global noise-width gate is driven by weakly
identified `log_jitter`; `total_error` is generally much more stable. A future
comparison should display both sites rather than allowing `log_jitter` to
hide a failure in the scientifically relevant total uncertainty.

## PRISM decision

I did not queue the optional PRISM 1000/1000 reference, as instructed by the
45-minute cutoff. The existing 200 warmup + 200 draw probe in
`gpu_queue/done/20_prism_low_joint_200.out` measured 1,011.288 s total chunk
wall with 18.548 s recorded compilation. Linear sampling extrapolation with
one compile,

`18.548 + 5 * (1011.288 - 18.548) = 4,982.250 s = 83.04 min`,

is well above 45 minutes. No production PRISM spectroscopic NUTS was run.

## Commands to reproduce

### Queue submission

The exact executable scripts are preserved in `gpu_queue/done/`. To queue
fresh copies through the orchestrator protocol:

```bash
install -m 775 acceleration_reports/gpu_queue/done/30_reference_soss_high_3seed.sh acceleration_reports/gpu_queue/pending/30_reference_soss_high_3seed.sh
install -m 775 acceleration_reports/gpu_queue/done/31_reference_soss_low_3seed.sh acceleration_reports/gpu_queue/pending/31_reference_soss_low_3seed.sh
install -m 775 acceleration_reports/gpu_queue/done/32_reference_g395h_low_3seed.sh acceleration_reports/gpu_queue/pending/32_reference_g395h_low_3seed.sh
install -m 775 acceleration_reports/gpu_queue/done/33_reference_g395h_high_3seed.sh acceleration_reports/gpu_queue/pending/33_reference_g395h_high_3seed.sh
```

Each script runs this exact driver form for seeds 0, 1, and 2, substituting
its dump, channel endpoint, and prefix shown in the script:

```bash
/usr/bin/time -p python tools/run_sampler_on_stage_inputs.py \
  "$DUMP" --backend joint_nuts --start 0 --end "$END" \
  --warmup 1000 --samples 1000 --chunk-size 40 --platform gpu \
  --seed "$SEED" --potential-atol 1e-4 --output-prefix "$PREFIX"
```

The queue logs and exit codes are the matching `.out` and `.exit` files under
`acceleration_reports/gpu_queue/done/`; all four exit files contain `0`.

### Pooling and noise floors

The following commands are exactly the four analyses used (line wrapping is
only for readability):

```bash
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/reference_noise_floor.py \
  /scratch/midway3/tfairnington/accel_stage_inputs/HAT-P-12_NIRISS_SOSS_order1_Rreference_high_resolution_inputs.pkl \
  /scratch/midway3/tfairnington/accel_gpu_results/30_reference_soss_high_3seed/soss_high_ch0_40_seed0_joint_1000x1000.pkl \
  /scratch/midway3/tfairnington/accel_gpu_results/30_reference_soss_high_3seed/soss_high_ch0_40_seed1_joint_1000x1000.pkl \
  /scratch/midway3/tfairnington/accel_gpu_results/30_reference_soss_high_3seed/soss_high_ch0_40_seed2_joint_1000x1000.pkl \
  --start 0 --end 40

/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/reference_noise_floor.py \
  /scratch/midway3/tfairnington/accel_stage_inputs/HAT-P-12_NIRISS_SOSS_order1_R20_low_resolution_inputs.pkl \
  /scratch/midway3/tfairnington/accel_gpu_results/31_reference_soss_low_3seed/soss_low_ch0_24_seed0_joint_1000x1000.pkl \
  /scratch/midway3/tfairnington/accel_gpu_results/31_reference_soss_low_3seed/soss_low_ch0_24_seed1_joint_1000x1000.pkl \
  /scratch/midway3/tfairnington/accel_gpu_results/31_reference_soss_low_3seed/soss_low_ch0_24_seed2_joint_1000x1000.pkl \
  --start 0 --end 24

/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/reference_noise_floor.py \
  /scratch/midway3/tfairnington/accel_stage_inputs/GJ-3470_NIRSPEC_G395H_nrs1_R20_low_resolution_inputs.pkl \
  /scratch/midway3/tfairnington/accel_gpu_results/32_reference_g395h_low_3seed/g395h_low_ch0_5_seed0_joint_1000x1000.pkl \
  /scratch/midway3/tfairnington/accel_gpu_results/32_reference_g395h_low_3seed/g395h_low_ch0_5_seed1_joint_1000x1000.pkl \
  /scratch/midway3/tfairnington/accel_gpu_results/32_reference_g395h_low_3seed/g395h_low_ch0_5_seed2_joint_1000x1000.pkl \
  --start 0 --end 5

/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/reference_noise_floor.py \
  /scratch/midway3/tfairnington/accel_stage_inputs/GJ-3470_NIRSPEC_G395H_nrs1_R300_high_resolution_inputs.pkl \
  /scratch/midway3/tfairnington/accel_gpu_results/33_reference_g395h_high_3seed/g395h_high_ch0_40_seed0_joint_1000x1000.pkl \
  /scratch/midway3/tfairnington/accel_gpu_results/33_reference_g395h_high_3seed/g395h_high_ch0_40_seed1_joint_1000x1000.pkl \
  /scratch/midway3/tfairnington/accel_gpu_results/33_reference_g395h_high_3seed/g395h_high_ch0_40_seed2_joint_1000x1000.pkl \
  --start 0 --end 40
```

### Tests

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
taskset -c 0-15 \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest \
  tests/test_stage_inputs_dump.py \
  tests/test_reference_noise_floor.py \
  tests/test_fit_independent_routing.py \
  tests/test_mcmc_runner_reuse.py \
  tests/test_spectro_safety_guards.py -x -q
```

Result: `22 passed, 2 warnings in 54.39s`. A separate `py_compile` check also
passed for both tools and both directly changed tests.

## Failures, limitations, and open risks

- Early V100 replay attempts exposed a CPU/GPU initial-potential reduction
  difference: `1.715e-5` for SOSS high and `2.346e-5` for SOSS low. The strict
  CPU loader test remains `1e-8`; the GPU driver defaults to/checks `1e-4` and
  records the tolerance in timing JSON. This does not change model inputs or
  sampling, but it is a cross-accelerator reproducibility limit.
- The G395H-high seed-0 channel-9 `log_jitter` ESS is only 6.91 despite zero
  divergences. The pooled reference is still the requested exact-NUTS
  ensemble, but this site/channel should not be treated as a sharply known
  width reference.
- Each seed is one NumPyro chain, so per-run R-hat is unavailable. Three
  independent seeds quantify between-run Monte-Carlo variation directly,
  which is the quantity used to set these gates.
- Only high-resolution channels 0:40 were calibrated. Do not assume identical
  noise floors for later 40-channel chunks without measuring them.
- As documented in `acceleration_reports/harness.md`, the dump geometries came
  from short white-light fits and the high-resolution dump initialization/mask
  was reached through a dump-only low-resolution bridge. These references are
  exact for the captured production-shaped stage calls, but not replacements
  for literature posterior products from a full upstream fit.
- PRISM has no 1000/1000 pooled reference because the measured projection was
  83.04 minutes, above the explicit 45-minute limit.
- The noise-ratio gate is intentionally loose because of `log_jitter` tails.
  Future candidate reports should show per-site results and `total_error`, not
  only a class-level PASS.

