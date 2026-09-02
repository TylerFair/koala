# Laplace-IS v3 Report

Worker: `laplace_is_v3_report`

## Executive Verdict

The on-disk v3 implementation matches the requested changes and the five requested CPU test files pass.  The completed V100 measurements show that `spectro_sampler: laplace_is` is much stronger than v2 on stellar-informed SOSS, G395H, and PRISM low resolution, but it is not generally equivalent to Hessian-metric NUTS/HMC within Monte-Carlo scatter.

For science sites under the calibrated gate, the completed v3 rows are:

| Run | Channels | Thin | Wall (s) | Compile (s) | Fallback lanes | Calibrated science pass | Mean depth offset (ppm) | Slope (ppm/um) | RMS channel median diff (ppm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SOSS high vs same-input control | 118 | 8 | 231.45 | 204.79 | 3/118 | 816/826 | +0.21 | -0.50 | 5.57 |
| G395H high vs pooled reference | 40 | 8 | 72.24 | 62.83 | 0/40 | 240/240 | +0.06 | +0.09 | 2.32 |
| PRISM low vs 1000/1000 joint | 21 | 8 | 170.91 | 107.23 | 1/21 | 174/189 | -0.40 | -0.44 | 2.99 |
| SOSS high early-exit repeat | 118 | 8 | 225.38 | 200.57 | 3/118 | 816/826 | not recomputed; same samples within roundoff | not recomputed | not recomputed |
| G395H high sensitivity | 40 | 16 | 74.99 | 64.04 | 0/40 | 238/240 | -0.29 | -0.89 | 1.87 |

The spectrum-level offsets are small in absolute ppm.  The remaining issue is per-site distribution fidelity and ESS, especially SOSS `v` and spot/trend/LD tails, plus PRISM trend/spot width failures.  Hessian-metric NUTS/HMC remains the safer production recommendation.

## Files Checked Or Created

| File | Status |
|---|---|
| `models/laplace_is.py` | Verified v3 defaults and fallback implementation. |
| `fit_jwst.py` | Verified v3 option parsing and wide/uniform-LD routing. |
| `tests/test_laplace_is.py` | Existing v3 coverage includes thin, fallback, runner reuse, and routing tests. |
| `acceleration_reports/data/laplace_v3_108_soss_high_calibrated.json` | Created calibrated summary. |
| `acceleration_reports/data/laplace_v3_109_g395_high_calibrated.json` | Created calibrated summary. |
| `acceleration_reports/data/laplace_v3_110_prism_low_calibrated.json` | Created calibrated summary. |
| `acceleration_reports/data/laplace_v3_112_g395_high_thin16_calibrated.json` | Created calibrated summary. |
| `acceleration_reports/gpu_queue/pending/113_laplace_v3_prism_high.sh` | Created queue script for missing PRISM high Laplace-IS diagnostics. |
| `acceleration_reports/laplace_is_v3.md` | This report. |

No model physics, likelihood, priors except explicit run-time `jitter_prior=lognormal`, masks, data, limb darkening, or precision were changed.

## Implementation Audit

The requested items are present on disk:

| Requested v3 item | Evidence |
|---|---|
| IMH thinning default 8 | `fit_jwst._resolve_laplace_is_stage_kwargs`, `build_laplace_is_runner`, and `get_samples_laplace_is` all default `laplace_is_imh_thin=8`. |
| Failed lanes use Laplace-metric independent NUTS | `models/laplace_is.py` fallback sets `mass_matrix="laplace"`, `laplace_hessian_method="finite_difference"`, `laplace_warmup=150`, `laplace_max_tree_depth=6`, `laplace_trust_radius=5.0`, and `laplace_start_at_map=True`. |
| Trust radius 5 and decrement stopping | Laplace-IS MAP default `laplace_is_trust_radius=5.0`, `laplace_is_map_tol=1e-4`; shared spectro Laplace defaults also use radius 5 and decrement tolerance `1e-4`. |
| Wide-Gaussian/uniform LD route away from Laplace-IS | `fit_jwst._run_spectroscopic_stage` routes `laplace_is` with `ld_mode in {"widegaussian", "uniform"}` to independent NUTS with Laplace metric unless `laplace_is_force=true`. |
| Gate unchanged | Gate still requires `k_hat < 0.7`, IMH acceptance `>= 0.2`, and IS-ESS `>= max(400, 0.2 N)`. |

The automatic route matters: the failed v2 wide-LD SOSS regime should not silently use Laplace-IS unless explicitly forced.

## Reproduction Commands

CPU tests:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
taskset -c 0-15 /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest \
  tests/test_laplace_is.py \
  tests/test_independent_nuts.py \
  tests/test_independent_hmc.py \
  tests/test_fit_independent_routing.py \
  tests/test_spectro_safety_guards.py -x -q
```

Result: `34 passed, 8 warnings in 166.56s`.

Completed GPU scripts are preserved in `acceleration_reports/gpu_queue/done/`:

```bash
bash acceleration_reports/gpu_queue/done/108_laplace_v3_soss_full.sh
bash acceleration_reports/gpu_queue/done/109_laplace_v3_g395_high.sh
bash acceleration_reports/gpu_queue/done/110_laplace_v3_prism_low.sh
bash acceleration_reports/gpu_queue/done/111_laplace_v3_soss_earlyexit.sh
bash acceleration_reports/gpu_queue/done/112_laplace_v3_g395_thin16.sh
```

Calibrated summaries:

```bash
JAX_PLATFORMS=cpu /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/summarize_laplace_is_v3.py \
  /scratch/midway3/tfairnington/accel_gpu_results/108_laplace_v3_soss_full/soss_high_0_118_thin8.pkl \
  /scratch/midway3/tfairnington/accel_campaign/HAT-P-12_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_SPOT/candidates/CTRL_joint_loguniform_high_resolution.pkl \
  /scratch/midway3/tfairnington/accel_campaign/HAT-P-12_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_SPOT/dumps/HAT-P-12_NIRISS_SOSS_order1_Rreference_high_resolution_inputs.pkl \
  --arviz /scratch/midway3/tfairnington/accel_gpu_results/108_laplace_v3_soss_full/soss_high_0_118_thin8.arviz.json \
  --output acceleration_reports/data/laplace_v3_108_soss_high_calibrated.json

JAX_PLATFORMS=cpu /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/summarize_laplace_is_v3.py \
  /scratch/midway3/tfairnington/accel_gpu_results/109_laplace_v3_g395_high/g395_high_0_40_thin8.pkl \
  /scratch/midway3/tfairnington/accel_stage_inputs/references/GJ-3470_NIRSPEC_G395H_nrs1_R300_high_resolution_inputs_ch0_40_pooled_joint_nuts.pkl \
  /scratch/midway3/tfairnington/accel_stage_inputs/GJ-3470_NIRSPEC_G395H_nrs1_R300_high_resolution_inputs.pkl \
  --arviz /scratch/midway3/tfairnington/accel_gpu_results/109_laplace_v3_g395_high/g395_high_0_40_thin8.arviz.json \
  --output acceleration_reports/data/laplace_v3_109_g395_high_calibrated.json

JAX_PLATFORMS=cpu /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/summarize_laplace_is_v3.py \
  /scratch/midway3/tfairnington/accel_gpu_results/110_laplace_v3_prism_low/prism_low_0_21_thin8.pkl \
  /scratch/midway3/tfairnington/accel_gpu_results/prism/prism_low_joint_1000_1000.pkl \
  /scratch/midway3/tfairnington/accel_stage_inputs/HAT-P-65_NIRSPEC_PRISM_nrs1_R10_low_resolution_inputs.pkl \
  --arviz /scratch/midway3/tfairnington/accel_gpu_results/110_laplace_v3_prism_low/prism_low_0_21_thin8.arviz.json \
  --output acceleration_reports/data/laplace_v3_110_prism_low_calibrated.json
```

Missing PRISM high diagnostic queued for the orchestrator:

```bash
bash acceleration_reports/gpu_queue/pending/113_laplace_v3_prism_high.sh
```

As of this report, `113_laplace_v3_prism_high.sh` is pending behind two white-light runs; no PRISM-high Laplace-IS timing claim is made here.

## Completed V100 Diagnostics

| Run | k-hat min/median/max | IS-ESS min/median/max | IMH accept min/median/max | Effective draw chunk | GPU memory max |
|---|---|---|---|---:|---:|
| SOSS high 0:118 thin8 | -0.030 / 0.277-0.359 / 0.718 by chunk | 940 / 1958-2620 / 2746 by chunk | 0.368 / 0.488-0.589 / 0.604 by chunk | 256 | 12580 MiB |
| G395H high 0:40 thin8 | -0.120 / 0.119 / 0.343 | 2519 / 2589 / 2656 | 0.554 / 0.574 / 0.587 | 256 | 12540 MiB |
| PRISM low 0:21 thin8 | 0.245 / 0.434 / 0.835 | 334 / 1351 / 2049 | 0.322 / 0.409 / 0.495 | 64 | 12568 MiB |
| SOSS high early-exit repeat | -0.030 / 0.277-0.359 / 0.718 by chunk | 940 / 1958-2620 / 2746 by chunk | 0.368 / 0.488-0.589 / 0.604 by chunk | 256 | 12580 MiB |
| G395H high thin16 | -0.120 / 0.120 / 0.343 | 2519 / 2589 / 2656 | 0.564 / 0.573 / 0.586 | 256 | 12540 MiB |

Fallback costs:

| Run | Fallback lanes | Fallback wall |
|---|---:|---:|
| SOSS high thin8 | 2 lanes in chunk 0, 1 lane in chunk 1, 0 in chunk 2 | 75.67 s + 74.34 s |
| G395H high thin8 | 0/40 | 0 s |
| PRISM low thin8 | 1/21 | 65.07 s |
| SOSS high early-exit repeat | 2 + 1 + 0 lanes | 73.99 s + 72.72 s |
| G395H high thin16 | 0/40 | 0 s |

Bulk ESS from ArviZ:

| Run | Science ESS min | Science ESS median | All-site ESS min | All-site ESS median |
|---|---:|---:|---:|---:|
| SOSS high thin8 | 22.1 | 916.3 | 18.5 | 905.9 |
| G395H high thin8 | 600.0 | 944.7 | 600.0 | 944.7 |
| PRISM low thin8 | 65.8 | 798.8 | 65.8 | 800.3 |
| SOSS high early-exit repeat | 22.1 | 916.3 | 18.5 | 905.9 |
| G395H high thin16 | 699.2 | 977.4 | 699.2 | 975.5 |

## Fidelity Notes

The flat diagnostic gate in `tools/run_sampler_on_stage_inputs.py` still reports failures because it requires `|dmedian| < 0.1 reference sigma` and sigma ratio in `[0.9, 1.1]` for every row, including noise sites affected by the jitter-prior change.  The calibrated gate is the relevant science gate from the orchestrator addenda.

Per-site calibrated science pass counts:

| Run | Main failures |
|---|---|
| SOSS high thin8 | 816/826 calibrated science rows pass.  Worst remaining science behavior is low ESS in a few SOSS rows and failures concentrated in `v`, `A_spot`, and LD/trend tails. |
| G395H high thin8 | 240/240 calibrated science rows pass with zero fallback lanes. |
| PRISM low thin8 | 174/189 calibrated science rows pass.  Depth medians are good in ppm, but `A`, `c`, and `v` still have distribution-shape failures. |
| G395H high thin16 | 238/240 calibrated science rows pass; thin16 raises ESS but slightly worsens the calibrated pass count relative to thin8 on this seed. |

The PRISM low handoff value is reproduced: science pass `174/189`, mean depth offset `-0.40 ppm`, slope `-0.44 ppm/um`, and RMS per-channel median difference `2.99 ppm`.

## What Failed Or Remains Open

1. PRISM high Laplace-IS diagnostics were missing from the completed v3 queue set.  I queued `113_laplace_v3_prism_high.sh` with chunk size 4, which is the largest known 16 GB-safe PRISM native-resolution chunk size from the campaign notes.  It had not started by report time.
2. SOSS high is faster than v2 and has only three fallback lanes, but compile dominates: 204.79 s recorded compile out of 231.45 s wall.  A deployment speedup should be measured end-to-end with white-light included and warmed/compiled behavior stated explicitly.
3. IMH output still has lower effective sample size than exact NUTS/HMC for some sites.  SOSS science ESS minimum is 22.1 and PRISM low science ESS minimum is 65.8, which is not equivalent to the Hessian-metric NUTS/HMC rows at equal returned draws.
4. The Laplace-IS gate can admit lanes whose finite IMH output misses calibrated distribution gates.  Thin=16 helped ESS on G395H but did not monotonically improve calibrated fidelity.
5. Comparisons involving `log_jitter` and `total_error` are not sampler-fidelity evidence when candidate runs force `jitter_prior=lognormal` against log-uniform references.

## Bottom Line

`spectro_sampler: laplace_is` is now a useful specialized candidate: it passes calibrated science gates on G395H high 0:40 and is ppm-clean on SOSS high and PRISM low.  It is not yet a drop-in equivalent to Hessian-metric NUTS/HMC within Monte-Carlo scatter across the tested regimes.  The production recommendation remains Hessian-metric independent NUTS/HMC for fidelity-critical runs, with Laplace-IS reserved for G395H-like near-Gaussian informed-LD chunks or for diagnostic acceleration where the lower IMH ESS is acceptable.
