# Stellar-informed accelerated-sampler validation campaign

## Outcome

I built a reproducible, fail-closed campaign driver and piloted it on one SOSS order 1, one G395H/NRS1, and one PRISM dataset. The driver regenerated the white-light geometry with production 1000/1000 NUTS, dumped both spectroscopic stages, verified saved-reference channel and wavelength alignment, concatenated checkpoint chunks in channel order, ran candidate samplers, and wrote machine-readable and Markdown results.

The two moderate-size pilots completed. Candidate A agreed well for the G395H high-resolution posterior (97.3% of gated science-site/channel entries passed), but not for SOSS (79.8%, with depth/rors the main failure). The independently regenerated white-light solutions moved by only 0.044 and 0.069 saved-posterior sigma respectively, so those high-resolution shifts are not explained primarily by white-light Monte Carlo variation.

The PRISM pilot completed its production white-light fit, both dumps, strict alignment, and all 42 low-resolution candidate-A channels. A 40-channel candidate chunk exceeded 16 GB; chunk size 4 worked. The 90-minute queue limit then stopped the high-resolution run after 324/368 channels (88.0%), so no high-resolution PRISM sample file or fidelity claim is reported. Extrapolating only its measured 81 chunks gives 4,997 s for the full high-resolution candidate stage and 7,197 s (2.00 h) for a complete PRISM A-only dataset including its measured white-light and low-resolution walls.

## Files built

- `tools/campaign/run_dataset.py`: config cloning, production white-light/dump run, strict alignment, reference concatenation, candidate execution, calibrated gates, diagnostics, geometry comparison, JSON/Markdown output, and dump-resume support.
- `tools/campaign/run_dataset.sh`: executable entry point.
- `tools/campaign/stage_reference.py`: stages the immutable saved-reference subset to scratch because `/cds2` is absent on GPU nodes; supports versioned `*_wavelengths_V1.csv` names.
- `tools/campaign/summarize.py`: aggregates result JSON files, emits per-run and per-site tables, and projects population cost only from complete A-stage pairs.
- `tools/campaign/__init__.py`.
- `tests/test_campaign.py`: reference concatenation/alignment, calibrated-gate, and projection tests.
- `configs_campaign/campaign_HAT-P-12_soss_order1_config.yaml`, `campaign_WASP-52_nrs1_g395h_config.yaml`, and `campaign_Kepler-12_nrs1_prism_v1_config.yaml`: generated campaign copies with `_CAMPAIGN` output directories, GPU host device, production white-light settings, and lognormal spectroscopic jitter.
- `acceleration_reports/campaign_summary.json` and `campaign_summary.md`: aggregate machine-readable and full per-site outputs.

All candidate comparisons exclude `log_jitter` and `total_error` from the gate because the saved population runs used the old jitter prior. The calibrated gates used were depth/rors 0.133 sigma and ratio [0.816, 1.226], trend 0.164 and [0.826, 1.210], LD 0.202 and [0.754, 1.327], and other sites 0.150 and [0.829, 1.207].

## Pilot inputs and dumps

| Dataset | Mode | Stage | Channels | Cadences | Active window | Dump bytes |
|---|---|---|---:|---:|---:|---:|
| HAT-P-12 | SOSS o1 | low R20 | 24 | 226 | 95 | 97,631 |
| HAT-P-12 | SOSS o1 | high Rreference | 118 | 225 | 94 | 451,560 |
| WASP-52 | G395H NRS1 | low R20 | 5 | 607 | 207 | 57,779 |
| WASP-52 | G395H NRS1 | high Rreference | 68 | 607 | 207 | 679,299 |
| Kepler-12 | PRISM NRS1 | low R20 | 42 | 29,127 | 12,276 | 19,866,169 |
| Kepler-12 | PRISM NRS1 | high Rnative | 368 | 29,090 | 12,259 | 171,634,414 |

The low-resolution saved runs contain best-fit summary CSVs but no low-resolution posterior checkpoint samples. Low-resolution fidelity therefore compares candidate draws to the saved median/error summary and is explicitly labeled a summary comparison. High-resolution fidelity uses the exact concatenated `[draw, channel, ...]` checkpoints. All wavelength arrays matched exactly after accepting the production `_V1` filename suffix.

## Timing and aggregate fidelity

Walls below are V100 measurements. “Total” includes white-light, LD-grid construction, bridge, dumps, candidates, comparisons, and orchestration. SOSS and G395H ran A/B/C; PRISM ran A only.

| Dataset | Production fit + dumps | Candidate | Low wall | Low gate fraction | High wall | High gate fraction | Divergences low/high | Min ESS low/high | max k-hat low/high |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| HAT-P-12 SOSS o1 | 583.3 s | A independent NUTS | 86.4 s | 59.5% | 113.9 s | 79.8% | 0 / 0 | 93.2 / 25.5 | n/a |
| HAT-P-12 SOSS o1 | 583.3 s | B independent HMC | 70.2 s | 61.3% | 80.9 s | 81.8% | 0 / 0 | 5.8 / 54.1 | n/a |
| HAT-P-12 SOSS o1 | 583.3 s | C Laplace IS | 72.9 s | 60.1% | 731.1 s | 71.9% | 0 / 0 | 62.5 / 8.1 | 0.624 / 0.718 |
| WASP-52 G395H | 411.6 s | A independent NUTS | 84.5 s | 53.3% | 113.2 s | 97.3% | 0 / 2 | 79.5 / 173.0 | n/a |
| WASP-52 G395H | 411.6 s | B independent HMC | 69.4 s | 56.7% | 78.6 s | 97.3% | 0 / 0 | 160.8 / 116.4 | n/a |
| WASP-52 G395H | 411.6 s | C Laplace IS | 74.5 s | 36.7% | 79.4 s | 96.6% | 0 / 0 | 255.5 / 47.1 | 0.399 / 0.270 |
| Kepler-12 PRISM | 1,289.5 s | A independent NUTS, chunk 4 | 911.0 s | 74.4% | timeout at 324/368 | not computed | 0 / unknown | 73.1 / unknown | n/a |

Complete end-to-end walls were 1,823.6 s (30.4 min) for SOSS A/B/C and 991.9 s (16.5 min) for G395H A/B/C. Complete A-only equivalents, excluding the optional B/C work, were 813.0 s and 637.1 s. The PRISM retry consumed the full 5,400 s queue allowance after its earlier 1,289.5 s fit/dump run; its completed low stage was 911.0 s.

Saved production reference wall time was not recoverable from the staged run logs, so no same-GPU speedup is claimed.

## Candidate A per-site fidelity

Values are `95th percentile / maximum` absolute median shifts in reference-sigma units, followed by the observed sigma-ratio range. The full A/B/C tables are in `campaign_summary.md`.

| Dataset/stage | Site | Median shift p95/max | Sigma-ratio min/max | Pass fraction |
|---|---|---:|---:|---:|
| SOSS high | depths/rors | 0.519 / 0.584 | 0.884 / 1.129 | 39.0% |
| SOSS high | c | 0.112 / 0.154 | 0.873 / 1.117 | 100% |
| SOSS high | v | 0.132 / 0.170 | 0.907 / 1.078 | 99.2% |
| SOSS high | c1 | 0.167 / 0.255 | 0.903 / 1.160 | 96.6% |
| SOSS high | c2 | 0.209 / 0.279 | 0.909 / 1.242 | 91.5% |
| G395H high | depths/rors | 0.114 / 0.189 | 0.902 / 1.110 | 95.6% |
| G395H high | c | 0.155 / 0.257 | 0.925 / 1.122 | 95.6% |
| G395H high | v | 0.149 / 0.246 | 0.918 / 1.078 | 97.1% |
| G395H high | c1 | 0.143 / 0.194 | 0.891 / 1.124 | 100% |
| G395H high | c2 | 0.120 / 0.134 | 0.896 / 1.101 | 100% |
| PRISM low summary | depths/rors | 0.332 / 0.354 | 0.844 / 1.126 | 50.0% |
| PRISM low summary | c | 0.140 / 0.248 | 0.882 / 1.325 | 88.1% |
| PRISM low summary | v | 0.116 / 0.213 | 0.912 / 1.219 | 95.2% |
| PRISM low summary | c1 | 0.140 / 0.156 | 0.899 / 1.238 | 100% |
| PRISM low summary | c2 | 0.119 / 0.160 | 0.884 / 1.159 | 100% |

## White-light geometry caveat

| Dataset | Maximum new-vs-saved white-light shift |
|---|---:|
| HAT-P-12 SOSS o1 | 0.0695 sigma |
| WASP-52 G395H | 0.0443 sigma |
| Kepler-12 PRISM | 0.0356 sigma |

These values quantify the geometry mismatch but are not a formal causal decomposition. Isolating the exact fraction of a spectroscopic shift caused by geometry would require a counterfactual candidate run conditioned on the old saved geometry. Here the geometry changes are an order of magnitude below the largest SOSS depth shifts (0.58 sigma), which is strong evidence that they are not the dominant explanation.

## Population cost projection

Using the complete A-only SOSS/G395H walls, proxying SOSS o1 for o2 and G395H for G395M, and using the measured PRISM extrapolation above:

| Mode | Runs | Wall per dataset | Projected GPU wall |
|---|---:|---:|---:|
| G395H | 32 | 637 s | 5.66 h |
| SOSS o1 | 18 | 813 s | 4.07 h |
| SOSS o2 (o1 proxy) | 13 | 813 s | 2.94 h |
| PRISM | 4 | 7,197 s extrapolated | 8.00 h |
| G395M (G395H proxy) | 3 | 637 s | 0.53 h |
| **Total** | **70** | — | **21.19 GPU-hours** |

An ideal two-V100 lower bound is 10.60 hours, before queue gaps and load imbalance. The PRISM estimate is based on 324/368 measured high-resolution channels: first chunk 136.4 s (85.4 s compile), subsequent 80 completed chunks averaging 53.41 s. It is an extrapolation, not a completed identical-input wall measurement.

## Reproduction commands

Stage references on a node that can see `/cds2`:

```bash
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/campaign/stage_reference.py \
  /scratch/midway3/tfairnington/accel_campaign_saved_refs \
  /cds2/ekempton/tfairnington/STELLARINFORMED/HAT-P-12_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_SPOT \
  /cds2/ekempton/tfairnington/STELLARINFORMED/WASP-52_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR \
  /cds2/ekempton/tfairnington/STELLARINFORMED/Kepler-12_PRISM_NRS1_V1_STELLARINFORMEDLD_POWER2_EXPLINEAR
```

Each queue script invoked one of these commands (the queue itself moved scripts from `pending/` and recorded `.exit`, `.out`, and `.gpu`):

```bash
tools/campaign/run_dataset.sh configs_fiducial_stellarinformed/HAT-P-12_soss_order1_config.yaml \
  --candidates A,B,C --saved-root /scratch/midway3/tfairnington/accel_campaign_saved_refs
tools/campaign/run_dataset.sh configs_fiducial_stellarinformed/WASP-52_nrs1_g395h_config.yaml \
  --candidates A,B,C --saved-root /scratch/midway3/tfairnington/accel_campaign_saved_refs
tools/campaign/run_dataset.sh configs_fiducial_stellarinformed/Kepler-12_nrs1_prism_v1_config.yaml \
  --candidates A --saved-root /scratch/midway3/tfairnington/accel_campaign_saved_refs
tools/campaign/run_dataset.sh configs_fiducial_stellarinformed/Kepler-12_nrs1_prism_v1_config.yaml \
  --candidates A --saved-root /scratch/midway3/tfairnington/accel_campaign_saved_refs --skip-fit
```

Aggregate results:

```bash
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/campaign/summarize.py \
  /scratch/midway3/tfairnington/accel_campaign/HAT-P-12_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_SPOT/result.json \
  /scratch/midway3/tfairnington/accel_campaign/WASP-52_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR/result.json \
  /scratch/midway3/tfairnington/accel_campaign/Kepler-12_PRISM_NRS1_V1_STELLARINFORMEDLD_POWER2_EXPLINEAR/result.json \
  --output-prefix acceleration_reports/campaign_summary
```

Verification:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  taskset -c 0-15 /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest \
  tests/test_campaign.py tests/test_stage_inputs_dump.py tests/test_reference_noise_floor.py \
  tests/test_fit_independent_routing.py tests/test_mcmc_runner_reuse.py \
  tests/test_spectro_safety_guards.py -x -q
```

Result: **25 passed, 2 warnings in 21.86 s**.

## Failures and open risks

- The first SOSS/G395H queue attempts failed immediately because GPU nodes cannot mount `/cds2`; staging to scratch fixed this. Both failures are preserved as queue exit 1 artifacts.
- PRISM initially stopped on the versioned wavelength filename. The matcher now accepts `*_wavelengths*.csv`; the failed exits are preserved.
- PRISM candidate chunks of 40 exceeded V100 memory (including a failed 7.11 GB allocation). Chunk size 4 completed low resolution and 88% of high resolution before queue exit 124.
- Candidate A had 2 G395H high-resolution divergences despite otherwise strong fidelity; it should not be accepted without addressing or bounding that tail behavior.
- SOSS depth/rors disagreement is much larger than the calibrated reference noise and fails the literature-level gate on most channels. Candidate A is therefore not validated population-wide by this pilot.
- Low-resolution comparisons are against saved summary CSVs, not posterior draws, and should not be pooled with high-resolution posterior-to-posterior metrics.
- Reference walls could not be recovered, so these results validate fidelity and campaign cost only; they do not establish same-input speedup over saved production.
- The 70-run estimate rests on one pilot per major class and proxy assumptions for SOSS o2 and G395M. A stratified expansion should sample at least several cadence/channel extremes before scheduling the full population.
