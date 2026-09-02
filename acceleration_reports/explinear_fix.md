# Explinear spectroscopic divergence fix

Status: **GPU validation pending; no production rule enabled.**

## What I built

| File | Change |
|---|---|
| `tools/campaign/run_parity_dataset.py` | Added additive candidate `A99` (Laplace independent NUTS, target acceptance 0.99, maximum tree depth 10, 200 Laplace warmup steps, 2,000 retained draws) and `--resume-existing`. The resume path consumes the immutable `dumps_A` low/high stage inputs and writes only `a99_replay/`, `fit_A99_*.log`, and `result_after_A99.json`; it never calls `fit_jwst.py` or regenerates white light. |
| `acceleration_reports/gpu_queue/pending/242_explinear_wasp63_a99.sh` | Queue entry for the WASP-63 SOSS explinear replay. |

No existing candidate A/B/C result, dump, checkpoint, or output directory was deleted or overwritten.

## Reproduction

CPU syntax validation:

```bash
cd /project/ekempton/tfairnington/JWST
JAX_PLATFORMS=cpu /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m py_compile tools/campaign/run_parity_dataset.py
```

The queued GPU command is:

```bash
python tools/campaign/run_parity_dataset.py \
  --config configs_fiducial_stellarinformed/WASP-63_soss_order1_config.yaml \
  --staged-reference /scratch/midway3/tfairnington/accel_parity_staged/WASP-63_soss_order1_config_reference.pkl \
  --run-tag 20260902a --candidates A99 --resume-existing
```

The runner expands each existing low/high dump to `tools/run_sampler_on_stage_inputs.py` with:

```text
--backend independent_nuts --warmup 200 --samples 2000 --chunk-size 40
--nuts-override mass_matrix=laplace
--nuts-override laplace_warmup=200
--nuts-override laplace_target_accept=0.99
--nuts-override laplace_max_tree_depth=10
--nuts-override laplace_start_at_map=True
--nuts-override laplace_fuse_program=True
```

## Measurements available

These are the pre-existing identical-input parity results motivating A99, not new speed measurements:

| Dataset / candidate | Divergences | Calibrated gate pass | Offset (ppm) | Notes |
|---|---:|---:|---:|---|
| WASP-63 SOSS A | 84 | 93.7% | -0.15 | TA 0.95, depth 6 |
| WASP-63 SOSS B | 37 | not passing | — | HMC-8 |
| WASP-63 SOSS C | — | not passing | — | 48 fallback lanes |
| HAT-P-11 G395H NRS1 V2 A | 10 | 91.8% | -0.77 | saved JSON gives slope -1.89 ppm/um and depth ESS min/median/p05 461/1449/686 |

No A99 measurements exist yet, so no A99 speedup or fidelity improvement is reported.

## Queue/blocker

`242_explinear_wasp63_a99.sh` was queued at 07:13 CDT and polled at intervals of at least 20 seconds through 07:26 CDT. It remained pending. Four older parity scripts remained in `gpu_queue/running/` throughout, with their last visible output timestamps between 06:21 and 06:49. The production control `241_parity_wasp63_joint_control.sh` also remained pending, so D could not be compared.

Per the one-pending-script rule, HAT-P-11 A99 was not queued behind WASP-63.

## Validation gates once the queue advances

For each dataset, harvest `result_after_A99.json` and require:

- zero divergences across both low- and high-resolution stages;
- calibrated `fidelity.gate_pass` and its pass fraction;
- weighted ppm offset, wavelength slope, and channel RMS versus the saved spectrum;
- depth ESS minimum, 5th percentile, and median.

Only if both datasets pass should production automatically select `spectro_laplace_target_accept: 0.99` and depth 10 when `detrending_type` contains `explinear`. Explicit stage-specific/user flags should retain precedence. This automatic selection was intentionally not implemented before validation.

## Open risks

- Higher target acceptance may eliminate divergences but materially increase leapfrog count and wall time.
- The saved spectra can reveal Monte-Carlo scatter but do not replace comparison with WASP-63 production control D on the regenerated inputs.
- A99 replay creates its output directory exclusively. If a killed run leaves a partial tree, a new suffixed output path is required; it must not be overwritten.

## 2026-09-02 continuation: WASP-63 harvested

The real replay artifacts were read from `result_after_A99.json` and `a99_replay/*.diagnostics.json`; the truncated queue log was not used.

| WASP-63 SOSS metric | A (TA .95/depth 6) | A99 (TA .99/depth 10) |
|---|---:|---:|
| Low + high divergences | 84 | **0** |
| Calibrated gate pass fraction | 93.69% (995/1062) | 94.26% (1001/1062) |
| Strict aggregate gate | fail | fail |
| Weighted offset | -0.151 ppm | -0.227 ppm |
| Slope | +0.464 ppm/um | -0.137 ppm/um |
| RMS channel-median difference | 6.541 ppm | 5.051 ppm |
| Median depth sigma ratio | 1.006 | 1.008 |
| Depth ESS min / p05 / median | 379 / 467 / 1135 | 241 / 346 / 908 |

A99 resolved the divergence symptom and improved spectral RMS, but did **not** pass the strict calibrated aggregate gate. Its replay walls were 247.5 s low resolution plus 491.7 s high resolution (739.2 s total sampler replay); these are not used as a speedup because A was measured through a different end-to-end path.

HAT-P-11 G395H NRS1 V2 was queued as `243_explinear_hatp11_nrs1_v2_a99.sh` at 07:47 CDT. It was polled every 20--40 seconds through 08:10 CDT and remained the oldest pending script while five older jobs occupied `running/`. Therefore no HAT-P-11 A99 result exists yet.

Conclusion at this checkpoint: do **not** automatically enable the config-level explinear rule yet. One of the two required datasets still lacks A99 measurements, and WASP-63 is divergence-free but its aggregate calibrated gate remains false. Consequently `fit_jwst.py` was not changed. The proposed rule remains: when Laplace independent NUTS is selected and `detrending_type` contains `explinear`, default `spectro_laplace_target_accept` to 0.99 and `spectro_laplace_max_tree_depth` to 10 while preserving explicit generic and stage-specific overrides.
