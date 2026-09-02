# Fixed-timescale `explinear_spectroscopic`

## Outcome

Implemented an opt-in spectroscopic trend

`c + v (t - min(t)) + A exp(-(t - min(t)) / tau_WL)`

where `tau_WL` is the posterior median saved by the unchanged white-light
`explinear` fit. The per-channel latent sites remain `c`, `v`, and `A`, with
the existing `A ~ Uniform(-0.1, 0.1)` prior. Each channel still emits a
deterministic `tau`, constant at `tau_WL`, so the existing CSV schema is
preserved. Existing configurations are unchanged unless
`flags.spectro_fixed_timescale_trends: true` is set.

This intentionally makes a small spectroscopic model change: uncertainty in
the white-light timescale is not propagated into each channel, and every
channel uses the same exponential shape.

## Files

| File | Change |
|---|---|
| `models/trends.py` | Added fixed-template forward kernel. |
| `models/detrend.py` | Registered `explinear_spectroscopic`. |
| `models/jaxoplanet/builder.py` | Added shared `exp_trend`/`fixed_tau`; samples `A`, emits deterministic `tau`, supports free and Gaussian-marginalized trend modes. |
| `models/trend_marginal.py` | Uses the supplied exponential vector as the analytic `A` design column. |
| `fit_jwst.py` | Added opt-in routing, WL-median tau handoff, stage-time alignment, LR/HR model kwargs and plotting/evaluation support. |
| `plotting.py` | Added fixed-template trend rendering. |
| `tests/test_explinear_spectroscopic.py` | Trace, forward equality, and opt-in routing tests. |
| `configs_explinear/WASP-63_soss_order1_EXPLINSPEC.yaml` | New immutable validation config/output family. |
| `configs_explinear/HAT-P-11_nrs1_g395h_v2_EXPLINSPEC.yaml` | New immutable validation config/output family. |
| `acceleration_reports/gpu_queue/pending/260_explinspec_wasp63_A.sh` | WASP-63 queue submission (now running). |
| `acceleration_reports/gpu_queue/done/261_explinspec_hatp11_A.sh` | First HAT-P-11 attempt; failed before Python because one dispatcher used the queue directory as cwd. |
| `acceleration_reports/gpu_queue/pending/262_explinspec_hatp11_A_retry.sh` | Absolute-path HAT-P-11 retry (now running). |

The independent NUTS/HMC and Laplace-IS backends need no special branch:
`exp_trend` and scalar `fixed_tau` are shared model kwargs, while their existing
channel partitioner slices only explicitly named channel-varying kwargs.
Both kwargs are already included in the checkpoint and sampling-workload
fingerprints through the complete `model_kwargs` payload. Compile-box cadence
padding also handles `exp_trend` because its final axis is time. The opt-in flag
fails early for Harmonica rather than silently ignoring the unsupported trend.

## CPU validation

Commands:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  -m pytest tests/test_explinear_spectroscopic.py -x -q

JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  -m pytest tests/test_trend_marginal_integration.py \
  tests/test_spectro_safety_guards.py tests/test_independent_nuts.py \
  tests/test_laplace_is.py -x -q
```

| Check | Result |
|---|---:|
| New focused tests | 3 passed in 27.27 s |
| Nearby trend/safety/independent/Laplace-IS tests | exit 0, 15 tests |
| Fixed-template versus free-tau forward model at identical tau | max tolerance `1e-14` |
| Trace latent `log_tau` | absent |
| Trace deterministic `tau` | shape `[channel]`, exactly constant |
| `A` support | exactly `[-0.1, 0.1]` |

## GPU validation

Exact queue commands are contained in the scripts listed above. Their effective
campaign invocations are:

```bash
python tools/campaign/run_parity_dataset.py \
  --config configs_explinear/WASP-63_soss_order1_EXPLINSPEC.yaml \
  --candidates A \
  --staged-reference /scratch/midway3/tfairnington/accel_parity_staged/WASP-63_soss_order1_config_reference.pkl \
  --run-tag 20260902_explinspec

python tools/campaign/run_parity_dataset.py \
  --config configs_explinear/HAT-P-11_nrs1_g395h_v2_EXPLINSPEC.yaml \
  --candidates A \
  --staged-reference /scratch/midway3/tfairnington/accel_parity_staged/HAT-P-11_nrs1_g395h_v2_config_reference.pkl \
  --run-tag 20260902_explinspec_retry
```

At report time both retries are running on separate queue dispatchers and are
still in the unchanged 1000-warmup/1000-draw white-light NUTS stage. No fixed-
timescale spectroscopic result exists yet, so the requested candidate metrics
are honestly reported as pending rather than inferred from partial logs.

| Dataset | Fixed tau (WL median) | Wall | Divergences | depth ESS min/p05/median | gate pass | offset | slope | RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WASP-63 SOSS order 1 | pending final posterior (`~0.1 d` optimized pre-NUTS value only; not used as a reported median) | pending | pending | pending | pending | pending | pending | pending |
| HAT-P-11 G395H NRS1 V2 | pending final posterior (`~0.0389 d` optimized pre-NUTS value only; not used as a reported median) | pending | pending | pending | pending | pending | pending | pending |

Existing identical-input free-tau candidate-A baselines from
`acceleration_reports/parity.md` are:

| Dataset | Wall (min) | depth ESS median/min | divergences | gate pass | offset (ppm) | slope (ppm/um) | RMS (ppm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| WASP-63 SOSS order 1 | 28.0 | 1140/379 | 84 | 93.69% | -0.151 | +0.464 | 6.54 |
| HAT-P-11 G395H NRS1 V2 | 27.5 | 1450/461 | 10 | 91.83% | -0.771 | -1.89 | 2.63 |

The fixed-tau depth difference relative to the free-tau production control
cannot be quantified until the running posterior checkpoints complete. No
speedup is claimed.

## Failures and open risks

- Queue attempt 261 failed with exit 2 before running Python because its
  dispatcher started in `gpu_queue/pending`; retry 262 uses absolute paths and
  an explicit project `cd`. No science output was overwritten.
- Fixing tau changes the spectroscopic model and may shift depths, especially
  for WASP-63 where the optimized white-light tau is near the prior ceiling.
- A plug-in posterior median ignores white-light tau uncertainty. This is the
  requested fixed-shape treatment but should only become a production default
  after the two parity results pass the calibrated and ppm-level gates.
- Harmonica rejects this opt-in mode cleanly; it is not implemented in the
  separate Harmonica model builder.
- GPU wall/fidelity/divergence and fixed-versus-free depth metrics remain open
  until jobs 260 and 262 finish.

## Continuation: completed GPU validation (2026-09-02)

The original jobs did not produce usable final results:

- `260_explinspec_wasp63_A` exited 143 when its allocation expired. Its
  automatic `_rq` copy exited 1 because the immutable campaign result directory
  already existed.
- `262_explinspec_hatp11_A_retry` ultimately completed with exit 0 just before
  its short-lived allocation ended. Per the continuation request, it was still
  rerun under a new output tree on a fresh allocation. The intervening v2
  replacement was claimed by GPU10 and exited 143.

Following the manifest rules, none of those partial trees was deleted or
overwritten. New scripts and new campaign/output trees were used:

| Script | Run tag | GPU | Result |
|---|---|---|---|
| `263_explinspec_wasp63_A_v2.sh` | `20260902_explinspec_v2` | Tesla V100, allocation 12/job 57401379 | exit 0 |
| `264_explinspec_hatp11_A_v2.sh` | `20260902_explinspec_v2` | expired GPU10 | exit 143; preserved |
| `265_explinspec_hatp11_A_v3.sh` | `20260902_explinspec_v3` | Tesla V100, allocation 13/job 57401380 | exit 0 |

Only one script was pending at a time. Both successful jobs were monitored with
a blocking ten-second polling loop until their `.exit` marker appeared. The
campaign tool rejects a pre-existing result directory, so completed white-light
products from interrupted trees could not be injected safely through its normal
interface. The successful runs therefore recomputed white light and reused only
immutable input/reference data.

### Final candidate-A results versus the saved production run

| Dataset | fixed `tau_WL` (d) | wall (s / min) | high-res divergences | depth ESS min / p05 / median | calibrated rows passed | depth/rors rows passed | offset (ppm) | slope (ppm/um) | RMS (ppm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WASP-63 SOSS order 1 | 0.0930182 | 973.6 / 16.23 | 0 | 2115 / 2461 / 3179 | 242/944 = 25.64% (FAIL) | 0/236 = 0% | -35.04 | +12.25 | 77.46 |
| HAT-P-11 G395H NRS1 V2 | 0.0397364 | 1353.8 / 22.56 | 0 | 1927 / 1990 / 2508 | 182/544 = 33.46% (FAIL) | 46/136 = 33.82% | -1.762 | +1.302 | 9.401 |

The aggregate calibrated fraction is unusually punitive for this intended model
change because deterministic `tau` has zero posterior width and therefore fails
every uncertainty-ratio row. The separate depth/rors fraction above avoids
mistaking that bookkeeping effect for spectral fidelity. Even on depth alone,
however, neither run passes the calibrated gate.

### Depth change caused by fixing tau

These comparisons use the same wavelength grids and compare channel posterior
median depths. “Free A” is the original Laplace-metric candidate A in
`20260902a`; control D is the regenerated production joint-NUTS control and
exists for WASP-63 only.

| Dataset | Comparison | weighted offset (ppm) | slope (ppm/um) | RMS (ppm) | median absolute channel change (ppm) | maximum absolute channel change (ppm) |
|---|---|---:|---:|---:|---:|---:|
| WASP-63 | fixed tau vs free A | -34.83 | +11.95 | 76.75 | 41.40 | 242.73 |
| WASP-63 | fixed tau vs production control D | -34.58 | +11.21 | 78.06 | 39.33 | 267.17 |
| HAT-P-11 V2 | fixed tau vs free A | -0.991 | +3.391 | 8.712 | 6.688 | 17.90 |

Thus the fixed-timescale model removes the curved per-channel `A`-`tau` ridge,
but for WASP-63 it changes the inferred transmission spectrum far beyond Monte
Carlo scatter and the ppm-level literature gate. The WASP-63 result is also
inconsistent with the regenerated production control D, not merely with an old
saved artifact. HAT-P-11 is much closer in mean depth but still fails the
calibrated per-channel gate and has a measurable wavelength slope.

### Wall and sampling comparison with free-tau candidate A

| Dataset | free-tau A wall (s) | fixed-tau wall (s) | whole-pipeline wall ratio | free / fixed divergences | free ESS min / p05 / median | fixed ESS min / p05 / median |
|---|---:|---:|---:|---:|---:|---:|
| WASP-63 | 1681.9 | 973.6 | 1.73x faster | 84 / 0 | 379 / 467 / 1135 | 2115 / 2461 / 3179 |
| HAT-P-11 V2 | 1649.0 | 1353.8 | 1.22x faster | 10 / 0 | 461 / 686 / 1449 | 1927 / 1990 / 2508 |

The wall ratios are measured end to end on the same dataset/config family and
V100-class GPUs, but include independently rerun unchanged white-light fits and
LD-cache construction. They are not spectroscopic-kernel-only speedups. Control
D for WASP-63 took 2390.1 s; its different joint-NUTS algorithm makes that wall
useful context but not an algorithm-isolated fixed-tau speedup.

### Exact successful commands and artifacts

```bash
# Queue script 263
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  tools/campaign/run_parity_dataset.py \
  --config configs_explinear/WASP-63_soss_order1_EXPLINSPEC.yaml \
  --candidates A \
  --staged-reference /scratch/midway3/tfairnington/accel_parity_staged/WASP-63_soss_order1_config_reference.pkl \
  --run-tag 20260902_explinspec_v2

# Queue script 265
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  tools/campaign/run_parity_dataset.py \
  --config configs_explinear/HAT-P-11_nrs1_g395h_v2_EXPLINSPEC.yaml \
  --candidates A \
  --staged-reference /scratch/midway3/tfairnington/accel_parity_staged/HAT-P-11_nrs1_g395h_v2_config_reference.pkl \
  --run-tag 20260902_explinspec_v3
```

Final campaign JSON files:

- `/scratch/midway3/tfairnington/accel_parity/20260902_explinspec_v2/WASP-63_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_EXPLINEAR_EXPLINSPEC/result_final.json`
- `/scratch/midway3/tfairnington/accel_parity/20260902_explinspec_v3/HAT-P-11_G395H_NRS1_V2_STELLARINFORMEDLD_POWER2_EXPLINEAR_EXPLINSPEC/result_final.json`

### Recommendation

Keep `spectro_fixed_timescale_trends` opt-in and **do not enable it for
production**. It succeeds as an engineering intervention (zero divergences,
higher ESS, modest whole-pipeline wall reduction), but fails the required
science-fidelity gate, decisively for WASP-63. A future exact alternative would
need to retain per-channel tau uncertainty while reparameterizing the ridge or
using a better geometry-aware inference method; merely plugging in the
white-light median is not fidelity-safe.
