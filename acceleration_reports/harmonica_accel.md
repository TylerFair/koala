# Harmonica spectroscopic Laplace acceleration

## Bottom line

**No fundamental problem.**  The Harmonica spectroscopic likelihood is exactly
factorized by wavelength channel once the white-light geometry is fixed.  On
the real WASP-94 SOSS order-1 R20 input (24 channels, 1,430 cadences, resident
width 4), finite-difference Laplace-metric independent NUTS was stable, had no
divergences, and reduced compile-inclusive wall time from 1,030.39 s to 230.87 s
at target acceptance 0.95 (4.46x) or 358.32 s at 0.99 (2.88x).  This is useful
but below the project's 10x goal on this already relatively well-behaved joint
NUTS case.

The Hessian is positive and only moderately conditioned (per-lane condition
numbers 114.35--118.07).  The local metric therefore does not encounter the
pathology feared for weak/bounded asymmetry parameters on this dataset.

## Changes

- `fit_jwst.py`: allow the generic independent NUTS/HMC backends for Harmonica;
  route Harmonica stage options through the existing validated Laplace option
  resolver; permit quadratic `ld_prior: sing`.
- `models/harmonica/builder.py`: implement the existing physical `(l, delta)`
  Sing parameterization for quadratic Harmonica, including `sing_free`, and
  pass the derived direct `(u1,u2)` coefficients to the unchanged transit
  kernel.
- `tools/run_sampler_on_stage_inputs.py`: include `delta_r`, `a1`, `q`,
  `limb_l`, and `limb_delta` in ESS reporting.
- `tests/test_harmonica_acceleration.py`: additive tests for Laplace option
  routing and Sing-site-to-kernel flow.
- `configs_harmonica/WASP-94_soss_order1_quadratic_accel_dump.yaml`: additive
  dump-only config with the requested resident width 4 and a new output path.
- Queue scripts `270_harmonica_accel_dump.sh` and
  `271_harmonica_accel_benchmark.sh` are preserved under
  `acceleration_reports/gpu_queue/done/`.

No existing output was deleted or overwritten.  The dump run used a copied
42 MB prior output tree at the new path
`/scratch/midway3/tfairnington/HARMONICA_ACCEL_WASP94_DUMP_20260902`.

## Model structure and posterior geometry

For the measured fixed-quadratic, linear-trend R20 run, each channel has four
sampled unconstrained coordinates:

| Site | Meaning | Prior/support |
|---|---|---|
| `rors[:,0]` | mean/base radius; `depths=rors^2` | Uniform in radius, sqrt(1e-5) to sqrt(0.5) |
| `delta_r` | morning/evening limb radius difference; `a1=delta_r/2` | Normal(0, 0.2 rors) |
| `c` | additive flux offset | Uniform(0.9, 1.1) |
| `v` | linear trend | Uniform(-0.1, 0.1) |

Quadratic `u1,u2` and `total_error` are deterministic in this config;
`harmonica_spectro_fit_jitter: false`.  Other configurations add per-channel
`log_jitter`, `v2..v4`, `A/log_tau`, or fractional/higher-order odd Fourier
coefficients.  All retain a channel axis.  `PERIOD`, `t0`, `b/cos_i`, `a_rs`,
eccentricity, and omega are fixed arguments from white light, not shared
latent variables.  The observation distribution also factorizes over rows.

The FD-MAP diagnostics over all 24 real R20 lanes were:

| Quantity | Range |
|---|---:|
| Hessian condition number | 114.35--118.07 |
| smallest raw Hessian eigenvalue | 9.72e4--1.06e6 |
| final MAP gradient norm (maximum by 4-lane chunk) | 0.0025--0.1805 |

An independent audit of the pre-existing real R50 production posterior (60
channels, same object/engine/parameterization) found median channel-wise
correlations: rho(rors,delta_r)=0.042, rho(c,rors)=0.400,
rho(c,delta_r)=0.226, rho(v,delta_r)=-0.292, and rho(v,c)=-0.771.  The largest
absolute values across channels were respectively 0.193, 0.456, 0.295, 0.389,
and 0.808.  Thus the main geometry is an ordinary trend correlation captured
by a dense local metric, not a radius/asymmetry funnel.

The old R50 `delta_r` posterior is not generically a zero-centered bounded
direction: median |median/sd| was 2.51 and maximum 6.93.  Its prior is Normal,
not bounded.  Radius and trend posterior locations are far from their wide
box boundaries.  Half-area `q` is truncated and higher-order coefficients can
be weak; those unmeasured parameterizations remain an open risk.

## Same-input GPU measurement

All three entries used the identical immutable R20 dump, seed/key, 1,000
retained draws, width 4, float64, and the same Tesla V100-PCIE-16GB allocation.
Joint NUTS used 1,000 adaptive warmup draws; Laplace NUTS used the standard 150
metric warmup draws after FD MAP/Hessian preparation.  Walls include cold
compilation and all six chunks.

| Sampler | Wall (s) | Speedup | Recorded compile (s) | Steps median / max | Divergences | min bulk ESS |
|---|---:|---:|---:|---:|---:|---:|
| joint NUTS, TA 0.8 | 1030.39 | 1.00x | 87.28 | 31 / 63 | 0 | 354.8 |
| Laplace independent NUTS, TA 0.95 | 230.87 | 4.46x | 98.56 | 7 / 7 | 0 | 791.1 |
| Laplace independent NUTS, TA 0.99 | 358.32 | 2.88x | 97.85 | 15 / 15 | 0 | 548.0 |

TA 0.95 is the better measured operating point here: higher ESS, half the
leapfrogs of TA 0.99, and zero divergences.

### Calibrated posterior comparison

The repository replay tool's deliberately strict per-row gate is
`|candidate median - reference median| < 0.1 reference SD` and SD ratio in
`[0.9,1.1]`.  It compares two finite, single-seed 1,000-draw Monte Carlo runs,
so a handful of marginal failures is expected even for exact samplers.  The
aggregate all-site gate nevertheless **failed**, and this must not be called a
formal fidelity pass.

| Candidate | Depth failed channels | max depth median shift | depth SD-ratio range | delta_r failed channels | max delta_r median shift | delta_r SD-ratio range |
|---|---:|---:|---:|---:|---:|---:|
| TA 0.95 | 5/24 | 0.131 sigma | 0.921--1.122 | 2/24 | 0.112 sigma | 0.921--1.082 |
| TA 0.99 | 4/24 | 0.112 sigma | 0.914--1.146 | 5/24 | 0.110 sigma | 0.909--1.086 |

There is no large depth or asymmetry displacement, but neither candidate
passes the literal all-row gate against this one-chain reference.  A
multi-seed pooled gate is still needed before enabling the backend by default.

## Sing prior

This was a small additive change.  Harmonica already consumes direct quadratic
coefficients, while the pipeline already builds per-channel means and sigmas
in `(l,delta)`.  The builder now samples `limb_l` and `limb_delta` with the same
physical truncations as the jaxoplanet quadratic path, deterministically maps

`u1 = 1 - l - 4 delta`, `u2 = 4 delta`,

and passes those to the unchanged Harmonica kernel.  The gray-offset
calibration/control flow and white-light model are otherwise unchanged.
Unit-level flow is tested, but no production GPU Sing-vs-fixed Harmonica
science comparison was run in this package; the prior changes the science
model and must be validated separately.

## Reproduction

CPU tests:

```bash
JAX_PLATFORMS=cpu /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  -m pytest tests/test_harmonica_acceleration.py tests/test_sing_ld.py \
  tests/test_harmonica_half_area.py -x -q
```

The dispatcher ran these preserved scripts (do not invoke Slurm):

```bash
bash acceleration_reports/gpu_queue/done/270_harmonica_accel_dump.sh
bash acceleration_reports/gpu_queue/done/271_harmonica_accel_benchmark.sh
```

Dump:

`/scratch/midway3/tfairnington/accel_gpu_results/270_harmonica_accel_dump/inputs/WASP-94_NIRISS_SOSS_order1_R20_low_resolution_inputs.pkl`

Machine-readable results:

`/scratch/midway3/tfairnington/accel_gpu_results/271_harmonica_accel_benchmark/{joint,laplace_ta095,laplace_ta099}.{timing,diagnostics,arviz,comparison}.json`

## Failures and risks

- The dump config did not reuse the copied white-light handoff because its
  provenance fingerprint changed, so it recomputed white light.  That control
  itself failed the current strict white-light quality gate after extension
  (24 divergences, 4,000 retained draws); this does not contaminate the
  same-input spectroscopic sampler comparison, but it is a warning about the
  current Harmonica white-light baseline.
- The strict single-seed posterior gate failed marginally for both Laplace
  settings; no default was changed.
- Only fixed quadratic LD, `delta_r`, linear trend, no spectroscopic jitter,
  and width 4 were measured.  Sing/free LD, half-area `q`, higher harmonic
  order, explinear trends, and fitted jitter increase dimension or introduce
  truncation and need separate measurements.
- The 4.46x wall speedup is measured and real, but does not meet 10x.  At this
  cadence/width, compilation is 43% of TA0.95 wall; persistent compilation
  reuse may improve repeat runs, but no unmeasured warm-cache speedup is
  claimed.

## Addendum (2026-09-02 15:30): multi-seed pooled gate — PASS

Queue script `272_harmonica_pooled_gate.sh` ran two more joint-NUTS seeds
(`--seed 1`, `--seed 2`, 1000/1000, same R20 dump, same V100 allocation) and
pooled them with the seed-0 run from 271 into a 3-seed reference
(`/scratch/midway3/tfairnington/accel_gpu_results/272_harmonica_pooled_gate/references/`,
3000 draws x 24 channels). `tools/reference_noise_floor.py` calibrated gates
from the seed-to-seed scatter (`harmonica_pooled_noise_floor.txt`):
median-shift limit 0.160σ depth/rors, 0.179σ trends, 0.142σ other
(`delta_r`, `a1`); sigma-ratio intervals [0.85, 1.18] / [0.86, 1.17] / [0.82, 1.22].
The exact joint sampler's own seed-to-seed depth shift reaches 0.152σ, which
is why the flat 0.1σ gate above rejected it.

Against the pooled reference, over all 24 channels x {depths, delta_r, c, v}:

| Candidate | max shift depth | max shift delta_r | max shift c / v | σ-ratio range | failed rows | weighted depth offset (ppm) |
|---|---:|---:|---:|---:|---:|---:|
| joint seed 1 (control, in pool) | 0.065 | 0.075 | 0.090 / 0.096 | 0.93–1.06 | 0 | −0.08 |
| joint seed 2 (control, in pool) | 0.070 | 0.072 | 0.086 / 0.064 | 0.94–1.06 | 0 | +0.11 |
| Laplace NUTS TA 0.95 | 0.100 | 0.106 | 0.105 / 0.121 | 0.93–1.10 | 0 | −0.00 |
| Laplace NUTS TA 0.99 | 0.075 | 0.104 | 0.087 / 0.074 | 0.91–1.10 | 0 | +0.08 |

Both Laplace candidates pass every calibrated row (0/96 failures each) with
sub-0.1 ppm weighted depth offsets. Controls are members of the pool and so
sit slightly closer than an independent chain would; the Laplace runs are
independent of the pool. Conclusion: Laplace-metric independent NUTS at
TA 0.95 (4.46x, zero divergences) is fidelity-safe for this Harmonica
configuration. Unmeasured parameterizations (truncated `q`, higher-order
odd coefficients, per-channel jitter) still need their own check before use.
