# Real white-light Laplace-metric NUTS

## Outcome

I implemented an opt-in dense inverse-Hessian metric for the real jaxoplanet
white-light fit.  The posterior model, likelihood, priors, masks, transit
physics, retained draw count, and posterior-median geometry handoff are
unchanged.  Harmonica does not enter the new branch.

The real-data conclusion differs materially from the synthetic benchmark.
SOSS and G395H pass stringent pooled-posterior comparisons, but the gain is
target dependent: compile-inclusive sampling is about 1.2x faster for SOSS
and 3.1x faster for G395H.  PRISM at the requested target acceptance of 0.9
has 7--93 divergences per seed.  Therefore **do not make Laplace the default**;
keep `whitelight_mass_matrix: adaptive` until PRISM has a divergence-free
configuration.

## Files

- `fit_jwst.py`: flag resolution, seed override, white-light-only stage,
  Laplace NUTS integration, MAP/timing diagnostics, unchanged geometry
  handoff.
- `models/independent_nuts.py`: reusable `prepare_laplace_metric` for a single
  arbitrary NumPyro posterior.
- `tests/test_whitelight_geometry_handoff.py`: default/override resolver test
  and a real small NUTS sample-layout test including deterministic sites.
- `tools/diag_whitelight/run_real_whitelight.py`: isolated real-pipeline
  white-light driver.
- `tools/diag_whitelight/compare_pooled.py`: draw-level pooled comparison and
  seed-null calculation.
- `acceleration_reports/diag_whitelight/{soss_pooled,gj3470_pooled}.csv`:
  full parameter comparisons.
- `acceleration_reports/gpu_queue/{done,running,held}/wl_*.sh`: exact GPU
  commands and logs.

## Configuration and implementation details

The new defaults are deliberately non-disruptive:

| flag | default |
|---|---:|
| `whitelight_mass_matrix` | `adaptive` |
| `whitelight_laplace_warmup` | 200 |
| `whitelight_laplace_target_accept` | 0.9 |
| `whitelight_laplace_max_tree_depth` | 10 |
| `whitelight_laplace_trust_radius` | 5 |
| `whitelight_laplace_hessian_method` | `finite_difference` |

For `laplace`, the existing optimizer solution initializes NumPyro, then the
shared JAX-only trust-region solver polishes the MAP in unconstrained
coordinates.  The Hessian is a symmetrized central difference of the exact
gradient, repaired spectrally, inverted, and passed as NumPyro's dense
`inverse_mass_matrix` with `adapt_mass_matrix=False`.  `MCMC.run` receives the
unconstrained MAP directly.  Only dual averaging remains active in warmup.

White-light coordinates span more curvature than a spectroscopic lane.  The
initial 1e-8 relative eigenvalue floor clipped genuine SOSS modes at condition
number 1e8 and forced every draw to depth 10.  A 1e-12 floor preserved the
positive measured minimum eigenvalue and was required for a usable metric.
This change is local to white light; spectroscopic defaults are untouched.

`flags.random_seed` and `FIT_JWST_SEED` now override the prior fixed seed 555.
`analysis_stage: whitelight` returns only after writing the same geometry
handoff used by the spectrum.

## CPU verification

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 0-15 \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest \
  tests/test_whitelight_geometry_handoff.py \
  tests/test_independent_nuts.py tests/test_independent_hmc.py -x -q
```

Result: **20 passed**, two dependency deprecation warnings, 84.29 s.

## Real V100 measurements

Each production reference used adaptive dense NUTS with 1000 warmup and 1000
retained draws.  Each candidate used 200 warmup and 1000 retained draws.
Times below are the instrumented `MCMC.run` wall; candidate preparation is
listed separately and includes JAX compilation plus MAP/Hessian execution.
JAX does not expose a reliable compile-only timer here, so I do not pretend
the preparation number is pure compilation.

| target/method | seeds | preparation (s) | MCMC wall (s) | mean steps | divergences |
|---|---:|---:|---:|---:|---:|
| SOSS adaptive | 3 | 0 | 171.1 | 78.2 | 0 |
| SOSS Laplace | 2 valid | 59.8 | 81.5 | 93.5 | 0 |
| G395H adaptive | 3 | 0 | 270.6 | 175.1 | 0 |
| G395H Laplace | 3 | 60.1 | 26.5 | 6.9 | 0 |
| PRISM adaptive | 2 complete, 1 running | 0 | 1104.0 | 104.8 | 0 |
| PRISM Laplace, TA=0.9 | 3 | 50.9 | 37.9 | 14.9 | **121** |
| PRISM Laplace, TA=0.99 diagnostic | 1 | 51.8 | 48.2 | 22.2 | 0 |

The SOSS means exclude the deliberately retained failed diagnostic seed with
the overly aggressive 1e-8 floor.  For the corrected seeds, median steps were
63; G395H median steps were 7.  PRISM median steps were 7--15.

Compile/preparation-inclusive ratios using identical 1000 returned draws are
`171.1 / (59.8 + 81.5) = 1.21x` for SOSS and
`270.6 / (60.1 + 26.5) = 3.12x` for G395H.  Sampling-only ratios are 2.10x
and 10.23x.  These runs used Tesla V100-PCIE-16GB devices; allocation markers
are retained beside every queue log.  A paired same-allocation timing script
is retained under `gpu_queue/held` because the replacement queue became
single-GPU while the long PRISM reference was running.

Science-site minimum/median bulk ESS per 1000-draw seed and minimum ESS per
compile-inclusive second were:

| target/method | min ESS | median ESS | min ESS/s |
|---|---:|---:|---:|
| SOSS adaptive | 400--502 | 541--624 | 2.39--2.87 |
| SOSS Laplace | 764--860 | 966--1028 | 5.63--5.87 |
| G395H adaptive | 459--666 | 688--789 | 1.75--2.42 |
| G395H Laplace | 1685--1950 | 1937--2149 | 19.3--22.7 |

SOSS `log_jitter/error` remains the known flat exception (bulk ESS 25--37),
while every listed science site has high ESS.  PRISM TA=0.9 has bulk ESS about
90 for `_b`, 99 for `log_tau`, and 157 for derived `b`, in addition to its
divergences.

## Posterior fidelity and geometry handoff

The reference is the three-seed pooled adaptive posterior.  Candidate values
pool the two corrected stored SOSS/G395H traces; the third timing seed was
created before trace persistence was enabled, although its diagnostics agree.
`seed null` is the range of per-seed medians divided by pooled reference sigma.

### HAT-P-12 SOSS, stellar-informed LD

| site | shift (reference sigma) | sigma ratio | candidate seed null |
|---|---:|---:|---:|
| `t0_0` | -0.051 | 1.006 | 0.080 |
| `b_0` | +0.006 | 0.959 | 0.024 |
| `logD_0` | -0.043 | 1.009 | 0.059 |
| `rors_0` | -0.024 | 0.960 | 0.049 |
| `c` | -0.057 | 1.029 | 0.001 |
| `v` | +0.027 | 1.028 | 0.079 |
| `c1` | -0.050 | 0.953 | 0.068 |
| `c2` | +0.013 | 0.956 | 0.026 |
| `log_jitter` | -0.010 | 1.022 | 0.029 |
| `spot_amp` | +0.016 | 0.989 | 0.025 |
| `spot_mu` | +0.059 | 0.988 | 0.045 |
| `spot_sigma` | +0.019 | 0.984 | 0.045 |

Maximum absolute shift over every stored site is 0.059 sigma; all sigma ratios
are 0.953--1.029.  The b handoff shift is 0.006 sigma, or **0.12 ppm** under
the requested 20 ppm per sigma-b proxy.

### GJ-3470 G395H NRS1

| site | shift (reference sigma) | sigma ratio | candidate seed null |
|---|---:|---:|---:|
| `t0_0` | +0.078 | 1.039 | 0.063 |
| `b_0` | +0.019 | 0.991 | 0.003 |
| `logD_0` | +0.058 | 0.996 | 0.010 |
| `rors_0` | +0.022 | 0.968 | 0.053 |
| `c` | -0.002 | 1.027 | 0.035 |
| `v` | +0.011 | 1.022 | 0.024 |
| `c1` | -0.053 | 0.954 | 0.008 |
| `c2` | -0.049 | 0.995 | 0.010 |
| `log_jitter` | +0.023 | 1.030 | 0.051 |

Maximum absolute shift is 0.078 sigma; all sigma ratios are 0.954--1.041.
The b handoff shift implies **0.37 ppm** by the proxy.

### HAT-P-65 PRISM

The 40,785-cadence white-light likelihood fits on the V100 without OOM.  The
16-iteration MAP polish converged in 10--16 iterations to decrement
`0.8e-4--2.0e-4`, gradient norm 0.09--0.48, and condition number about
`2.9e8`.  However the three TA=0.9 chains had 7, 93, and 21 divergences.  This
fails the exact-MCMC gate regardless of posterior shifts.  A one-seed
diagnostic at target acceptance 0.99 had zero divergences, mean 22.2 steps,
48.2 s sampling, and 51.8 s preparation.  Against the first two adaptive
seeds, all reported science-site median shifts are at most 0.078 sigma and
sigma ratios are 0.934--1.132.  (The signed latent `_b_0` is bimodal by
construction and is not a geometry handoff site.)  Its minimum science-site
ESS is 148 (`log_tau`), or 1.48 ESS/s including preparation, compared with the
adaptive references' roughly 18-minute sampling wall.  This is promising but
one seed is not enough to validate TA=0.99 as a new default.

| site | TA=0.99 shift (reference sigma) | sigma ratio |
|---|---:|---:|
| `t0_0` | +0.001 | 1.064 |
| `b_0` | +0.069 | 0.948 |
| `logD_0` | +0.029 | 0.934 |
| `rors_0` | +0.078 | 0.958 |
| `c` | +0.006 | 1.086 |
| `v` | +0.006 | 1.079 |
| `log_jitter` | -0.078 | 1.076 |
| `A` | +0.006 | 1.050 |
| `log_tau` | -0.037 | 1.132 |

The TA=0.99 `b` handoff shift is 0.069 sigma, corresponding to 1.39 ppm by
the requested proxy.  `c1` and `c2` are fixed in this PRISM configuration.

## Exact reproduction

Representative direct command (the queue scripts enumerate all seeds):

```bash
python tools/diag_whitelight/run_real_whitelight.py \
  --config configs_accel/HAT-P-12_soss_order1_stellarinformed_accel_dump.yaml \
  --output-dir wl_validation/soss_laplace_seed556 \
  --seed 556 --mass-matrix laplace --warmup 200 --samples 1000
```

Pooled comparison:

```bash
JAX_PLATFORMS=cpu /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  tools/diag_whitelight/compare_pooled.py \
  --reference '/scratch/midway3/tfairnington/wl_validation/soss_adaptive_seed*/whitelight_trace_1planets.nc' \
  --candidate '/scratch/midway3/tfairnington/wl_validation/soss_laplace_seed*/whitelight_trace_1planets.nc' \
  --output acceleration_reports/diag_whitelight/soss_pooled.csv
```

GPU execution used the file queue only; no worker Slurm command was issued.

## Failures and open risks

- The first real SOSS metric used the spectroscopic 1e-8 relative eigenvalue
  floor.  It saturated 1023 steps/draw and is excluded from speed/fidelity
  claims.  This directly motivated the white-light-local 1e-12 floor.
- Real SOSS does not reproduce the synthetic 312-to-14 step result.  Its
  corrected candidate uses median 63 and mean 82--105 steps, very close to
  adaptive's median 63; most first-run gain is reduced warmup and better ESS.
- PRISM TA=0.9 is divergent and must not be enabled in production.
- Preparation is about 51--60 s and is not amortized across targets.  It
  includes compilation and MAP work; compile-only time was not separately
  observable without perturbing the production execution.
- The white-light pipeline has one chain per process; validation pools
  independent seeded processes.  Per-file ArviZ `r_hat` is therefore
  undefined, and the reported seed-null is the relevant cross-run check.
- The exact spectrum-level sensitivity to b was not recomputed; ppm values
  use the explicitly requested 20 ppm/sigma-b proxy.

## Recommendation

Keep the default **adaptive**.  The opt-in Laplace path is scientifically
validated for SOSS and G395H and is useful especially for G395H, but a global
default requires zero-divergence PRISM validation.  If enabled selectively,
use the implemented 200 warmup, full depth-10 NUTS, finite-difference Hessian,
and target acceptance 0.9 for SOSS/G395H; do not use it for PRISM at TA=0.9.
