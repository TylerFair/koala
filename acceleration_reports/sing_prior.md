# Sing et al. (2026) quadratic limb-darkening prior

## Status

An opt-in first-pass implementation is in place. No defaults changed and no
Slurm/GPU job was submitted. The real-data demonstration is **not run** because
the shared queue already contained pending work; consequently there are no
honest wall-time, spectrum-parity, divergence, LD-posterior, or Pareto-k numbers
to report yet.

The current implementation supports a tabulated offset and consumption of a
previously fitted offset JSON. It does **not yet automatically turn the free
white-light fit into that JSON**; without `flags.ld_sing_offset_path`, the
requested default `fit` mode prints a warning and uses the documented tabulated
fallback. This missing orchestration is the main incomplete item.

## What was built

| File | Change |
|---|---|
| `models/sing_ld.py` | Exact coefficient transforms, tabulated constants, inverse-variance gray-offset estimator, exclusive-create fingerprinted JSON writer. |
| `models/jaxoplanet/builder.py` | New `ld_mode='sing'`; samples `(limb_l, limb_delta)` and saves deterministic `c1`, `c2`, and `u`. Coupled support enforces `l in [0,1]`, `c1>=0`, `c1+2c2>=0`, and `c1+c2<=1`. |
| `fit_jwst.py` | Resolves `ld_prior: sing`, requires quadratic LD, forwards `ld_mu_min` (default 0.2), constructs channel-varying Sing centers/scales, handles fitted-artifact/tabulated fallback, and initializes the new latent sites. |
| `tests/test_sing_ld.py` | Transform identity, synthetic offset recovery, and vectorized-model physical-support trace tests. |
| `configs_sing/HAT-P-12_soss_order1_sing_candidate_A.yaml` | New candidate-A config (independent NUTS, Laplace metric, FD Hessian, 150 warmup, target 0.95, depth 6). |
| `configs_sing/HAT-P-12_soss_order1_sing_laplace_is.yaml` | New Laplace-IS config. |
| `acceleration_reports/sing_gpu_scripts/*.sbatch` | Two new scripts for orchestrator submission; neither was submitted. |

`models/laplace_is.py` was not modified. Its existing dynamic channel-kwarg and
deterministic postprocessing machinery accepts `mu_u_ld`/`sigma_u_ld`; the new
latent names are ordinary scalar-per-lane sites.

## Mathematical implementation

The transformations are

    u+ = c1 + c2; u- = c1 - c2
    l = 1 - u+; delta = (u+ - u-)/8
    c1 = 1 - l - 4 delta; c2 = 4 delta

The prior center is model `(l,delta)` plus either the fitted artifact or
`(+0.020,-0.003)`. Per-channel scales are `(0.031,0.016)`. `l` is a truncated
normal on `[0,1]`; conditional on `l`, delta is truncated to
`[-(1-l)/4, +(1-l)/4]`. These bounds impose the usual physically admissible
quadratic profile over mu in `[0,1]`.

## CPU measurements

| Command / input | Result | Wall |
|---|---:|---:|
| `pytest tests/test_sing_ld.py -x -q` | 3 passed | 2.36 s |

Exact reproduction:

```bash
cd /project/ekempton/tfairnington/JWST
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  -m pytest tests/test_sing_ld.py -x -q

JAX_PLATFORMS=cpu \
  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  -m py_compile fit_jwst.py models/jaxoplanet/builder.py models/sing_ld.py
```

The broader `test_laplace_is.py` and `test_independent_nuts.py` run was started
and exited without a failure diagnostic, but the captured launcher output did
not include a pytest summary; it is therefore not counted as a passed test run.

## GPU reproduction (orchestrator only)

Submit, one at a time, the new scripts:

```bash
sbatch acceleration_reports/sing_gpu_scripts/run_hatp12_sing_candidate_A.sbatch
sbatch acceleration_reports/sing_gpu_scripts/run_hatp12_sing_laplace_is.sbatch
```

The scripts write only to new `_SING_...` output directories. Before running a
fit-calibrated prior, produce a new JSON with `estimate_gray_offset` and
`write_offset_artifact`, then add its path as `flags.ld_sing_offset_path` to a
new config (do not replace tonight's configs in place).

## What failed / remains incomplete

- No real HAT-P-12 run was made, so no comparison to STELLARINFORMED exists.
- Automatic extraction of gray offsets from the free coarse/white posterior is
  not wired into the main pipeline. Current `fit` behavior falls back loudly to
  the population values unless an artifact path is supplied.
- The requested explicit tiny independent-NUTS and Laplace-IS tests using the
  new prior remain to be added. The vectorized model trace passes, and the
  unchanged backend suites produced no recorded failure, but that is not an
  end-to-end proof.
- Only the jaxoplanet vectorized builder implements `sing`; harmonica does not.
- The gray-offset estimator propagates marginal c1/c2 errors without covariance.
  A production artifact should use posterior draws directly so covariance is
  retained.
- No same-input speedup was measured; this feature makes no speed claim.

## Note for later

Once the parity checks pass, test fitted-offset versus tabulated-offset runs on
the same GPU and identical HAT-P-12 inputs. Record spectrum mean offset (ppm),
wavelength slope, per-channel scatter, radius uncertainty ratios, LD posterior
coverage, divergences, ESS, Pareto-k/IS fallback rate, compilation time, and
sampling wall. Then repeat on at least one long-cadence target and one hotter
FGK star before treating the population scatter as portable.

The saved STELLARINFORMED reference uses the **power-2** law, whereas Sing et al.
recommend and this feature implements the **quadratic** law. Any spectrum
difference is therefore apples-to-oranges: it mixes prior calibration with a
change of limb-darkening family and must not be described as sampler bias or a
fidelity speedup result.

# Second pass

## Completed implementation

`fit_jwst.py` now runs a distinct gray-offset calibration after the white-light
fit and before the normal low-resolution Sing fit when `ld_prior: sing` and
`ld_sing_offset: fit` (the default) are selected. It uses the existing R20
channels, a free/uniform quadratic-LD model, the configured accelerated backend,
and new calibration controls:

- `ld_sing_calibration_warmup` (default 150)
- `ld_sing_calibration_samples` (default 300)
- `ld_sing_calibration_min_ess` (default 100)

Laplace-IS configurations are automatically routed through the existing
finite-difference Laplace-metric independent-NUTS fallback for this free-LD
calibration, because the production Laplace-IS guard rejects broad uniform LD.
The calibration is accepted only if every c1/c2 bulk ESS meets the threshold,
all samples are finite, every chunk diagnostics file is present, and there are
zero divergences. Accepted offsets are written with exclusive-create semantics
to a fingerprinted `*_sing_gray_offset_<hash>.json`; the path is then consumed
by both the R20 and Rreference Sing priors. A failed gate prints the full reason
and uses the tabulated Stagger offsets.

The estimator's delta uncertainty propagation was also corrected to use
`sigma(c2)/4`, since `delta=c2/4` exactly. No change was made to
`models/laplace_is.py`.

New second-pass files:

- `configs_sing/HAT-P-12_soss_order1_sing_fitoffset_candidate_A_v2.yaml`
- `configs_sing/HAT-P-12_soss_order1_sing_fitoffset_laplace_is_v2.yaml`
- queue records `220_sing_hatp12_candidate_A_v2.*` and
  `221_sing_hatp12_laplace_is_v2.*`

## CPU tests

| Test | Recorded result |
|---|---:|
| `tests/test_sing_ld.py` | 5 passed in 60.30 s |
| Sing independent NUTS with FD Laplace metric alone | 1 passed in 48.37 s |
| Sing Laplace-IS alone | 1 passed in 27.29 s |
| `tests/test_laplace_is.py` | 10 passed, 2 warnings in 87.41 s |
| `tests/test_independent_nuts.py` | 11 passed, 2 warnings in 86.28 s |

Reproduce with:

```bash
cd /project/ekempton/tfairnington/JWST
taskset -c 0-15 env JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  -m pytest tests/test_sing_ld.py -q

taskset -c 0-15 env JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  -m pytest tests/test_laplace_is.py -q

taskset -c 0-15 env JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  -m pytest tests/test_independent_nuts.py -q
```

## HAT-P-12 SOSS order-1 GPU demonstrations

Both jobs were submitted only through the file queue and exited 0. They ran on
different Tesla V100 allocations, so the wall times below must not be used as a
backend speed comparison.

| Run | GPU marker | Whole-process wall | Production diagnostics |
|---|---|---:|---|
| candidate A, independent NUTS + Laplace metric | Tesla V100, job 57220760 | 469 s | 0/118 divergent lanes/draw events; mean 8.94 leapfrog steps/channel/draw; maximum 63 |
| Laplace-IS | Tesla V100, job 57224760 | 535 s | 116/118 IS gates passed; 2 fallback lanes; median/max k-hat 0.341/0.800; 2 lanes above 0.7 |

Additional Laplace-IS diagnostics: median/min IMH acceptance was 0.523/0.369;
median/min raw IS ESS was 2182/835. Candidate A white-light sampling wall was
134.18 s with 0 divergences. Laplace-IS required an adaptive white-light
extension to 2000 retained draws; its final white-light sampling wall was
136.84 s with 0 divergences.

Exact file-queue commands were embodied in the preserved scripts:

```bash
bash acceleration_reports/gpu_queue/done/220_sing_hatp12_candidate_A_v2.sh
bash acceleration_reports/gpu_queue/done/221_sing_hatp12_laplace_is_v2.sh
```

## Gray-offset calibration result

Both real-data free-LD calibrations failed their own diagnostics, and therefore
the production fits correctly used the tabulated `(delta_l, delta_delta) =
(+0.020, -0.003)` values. No accepted fingerprinted offset JSON was written.

| Run | Rejected fitted delta_l | Rejected fitted delta_delta | min bulk ESS | divergences | action |
|---|---:|---:|---:|---:|---|
| candidate A | +0.04527 +/- 0.00693 | -0.03377 +/- 0.00144 | 6.65 | 2 | loud tabulated fallback |
| Laplace-IS config | +0.04572 +/- 0.00668 | -0.03419 +/- 0.00138 | 9.70 | 2 | loud tabulated fallback |

The fitted values in this table were calculated post hoc from the preserved
rejected checkpoints for diagnosis only. They were not allowed to inform either
production spectrum. The failures show that 150 warmup + 300 draws is not enough
for this very broad free quadratic calibration on these channels.

## Limb-darkening posteriors

These summaries cover 118 Rreference channels. Here `c1/u1` and `c2/u2` are the
same saved deterministic quadratic coefficients.

| Run | median c1 (channel range) | median c1 uncertainty | median c2 (channel range) | median c2 uncertainty |
|---|---:|---:|---:|---:|
| candidate A | 0.17957 (0.06839--0.46343) | 0.03494 | 0.13685 (0.04014--0.21740) | 0.04627 |
| Laplace-IS | 0.17971 (0.06757--0.46533) | 0.03571 | 0.13699 (0.04014--0.21803) | 0.04604 |

## Spectrum comparison to saved STELLARINFORMED

This is explicitly **apples-to-oranges**: the reference is stellar-informed
power-2 LD, while these runs use a tabulated-offset quadratic Sing prior. The
118 wavelength centers matched exactly. Differences below are Sing minus the
saved STELLARINFORMED depth.

| Run | mean offset (ppm) | linear slope (ppm/micron) | raw channel scatter (ppm) | scatter after mean+slope removal (ppm) | max absolute difference (ppm) |
|---|---:|---:|---:|---:|---:|
| candidate A | +86.43 | -22.60 | 40.94 | 38.55 | 183.67 |
| Laplace-IS | +88.15 | -25.33 | 42.47 | 39.56 | 187.68 |

These are measured differences, not a sampler speed/fidelity claim. The two
Sing runs agree closely with one another, but both inherit the same tabulated
fallback and differ systematically from the power-2 reference.

## Remaining risk / next test

The automatic stage and fallback are now exercised end to end, but the fitted
offset branch has not yet passed on real data. A new config (new output path)
should increase calibration warmup/draws and/or use more conservative NUTS
controls until zero divergences and minimum bulk ESS >=100 are achieved. Only
then should the accepted JSON path and truly fitted-offset spectrum be compared
against this tabulated-fallback result. The power-2-versus-quadratic caveat from
the first pass remains fundamental.

# Third pass

## Physical calibration parameterization

The calibration no longer samples broad independent `c1,c2`. The new
`ld_mode='sing_free'` uses exactly the paper coordinates and a broad physical
uniform prior:

```text
l ~ Uniform(0, 1)
delta | l ~ Uniform(-(1-l)/4, +(1-l)/4)
c1 = 1 - l - 4 delta                 [deterministic]
c2 = 4 delta                         [deterministic]
```

The conditional delta band enforces `c1 >= 0`, `c1 + 2*c2 >= 0`, and
`c1+c2 <= 1`. The calibration gate is evaluated directly on all 24 channels of
both sampled physical sites (`limb_l`, `limb_delta`): every sample must be
finite, minimum bulk ESS must be at least 100, all chunk diagnostic files must
exist, and total divergences must be zero.

For v3 the physical calibration uses finite-difference Laplace-metric
independent NUTS with 1000 warmup, 1000 retained draws, target acceptance 0.99,
and maximum tree depth 10. The normal stage-two candidate remains target 0.95,
depth 6, and 150 Laplace warmup.

## Exact stage-two prior

For Stagger quadratic coefficients computed after discarding model intensities
at `mu < 0.2`, define for channel `j`:

```text
l_model,j     = 1 - (c1_model,j + c2_model,j)
delta_model,j = c2_model,j / 4

l_center,j     = l_model,j     + Delta_l_fit
delta_center,j = delta_model,j + Delta_delta_fit

l_j ~ TruncatedNormal(l_center,j, 0.031; low=0, high=1)
delta_j | l_j ~ TruncatedNormal(delta_center,j, 0.016;
                                low=-(1-l_j)/4,
                                high=+(1-l_j)/4)

c1_j = 1 - l_j - 4 delta_j             [saved deterministic]
c2_j = 4 delta_j                        [saved deterministic]
u_j  = (c1_j, c2_j)                     [saved deterministic]
```

Thus the untruncated target densities are Gaussians in the paper's `(l,delta)`
coordinates, centered on the `mu_min=0.2` model plus the fitted gray offsets,
with exactly the tabulated Stagger scatters `(0.031,0.016)`. Truncation only
enforces the physical support. The high-resolution log confirms consumption of
the newly written fitted-offset JSON rather than the tabulated fallback.

## Tests

The added physical-prior trace test checks the support band and confirms `c1`
and `c2` are deterministic sites. The complete Sing-specific suite result was:

```text
6 passed in 79.41s
```

Reproduce:

```bash
cd /project/ekempton/tfairnington/JWST
taskset -c 0-15 env JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  -m pytest tests/test_sing_ld.py -x -q
```

## GPU run and queue outcome

The first immutable v3 attempt used
`configs_sing/HAT-P-12_soss_order1_sing_fittedoffset_candidate_A_v3.yaml`
and `_SING_v3`. It obtained a GPU immediately but exited 143 because allocation
57220760 reached its Slurm time limit during white-light warmup (draw 279/2000),
before the gray calibration began. Its partial directory was left untouched.

The retry used the new config
`configs_sing/HAT-P-12_soss_order1_sing_fittedoffset_candidate_A_v3_retry1.yaml`
and new output directory `HAT-P-12_SOSS_ORDER1_SING_v3_RETRY1`. It obtained a
fresh Tesla V100 within 20 seconds, exited 0, and completed in 618 s. The GPU
marker is allocation 57396794. White-light sampling took 164.99 s and required
one adaptive extension to 2000 retained draws. The calibration checkpoint
interval was approximately 91.6 s from manifest creation to completed
diagnostics; this is a filesystem-timestamp proxy rather than an internally
instrumented wall.

Preserved queue script:

```bash
bash acceleration_reports/gpu_queue/done/261_sing_hatp12_fittedoffset_v3_retry1.sh
```

## Accepted fitted offsets and calibration diagnostics

The v3 physical calibration passed and wrote the exclusive-create artifact
`HAT-P-12_NIRISS_SOSS_order1_R20_sing_gray_offset_582a966234fe.json`.

| Quantity | Fitted v3 | Tabulated Stagger | Difference |
|---|---:|---:|---:|
| Delta_l | +0.052593 +/- 0.010092 | +0.020000 +/- 0.031 | +0.032593 |
| Delta_delta | -0.037065 +/- 0.002127 | -0.003000 +/- 0.016 | -0.034065 |

Minimum bulk ESS across all `l/delta` channel sites was **168.71**, above the
100 threshold. There were **0 divergences**, no missing diagnostics, and all
posterior values were finite. Both low- and high-resolution stage-two fits
therefore used these fitted values; no fallback occurred.

## Stage-two LD posterior

The 118-channel Rreference posterior summaries are:

| Coefficient | Median | Channel range | Median posterior uncertainty |
|---|---:|---:|---:|
| c1 | 0.24681 | 0.14038 to 0.50875 | 0.03402 |
| c2 | 0.03700 | -0.03679 to 0.13318 | 0.04570 |

Production candidate-A sampling had **0 divergences** across 118 channels,
mean 9.19 leapfrog steps per channel/draw, and maximum 63 steps.

## Spectrum comparisons

All comparisons use the same 118 wavelength centers. The STELLARINFORMED
comparison is explicitly **apples-to-oranges**, because that saved reference
uses power-2 LD whereas v3 uses quadratic LD with fitted Sing offsets.

| Difference (first minus second) | Mean ppm | Slope ppm/micron | Raw channel scatter ppm | Scatter after mean+slope removal ppm | Max absolute ppm |
|---|---:|---:|---:|---:|---:|
| v3 fitted-offset minus STELLARINFORMED power-2 | +108.60 | -31.75 | 43.08 | 38.47 | 206.78 |
| v3 fitted-offset minus second-pass tabulated-offset candidate A | +22.17 | -9.15 | 8.10 | 5.87 | 32.92 |

For context, the second-pass tabulated-offset candidate-A difference from
STELLARINFORMED was +86.43 ppm mean, -22.60 ppm/micron slope, and 40.94 ppm raw
channel scatter. The fitted offset therefore produces a measurable additional
shift relative to the tabulated population correction. No speedup claim is
made: the v3 and second-pass runs include different calibration workloads, and
the STELLARINFORMED comparison changes the LD law as well as the prior.

## 2026-09-02 addendum: calibration coordinates and ESS-aware pooling

Section 3.3 of [Sing et al. (2026), arXiv:2609.00263](https://arxiv.org/abs/2609.00263)
says the `(l, delta)` reparameterization “is not designed to be used as the
variables fit in a transit light curve model.” The paper instead advocates
fitting `(u+, u-)` or `(c1, c2)` with sufficiently wide uninformative priors and
transforming afterward. The first implementation missed that distinction: its
free calibration sampled `l ~ Uniform(0,1)` and conditionally sampled
`delta` inside `+/-(1-l)/4`. That interval collapses for weak limb darkening as
`l` approaches one.

The free calibration now samples independent `u+ ~ Uniform(-1,2)` and
`u- ~ Uniform(-2,2)`. This broad rectangular support follows the paper's
uninformative-prior recommendation and, unlike a physically truncated wedge,
has no channel-dependent collapsing boundary. The model saves `l=1-u+`,
`delta=(u+-u-)/8`, `c1=(u++u-)/2`, and `c2=(u+-u-)/2` as deterministic sites.
The affine `(l,delta)` prior and its physical truncation in the second-stage fit
are unchanged.

Calibration pooling is also ESS-aware. For each channel and each transformed
coefficient, its posterior uncertainty is multiplied by
`max(1, sqrt(N/ESS_bulk))` before inverse-variance weighting. A bulk ESS below
100 now emits a warning and is recorded in the artifact; it no longer discards
an otherwise finite, zero-divergence calibration. Missing/non-finite diagnostics,
non-finite pooled results, or any divergence remain hard failures. Artifacts now
store both per-channel ESS vectors and the Table 3 population offsets and
scatters for comparison.

HAT-P-18 G395M queue 420 validated the change on 11 R20 calibration channels.
All calibration ESS values exceeded 100: `l` ESS was
`[397.90, 328.59, 335.59, 364.42, 328.53, 370.97, 321.85, 268.57, 300.78, 356.75, 343.69]`;
`delta` ESS was
`[402.17, 324.62, 325.12, 319.75, 275.26, 424.17, 396.19, 235.89, 300.22, 348.99, 346.01]`.
There were zero calibration divergences. The fitted values were
`Delta_l=+0.007245 +/- 0.017447` and
`Delta_delta=+0.003486 +/- 0.003655`, compared with Table 3 values
`+0.020 +/- 0.031` and `-0.003 +/- 0.016`. Each difference is smaller than the
corresponding star-to-star scatter. The stage-input-to-artifact timestamp
interval was approximately 76 seconds; this is a filesystem proxy, while the
full pipeline wall was 982 seconds.

## 2026-09-03 completion note

Queue 420 subsequently reached a terminal exit code of zero. The reference
stage explicitly logged the fitted artifact as its offset source and completed
with all 208 lanes passing the production gate: 200 used independent NUTS and
8 used independent HMC after swap, with zero accepted divergences and minimum
accepted depth ESS 718.07. The fitted values and calibration ESS table above
are therefore final rather than pending. The updated HAT-P-18 model stack also
replaces the pre-default uniform run with queue 421's wide-`u+`/`u-` posterior;
full numerical consequences are recorded in `stacking/stacking.md`.
