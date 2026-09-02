# Spectroscopic light-curve fitting acceleration: orchestrator summary

Date: 2026-09-01. All timings are Tesla V100-PCIE-16GB, float64, 1000 returned
draws per channel, identical dumped production inputs, same GPU for baseline
and candidate. Worker reports with every number, command, and artifact path:
`harness.md`, `gpu_bench.md`, `reference.md`, `diagnosis.md`, `laplace_is.md`,
`precond.md`, `precond2.md`, `laplace_is_v2.md` (this directory).

## Headline

| Stage (real data) | Production joint NUTS | Best validated candidate | Full-stage speedup | Cached-chunk speedup |
|---|---:|---|---:|---:|
| HAT-P-12 SOSS order 1, stellar-informed LD, 118 channels | 623.8 s | Laplace-IS (Newton MAP + PSIS + IMH), no fallback | **7.9x** (79.0 s) | **70-140x** (1.4 s and 3.0 s per chunk vs ~200 s) |
| same | 623.8 s | FD-Laplace HMC-8 ±25% jitter | **8.0x** (77.8 s) | **72x** (3.4 s) |
| same | 623.8 s | FD-Laplace NUTS, depth 5 | 5.5x (112.9 s) | 21x (11.4 s) |
| GJ-3470 G395H NRS1, 74 channels | 544.7 s | FD-Laplace HMC-8 ±25% jitter | **6.2x** (87.5 s) | **28x** (9.9 s) |
| same | 544.7 s | Laplace-IS + fallback (0 fallbacks) | 3.1x first chunk (76.9 s vs 238 s) | ~31x derived |
| HAT-P-65 PRISM low-res, 21 channels | 1718.5 s (1000/1000) | FD-Laplace NUTS depth 5, trust radius 5 (see addendum) | **7.0x** (246 s) | ~9x (181 s post-compile) |

In every validated row the science sites (rors/depth, c, v, c1, c2, A_spot)
agree with the production posterior within the calibrated gates derived from
seed-to-seed scatter of exact NUTS (0.133 sigma depth, 0.164 trends, 0.202 LD;
see `reference.md`), with one marginal miss each (c at 0.169 vs 0.164 for HMC;
c at 0.228 vs 0.164 for Laplace-IS resample, where the exact-sampler prior
change alone already moves c by 0.194).

## What the fundamental bottleneck was, quantitatively

1. **Posterior geometry, not the forward model.** In NumPyro's unconstrained
   coordinates each channel posterior has Hessian condition number 1e6-1e7
   (SOSS median 9.6e6, G395H 1.7e6, PRISM 1.3e6) with correlations c1-c2 =
   -0.98, c-v = -0.80 to -0.94, c-rors = +0.57. A diagonal mass matrix cannot
   remove correlations, so production joint NUTS runs at tree depth 7 (127
   leapfrogs per draw, 254k gradient calls per 40-channel chunk). One 40-lane
   SOSS value+gradient costs 0.39 ms on V100 and the per-leapfrog cost
   including NUTS tree logic is about 0.86 ms; that is the 195 s per chunk.
2. **Fixing the metric per lane removes ~90 % of that work.** With each lane's
   inverse Hessian as a fixed dense mass matrix, lane-mean leapfrogs drop to
   8-9 (NUTS) or exactly 8 (jittered HMC); with HMC there is no lane
   synchronisation waste and no tree overhead, so a cached 40-channel chunk
   samples in 3.4 s (SOSS) / 9.9 s (G395H).
3. **A near-Gaussian posterior can skip gradient MCMC entirely.** After a
   converged Newton MAP, a Student-t proposal passes Pareto-smoothed
   importance sampling with k-hat median 0.31, max 0.57 on stellar-informed
   SOSS (all 40 lanes), IS-ESS median 4567 of 8192; cached chunks then cost
   1-3 s including exact independence-Metropolis output.
4. **The remaining wall is one-time XLA compilation: 55-80 s per lane width**
   (about 330 compile events through the transit kernel inside the
   MAP/Hessian program), versus 22 s for the plain joint runner. Finite-
   difference Hessians (accurate to 1e-5 relative) and fusing the programs
   did not reduce it. The pipeline compiles once per lane width (low-res and
   high-res widths differ), so a full fit pays it about twice. This, plus the
   white-light stage (4-9 min, unchanged in the pipeline, 27x faster in the
   CPU diagnostic with the same Laplace preconditioning), bounds the
   end-to-end fit speedup at roughly 5-10x today even though the samplers
   themselves are 30-140x faster per chunk.

## Two things that were necessary and are justified

- **Log-normal jitter prior** (`spectro_jitter_prior: lognormal`,
  log_jitter ~ Normal(log(0.5 * median yerr), 2)). The historical
  Uniform(log 1e-6, 0) prior left a flat plateau to the prior floor in 19/40
  real SOSS channels (skew 3.9, kurtosis 22). With exact joint NUTS on the same
  data, every science site moved by at most 0.13 sigma (sigma ratios
  0.93-1.10) while only log_jitter/total_error changed in channels where the
  jitter is not identified, i.e. the science posterior is likelihood-driven.
  On stellar-informed SOSS the prior alone also halves the NUTS tree (127 to
  63 steps, 198 s to 149 s).
- **Stellar-informed LD priors matter.** The older wide-Gaussian LD SOSS
  configuration produces a c1-c2 ridge (correlation -0.977, single-digit ESS)
  that breaks every Laplace-based method (34/40 Laplace-IS fallbacks, 26 NUTS
  divergences). The fiducial `ld_prior: stellarprior` configurations do not
  have this problem and are where all the headline numbers were measured.

## What did not work / open items

- **PRISM (resolved in the addendum below; kept for the record).** The Newton MAP did not converge on the PRISM low-res dump
  (gradient norm 4e4 after 16-32 iterations; the Laplace-IS worker needed
  ~173 iterations and 7/21 lanes still stalled). The dump's geometry came
  from a 20-draw white-light bridge, so its start point is poor. Both fast
  paths therefore fail fidelity on PRISM today. Needed: production white-light
  geometry in the dump plus a robust MAP (L-BFGS warm start, then Newton).
  The forward-only arithmetic (1.2 ms per 21-lane evaluation) says a
  converged Laplace-IS PRISM stage would take ~15-30 s plus compile, versus
  ~80 min for production 1000/1000.
- **Compile time** (see 4 above) is the single item standing between the
  measured 6-8x full-stage results and the 10x target; the lever is
  restructuring the MAP/Hessian program (e.g. Hessian outside `scan`,
  fewer traced transit sub-programs) or persisting compiled executables
  across runs.
- Fixed-length HMC without trajectory jitter shows resonances (near-unit ESS
  in some sites); jitter of ±25 % is required. 16 and 32 steps are worse than
  8.
- Laplace-IS IMH output has acceptance ~0.5, so 1000 IMH draws carry fewer
  effective samples than 1000 NUTS draws; the `resample` output mode (from
  ~4500 effective importance draws) is more precise but is an approximate
  (not exact-MCMC) representation.

## How to use it

Opt-in flags (all defaults unchanged; existing tests pass, 61/61 in the
touched suites):

```yaml
flags:
  spectro_jitter_prior: lognormal
  spectro_sampler: independent_hmc          # or independent_nuts / laplace_is
  spectro_mass_matrix: laplace
  spectro_laplace_hessian_method: finite_difference
  spectro_laplace_fd_relative_step: 0.0002
  spectro_laplace_warmup: 150
  spectro_laplace_target_accept: 0.85       # 0.95 for independent_nuts
  spectro_laplace_fuse_program: true
  spectro_hmc_num_steps: 8
  spectro_hmc_trajectory_jitter: 0.25
  # for independent_nuts add: spectro_laplace_max_tree_depth: 5
```

Offline replay of any stage on any backend:
`tools/run_sampler_on_stage_inputs.py <dump> --backend ... --builder-override
jitter_prior=lognormal --nuts-override mass_matrix=laplace ... --compare
<reference.pkl>`; dumps in `/scratch/midway3/tfairnington/accel_stage_inputs*/`,
pooled 3-seed references in `.../references/`.

A pipeline bug was also fixed on the way (harness worker): post-fit
white-light diagnostics on PRISM used pre-mask phase offsets after outlier
clipping and crashed on a shape mismatch.

## Addendum (evening of 2026-09-01)

### Default changed
`spectro_jitter_prior` now defaults to `lognormal`; builder keyword arguments
(including the prior) are part of the chunk-checkpoint fingerprint. Evidence:
exact joint NUTS under old vs new prior moves every science site by at most
0.13 sigma (SOSS, both LD priors) and 0.10 sigma (G395H), sigma ratios
0.90-1.10; only log_jitter/total_error change, in unidentified channels.

### PRISM (see `prism.md`)
The PRISM failure was the MAP optimizer's unit trust radius, not the metric.
With `spectro_laplace_trust_radius: 5`, decrement stopping and a 200-iteration
cap (early exit when all lanes converge), FD-Laplace NUTS (depth 5, target
0.99, MAP start) returned 1000 draws for the 21-channel low-res stage in
246 s (65 s compile) vs 1718 s for production joint NUTS at 1000/1000 on the
same V100: **7.0x**, zero divergences, science medians within 0.25 sigma;
one caveat, the depth-5 cap under-estimates the `A` width by 37 % in one
channel (depth 6 is the likely production setting). The 106-channel
high-res stage took 18.0 min (409 + 335 + 336 s) vs an estimated ~4 h.

### Campaign pilot against the saved STELLARINFORMED posteriors (`campaign.md`)
On HAT-P-12 SOSS order 1 (118 channels) all three candidates disagreed with
the saved run's posterior on depth by a *coherent* -0.23 sigma (95 % of
channels the same sign). A control run of the **production sampler with the
old prior on the same regenerated inputs** showed the same disagreement
(377/1062 rows failing, worst depth 0.61 sigma), so the offset is between the
two pipeline runs' inputs (white-light geometry realisation / pipeline
revision), not the samplers. Against that same-input control:

| Candidate | Science-site pass fraction | Depth median shift median / max |
|---|---:|---:|
| Laplace NUTS depth 5 | 99.2 % | 0.02 / 0.12 sigma |
| Laplace HMC-8 ±25 % | 99.8 % | 0.02 / 0.11 sigma |
| Laplace-IS + fallback (IMH output) | 87.9 % | 0.07 / 0.38 sigma (lower ESS per 1000 IMH draws, unbiased widths) |

Implication for the 70-dataset validation: the reference must be a
same-input production run (or the saved run's white-light geometry must be
injected), otherwise the comparison measures white-light realisation drift.

### Campaign pilot tooling and cost (`campaign.md`)
`tools/campaign/run_dataset.sh <config> --candidates A,B,C --saved-root ...`
regenerates the production white-light geometry, dumps both stages, runs the
candidates, and gates against the saved posteriors; `tools/campaign/summarize.py`
aggregates. Pilot results: G395H (WASP-52) 97.3 % of science coordinates
pass against the *saved* run for both Laplace NUTS and HMC; SOSS 80 % against
the saved run but 99+ % against the same-input control (see above); PRISM
native resolution (368 channels) needs chunk size 4 on a 16 GB V100 and ran
at ~53 s per chunk. Projected cost for all 70 datasets: ~21 GPU-hours
(~11 h on two V100s). Note the campaign worker's SOSS verdict ("not
validated") predates the same-input control run and is superseded by it.

### Spectrum-level bias test (ppm), same inputs, candidate vs production sampler
Weighted mean depth offset across channels and linear slope with wavelength,
with the Monte-Carlo error of the median estimates (ESS 300 assumed, conservative):

| Comparison | Channels | Mean offset (ppm) | Slope (ppm/µm) | rms per-channel median diff |
|---|---:|---:|---:|---:|
| SOSS 118 ch: Laplace NUTS d5 vs production joint | 118 | +0.0 ± 0.8 | −0.9 ± 1.7 | 5 ppm (σ_ch 88) |
| SOSS: Laplace HMC-8 vs production joint | 118 | +0.2 ± 0.8 | +0.5 ± 1.7 | 4 ppm |
| SOSS: Laplace-IS + fallback vs production joint | 118 | −0.2 ± 0.8 | −2.5 ± 1.7 | 12 ppm |
| G395H 40 ch: Laplace NUTS vs pooled 3-seed production | 40 | −0.3 ± 0.7 | +2.3 ± 6.3 | 2 ppm (σ_ch 46) |
| G395H: Laplace HMC-8 (FD) vs pooled production | 40 | −0.3 ± 0.7 | −0.9 ± 6.3 | 1 ppm |
| G395H: production seed vs seed (null) | 40 | −0.3 ± 0.7 | −0.4 ± 6.2 | 2 ppm |
| **SOSS: fresh production run vs SAVED production run** | 118 | **−21.7 ± 0.8** | **+13.5 ± 1.7** | 20 ppm |

The samplers introduce no detectable offset (< 1 ppm) or slope (< 2 ppm/µm)
relative to the production sampler on identical inputs. The one large
coherent effect found today is between two *production* runs of the same
dataset (white-light geometry handoff / pipeline revision): −22 ppm level and
+13 ppm/µm slope. That is a property of the fixed-geometry pipeline, not of
the accelerated samplers, and it is the effect a population re-run will
mostly measure unless same-input controls are used.

### Laplace-IS v3 (2026-09-02, `laplace_is_v3.md`)
Changes: IMH thinning 8 (8000-step chain per 1000 draws), fallback = Laplace-metric NUTS,
trust radius 5 with early exit, wide-Gaussian/uniform-LD chunks auto-routed to Laplace NUTS,
gate unchanged (k-hat < 0.7, acceptance >= 0.2). Same-input results, V100, 1000 draws:

| Stage | Science-site pass (calibrated) | Mean offset | Slope | rms per-channel | ESS min / median | Fallback lanes | Wall (compile) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SOSS 118 ch vs production control | 816/826 (98.8 %) | +0.2 ppm | −0.5 ppm/µm | 5.6 ppm | 22 / 916 | 3 | 231 s (205 s) |
| G395H 40 ch vs pooled reference | 240/240 | +0.1 ppm | +0.1 ppm/µm | 2.3 ppm | 600 / 945 | 0 | 72 s (63 s) |
| PRISM low 21 ch vs 1000/1000 joint | 174/189 (92 %) | −0.4 ppm | −0.4 ppm/µm | 3.0 ppm | 66 / 799 | 1 | 171 s (107 s) |

Status: ppm-clean everywhere; distribution-shape gates now pass at the NUTS/HMC
level on SOSS and G395H, PRISM trend/spot widths still miss in some channels.
Open engineering: the fallback runner recompiles per chunk (SOSS compile 205 s
because fallback lane counts differ per chunk; pad the fallback to a fixed
width), and a few lanes retain low ESS. Recommendation unchanged: Laplace
NUTS/HMC for fidelity-critical runs; Laplace-IS acceptable for G395H-like
informed-LD chunks.

## Overnight addendum (2026-09-02, 01:30-09:30) — see OVERNIGHT_MANIFEST.md for the log

Rules: no file was deleted, moved, or overwritten in place; all new outputs
are under new paths (`/scratch/midway3/tfairnington/accel_parity/`,
`accel_campaign_saved_refs/`, `configs_parity/`, `configs_sing/`,
`configs_speed/`, `_SPEED_*`/`_SING_*`/`_PARITY_*` output directories).

### Defaults changed tonight
- `whitelight_geometry_estimator: posterior_median` (was max_likelihood_draw;
  the draw estimator produced the −22 ppm / +13 ppm/µm coherent offset).
- `spectro_jitter_prior: lognormal` (earlier in the day).

### Parity against the saved STELLARINFORMED spectra (`parity.md`)
Full pipeline end to end (production adaptive white light 1000/1000, median
handoff, accelerated spectroscopic samplers A = Laplace NUTS, B = Laplace
HMC-8, C = Laplace-IS v3), compared to the saved posteriors with the
calibrated gates and the ppm offset/slope test. 20 datasets, 60 candidate
rows: 49 PASS, 3 CONTROL-PASS (HAT-P-30: the production sampler itself
disagrees with the saved run identically -> input difference between
pipeline revisions), 1 INPUT-DIFF (that control), 6 FLAG, all on the three
`explinear`-trend datasets (WASP-63 SOSS, HAT-P-11 G395H V1/V2): negligible
bias (|offset| < 0.8 ppm) but 89-94 % gate pass and divergences at target
0.95; the A99 variant (target 0.99, depth 10) removes the divergences but
gate pass stays ~94 %, so the production control D on WASP-63 decides
whether it is an input difference (pending at 08:00). Typical passing
dataset: |offset| < 3.5 ppm, |slope| < 2.3 ppm/µm, rms 1-11 ppm per channel,
error-bar ratio 0.99-1.01, 96-100 % of calibrated gates. End-to-end walls
8-17 min per candidate for SOSS/G395H (production: 47 min to 5.7 h).

### White light (`whitelight.md`, `speed_p2.md`)
`whitelight_mass_matrix: laplace` (opt-in; FD Hessian, curvature floor 1e-12,
200 step-size-only warmup): SOSS/G395H pass pooled 3-seed gates (largest
shift 0.08 sigma, ~0.4 ppm on the spectrum level), PRISM passes at target
0.99 (0 divergences in 3 seeds, 48-69 s sampling + 52 s prep vs ~1100 s).
Gains: SOSS 1.2x incl. prep (2x sampling), G395H 3.1x (10x sampling), PRISM
~10x. Recommended per-mode target: SOSS/G395H 0.9, PRISM 0.99; global
default left adaptive pending a wider validation.
New gates: white-light ESS/divergence gate with automatic chain extension
and adaptive fallback; per-channel depth-ESS gate with selective
Laplace-NUTS fallback (`spectro_min_depth_ess`, default 400).

### Speed engineering (`speed_p2.md`)
- Compile box (cadence padding to 256, likelihood mask, persistent JAX
  cache): potential/gradient identical to 1e-12, but only 1.3x warm-process
  gain and the padded program changes the MCMC realisation (8-15 ppm max
  channel differences, MC-level, not bias) -> opt-in, not recommended.
- Laplace-IS fallback padded to a fixed width (compile once per width).
- Full-pipeline PRISM (HAT-P-65, 106 high-res channels): 3367 s whole
  process (WL 238 s, low 719 s, high 2393 s) vs ~5-6 h production; two-GPU
  `chunk_mode: parallel` split of the high-res stage measured overnight
  (equality vs serial: see speed_p2.md final section).

### Sing (2026) LD prior (`sing_prior.md`, opt-in `ld_prior: sing`)
Implemented with the paper's (l, delta) parameterisation, mu_min 0.2,
tabulated Stagger offsets, an automatic gray-offset calibration stage
(free quadratic LD on the R20 channels, gated on ESS/divergences), and
Laplace-IS compatibility (tests pass). On HAT-P-12 SOSS the calibration fit
failed its own gate (ESS < 10) so the tabulated offsets were used; the
resulting spectrum differs from the power-2 stellar-informed one by +86 ppm
mean and −23 ppm/µm slope (apples-to-oranges: quadratic vs power-2 law and
prior). Left for later testing with a longer calibration fit.

Correction (08:30): the WASP-63 production control also fails the saved
gate at 93.97 % (input difference between pipeline revisions, as for
HAT-P-30); A/B/C match that control to 0.2-0.8 ppm, so WASP-63 is
CONTROL-PASS. The only unresolved flags are the two HAT-P-11 G395H
explinear visits, whose parity runs hit the 90-minute queue limit.
