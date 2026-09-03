# Production defaults and limb-darkening reparameterization

Date: 2026-09-02 (CDT)

## Outcome

The production defaults are now Laplace-metric independent NUTS for
spectroscopy, Laplace-metric NUTS with adaptive fallback for white light, and
fixed white-light-median timescales for explinear spectroscopy. Spectroscopic
lanes are gated at depth ESS >= 400 and zero divergences. Failed Laplace-NUTS
lanes run exact HMC-8 (25% trajectory jitter, target 0.85), then exact legacy
adaptive joint NUTS. An explicit HMC selection reverses the first two steps.

The decorrelated LD parameterization is implemented as an opt-in exact
change of variables. It did **not** work with the tested finite-difference
Laplace preparation, so it is **not** a default. Wide-Gaussian and uniform LD
continue to default to `coefficients`.

## Files

| File | Change |
|---|---|
| `fit_jwst.py` | New production defaults, mode-dependent targets/depths, two-stage selective sampler swap, ESS/divergence gate, checkpoint schema v4, per-channel sampler provenance, CSV provenance. |
| `models/ld_parameterization.py` | New Maxted power-2 bijection with an analytic Jacobian. |
| `models/jaxoplanet/builder.py` | Opt-in exact transformed priors for white-light and spectroscopic free/wide/uniform LD; physical deterministic sites retained. |
| `tests/test_spectro_safety_guards.py` | Directional swap-order and real tiny-model selective-swap test. |
| `tests/test_ld_parameterization.py` | Exact induced-prior and posterior-MC parity tests. |
| `tests/test_whitelight_geometry_handoff.py` | Updated Laplace white-light default assertion. |
| `SPECTRO_ACCELERATION.md` | Documented defaults, swap behavior, fixed timescale, and LD flags. |
| `configs_defaults/*defaults*.yaml` | New no-sampler-flag SOSS/G395H smoke configs and isolated retries. |
| `acceleration_reports/gpu_queue/{done,pending}/28*.sh` | Full-pipeline default smoke scripts. |
| `acceleration_reports/gpu_queue/done/29*.sh` | Three wide-LD stage benchmark scripts. |

No existing data or output was deleted or renamed.

## Defaults and exactness

The no-flag defaults are:

```yaml
spectro_sampler: independent_nuts
spectro_mass_matrix: laplace
spectro_laplace_warmup: 150
spectro_laplace_hessian_method: finite_difference
spectro_laplace_trust_radius: 5
spectro_laplace_target_accept: 0.95   # 0.99 PRISM/explinear
spectro_laplace_max_tree_depth: 5    # 6 PRISM
spectro_min_depth_ess: 400
spectro_max_divergences: 0
spectro_hmc_num_steps: 8
spectro_hmc_trajectory_jitter: 0.25
whitelight_mass_matrix: laplace
whitelight_laplace_target_accept: 0.9 # 0.99 PRISM
whitelight_min_ess: 400
whitelight_max_divergences: 0
whitelight_max_extra_blocks: 3
whitelight_geometry_estimator: posterior_median
spectro_fixed_timescale_trends: true
spectro_ld_parameterization: coefficients
whitelight_ld_parameterization: coefficients
```

All sampler choices in the swap chain are exact MCMC kernels. Checkpoint
payloads store `sampler_used` per channel, diagnostics store the complete gate
attempt list, and final transmission CSVs contain a `sampler_used` column.
The checkpoint schema/target revision changed, and ESS/divergence thresholds,
sampler controls, builder arguments, and fixed-timescale model arguments enter
the fingerprint. Old checkpoints therefore do not silently match.

## CPU verification

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest \
  tests/test_independent_nuts.py tests/test_independent_hmc.py \
  tests/test_spectro_safety_guards.py tests/test_explinear_spectroscopic.py \
  tests/test_whitelight_geometry_handoff.py tests/test_ld_parameterization.py \
  -x -q
```

Result: **36 passed**, 3 dependency warnings, 130.69 s.

After the final fingerprint edits, an additional regression run of
`test_mcmc_runner_reuse.py`, `test_whitelight_geometry_handoff.py`,
`test_spectro_safety_guards.py`, and `test_ld_parameterization.py` passed
**26/26** in 29.02 s (the same 3 dependency warnings).

Real wide-LD one-lane replay:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  tools/run_sampler_on_stage_inputs.py \
  /scratch/midway3/tfairnington/accel_stage_inputs/HAT-P-12_NIRISS_SOSS_order1_Rreference_high_resolution_inputs.pkl \
  --backend independent_nuts --start 0 --end 1 --warmup 5 --samples 5 \
  --platform cpu --seed 293 --output-prefix /tmp/ld_decor_cpu_smoke \
  --builder-override ld_parameterization=decorrelated \
  --nuts-override mass_matrix=adaptive --nuts-override max_tree_depth=2
```

Result: exit 0, 29.762 s. The 20,000-draw induced-prior test recovers the
original physical samples to maximum roundoff of 3.15e-13. The two small
posterior parameterizations agree within the declared 8%/0.02 MC tolerance.

## Full-pipeline default smokes

Both configs contain no sampler or mass-matrix flags. The clean SOSS retry 283
finished with dispatcher exit 0. The clean G395H retry 282 printed `Analysis
complete!`, wrote all expected products and the final wall marker, but the
dispatcher recorded exit 127 after the script completed. The same late exit
occurred for 281; this is reported as a queue/wrapper failure, not hidden as a
clean process exit.

| Dataset | Queue | Pipeline wall | Initial gate | Final gate | sampler swaps | offset ppm | slope ppm/um | error ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HAT-P-12 SOSS order 1 | 283 | 793 s | 118/118 | 118/118 | 0 | -0.646 | +1.369 | 1.0012 |
| WASP-52 G395H NRS1 | 282 | 666 s | 67/68 | 68/68 | 1 to HMC-8 | -2.185 | +2.742 | 0.9893 |

Spectrum metrics are candidate minus the immutable files in
`/scratch/midway3/tfairnington/accel_parity_staged/`. The offset is inverse-
combined-variance weighted; the slope is a weighted linear fit; the error
ratio is the median candidate/reference depth standard-deviation ratio.

Reproduction scripts are preserved as queue records 280--283. Queue 280 was
accidentally recreated after its first dispatcher had moved it, causing a
second dispatcher to run the same path concurrently; its artifacts are not
used above. Queue 281 used `/usr/bin/time` and completed the analysis in
602.39 s but recorded exit 127. New output roots were used for both retries.

## Wide-LD GPU benchmark

Input: the existing HAT-P-12 SOSS order-1 wide-Gaussian high-resolution dump,
channels 0:40, 1,000 returned draws. Scripts 290--292 all exited 0.

| Sampler / coordinates | Wall | compile | divergences | depth ESS med/min | c1 ESS med/min | c2 ESS med/min | Hessian cond. med/max |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive joint NUTS / coefficients | 431.7 s | 22.3 s | 0 | 1467/900 | 703/310 | 688/302 | n/a |
| Laplace NUTS / coefficients | 120.4 s | 84.8 s | 4 | 883/512 | 77.3/9.75 | 71.1/8.34 | 9.60e6/1.44e7 |
| Laplace NUTS / decorrelated | 118.3 s | 81.7 s | 11,988 | 4.74/1.17 | 1.81/1.17 | 2.10/1.17 | 2.64e4/6.95e4 |

Against the new joint-NUTS reference, coefficient-basis Laplace NUTS has
median/max absolute posterior shifts of 0.036/0.142 sigma for depth,
0.087/0.407 for c1, and 0.101/0.469 for c2; its median width ratios are
0.999, 0.929, and 0.974. It nevertheless fails the zero-divergence and LD-ESS
gates. The decorrelated run is unusable: MAP gradients were about 1.29e7 at
the median, all lanes reached 200 MAP iterations, its median step size was
1.87e-7, and its posterior ESS collapsed. The smaller reported Hessian
condition number therefore does not indicate successful geometry.

Exact GPU commands are the contents of preserved scripts
`290_ld_joint_coeff.sh`, `291_ld_laplace_coeff.sh`, and
`292_ld_laplace_decorrelated.sh` in `acceleration_reports/gpu_queue/done/`.

## Failures and open risks

- The exact transformed prior is correct, but the current Newton/FD Laplace
  preparation is not robust in those coordinates. `decorrelated` remains
  opt-in; no speed or convergence claim is made.
- A bounded-triangle quadratic reparameterisation was CPU prior-tested, but the real GPU
  benchmark exercised power-2/Maxted only.
- G395H completed twice but the dispatcher recorded exit 127 after each shell
  script had produced its final marker. The products and gate results exist,
  but this unexplained dispatcher status remains an operational risk.
- Queue 280 is excluded because two executions shared its output directory.
- The smoke comparison is against saved production spectra, so it includes
  white-light realization/pipeline-version differences as documented in
  `parity.md`; it is not a same-input sampler-only comparison.

## Follow-up: wide-LD MAP failure and linear basis (2026-09-02 CDT)

### Diagnosis

The immediate failure was more specific than an ill-conditioned Maxted
Jacobian. Production/stage initialization contains physical `c1,c2`, while the
independent sampler previously did not derive `ld_decorrelated` from them. The
omitted latent therefore inherited the random model-trace value. For channel 0
one reproduced trace value was `h=(-0.5933,-0.3439)`, outside the Maxted image;
the potential and gradient were both NaN. The intended physical start
`(c,alpha)=(0.6713195,0.6616653)` maps to
`h=(0.7530539,0.4243734)` and has a finite initial potential gradient.

The boundary singularity is nevertheless real. Holding channel 0's `h1`
fixed and following the problematic direction toward `h2=0` gave:

| h2 | induced log-Jacobian | potential gradient norm |
|---:|---:|---:|
| 0.4244 | -1.224 | 6.84e2 |
| 0.0100 | -4.972 | 1.39e4 |
| 0.0010 | -7.274 | 2.61e5 |
| 0.0003 | -8.478 | 1.08e6 |
| 0.0001 | -9.577 | 3.82e6 |

Thus a random or wandering MAP point near the transformed boundary explains
the earlier approximately `1e7` gradients. After deriving the transformed
initial value from physical coefficients, the channel-0 CPU Laplace MAP took
6 iterations, ended at gradient norm `9.25e-4`, and had zero divergences in a
two-draw smoke run.

### Changes

| File | Follow-up change |
|---|---|
| `models/independent_nuts.py` | Derive transformed LD initialization exactly from staged `c1,c2`, instead of using a random trace value. |
| `models/ld_parameterization.py` | Add exact constant-Jacobian `(s,d)=(c1+c2,c1-c2)` transform. |
| `models/jaxoplanet/builder.py` | Add internal opt-in `decorrelated_linear` builder selection for the ordered experiment. |
| `tests/test_ld_parameterization.py` | Verify linear round trip and constant `log|J|=log(2)`. |
| `acceleration_reports/gpu_queue/done/293_ld_maxted_physical_init.sh` | 40-lane corrected-Maxted benchmark. |
| `acceleration_reports/gpu_queue/done/294_ld_linear.sh` | 40-lane linear-basis benchmark. |

### GPU benchmark and decision

All rows use the queue-290 joint-NUTS coefficient reference, the same HAT-P-12
wide-Gaussian dump, channels 0:40, 150 Laplace warmup steps, and 1,000 draws.

| Coordinates | Wall | compile | divergences | depth ESS med/min | c1 ESS med/min | c2 ESS med/min | Hessian cond. med/max |
|---|---:|---:|---:|---:|---:|---:|---:|
| joint NUTS coefficients (290) | 431.7 s | 22.3 s | 0 | 1467/900 | 703/310 | 688/302 | n/a |
| Maxted, physical mapped init (293) | 112.7 s | 79.9 s | 1 | 1380/860 | 867/406 | 888/397 | 9.63e6/1.45e7 |
| linear sum/difference (294) | 116.7 s | 82.0 s | 69 | 974/675 | 192/77 | 192/87 | 9.61e6/1.44e7 |

Maxted posterior median shifts relative to queue 290 were median/max
0.029/0.118 sigma for depth, 0.043/0.190 for c1, and 0.041/0.190 for c2;
median width ratios were 0.997, 1.027, and 1.041. Linear shifts were
0.033/0.163, 0.055/0.182, and 0.073/0.191 sigma, with median width ratios
0.995, 0.995, and 1.025. Posterior location/width parity is reasonable, but
neither run passes the requested sampler gate: Maxted has one divergence and
minimum c2 ESS 397.1, while linear has 69 divergences and LD ESS far below 400.

Recommendation: retain `coefficients` as the production default for
`widegaussian` and `uniform`. Stellar-informed and Sing modes are unaffected.
No automatic default was changed.

The requested full coefficient-space MAP/Hessian pullback was not completed
within the time box. Diagnosis exposed the invalid random transformed start,
so the first controlled experiment repaired that exact failure and evaluated
the transformed Hessian at the resulting stable MAP. It nearly, but did not,
pass. This remains an explicit open item rather than being reported as the
requested analytic metric transformation.

### Exact commands and verification

The full GPU commands are preserved verbatim in queue scripts 293 and 294.
The CPU channel-0 diagnosis used the required CPU environment and
`initialize_model`/the production potential on the real stage dump. Tests:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest -q \
  tests/test_ld_parameterization.py tests/test_independent_nuts.py
```

Result: **13 passed**, 3 dependency warnings, 91.59 s. Queue 293 and 294 both
exited 0. Queue 293 waited about 75 minutes for a busy V100 step before its
112.6 s benchmark; that queue delay is excluded from the sampler wall above.

## Follow-up: inverse-CDF latent Gaussian LD (2026-09-02 CDT)

### Outcome and exactness

`latent_gaussian` is implemented for the bounded free/wide/uniform power-2
and quadratic priors in both white-light and spectroscopic builders. It is
opt-in and **is not a production default** because the real wide-Gaussian
benchmark failed both the zero-divergence and per-lane LD-ESS gates.
Stellar-informed and Sing priors are unchanged.

For a uniform prior, `U = lo + (hi-lo) Phi(Z)` with `Z ~ N(0,1)` is uniform
because `Phi(Z) ~ Uniform(0,1)`. For a truncated normal, applying its inverse
CDF to the same `Phi(Z)` gives exactly that truncated-normal law. The model
therefore samples only the standard-normal `ld_latent`; the physical `c1,c2`
or `u` values are deterministic outputs. No extra Jacobian factor is required:
the probability-integral transform already defines the pushforward prior.

### Files changed

| File | Change |
|---|---|
| `models/ld_parameterization.py` | Added standard-normal-to-uniform and standard-normal-to-truncated-normal inverse-CDF maps using `ndtr`/`ndtri`. |
| `models/jaxoplanet/builder.py` | Added `latent_gaussian` for free/wide/uniform power-2 and quadratic LD in both builders; physical output sites retained. |
| `fit_jwst.py` | Accepted the new opt-in flag value without changing the `coefficients` default. |
| `tests/test_ld_parameterization.py` | Added 20,000-draw prior-quantile checks and tiny-posterior parity against coefficient sampling. |
| `SPECTRO_ACCELERATION.md` | Documented the opt-in parameterization and failed production gate. |
| queue scripts `294_ld_laplace_latent_gaussian.sh`, `295_ld_uniform_quadratic_latent.sh` | Preserved exact GPU commands and isolated result roots. |

No existing output or data was deleted, renamed, or overwritten. The queue
number 294 was also used by the earlier linear experiment after the terminal
crash; the script names and result roots are distinct, so neither output was
overwritten.

### GPU results

The primary comparison uses the identical HAT-P-12 SOSS wide-Gaussian
high-resolution dump, channels 0:40, 1,000 draws, and queue-290 reference.

| Sampler / coordinates | Wall | compile | divergences | depth ESS med/min | c1 ESS med/min | c2 ESS med/min | Hessian cond. med/max |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive joint NUTS / coefficients (290) | 431.7 s | 22.3 s | 0 | 1467/900 | 703/310 | 688/302 | n/a |
| Laplace NUTS / coefficients (291) | 120.4 s | 84.8 s | 4 | 883/512 | 77.3/9.75 | 71.1/8.34 | 9.60e6/1.44e7 |
| Laplace NUTS / latent Gaussian (294) | 123.6 s | 88.3 s | 38 | 933/233 | 139/4.07 | 134/4.10 | 9.61e6/1.44e7 |

Against queue 290, latent-Gaussian absolute median shifts (median/p95/max,
in reference sigma) were `0.032/0.100/0.108` for depth,
`0.056/0.201/0.674` for c1, and `0.050/0.238/0.577` for c2. Candidate/reference
sigma-ratio ranges (min/median/max) were `0.917/0.994/1.134`,
`0.782/0.960/1.488`, and `0.866/0.985/1.135`, respectively. These outliers
also fail the calibrated parity gates; the failure is not merely an ESS label.

The requested uniform/quadratic coverage used a real five-channel WASP-39
G395H R20 uniform stage dump, rebuilding only its LD profile as quadratic and
its parameterization as latent Gaussian. It completed in 67.9 s (51.5 s
compile), with zero divergences and Hessian condition number median/max
`1.23e6/1.63e6`. Depth ESS was 969/548 median/min, but u1 and u2 ESS were only
155/87.5 and 169/102. This short coverage case therefore also fails the
per-lane LD ESS > 400 rule. It is not presented as posterior parity because
changing the recorded power-2 builder to quadratic intentionally changes the
model and no matched quadratic coefficient reference dump existed.

### Commands, tests, failures, and recommendation

The exact long GPU invocations are the contents of the preserved queue
scripts; both dispatcher exits were 0 and results are under
`/scratch/midway3/tfairnington/accel_gpu_results/294_ld_laplace_latent_gaussian/`
and `/scratch/midway3/tfairnington/accel_gpu_results/295_ld_uniform_quadratic_latent/`.
The CPU verification command was:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest -q \
  tests/test_ld_parameterization.py tests/test_independent_nuts.py \
  tests/test_spectro_safety_guards.py
```

Final result: **24 passed**, 3 dependency warnings, 93.11 s. An earlier run
failed only because the new prior test referenced a local RNG key before it
was defined; that test-code error was fixed and the complete command rerun.

Recommendation: keep `coefficients` as the default for free, wide-Gaussian,
and uniform LD. Keep `latent_gaussian`, `decorrelated`, and
`decorrelated_linear` opt-in only. The inverse-CDF prior is exact, but on this
ridge it did not improve the finite-difference Laplace metric: it produced 38
divergences, LD ESS minima near four, and parity outliers up to 0.674 sigma.

## Default flip to Jacobian-corrected LD coordinates (2026-09-02 CDT)

### Implementation and prior meaning

The prior-dependent default is now `decorrelated` for `widegaussian`,
`uniform`, and the `free` alias in both white-light and spectroscopic model
builders. Power-2 LD uses Maxted h1,h2; quadratic LD uses physical coefficients.
All other prior modes default to `coefficients`; in particular,
stellar-informed and Sing are unchanged. Either parameterization flag still
overrides this resolution explicitly.

This is a change of sampling coordinates only. NumPyro's transformed
distribution evaluates the original density on the physical coefficients and
includes the analytic change-of-variables Jacobian. Physical c1,c2 or u1,u2
remain deterministic output sites. The model-builder keyword is part of
`_chunk_checkpoint_fingerprint`, and the regression test directly verifies
that coefficient and decorrelated builders produce different fingerprints.

### Files changed

| File | Change |
|---|---|
| `fit_jwst.py` | Added prior-dependent LD parameterization resolution for both stages. |
| `tests/test_spectro_safety_guards.py` | Asserted wide/free/uniform defaults, unchanged modes, and explicit override. |
| `tests/test_mcmc_runner_reuse.py` | Asserted the LD coordinate choice changes the checkpoint fingerprint. |
| `SPECTRO_ACCELERATION.md` | Documented the new defaults and exact prior preservation. |
| `docs/guides/limb_darkening.md`, `docs/guides/samplers.md` | Added the requested plain-language sampling-coordinate notes. |
| `tools/campaign/compare_ld_coordinate_runs.py` | Added matched-sample spectrum metrics and the two-panel comparison plot. |
| `configs_defaults/WASP-39_nrs1_g395h_uniform_quadratic_dump342.yaml`, `...dump347.yaml`, `...dump348.yaml` | Added immutable initial and retry configs for a real uniform-quadratic dump. |
| queue scripts 340--348 | Added immutable GPU invocations; 345/346 became recorded no-ops after the user dropped stellar reruns. |
| `ld_jacobian_vs_legacy_HAT-P-12_SOSS.{png,json}`, `ld_jacobian_vs_legacy_WASP-39_G395H.{png,json}` | Saved the requested visual checks and full machine-readable metrics. |

No existing data or output was deleted, renamed, or overwritten.

### Full-spectrum visual and numerical shape checks

Both comparisons use identical stage inputs and the same seed within each
pair. Depths and error bars are posterior median and posterior standard
deviation. The lower figure panels plot candidate minus legacy depth with the
quadrature sum of the two posterior standard deviations.

| Dataset | Channels | Legacy wall / compile | New-coordinate wall / compile | Legacy / new div. | offset (ppm) | slope (ppm/um) | slope uncertainty (ppm/um) | RMS after offset (ppm) | median-depth MC RMS (ppm) | error ratio p05 / med / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HAT-P-12 SOSS order 1, wide Gaussian | 118 | 604.24 / 55.05 s | 121.89 / 76.92 s | 0 / 75 | -0.470 | -2.236 | 20.023 | 6.616 | 6.858 | 0.918 / 1.002 / 1.083 |
| WASP-39 G395H NRS1, uniform quadratic | 68 | 217.14 / 35.93 s | 83.78 / 56.37 s | 0 / 18,761 | +144.642 | -99.916 | 59.484 | 239.272 | 5.599 | 0.00045 / 0.00662 / 1.157 |

The SOSS spectra are roughly the same: the fitted slope is consistent with
zero, and its 6.62 ppm offset-removed RMS matches the 6.86 ppm Monte Carlo
expectation. The figure is `acceleration_reports/ld_jacobian_vs_legacy_HAT-P-12_SOSS.png`.

The uniform-quadratic G395H raw Laplace run **does change the recovered
spectral shape** relative to the converged legacy run. Its slope is not a
clean zero-shape result and its 239 ppm RMS is about 43 times the 5.60 ppm
Monte Carlo expectation. The visual discrepancy is clear in
`acceleration_reports/ld_jacobian_vs_legacy_WASP-39_G395H.png`. This is not
evidence that the Jacobian changed the posterior target: the candidate has
18,761 divergences, median depth ESS 8.42 (minimum 1.32), and median depth
error ratio 0.0066, so it did not sample that target. MAP preparation also
ended at 200 iterations in many lanes, with gradient norms as large as
4.84e3 and reported condition-number caps of 1e8. The production automatic
sampler gate must therefore reject and swap these lanes; an un-gated raw
Laplace result is unsafe.

For completeness, the SOSS candidate depth ESS was 1266 median / 865 minimum,
c1 was 1024 / 428, and c2 was 1075 / 423. Its 75 divergences likewise mean
the automatic gate will swap affected lanes even though the aggregate shape
check is clean.

### Exact commands

The exact GPU commands and environment are preserved in executable queue
scripts:

```text
acceleration_reports/gpu_queue/done/340_ld_soss_full_joint_coeff.sh
acceleration_reports/gpu_queue/done/341_ld_soss_full_decorrelated.sh
acceleration_reports/gpu_queue/done/342_uniform_quadratic_dump.sh
acceleration_reports/gpu_queue/done/343_ld_uniform_quadratic_full_joint.sh
acceleration_reports/gpu_queue/done/344_ld_uniform_quadratic_full_decorrelated.sh
acceleration_reports/gpu_queue/done/347_uniform_quadratic_dump_retry.sh
acceleration_reports/gpu_queue/done/348_uniform_quadratic_dump_retry2.sh
```

The successful dump was produced by queue 348. The two analysis commands
were:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/campaign/compare_ld_coordinate_runs.py \
/scratch/midway3/tfairnington/accel_gpu_results/340_ld_soss_full_joint_coeff/joint_coeff.npz \
/scratch/midway3/tfairnington/accel_gpu_results/341_ld_soss_full_decorrelated/laplace_decorrelated.npz \
/scratch/midway3/tfairnington/accel_stage_inputs/HAT-P-12_NIRISS_SOSS_order1_Rreference_high_resolution_inputs.pkl \
--gate-json /scratch/midway3/tfairnington/accel_stage_inputs/references/HAT-P-12_NIRISS_SOSS_order1_Rreference_high_resolution_inputs_ch0_40_pooled_joint_nuts_noise_floor.json \
--figure acceleration_reports/ld_jacobian_vs_legacy_HAT-P-12_SOSS.png \
--title 'HAT-P-12 SOSS order 1: LD sampling coordinates' \
--output acceleration_reports/ld_jacobian_vs_legacy_HAT-P-12_SOSS.json

JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/campaign/compare_ld_coordinate_runs.py \
/scratch/midway3/tfairnington/accel_gpu_results/343_ld_uniform_quadratic_full_joint/joint_coeff.npz \
/scratch/midway3/tfairnington/accel_gpu_results/344_ld_uniform_quadratic_full_decorrelated/laplace_decorrelated.npz \
/scratch/midway3/tfairnington/accel_gpu_results/348_uniform_quadratic_dump_retry2/stage_inputs/WASP-39_NIRSPEC_G395H_nrs1_Rreference_high_resolution_inputs.pkl \
--figure acceleration_reports/ld_jacobian_vs_legacy_WASP-39_G395H.png \
--title 'WASP-39 G395H NRS1: uniform quadratic LD coordinates' \
--output acceleration_reports/ld_jacobian_vs_legacy_WASP-39_G395H.json
```

The final verification commands were:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest -q \
tests/test_ld_parameterization.py tests/test_spectro_safety_guards.py \
tests/test_mcmc_runner_reuse.py tests/test_independent_nuts.py

JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m sphinx -W -b html docs docs/_build/html
```

Final results were **40 passed** with three dependency warnings in 130.37 s,
and the warning-as-error HTML documentation build succeeded.

### Failures and open risks

Queue 342 spent 41.1 s in failed white-light Laplace preparation, fell back
to adaptive NUTS, retained 4,000 draws after three extension blocks, and then
failed in the low-resolution automatic HMC swap because
`laplace_fd_batch_size` was passed to independent HMC. Its dispatcher exit was
1. Queue 347 failed immediately because explicit `joint_nuts` was combined
with the then-default `spectro_mass_matrix: laplace`; queue 348 corrected this
to `adaptive`, shortened only the dump-producing fit, and exited 0. The
queue-348 dump is the sole input to both reported G395H 1,000-draw samplers.

The principal open risk is the failed uniform-quadratic raw Laplace shape
check. The requested default flip is implemented as directed, but production
correctness for this case depends on the already-implemented divergence/ESS
gate and exact-MCMC sampler swap. This task did not run a full automatic-swap
uniform-quadratic production spectrum after the scope was narrowed to the
two direct sampler overlays. Stellar-informed reruns were explicitly dropped;
queued scripts 345 and 346 exited 0 after printing `SKIPPED` and ran no fit.

### Quadratic-default correction and production swap (2026-09-02 CDT)

The automatic Jacobian-coordinate default is now restricted to power-2
wide-Gaussian and uniform/free LD. Quadratic LD resolves to `coefficients` for
both white light and spectroscopy; an explicit `decorrelated` flag raises an
error for quadratic LD. Stellar-informed, Sing, and fixed modes are unchanged.
Because the resolved builder parameterization is part of the checkpoint
signature, quadratic wide/free jobs no longer reuse checkpoints made with the
old automatic bounded-triangle quadratic choice.

| File | Change |
|---|---|
| `fit_jwst.py` | Made LD resolution depend on both prior and profile; restored depth 10 for the legacy adaptive fallback. |
| `models/independent_hmc.py` | Carried `laplace_fd_batch_size` through HMC option resolution so selective fallback accepts production NUTS inputs. |
| `models/jaxoplanet/builder.py` | Removed the experimental quadratic coordinate branches; quadratic LD now samples physical coefficients or the inverse-CDF latent alternative. |
| `tools/diagnose_quadratic_map_path.py` | Retired the campaign-only quadratic MAP diagnostic. |
| `tools/run_production_swap_on_stage_inputs.py` | Added a stage-dump replay of the production gate and exact-MCMC swap chain with sampler provenance. |
| `tools/campaign/compare_ld_coordinate_runs.py` | Added an explicit candidate legend label for mixed production samplers. |
| `tests/test_ld_parameterization.py`, `tests/test_spectro_safety_guards.py`, `tests/test_independent_hmc.py` | Added support, profile-resolution, explicit-override, and HMC FD-batching coverage. |
| `SPECTRO_ACCELERATION.md`, `docs/guides/limb_darkening.md`, `docs/guides/samplers.md` | Documented the power-2-only automatic transform and coefficient default for quadratic LD. |
| Quadratic MAP trace artifact | Saved the channel-40 MAP trace during the experiment; the artifact is now cleared. |
| `acceleration_reports/ld_coefficients_production_swap_vs_legacy_WASP-39_G395H_corrected.png` | Legacy-versus-production spectrum and per-channel difference figure. |

#### Why the bounded-triangle quadratic reparameterisation failed here

Channel 40 did not start on an edge: q=(0.04801, 0.18581), corresponding to
u=(0.08142, 0.13768), with gradient norm 457 and Hessian condition number
7.19e5. By iteration 20 the optimizer had reached q=(0.000375, 4.973), whose
inverse is u=(0.1927, -0.1733). It ended at iteration 200 with q=(0.001048,
6.512), u=(0.4216, -0.3893), gradient norm 693, a nearly zero/negative raw
minimum Hessian eigenvalue, and the repaired condition number pinned at 1e8.
Thus the initial point was not the problem, and the 1e8 cap was a downstream
symptom rather than the initiating cause.

The direct cause was a support leak, not evidence that the physical posterior
sat on the physical triangle edge. The custom transform advertised a real-vector
codomain; NumPyro therefore did not apply a bounded support transform, while
the base Uniform log density assumed support had already been enforced. The
MAP could consequently obtain a finite potential after its inverse left
[0,1]^2, then follow an almost flat, nonphysical direction until Hessian repair
hit 1e8. The experimental coordinate implementation and its support path have
now been removed completely. Quadratic uses physical coefficients, with the
inverse-CDF latent option as its only alternative.

#### Full 68-channel production result

| Sampler result | wall (s) | retained lanes: Laplace NUTS / HMC-8 / adaptive | attempted divergences: NUTS / HMC / adaptive | depth ESS min / median | u1 ESS min / median | u2 ESS min / median | offset (ppm) | slope (ppm/um) | RMS after offset / MC RMS (ppm) | error ratio p05 / med / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Legacy joint NUTS, coefficients | 217.14 | 0 / 0 / 68 | 0 | not recomputed | not recomputed | not recomputed | reference | reference | reference | reference |
| Raw Laplace NUTS, bounded-triangle coordinates (rejected) | 83.78 | 68 / 0 / 0 | 18,761 / 0 / 0 | 1.32 / 8.42 | unusable | unusable | +144.642 | -99.916 +/- 59.484 | 239.272 / 5.599 | 0.00045 / 0.00662 / 1.157 |
| Production coefficient gate + swap | 326.44 | 53 / 9 / 6 | 35 / 1 / 0 | 389.94 / 782.23 | 8.32 / 100.48 | 8.67 / 103.82 | -1.538 | +1.686 +/- 59.484 | 6.266 / 6.349 | 0.928 / 0.999 / 1.075 |

The final retained mixed posterior has zero divergences by construction: each
retained lane passed the production zero-divergence/depth-ESS gate. The 389.94
depth ESS quoted above is ArviZ bulk ESS; the production NumPyro gate uses its
own estimator and accepted every retained lane at >=400. Fifteen lanes were
sent to HMC-8; nine passed there, and six continued to adaptive joint NUTS.
The LD coefficients themselves remain slow (minimum bulk ESS about 8), which
is an open mixing limitation of quadratic uniform LD even though the depth
spectrum is fidelity-safe.

The spectrum check passes the requested shape criterion. Its slope is
consistent with zero (1.69 +/- 59.48 ppm/um), and the 6.27 ppm offset-removed
RMS agrees with the 6.35 ppm Monte Carlo expectation. The weighted offset is
-1.54 ppm and median error ratio is 0.999. The visual overlay and difference
panel are in
`acceleration_reports/ld_coefficients_production_swap_vs_legacy_WASP-39_G395H_corrected.png`.

#### Exact commands, failures, and risks

CPU diagnosis and comparison:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/diagnose_quadratic_map_path.py \
/scratch/midway3/tfairnington/accel_gpu_results/348_uniform_quadratic_dump_retry2/stage_inputs/WASP-39_NIRSPEC_G395H_nrs1_Rreference_high_resolution_inputs.pkl \
--channel 40 --iterations 200 --output acceleration_reports/quadratic_map_path_channel40.json

JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/campaign/compare_ld_coordinate_runs.py \
/scratch/midway3/tfairnington/accel_gpu_results/343_ld_uniform_quadratic_full_joint/joint_coeff.npz \
/scratch/midway3/tfairnington/accel_gpu_results/392_uniform_quadratic_production_swap_retry2/production_coefficients_swap.npz \
/scratch/midway3/tfairnington/accel_gpu_results/348_uniform_quadratic_dump_retry2/stage_inputs/WASP-39_NIRSPEC_G395H_nrs1_Rreference_high_resolution_inputs.pkl \
--figure acceleration_reports/ld_coefficients_production_swap_vs_legacy_WASP-39_G395H_corrected.png \
--candidate-label 'Production Laplace NUTS + selective swaps (coefficients)' \
--title 'WASP-39 G395H NRS1: production coefficient sampler swap' \
--output acceleration_reports/ld_coefficients_production_swap_vs_legacy_WASP-39_G395H_corrected.json
```

The exact GPU commands are preserved in queue scripts 390--392. Queue 390
failed after primary NUTS when HMC option resolution omitted
`laplace_fd_batch_size`. Queue 391 fixed that and reached adaptive NUTS, but
failed the gate in two lanes because fallback inherited max tree depth 5.
Queue 392 used the restored legacy depth 10, exited 0, and wrote to the fresh
root `/scratch/midway3/tfairnington/accel_gpu_results/392_uniform_quadratic_production_swap_retry2/`.

The main open risk is that the earlier HAT-P-12 Maxted shape benchmark was run
before the transformed-support leak was identified; prior sampling itself was
correct, and its depth spectrum agreed with legacy, but some retained physical
LD coefficients were outside their declared bounds. The support is now
enforced, so the default is mathematically prior-preserving as documented, but
the corrected-support power-2 full-spectrum performance has not been rerun in
this narrowly scoped follow-up. No GPU jobs remain pending.

Final verification used the mandated CPU environment. The LD/default/HMC
group passed 28 tests in 38.49 s, `tests/test_independent_nuts.py` passed 11
tests in 93.45 s (three dependency warnings in each invocation), and
`python -m sphinx -W -b html docs docs/_build/html` succeeded. Queue exit
codes were 390=1, 391=1, and 392=0; all are terminal and none is pending.

### White-light optimized-start validation and fallback (2026-09-02)

The white-light optimizer result is now validated before it can seed NUTS.
For power-2 and quadratic LD the deterministic physical coefficients must be
finite and inside the prior support; radius ratio must be inside its prior
range; and each planet must satisfy `0 <= b < 1 + rprs`.  A failed check is
discarded, logged with its reasons, and replaced by the already transformed
prior/physical-coordinate start.  This is a start-selection safeguard only;
it does not change the posterior.

The observed regression is covered literally.  The test supplies
`ld_decorrelated=(316.73443542, 0.44949058)`, `b=2`, and `rprs=0.70710678`;
validation rejects the resulting non-finite `c2` and invalid geometry. Queue
395 exercised the same branch on the real data and logged the fallback to
finite `ld_decorrelated=(0.92406627, 0.29518621)`, `b=0.4498`, and
`rprs=0.1457`. Its pre-NUTS directional gradient check then passed at
2.495e-5, proving that the invalid-location failure no longer reaches NUTS.

The spectroscopic path already enforced transformed physical support. During
this work that guard was made safe under NumPyro validation: invalid inverse
points receive a finite `-1e100` log factor (zero probability in float64), and
sanitized coefficients are used only to keep eager likelihood evaluation
finite. Spectroscopic initial values are now constructed in the model's
actual batched latent coordinates. The final adaptive safety net now rebuilds
the coefficient-coordinate model and starts from physical coefficients, so it
is genuinely the legacy exact-MCMC fallback rather than another run against
the difficult transformed boundary.

| File | Change |
|---|---|
| `fit_jwst.py` | Added white-light physical/geometry validation and logged fallback; made power-2 initialization batch-safe; supplied a coefficient-basis model and physical starts to the legacy spectroscopic adaptive fallback. |
| `models/jaxoplanet/builder.py` | Made transformed support rejection NumPyro-safe and prevented non-finite inverse values from entering likelihood construction. |
| `tests/test_ld_initialization.py` | Added the observed bad-value regression, valid-start coverage, and batched spectroscopic transform coverage. |
| `tests/test_ld_parameterization.py` | Updated invalid-support coverage for the finite zero-probability penalty and sanitized inverse. |
| `configs_stacking/wasp39_nrs1_ref_wide_linear_safety395.yaml` through `safety399.yaml` | Added immutable retry configs with fresh output roots. |
| `acceleration_reports/gpu_queue/{done,pending}/395_*.sh` through `399_*.sh` | Preserved the exact GPU launch scripts and terminal records. |

#### Real stacking run and gates

No run produced the requested complete reference-grid spectrum before the
authorized queue range ended. The strongest completed evidence is queue 399:
white light initialized with finite physical values, the complete five-lane
R20 spectrum was written, and its retained sampler mix was 0 Laplace NUTS / 2
HMC-8 / 3 coefficient-basis adaptive NUTS. The three adaptive lanes had zero
divergences and depth ESS 1092.79, 1564.94, and 1177.15; the two HMC lanes had
zero divergences and depth ESS 3486.94 and 1891.19. Thus all five low-resolution
lanes passed the production gate.

White light retained 4000 adaptive-fallback draws with geometry ESS
3776.31/1815.09/3316.17/2121.36 for t0/b/duration/rprs, but had eight
divergences and therefore did not pass its strict zero-divergence gate after
all three extra blocks. As designed, the pipeline continued using the exact
adaptive posterior. The high-resolution 0:40 primary run and HMC retry
finished; 35 lanes then entered valid coefficient-basis adaptive sampling.
Allocation 57474887 expired during that sampling and the dispatcher recorded
exit 143 at 23:53:59 CDT, about 24 minutes after launch. No high-resolution
checkpoint or final spectrum was claimed.

| Queue | exit | outcome |
|---:|---:|---|
| 395 | 1 | White fallback succeeded; model then hit NumPyro validation on an infinite support factor. |
| 396 | 1 | Finite support guard worked; adaptive spectroscopic fallback still started in unrestricted Maxted coordinates and failed its gate. |
| 397 | 1 | Batched latent start worked; R20 adaptive fallback had five divergences because it was not yet the coefficient-basis legacy model. |
| 398 | 1 | R20 spectrum completed; high-resolution coefficient fallback received an unsliced 68-channel start. |
| 399 | 143 | R20 passed all lane gates; high-resolution coefficient fallback began correctly, then the GPU allocation expired. |

Exact CPU verification command:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest -q \
tests/test_ld_initialization.py tests/test_ld_parameterization.py \
tests/test_spectro_safety_guards.py
```

This passed 29 tests in 27.57 s. A subsequent focused rerun of initialization
and safety guards passed 25 tests in 14.84 s. The GPU command in every case was
`python fit_jwst.py --config <new safety config>` under the mandated dispatcher
environment; exact paths and timing wrappers are in queue scripts 395--399.

Open risk: the requested full 68-channel completion and its final wall/gate
table remain unverified because the last authorized job was externally killed.
All queue 395--399 jobs are terminal and none remains pending.

After the final full-channel slicing correction, the exact three-file CPU
command above was rerun and passed 29 tests in 51.56 s (nine warnings).

## 2026-09-03: wide u-plus/u-minus default for quadratic uniform LD

Following Section 3.3 of Sing et al. (2026), quadratic `ld_prior: uniform` now
samples independent `u_plus=u1+u2 ~ Uniform(-1,2)` and
`u_minus=u1-u2 ~ Uniform(-2,2)`. The rectangular box is intentionally wide and
uninformative. It is not the old prior in different coordinates: it admits
quadratic coefficients outside the former independent `u1,u2 in [0,1]` square.
The model emits deterministic `u1`, `u2`, `l=1-u_plus`,
`delta=(u_plus-u_minus)/8`, and joint `u`, preserving downstream quantities.
Set `flags.ld_uniform_basis: coefficients` to recover the exact historical
coefficient prior. New pipeline builder metadata and checkpoint signatures
include `ld_uniform_basis`; pre-change stage dumps with the absent field are
explicitly reconstructed as `coefficients` before potential validation.

`free` remains an alias for the calculated-center wide-Gaussian mode, and
`widegaussian` remains a truncated Gaussian rather than being silently changed
to a flat prior. This isolates the deliberate prior decision to `uniform`.
`ld_prior: sing` remains the offset-corrected informed quadratic mode.

### Same-input, same-seed WASP-39 comparison

Queues 432 and 433 replayed all 68 channels from the queue-348 G395H NRS1
high-resolution dump with its dumped pipeline key and 1,000 retained draws.
Queue 432 used the legacy coefficient box with adaptive joint NUTS. Queue 433
used the wide sum/difference box with the production Laplace-NUTS gate/swap
chain.

| Metric | Legacy coefficients | Wide u-plus/u-minus |
|---|---:|---:|
| wall | 184.62 s | 117.48 s |
| minimum depth bulk ESS | 1033.90 | 792.54 |
| primary-attempt divergences | 0 | 1 |
| accepted lanes | 68 joint NUTS | 67 independent NUTS, 1 independent HMC |
| lanes swapped | 0 | 1 |

The one wide-prior NUTS lane was rerun by the production chain with HMC, which
had zero divergences; the aggregate chunk diagnostics retain the one rejected
primary-attempt divergence. Wide minus legacy has inverse-variance weighted
offset `-26.53 ppm` and slope `-55.11 ppm/um`. After removing both, its RMS is
`44.35 ppm`. The estimated median Monte Carlo floor is `5.87 ppm`, using
`1.2533*sigma/sqrt(ESS)` per chain and channel, so the residual shape difference
is `7.55` times that floor. The median wide/legacy 68-percent error-bar ratio is
`1.123`. This confirms a measurable prior effect rather than sampling-coordinate
parity, as expected from the user-selected wider support.

The two-panel product is
`acceleration_reports/ld_uplus_uminus_vs_legacy_WASP-39_G395H.png`; its machine
readable metrics are in the same-named JSON.

### Files, commands, failures, and risks

| File | Change |
|---|---|
| `models/jaxoplanet/builder.py` | Wide quadratic-uniform prior and deterministic legacy output sites |
| `fit_jwst.py` | Flag validation, builder plumbing, and coordinate-aware initial sites |
| `tools/spectro_stage_inputs.py` | Backward-compatible interpretation of pre-change dumps |
| `tools/run_production_swap_on_stage_inputs.py` | Selectable uniform basis and transformed replay initial point |
| `tools/compare_ld_uplus_prior.py` | Matched spectra, diagnostics, MC-floor calculation, and figure |
| `tests/test_quadratic_uniform_prior.py` | Default/legacy supports, transforms, white-light outputs, old-dump replay, fingerprint coverage |
| `SPECTRO_ACCELERATION.md`, `docs/guides/limb_darkening.md` | User-facing default and legacy escape hatch |

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false JAX_ENABLE_X64=1 /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest tests/test_quadratic_uniform_prior.py tests/test_sing_ld.py tests/test_spectro_safety_guards.py tests/test_mcmc_runner_reuse.py -q
python tools/run_sampler_on_stage_inputs.py /scratch/midway3/tfairnington/accel_gpu_results/348_uniform_quadratic_dump_retry2/stage_inputs/WASP-39_NIRSPEC_G395H_nrs1_Rreference_high_resolution_inputs.pkl --backend joint_nuts --platform gpu --seed 0 --warmup 1000 --samples 1000 --chunk-size 40 --builder-override ld_uniform_basis=coefficients --output-prefix /scratch/midway3/tfairnington/accel_gpu_results/432_ld_uniform_legacy_coefficients_v2/legacy_coefficients
python tools/run_production_swap_on_stage_inputs.py /scratch/midway3/tfairnington/accel_gpu_results/348_uniform_quadratic_dump_retry2/stage_inputs/WASP-39_NIRSPEC_G395H_nrs1_Rreference_high_resolution_inputs.pkl --output-prefix /scratch/midway3/tfairnington/accel_gpu_results/433_ld_uniform_uplus_uminus_v2/uplus_uminus --seed 0 --chunk-size 40 --warmup 1000 --samples 1000 --ld-uniform-basis uplus_uminus
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false JAX_ENABLE_X64=1 /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/compare_ld_uplus_prior.py
```

Queues 430 and 431 failed before sampling because the changed builder default
made the old dump reconstruct with the new prior during its stored-potential
validation. The compatibility rule above fixed the cause; fresh-root retries
432 and 433 both exited zero. No failed artifact was removed or overwritten.
Open risks are the intentionally nonphysical portions of the broad rectangular
prior and the substantial prior-sensitive chromatic slope seen here. Users who
require physical intensity profiles should compare the informed Sing mode or
explicitly choose a physically constrained prior rather than interpreting
`uniform` as such a constraint.

Final verification: the four-file focused regression command above passed 42
tests in 79.82 s (three dependency deprecation warnings), and
`python -m sphinx -E -W -b html docs docs/_build/html` completed successfully.
The final queue audit found 430/431 terminal with exit 1 and their corrected,
fresh-output-root replacements 432/433 terminal with exit 0; no job numbered
430--439 remains pending or running.
