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
| `models/ld_parameterization.py` | New Maxted power-2 and Kipping quadratic bijections with analytic Jacobians. |
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
- The Kipping transform is implemented and CPU prior-tested, but the real GPU
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
