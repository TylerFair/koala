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
