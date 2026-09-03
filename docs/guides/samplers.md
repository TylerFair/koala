# Samplers

The standard spectroscopic configuration is:

```yaml
flags:
  spectro_sampler: independent_nuts
  spectro_mass_matrix: laplace
  spectro_jitter_prior: lognormal
```

White-light fits use NumPyro NUTS. Set `whitelight_mass_matrix: laplace` for Hessian preconditioning; a failed Laplace preparation falls back to adaptive mass-matrix NUTS.

Wide-Gaussian and uniform/free power-2 limb-darkening priors use Maxted decorrelated coordinates by default in both white-light and spectroscopic fits. The analytic Jacobian leaves the physical prior and reported coefficients unchanged; quadratic priors use physical coefficient coordinates, with `latent_gaussian` as the only alternative, and stellar-informed and Sing modes are unaffected.

Independent NUTS adapts a low-dimensional chain per wavelength channel. With `spectro_mass_matrix: laplace`, each chain uses a local Laplace metric. The production policy switches a channel failing the depth-ESS/divergence gate to Laplace-metric fixed-step HMC-8, and switches back when the alternative fails. Select HMC directly with `spectro_sampler: independent_hmc` and `spectro_hmc_num_steps: 8`. `joint_nuts` is the legacy adaptive joint-channel fallback.

`laplace_is` is opt-in approximate inference centered on a MAP/Laplace proposal. It reports importance diagnostics and can fall back for poor channels; use it only when its approximation and quality thresholds are acceptable for the analysis. The gate requires zero divergences and a minimum bulk ESS for transit depth. Logs such as `depth ESS below ...`, `divergences=...`, and `PASSED/FAILED ... GATE` report the channel or stage decision. JSON diagnostics beside chunk checkpoints contain the machine-readable values.

On a V100 or A100, the first chunk of a new static width typically spends about 1–1.5 minutes compiling. Equal-width chunks reuse the compiled runner and then take seconds to a few minutes each, depending on cadence count and sampler work.

```yaml
flags:
  vmap_chunk: 40
  lowres_num_warmup: 1000
  lowres_num_samples: 1000
  highres_num_warmup: 1000
  highres_num_samples: 1000
```

Completed chunks are saved under `output_dir/chunks/`. Re-running an identical configuration loads fingerprint-matching checkpoints. `chunk_mode: parallel` writes assigned chunks; `chunk_mode: combine` requires every expected checkpoint and concatenates them in wavelength order.

## Decision table

| Stage or condition | Sampler | Metric | Action on failure |
|---|---|---|---|
| White light | NUTS | Laplace by default | Adaptive-metric fallback if Laplace preparation fails |
| Spectroscopic default | Independent NUTS | Per-channel Laplace | Retry failed work with HMC-8 |
| NUTS gate failure | Independent HMC | Per-channel Laplace, 8 steps | Retry with NUTS if HMC is the selected primary path |
| Explicit legacy fallback | Joint NUTS | Adaptive | Inspect joint diagnostics |
| Explicit approximate option | Laplace importance sampling | Laplace proposal | Internal fallback when enabled and diagnostics fail |

The automatic swap remains within exact HMC/NUTS inference. Laplace importance sampling is separate and must be requested explicitly.

## White-light NUTS

White light samples shared geometry, broadband limb darkening, jitter, and trend parameters. The default Laplace preparation finds a mode and constructs a curvature-based inverse mass matrix.

```yaml
flags:
  whitelight_mass_matrix: laplace
  whitelight_laplace_warmup: 200
  whitelight_laplace_target_accept: 0.9
  whitelight_laplace_max_tree_depth: 10
  whitelight_laplace_trust_radius: 5.0
  whitelight_laplace_hessian_method: finite_difference
  whitelight_min_ess: 400
  whitelight_max_divergences: 0
  whitelight_max_extra_blocks: 3
```

PRISM raises the production white-light target acceptance to 0.99. Additional blocks can extend a chain that has not yet reached the gate.

## Independent NUTS

Each lane contains one channel posterior and its own adaptation state. The production finite-difference Laplace metric starts sampling at the local mode.

```yaml
flags:
  spectro_sampler: independent_nuts
  spectro_mass_matrix: laplace
  spectro_laplace_warmup: 150
  spectro_laplace_target_accept: 0.95
  spectro_laplace_max_tree_depth: 5
  spectro_laplace_trust_radius: 5.0
  spectro_laplace_hessian_method: finite_difference
  spectro_laplace_start_at_map: true
  spectro_min_depth_ess: 400
  spectro_max_divergences: 0
```

PRISM or explinear configurations default to target acceptance 0.99. PRISM defaults to maximum tree depth 6.

## Independent HMC

HMC uses a fixed number of leapfrog steps rather than building a NUTS tree. The production alternate uses eight steps and trajectory jitter 0.25.

```yaml
flags:
  spectro_sampler: independent_hmc
  spectro_mass_matrix: laplace
  spectro_hmc_num_steps: 8
  spectro_hmc_trajectory_jitter: 0.25
```

Fixed work can be faster and more predictable for a difficult lane. It can also fail when eight steps do not explore a posterior adequately, which is why NUTS remains the alternate.

## Joint adaptive NUTS

```yaml
flags:
  spectro_sampler: joint_nuts
  spectro_mass_matrix: adaptive
```

This samples all channels in a chunk within one joint kernel. It is retained for compatibility and as a fallback path. One hard channel can increase tree work for every lane.

## Laplace importance sampling

```yaml
flags:
  spectro_sampler: laplace_is
  spectro_laplace_is_num_draws: 4096
  spectro_laplace_is_rounds: 2
  spectro_laplace_is_student_df: 3.0
  spectro_laplace_is_khat_threshold: 0.7
  spectro_laplace_is_min_ess: 400
  spectro_laplace_is_fallback: true
```

The method draws from a heavy-tailed proposal centered on the local Laplace approximation and reweights against the exact posterior. It is approximate because finite importance samples and Pareto smoothing replace a Markov chain. Inspect Pareto-$k$, importance ESS, and fallback counts before using its summaries.

## Reading timing messages

`preparing Laplace metric` marks mode/Hessian work. `COMPUTING` means no usable checkpoint exists. The first equal-width chunk includes JIT compilation and commonly takes about 1--1.5 minutes on V100/A100.

`reusing compiled independent-NUTS runner` means subsequent equal-width chunks avoid that compile. Sampling then usually takes seconds to a few minutes per chunk. The last partial-width chunk can compile once more because its static shape differs.

## Gate and swap messages

`depth ESS below` identifies insufficient effective samples. `num_divergences` counts Hamiltonian integration failures. `PASSED ... GATE` means the configured ESS and divergence criteria were satisfied.

A failure is followed by a message naming the alternate backend. The replacement chain receives a distinct diagnostic record. Do not infer success solely from `SAVED checkpoint`; read the gate status.

## Checkpoints

Checkpoint filenames contain the target/stage prefix, backend, fingerprint fragment, and channel range. The manifest binds the family to model arguments, arrays, priors, sampler controls, and relevant source files. Resume loads only current, readable, matching files.

Corrupt or stale files are not silently concatenated. Parallel jobs assign deterministic chunk ranges. `combine` checks that all expected chunks exist and restores wavelength order.

## Reproducibility

The default master seed is 555. Set it explicitly with `flags.random_seed`. `FIT_JWST_SEED` overrides the YAML value.

Every chunk derives a deterministic distinct key from the master seed and channel range. Resuming does not change later chunk keys because they do not depend on which earlier files were loaded. Record the seed, software versions, backend, metric, warmup, samples, target acceptance, and GPU type.

## Troubleshooting

**Repeated compilation:** compare chunk widths and static model settings. A different final width legitimately compiles separately. **Low ESS without divergences:** increase retained samples, inspect autocorrelation and tree depth, and allow the alternate sampler.

**Divergences:** inspect LD boundaries, trend degeneracy, reported uncertainties, and the local MAP/Hessian diagnostics. **One bad lane slows a chunk:** reduce `vmap_chunk` or use a verified batch plan to isolate it. **Resume does not load:** compare the manifest fingerprint inputs; a scientifically relevant change must create new checkpoints.

**Out of memory:** lower resident width. Do not reduce wavelength resolution unless that is also the intended analysis. **Both exact samplers fail:** inspect the data channel and model rather than lowering the quality gate.
