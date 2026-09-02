# Samplers

The standard spectroscopic configuration is:

```yaml
flags:
  spectro_sampler: independent_nuts
  spectro_mass_matrix: laplace
  spectro_jitter_prior: lognormal
```

White-light fits use NumPyro NUTS. Set `whitelight_mass_matrix: laplace` for Hessian preconditioning; a failed Laplace preparation falls back to adaptive mass-matrix NUTS.

Independent NUTS adapts a low-dimensional chain per wavelength channel. With `spectro_mass_matrix: laplace`, each chain uses a local Laplace metric. The production policy switches a channel failing the depth-ESS/divergence gate to Laplace-metric fixed-step HMC-8, and switches back when the alternative fails. Select HMC directly with `spectro_sampler: independent_hmc` and `spectro_hmc_num_steps: 8`. `joint_nuts` is the legacy adaptive joint-channel fallback.

`laplace_is` is opt-in approximate inference centered on a MAP/Laplace proposal. It reports importance diagnostics and can fall back for poor channels; use it only when its approximation and quality thresholds are acceptable for the analysis.

The gate requires zero divergences and a minimum bulk ESS for transit depth. Logs such as `depth ESS below ...`, `divergences=...`, and `PASSED/FAILED ... GATE` report the channel or stage decision. JSON diagnostics beside chunk checkpoints contain the machine-readable values.

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

