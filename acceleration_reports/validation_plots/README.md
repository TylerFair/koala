# Posterior validation plots

- Random seed: `20260902`
- Eligible completed A+B datasets: `27`
- Selection: mode-stratified seeded draw (one each from available SOSS order 1, G395H, SOSS order 2, G395M, PRISM; remaining slots sampled without replacement).

| Dataset | Channels (zero-based) | Optional D |
|---|---|---|
| HAT-P-12_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_SPOT | 14, 48, 73, 107 | no |
| WASP-52_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR | 26, 35, 54, 61 | no |
| HAT-P-26_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_LINEAR | 0, 7, 13, 31 | no |
| HAT-P-18_G395M_NRS1_STELLARINFORMEDLD_POWER2_LINEAR | 46, 76, 135, 163 | no |
| HAT-P-12_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_SPOT | 19, 20, 26, 29 | yes |
| HAT-P-30_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR | 16, 58, 59, 66 | no |

## Aggregate numbers

| Candidate | Site | N channels | median shift/saved σ | 95% | median σ ratio | 5–95% |
|---|---|---:|---:|---:|---:|---:|
| Laplace NUTS | depth | 536 | -0.0009 | [-0.1035, +0.0948] | 0.9963 | [0.9178, 1.0825] |
| Laplace NUTS | c | 536 | -0.0175 | [-0.1142, +0.0820] | 1.0012 | [0.9326, 1.0823] |
| Laplace NUTS | v | 536 | +0.0051 | [-0.0955, +0.1013] | 1.0019 | [0.9309, 1.0796] |
| Laplace NUTS | c1 | 536 | +0.0025 | [-0.0984, +0.0976] | 0.9994 | [0.9283, 1.0935] |
| Laplace NUTS | c2 | 536 | -0.0010 | [-0.0882, +0.0880] | 1.0047 | [0.9314, 1.0932] |
| Laplace HMC-8 | depth | 536 | -0.0019 | [-0.1042, +0.1022] | 1.0027 | [0.9066, 1.1047] |
| Laplace HMC-8 | c | 536 | -0.0139 | [-0.1179, +0.0850] | 1.0026 | [0.9134, 1.0998] |
| Laplace HMC-8 | v | 536 | +0.0021 | [-0.0962, +0.0883] | 1.0026 | [0.9229, 1.0906] |
| Laplace HMC-8 | c1 | 536 | +0.0046 | [-0.1031, +0.0912] | 1.0008 | [0.9154, 1.0861] |
| Laplace HMC-8 | c2 | 536 | -0.0026 | [-0.0806, +0.0815] | 1.0038 | [0.9182, 1.1074] |

The dashed references in `summary.png` use the finite-draw normal approximations: median-shift SD `sqrt(pi/2 * (1/N_saved + 1/N_candidate))`; sigma-ratio SD `sqrt(1/(2(N_saved-1)) + 1/(2(N_candidate-1)))`. These are Monte-Carlo references, not fitted claims.

## Reproduce

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 0-15 \
  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/plot_validation_posteriors.py \
  --seed 20260902 --output acceleration_reports/validation_plots_reproduction
```
