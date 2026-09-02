# v2: ESS-aware summary addendum

- Seed and six datasets: unchanged from `README.md` (`20260902`).
- Output: `summary_v2.png` (150 dpi).
- Dashed curves: averages of per-channel Gaussian references using actual ArviZ bulk ESS.
- Grey fill: matching high-resolution production seed-to-seed null rows. The source JSON stores absolute median shifts, so shift values are mirrored about zero for display. Sigma ratios are plotted as stored.

## Median bulk ESS by site

| Chain | Site | N channel-datasets | Median ESS | 5–95% ESS |
|---|---|---:|---:|---:|
| saved | depth | 536 | 1543.7 | [800.3, 2476.4] |
| saved | c | 536 | 1077.7 | [730.1, 2016.9] |
| saved | v | 536 | 1050.4 | [581.0, 2507.2] |
| saved | c1 | 536 | 1288.2 | [697.9, 2658.4] |
| saved | c2 | 536 | 1344.0 | [745.2, 2898.1] |
| Laplace NUTS | depth | 536 | 2479.7 | [1845.2, 3609.7] |
| Laplace NUTS | c | 536 | 2439.6 | [1844.2, 3717.6] |
| Laplace NUTS | v | 536 | 2119.2 | [1521.0, 3063.2] |
| Laplace NUTS | c1 | 536 | 2464.5 | [1822.1, 3738.6] |
| Laplace NUTS | c2 | 536 | 2498.4 | [1828.1, 3772.3] |
| Laplace HMC-8 | depth | 536 | 6602.1 | [1002.8, 6602.1] |
| Laplace HMC-8 | c | 536 | 6602.1 | [1039.4, 6602.1] |
| Laplace HMC-8 | v | 536 | 6602.1 | [1615.7, 6602.1] |
| Laplace HMC-8 | c1 | 536 | 6602.1 | [1013.4, 6602.1] |
| Laplace HMC-8 | c2 | 536 | 6602.1 | [1110.6, 6602.1] |

## Production-null inputs

- `/scratch/midway3/tfairnington/accel_stage_inputs/references/GJ-3470_NIRSPEC_G395H_nrs1_R300_high_resolution_inputs_ch0_40_pooled_joint_nuts_noise_floor.json`
- `/scratch/midway3/tfairnington/accel_stage_inputs/references/HAT-P-12_NIRISS_SOSS_order1_Rreference_high_resolution_inputs_ch0_40_pooled_joint_nuts_noise_floor.json`

## Reproduce

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 0-15 \
  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/plot_validation_summary_v2.py \
  --seed 20260902 --output-dir acceleration_reports/validation_plots_reproduction
```
