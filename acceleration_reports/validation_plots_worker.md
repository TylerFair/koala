# Posterior validation plots: saved production vs Laplace NUTS and HMC-8

## Outcome

Generated six seeded, posterior-level comparison figures and one aggregate
figure at 150 dpi.  The eligible pool contained 27 datasets with completed A
and B results and staged saved posterior draws.  The seeded selection covers
SOSS order 1, G395H, SOSS order 2, and G395M.  No PRISM dataset had a completed
A+B pair, so no PRISM result is represented.

These are parity plots only; no speedup was measured or claimed here.

## Files built

- `tools/plot_validation_posteriors.py`: immutable-output plot generator and
  seeded campaign/reference discovery.
- `acceleration_reports/validation_plots/README.md`: seed, selected datasets,
  exact channels, aggregate statistics, and reproduction command.
- `acceleration_reports/validation_plots/*.png`: six dataset figures plus
  `summary.png` (all 150 dpi).
- `acceleration_reports/validation_plots_worker.md`: this report.  The worker
  report uses this name because `acceleration_reports/validation_plots/` is the
  required output directory and a file cannot share that pathname.

Each dataset figure contains step histograms for four seeded channels, saved
production as grey fill, A in blue, B in orange, and D as black dashed when
available.  Every histogram is annotated with median, central-68% half-width,
and signed median shift in saved-sigma units.  The lower panels contain the
full depth spectrum, residuals in ppm with saved error bars, weighted offset
and slope, and candidate/saved error-bar ratios.

## Seeded selection

Seed: `20260902`.  Channels are zero-based.

| Dataset | Mode | Channels | Control D |
|---|---|---|---|
| HAT-P-12 SOSS order 1 spot | SOSS O1 | 14, 48, 73, 107 | no |
| WASP-52 G395H NRS1 linear | G395H | 26, 35, 54, 61 | no |
| HAT-P-26 SOSS order 2 linear | SOSS O2 | 0, 7, 13, 31 | no |
| HAT-P-18 G395M NRS1 linear | G395M | 46, 76, 135, 163 | no |
| HAT-P-12 SOSS order 2 spot | SOSS O2 | 19, 20, 26, 29 | yes |
| HAT-P-30 G395H NRS1 linear | G395H | 16, 58, 59, 66 | no |

## Measured aggregate posterior differences

All 536 high-resolution wavelength channels from the six datasets enter each
row.  Sigma is the central-68% half-width.

| Candidate | Site | median signed shift / saved sigma | central 95% | median sigma ratio | 5--95% sigma ratio |
|---|---|---:|---:|---:|---:|
| Laplace NUTS | depth | -0.0009 | [-0.1035, +0.0948] | 0.9963 | [0.9178, 1.0825] |
| Laplace NUTS | c | -0.0175 | [-0.1142, +0.0820] | 1.0012 | [0.9326, 1.0823] |
| Laplace NUTS | v | +0.0051 | [-0.0955, +0.1013] | 1.0019 | [0.9309, 1.0796] |
| Laplace NUTS | c1 | +0.0025 | [-0.0984, +0.0976] | 0.9994 | [0.9283, 1.0935] |
| Laplace NUTS | c2 | -0.0010 | [-0.0882, +0.0880] | 1.0047 | [0.9314, 1.0932] |
| Laplace HMC-8 | depth | -0.0019 | [-0.1042, +0.1022] | 1.0027 | [0.9066, 1.1047] |
| Laplace HMC-8 | c | -0.0139 | [-0.1179, +0.0850] | 1.0026 | [0.9134, 1.0998] |
| Laplace HMC-8 | v | +0.0021 | [-0.0962, +0.0883] | 1.0026 | [0.9229, 1.0906] |
| Laplace HMC-8 | c1 | +0.0046 | [-0.1031, +0.0912] | 1.0008 | [0.9154, 1.0861] |
| Laplace HMC-8 | c2 | -0.0026 | [-0.0806, +0.0815] | 1.0038 | [0.9182, 1.1074] |

The aggregate plot overlays finite-draw Gaussian references.  For posterior
median shifts it uses
`sqrt(pi/2 * (1/N_saved + 1/N_candidate))`; for sigma ratios it uses
`sqrt(1/(2(N_saved-1)) + 1/(2(N_candidate-1)))`.  With 1,000 draws on both
sides, the median-shift reference width is 0.056 and the sigma-ratio reference
width is 0.032.  These curves are references, not fitted claims.

## Exact reproduction

The required destination must not already exist because the script opens the
output tree immutably.

```bash
JAX_PLATFORMS=cpu \
  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  -m py_compile tools/plot_validation_posteriors.py

JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  taskset -c 0-15 \
  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  tools/plot_validation_posteriors.py \
  --seed 20260902 \
  --output acceleration_reports/validation_plots_reproduction
```

Observed generation output reported the six dataset names and eight outputs
(six dataset PNGs, summary PNG, README).  PNG dimensions are 2362 or 2835 by
2250 pixels for dataset figures and 2400 by 975 pixels for the summary.

## Failures and open risks

- Neither available PRISM campaign completed an A+B pair: Kepler-12 timed out
  and HAT-P-65 remained incomplete at campaign harvest.  PRISM posterior-level
  behavior is therefore untested in these figures.
- Only HAT-P-12 SOSS order 2 among the seeded six has a completed regenerated
  production control D; other panels compare against historical saved runs.
- Candidate/saved sigma-ratio distributions are centered near one but are
  visibly wider than the simple 1,000-independent-draw reference, especially
  for HMC-8.  Autocorrelation, unequal effective sample sizes, posterior
  non-normality, and run/input differences can all broaden this distribution;
  the figure does not identify which mechanism dominates.
- Step histograms preserve non-Gaussian shapes without bandwidth choices, but
  30-bin appearance is still bin-edge dependent.  Numerical annotations and
  aggregate tables should be used alongside the shapes.
- The random selection is mode-stratified rather than a uniform draw over all
  27 eligible datasets: one dataset was first drawn from each available
  requested mode, then remaining slots were drawn without replacement.
