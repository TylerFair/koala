# Free-width sigmoid discontinuity

Date: 2026-09-03

## Result

The discontinuity is now `jump * sigmoid((t - t_jump) / width)`.  White light samples `log_width` uniformly between `log(0.5 * dt)` and `log(30 minutes)`, where `dt` is the median cadence.  It emits `width` in days and `width_minutes`.  Spectroscopy fixes the sigmoid shape to the white-light posterior medians of `t_jump` and `width` and continues to fit only `A_jump`.  `step_width_mode: fixed` with `step_width_days` retains the fixed-width behavior.

The full WASP-39 NRS1 run completed, but the free width did **not** make the white-light Laplace metric pass.  The first Laplace block had zero divergences but geometry ESS 6.77--13.53.  The new ESS<50 fail-fast correctly skipped all three extension blocks.  Adaptive fallback then passed its first block with zero divergences and ESS 552--925.  Thus the production result is exact MCMC and completes in 37.83 minutes instead of repeating three unproductive Laplace blocks, but the requested one-block Laplace success was not achieved.

## Files

| File | Change |
|---|---|
| `models/trends.py` | Sigmoid step, cadence-scaled width prior, deterministic day/minute sites |
| `models/builder.py` | Free/fixed width in the legacy white-light builder |
| `models/jaxoplanet/builder.py` | Free/fixed width in the production white-light builder |
| `models/harmonica/builder.py` | Free/fixed width and shared sigmoid evaluation |
| `fit_jwst.py` | Defaults/validation, static model-factory routing, interior optimizer reset, fitted-width templates and outputs, fail-fast gate, white-light artifact fingerprint inputs |
| `tests/test_step_sigmoid.py` | Sharp-limit, prior/site, fixed-mode, support-edge, and fail-fast tests |
| `docs/guides/trends.md` | Astronomer-facing sigmoid and compatibility description |
| `configs_stacking/wasp39_nrs1_ref_informed_linear_step_sigmoid*.yaml` | New isolated validation and retry configs |
| `acceleration_reports/gpu_queue/*/400_*.sh`, `401_*.sh`, `402_*.sh` | GPU validation scripts |

## GPU validation

Final job: queue 402, Tesla V100-PCIE-16GB, exit 0.  Total wall was 2,270 s (37.83 min).  File timestamps place completion of the white-light diagnostics 1,806 s (30.10 min) after the GPU marker; this includes input/LD setup, optimizer, 63.41 s Laplace preparation, 721.98 s first-block Laplace sampling, adaptive fallback, and white-light output.  The code does not separately time the adaptive fallback, so no narrower white-light MCMC wall is claimed.

| White-light attempt | Retained blocks | Divergences | ESS (`t0`, `b`, duration, `rors`) | Outcome |
|---|---:|---:|---|---|
| Laplace metric | 1 | 0 | 6.77, 7.42, 7.93, 13.53 | Fail-fast; no extensions |
| Adaptive fallback | 1 | 0 | 924.85, 552.28, 740.79, 682.40 | Pass |

The Laplace preparation reached 200 iterations with gradient norm 115,530, Hessian minimum eigenvalue -723,114, and reported condition number `1e12`.  The final adaptive chain had 591/1000 retained draws at maximum tree depth.  The sigmoid is physically smoother, but this posterior favors the lower-width edge and remains difficult for a local Laplace metric.

White-light posterior medians and 16th/84th offsets were:

| Site | Median | - error | + error |
|---|---:|---:|---:|
| `t_jump` (BJD) | 59791.12062468 | 0.00010443 | 0.00010276 |
| `width` (day) | 0.000408706 | 0.000028841 | 0.000051748 |
| `width_minutes` | 0.588537 | 0.041531 | 0.074517 |
| `jump` | -0.00152628 | 0.00002437 | 0.00002670 |

The reference-grid spectroscopic stage passed: minimum/median depth ESS 724.93/1182.57.  One of 68 lanes had one initial NUTS divergence and was selectively rerun with HMC-8; the final sampler assignment was 67 Laplace NUTS and one Laplace HMC, with the rerun divergence count zero.

## Spectrum comparisons (68 identical channels)

Differences are sigmoid result minus comparator.  Offset uses inverse combined variance; slope is a weighted line fit; RMS is computed after removing the weighted offset; error ratio is the median sigmoid/comparator uncertainty ratio.

| Comparator | Offset (ppm) | Slope (ppm/um) | Offset-removed RMS (ppm) | Error ratio |
|---|---:|---:|---:|---:|
| Previous sharp-step run (queue 324) | -1.40 +/- 15.92 | +2.09 +/- 69.23 | 3.39 | 0.9982 |
| Saved production `LINEAR_DISCONTINUITY` posterior | -1.60 +/- 15.91 | +7.13 +/- 69.37 | 6.48 | 0.9987 |

Both slopes are consistent with zero and both RMS values are far below a typical channel uncertainty (about 90--100 ppm).  There is no measurable spectral-shape change at this Monte Carlo precision.

## Failures

- Queue 400 exited 1 before sampling because the staged optimizer returned `log_width` exactly at its Uniform lower support edge.  The unconstrained coordinate was non-finite.  The code now detects this and resets to the interior log-prior midpoint before Laplace preparation.
- Queue 401 exited 1 after preparation because the string `step_width_mode` was passed as a dynamic JIT model argument.  It is now a static model-factory choice; only numeric width values enter `prior_params`.
- A broad combined CPU test invocation was terminated externally after partial progress, and the independent-NUTS/safety subset was also terminated without a pytest result.  They are not reported as passes.  The focused new suite passed 5 tests in 4.61 s (and again after both fixes), and an earlier focused run including safety/LD initialization passed 29 tests in 15.78 s.  Sphinx 8.1.3 passed with `-W`.

## Exact commands

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest -q tests/test_step_sigmoid.py tests/test_spectro_safety_guards.py tests/test_ld_initialization.py
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest -q tests/test_step_sigmoid.py
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m sphinx -W -b html docs docs/_build/html
python fit_jwst.py --config /project/ekempton/tfairnington/JWST/configs_stacking/wasp39_nrs1_ref_informed_linear_step_sigmoid.yaml
python fit_jwst.py --config /project/ekempton/tfairnington/JWST/configs_stacking/wasp39_nrs1_ref_informed_linear_step_sigmoid_retry401.yaml
python fit_jwst.py --config /project/ekempton/tfairnington/JWST/configs_stacking/wasp39_nrs1_ref_informed_linear_step_sigmoid_retry402.yaml
```

## Open risks

The default prior is well defined only when half the median cadence is below 30 minutes; normal JWST time series satisfy this, but the builder does not currently provide a tailored error for coarser synthetic inputs.  White-light diagnostics currently serialize only geometry ESS, not ESS for `t_jump`, `jump`, or `width`; their posterior intervals are available in the CSV but their ESS cannot be reconstructed because this config did not request the full white-light trace.  Most importantly, smoothing removed the staircase likelihood but did not cure this dataset's boundary-favoring white-light geometry; production remains dependent on the exact adaptive fallback.
