# Loop mode

## Status (2026-09-03)

This task is not complete. No GPU validation was queued and no timing or parity
result is reported.

The builder now has an opt-in invariant limb-darkening data form. It always
samples one two-component standard-normal latent and replaces that base density
with the selected coordinate density using a NumPyro factor. The selector is
traced data. Fixed coefficients ignore the auxiliary latent, which therefore
integrates to one and leaves the physical posterior unchanged. Tests compare
the resulting log density with the structural coefficient, transformed
power-2, quadratic sum/difference, and Sing densities to an absolute tolerance
of `1e-12` (the requested threshold was `1e-10`).

| File | Change |
|---|---|
| `models/jaxoplanet/builder.py` | Added the invariant LD data-form helper and opt-in white-light/spectroscopic model arguments. |
| `tests/test_loop_mode.py` | Added potential equality, fixed-nuisance, and ordering tests. |
| `tools/loop_fit.py` | Added the six-variant plan and cache setup; execution deliberately fails rather than pretending six subprocesses reuse resident runners. |

The blocking integration issue is that `fit_jwst.py::main` owns data loading,
white-light optimization, MCMC construction, geometry handoff, low-resolution
calibration, high-resolution fitting, and product writing in one monolithic
function. It exposes no staged in-process API. A correct loop must first split
those responsibilities into reusable stage objects; a subprocess loop would
reload inputs and reconstruct runners six times and does not meet the user's
compile-once requirement. The new builder form is not yet passed through the
pipeline and must not be used for production.

## Commands run

```text
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest -q tests/test_loop_mode.py
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest -q tests/test_ld_parameterization.py tests/test_quadratic_uniform_prior.py tests/test_sing_ld.py tests/test_loop_mode.py
```

The first command passed 4 tests. The combined command passed 21 tests in
81.00 seconds with three third-party deprecation warnings. Documentation and
GPU validation were not run because the production integration is incomplete.

## Open risks

The opt-in builder path has unit-level density coverage but no full-model MCMC
coverage. In particular, initialization and finite rejection behavior at the
bounded coordinate supports still need pipeline tests before GPU use.
