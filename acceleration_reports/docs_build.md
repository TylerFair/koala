# Documentation build report

Date: 2026-09-02

## Files written

No pre-existing file was deleted, renamed, or overwritten. The following new source files were added.

| File | Purpose |
|---|---|
| `.readthedocs.yaml` | Read the Docs Python 3.11 build configuration |
| `docs/conf.py` | Sphinx, MyST, Furo, autodoc, and single project-name definition |
| `docs/requirements.txt` | Documentation dependencies |
| `docs/index.md`, `docs/install.md`, `docs/quickstart.md`, `docs/api.md`, `docs/citing.md` | Landing, setup, first fit, API, and citation pages |
| `docs/tutorials/*.md` | SOSS order 1, G395H, PRISM, and Harmonica tutorials |
| `docs/guides/*.md` | Limb darkening, trends, samplers, configuration, outputs, and cluster guides |
| `docs/_build/html/**` | Generated HTML from the successful local build |

There are 17 documentation source/configuration files under `docs/`, totaling 562 lines before the final reference-table addition.

## Commands and results

Dependency installation:

```bash
/home/tfairnington/miniconda3/envs/jaxoplanet/bin/pip install sphinx myst-parser furo sphinx-copybutton sphinx-design
```

Result: success. Installed Sphinx 8.1.3, MyST Parser 4.0.1, Furo 2025.12.19, sphinx-copybutton 0.5.2, and sphinx-design 0.6.1.

Required warning-as-error build:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false /home/tfairnington/miniconda3/envs/jaxoplanet/bin/sphinx-build -W -b html docs docs/_build/html
```

Result: success, 15 source pages, zero Sphinx warnings. HTML is in `docs/_build/html`.

Focused tests:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest -q tests/test_fit_independent_routing.py tests/test_explinear_spectroscopic.py tests/test_sing_ld.py tests/test_harmonica_limb_products.py
```

Result: 14 passed in 130.93 seconds. Three dependency deprecation warnings were emitted (`pkg_resources` twice and JAXopt once); no test failed.

## Failures and open risks

The first build failed with six autodoc warnings because `numpy` was included in `autodoc_mock_imports`; mocked `numpy.pi` broke import-time arithmetic in `models.harmonica.core`. Removing `numpy` from the mock list fixed all six warnings, and both subsequent warning-as-error builds passed.

The documented production sampler defaults were verified after a concurrent code update: independent NUTS, Laplace mass matrix, automatic HMC-8 fallback, lognormal jitter, posterior-median white-light geometry, and fixed spectroscopic explinear timescale are now set in `fit_jwst.py`. Existing configurations may explicitly override them.

The example spectrum filename stems depend on target/instrument/resolution naming. Users should list the output directory rather than assume a stem. No end-to-end science fit was run because it requires target FITS data and a GPU; documentation construction and CPU-focused tests do not verify a complete observational dataset run.
