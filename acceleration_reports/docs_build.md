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

## Full expansion pass — 2026-09-02

The public documentation was expanded from 545 to 3,232 MyST lines across the 17 built pages. Every built page is at least 120 lines; the four tutorials are 204--263 lines. `concepts.md` and `faq.md` were added, and `index.md` now links both.

Expanded content includes the staged white-light/channel model, geometry handoff, lanes and chunks, Laplace metrics, quality gates, full commented tutorial configurations, excerpts from completed logs, output walkthroughs, equations and prior bounds, sampler routing, checkpoint semantics, configuration examples, CSV schemas, and troubleshooting.

### Figures copied

All copies are PNG files below 500 kB. No fit was generated.

| Existing source | Documentation copy | Bytes |
|---|---|---:|
| `/scratch/midway3/tfairnington/WASP-39_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR_PARITY_20260902a_A/11_WASP-39_NIRISS_SOSS_order1_whitelightmodel.png` | `docs/_static/soss_wasp39_whitelight.png` | 27,575 |
| `/scratch/midway3/tfairnington/WASP-39_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR_PARITY_20260902a_A/31_WASP-39_NIRISS_SOSS_order1_Rreference_spectrum_00.png` | `docs/_static/soss_wasp39_spectrum.png` | 68,672 |
| `/scratch/midway3/tfairnington/POPULATION_FITS_FIRSTRUN_NONPUBLICATIONREADY/WASP-121_G395H_NRS1_GAUSSIANLD_POWER2_QUADRATIC/11_WASP-121_NIRSPEC_G395H_nrs1_whitelightmodel.png` | `docs/_static/g395h_wasp121_whitelight.png` | 32,306 |
| `/scratch/midway3/tfairnington/ASYMM_WASP-121_G395H_NRS1_FIXEDLD_QUADRATIC_QUADRATIC/31_WASP-121_NIRSPEC_G395H_nrs1_R100_spectrum_00.png` | `docs/_static/g395h_wasp121_spectrum.png` | 50,149 |
| `/scratch/midway3/tfairnington/HAT-P-65_PRISM_NRS1_V1_STELLARINFORMEDLD_POWER2_EXPLINEAR_PARITY_20260902a_A/11_HAT-P-65_NIRSPEC_PRISM_nrs1_whitelightmodel.png` | `docs/_static/prism_hatp65_whitelight.png` | 52,727 |
| `/scratch/midway3/tfairnington/POPULATION_FITS_FIRSTRUN_NONPUBLICATIONREADY/HAT-P-65_PRISM_NRS1_V1_GAUSSIANLD_POWER2_EXPLINEAR/31_HAT-P-65_NIRSPEC_PRISM_nrs1_Rnative_spectrum_00.png` | `docs/_static/prism_hatp65_spectrum.png` | 67,461 |
| `/scratch/midway3/tfairnington/HARMONICA_ACCEL_WASP94_DUMP_20260902/11_WASP-94_NIRISS_SOSS_order1_whitelightmodel.png` | `docs/_static/harmonica_wasp94_whitelight.png` | 36,360 |
| `/scratch/midway3/tfairnington/HARMONICA_ACCEL_WASP94_DUMP_20260902/31_WASP-94_NIRISS_SOSS_order1_R50_spectrum_00.png` | `docs/_static/harmonica_wasp94_spectrum.png` | 53,750 |
| `/scratch/midway3/tfairnington/HARMONICA_ACCEL_WASP94_DUMP_20260902/35_WASP-94_NIRISS_SOSS_order1_R50_limb_spectra.png` | `docs/_static/harmonica_wasp94_limb_spectra.png` | 107,910 |

### Build command and result

```bash
PATH=/home/tfairnington/miniconda3/envs/jaxoplanet/bin:$PATH JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false sphinx-build -E -W -b html docs docs/_build/html
```

Result: success. Sphinx rebuilt 17 source pages, copied all nine figures, and emitted zero warnings.

### Failures and open risks

No build failure occurred during the expansion pass. A separately added `docs/guides/model_stacking.md` was present in the shared workspace but is outside this documentation task; it is excluded in `docs/conf.py` and is not counted among the 17 built pages.

The tutorial log excerpts are shortened from existing completed logs. Older logs can show sampler settings predating current production defaults; explanatory prose and full tutorial YAML use the current verified defaults. The documentation figures are illustrative completed products for the named target/mode. In the G395H and PRISM tutorials, some figure sources use a different verified limb-prior or engine configuration than the tutorial YAML, and the captions avoid presenting them as outputs of that exact configuration.
2026-09-02 prose pass: merged one-sentence notes into connected 3--6 sentence paragraphs across all 17 built pages, retained code/equations/tables/figures, and added tutorial outcomes and Next steps links.
Build check: `sphinx-build -E -W -b html docs docs/_build/html` completed successfully with zero warnings.
