# Documentation figures

These scripts draw the documentation PNGs without running Koala. Run them from the repository root with NumPy, Matplotlib, and pandas installed; FITS ingestion also needs Astropy.

```bash
OPENBLAS_NUM_THREADS=1 python tools/docs/generate_tutorial_figures.py
python tools/docs/plot_run_outputs.py --runs /path/to/docs_runs_20260927
```

The first script uses the archived tables in `docs/_static/data/`. It generates the workflow, limb-darkening profiles, trend gallery, archived Harmonica limb spectra, HAT-P-18 stacked spectrum, and realistic MIRI timing scan. `wasp94_limbs.csv` preserves the archived schema-v2 limb summaries; the plot maps the old physical labels to neutral angular indices using that file's stored convention. Schematics have illustrative coefficients and are labeled as such in their captions. The MIRI scan uses the 2,000-integration white-light series summed from the finite channels of the supplied realistic synthetic FITS file. It fits a nonnegative uniform-disk eclipse by weighted least squares; the exponential timescale is estimated outside the injected event and then held fixed. The reported uncertainty is conditional on that timescale and geometry.

To re-ingest the original scan or stacking input:

```bash
python tools/docs/generate_tutorial_figures.py \
  --eclipse-fits /path/to/realistic_eclipse.fits \
  --stacked-csv /path/to/hatp18_stacked.csv
```

The second script requires completed `niriss_soss_order1/`, `eclipse/`, and `phase_curve/` directories. It reads CSV products only, copies the plotting inputs to `docs/_static/data/runs/`, and records their paths, SHA-256 checksums, and columns in `manifest.json`. It writes four SOSS figures plus eclipse and phase-curve recovery figures. The SOSS detrended curves divide the exported channel flux by `c + v * (time - min(time))`; no private model functions are imported.

Redraw these archived products without a fit:

```bash
python tools/docs/plot_run_outputs.py --archive docs/_static/data/runs
```

`--output /path/to/preview` sends the run figures and copied tables to a separate directory. Images use the `koala.plotting_style` house style with Matplotlib text rendering, 180 dpi, and a width of 1,620 pixels.

The source of `hatp18_stacked.csv` is the archived three-candidate HAT-P-18 NIRSpec/G395M comparison, `hatp18_nrs1_g395m_reference_fitted_sing_uplus_v2_stacked.csv`. Its weights correspond to fixed power-2, uniform quadratic, and fitted Sing quadratic. The example matrix now includes a fourth candidate; the caption makes this distinction explicit.
