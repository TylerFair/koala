# Documentation figures

These scripts draw the figures in `docs/_static/`.
Run them from the top of the repository.
They need NumPy, Matplotlib, pandas and Astropy, but not JAX, and none of them runs a fit.

`first_fit_figures.py` makes the figures for "Your first fit", the both-orders case study and the front page.
The plotting code is the same as the code on those pages:

```bash
python tools/docs/first_fit_figures.py \
  --run results/WASP-39_SOSS_ORDER1 --joint results/WASP-39_joint
```

Without any options it makes only the first-look figure of the data, which needs just the example FITS file.

`generate_tutorial_figures.py` makes the trend gallery, the archived WASP-94 b Harmonica limbs, the archived HAT-P-18 b stacked spectrum, and the shallow-eclipse timing scan, from the tables in `docs/_static/data/`:

```bash
python tools/docs/generate_tutorial_figures.py
```

To redo the eclipse scan or the stacking figure from their original inputs, add `--eclipse-fits PATH` or `--stacked-csv PATH`.
The scan sums the finite channels of the synthetic FITS file into a white-light curve and fits a uniform-disk eclipse by weighted least squares at each trial time, with the ramp timescale estimated outside the eclipse and held fixed.

`plot_run_outputs.py` makes the eclipse and phase-curve recovery figures from finished runs of `examples/eclipse.yaml` and `examples/phase_curve.yaml`:

```bash
python tools/docs/plot_run_outputs.py --runs /path/to/runs
```

where `/path/to/runs` has `eclipse/` and `phase_curve/` output directories.
The points binned in time in some figures are for display only, and the fits themselves use every integration.
