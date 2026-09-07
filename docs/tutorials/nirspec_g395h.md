# NIRSpec detector notes

NIRSpec time-series extractions use the same fit sequence as the [first-transit tutorial](soss_order1.md). The important extra choice is the detector:

```yaml
instrument: NIRSPEC/G395H
nrs: 1
fits_file: WASP-121_nrs1_box_spectra_fullres_G395H.fits
```

Fit NRS1 and NRS2 separately with their matching extractions. Changing only `nrs` while keeping an NRS1 filename is an easy way to mislabel a result.

G395M and G140H use the same `nrs` field with their corresponding instrument strings. PRISM is covered separately because its long time series and early ramp often need different choices.

## A G395H example

```bash
cp examples/nirspec_g395h.yaml wasp121_nrs1.yaml
python fit_jwst.py -c wasp121_nrs1.yaml
```

The checked configuration uses power-2 limb darkening with `stellarprior`, a quadratic baseline, an $R=20$ bridge, and a supplied final wavelength grid:

```yaml
instrument: NIRSPEC/G395H
nrs: 1

resolution:
  low: 20
  high: reference
  reference_grid: prism_template.csv

flags:
  need_lowres: true
  detrending_type: quadratic
  ld_profile: power2
  ld_prior: stellarprior
```

Use a numeric `resolution.high` if you want constant resolving power without an external grid file. The bridge and final grid are distinct: the coarse stage stabilizes the handoff, while the high-resolution stage produces the reported spectrum.

```{image} ../_static/g395h_wasp121_whitelight.png
:alt: Example WASP-121 b G395H NRS1 white-light fit and residuals
:width: 760px
:align: center
```

Check that the broadband baseline constrains the quadratic curvature. If a linear trend leaves no coherent residual structure and gives stable geometry, prefer the simpler model.

```{image} ../_static/g395h_wasp121_spectrum.png
:alt: Example WASP-121 b G395H NRS1 transmission spectrum
:width: 760px
:align: center
```

The detector name appears in output stems such as `*_nrs1_*`. Before joining NRS1 and NRS2 spectra, compare their overlap and decide explicitly how detector offsets will be handled; the fitter does not hide that choice.

For model selection, read [systematics trends](../guides/trends.md). For column definitions and fit diagnostics, use [outputs](../guides/outputs.md).
