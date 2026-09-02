# NIRSpec/PRISM

The repository PRISM example uses native high-resolution channels and an R=20 low-resolution stage:

```bash
cp configs_fiducial_stellarinformed/HAT-P-65_nrs1_prism_v1_config.yaml prism.yaml
python fit_jwst.py -c prism.yaml
```

```yaml
instrument: NIRSPEC/PRISM
nrs: 1
fits_file: HAT-P-65_nrs1_box_spectra_fullres_PRISM_V1.fits
resolution: {high: native, low: 20}
flags:
  detrending_type: explinear
  need_lowres: true
```

PRISM spans a large dynamic range. Remove saturated wavelengths in the extraction or with the configured wavelength filters/masks before fitting. Use `resolution.high: 20`, `100`, or `300` to bin at constant resolving power, or `native` to retain input channels. `need_lowres: true` runs the R=20 stage before the high-resolution stage. For staged cluster runs, use `analysis_stage: prep` to finish preparation/low resolution and `analysis_stage: highres` for the final stage.

