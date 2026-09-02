# NIRISS/SOSS order 1

Start from the checked-in WASP-39 configuration:

```bash
cp configs_fiducial_stellarinformed/WASP-39_soss_order1_config.yaml wasp39.yaml
python fit_jwst.py -c wasp39.yaml
```

Its defining entries are:

```yaml
instrument: NIRISS/SOSS
order: 1
fits_file: WASP-39_box_spectra_fullres.fits
resolution: {high: reference, low: 20, reference_grid: prism_template.csv}
flags:
  need_lowres: true
  ld_profile: power2
  ld_prior: stellarprior
  detrending_type: linear
```

Use `order: 2` with the corresponding order-2 configuration and extraction. Orders are fitted separately. The reference grid is used for high-resolution binning when `high: reference`; `low: 20` supplies the calibration stage.

