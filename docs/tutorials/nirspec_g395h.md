# NIRSpec/G395H

NRS1 and NRS2 are separate inputs and fits:

```bash
cp configs_fiducial_stellarinformed/WASP-121_nrs1_g395h_config.yaml wasp121_nrs1.yaml
python fit_jwst.py -c wasp121_nrs1.yaml
```

```yaml
instrument: NIRSPEC/G395H
nrs: 1
fits_file: WASP-121_nrs1_box_spectra_fullres_G395H.fits
resolution: {high: reference, low: 20, reference_grid: prism_template.csv}
flags:
  detrending_type: quadratic
  need_lowres: true
  ld_profile: power2
  ld_prior: stellarprior
```

Set `nrs: 2` and use the NRS2 FITS file for the long-wavelength detector. Do not combine detector files in one configuration. G395M and G140H use the same detector key and stage structure with their corresponding `instrument` strings.

