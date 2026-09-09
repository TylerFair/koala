# HAT-P-18 b: modelling a starspot crossing

The NIRISS/SOSS transit of HAT-P-18 b contains a spot-crossing event: a
short brightening near mid-transit where the planet covers a dark region of
the star. Instead of removing it with a template in the trend, this tutorial
puts a spot on the stellar surface and fits its contrast with Koala's
JAXoplanet/starry surface model.

## Configuration

Start from `examples/hatp18_soss_starspot.yaml`:

```yaml
planet:
  name: HAT-P-18
  period: 5.50802941
  t0: 59743.353393          # BMJD_TDB
  duration: 0.11            # days; a_rs is derived from it
  b: 0.148
  rprs: 0.1356
  ecc: 0.0
  omega: 0.0

stellar:
  teff: 4710
  logg: 4.17
  feh: 0.06
  teff_sigma: 238
  logg_sigma: 0.58
  feh_sigma: 0.08
  ld_model: stagger
  ld_data_path: exotic_ld_data
  rotation_period: 25.0     # days
  spots:
    - latitude_deg: -8.5    # on the transit chord for b = 0.148
      longitude_deg: 4.5    # at t0; positive is crossed after mid-transit
      radius_deg: 8.0
      contrast: 0.15
      contrast_prior_width: 0.15

instrument: NIRISS/SOSS
order: 1

path: /path/to/analysis
input_dir: FITS
fits_file: HAT-P-18_box_spectra_fullres.fits
output_dir: HAT-P-18_SOSS_ORDER1_STARSPOT

resolution:
  high: 100

wavelength_masks: [0.853, 0.87, 1.048, 1.061, 1.366, 1.384, 1.972, 2.011]

flags:
  transit_engine: jaxoplanet
  light_curve_model: transit
  detrending_type: linear
  ld_profile: power2
  ld_prior: stellarprior
  mask_start: [59743.15, 59743.47]
  mask_end: [59743.225, 59743.55]

outlier_clip:
  whitelight_sigma: 5
  spectroscopic_sigma: 5

host_device: gpu
```

The spot's position and size are fixed; only its `contrast` is fitted, with
a prior of width `contrast_prior_width` truncated to the range 0 (photosphere)
to 1 (dark). The spot sits on the transit chord: for an impact parameter of
0.148 the planet crosses the star at a latitude of about −8.5°, and a
longitude of 4.5° at `t0` places the crossing a few minutes after
mid-transit, where the bump is seen in the data. `stellar.rotation_period` is required but barely
matters over one transit. There is no `resolution.low`, so the pipeline
goes straight from the white-light fit to the R = 100 spectrum.

## Run

```bash
python fit_jwst.py -c hatp18_starspot.yaml
```

## White-light fit

<!-- TODO figure: generate with
     python tools/docs/hatp18_starspot_figures.py /path/to/HAT-P-18_SOSS_ORDER1_STARSPOT
     then uncomment.
```{image} ../_static/hatp18_starspot_whitelight.png
:alt: HAT-P-18 b SOSS white-light transit with the fitted spotted-star model and residuals
:width: 760px
:align: center
```
-->

The surface model reproduces the crossing as the planet passes over the
spot, and the residuals around mid-transit are flat rather than showing the
bump that a spot-free transit leaves behind.

## Transmission spectrum and spot contrast

<!-- TODO figure: hatp18_starspot_spectrum.png from the same script.
```{image} ../_static/hatp18_starspot_spectrum.png
:alt: HAT-P-18 b SOSS order-1 transmission spectrum with the spot modelled
:width: 760px
:align: center
```
-->

<!-- TODO figure: hatp18_starspot_contrast.png from the same script.
```{image} ../_static/hatp18_starspot_contrast.png
:alt: Fitted spot contrast as a function of wavelength
:width: 760px
:align: center
```
-->

Each wavelength channel fits its own contrast, written to
`*_stellar_spots.csv`. A cool spot is darker at short wavelengths, so the
contrast should fall with wavelength; the transmission spectrum in
`*_R100.csv` is measured with that chromatic crossing already in the model.
