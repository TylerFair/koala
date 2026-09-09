# WASP-39 b secondary eclipse with NIRSpec/G395H

This tutorial fits the NRS1 and NRS2 secondary-eclipse time series
separately with JAXoplanet's uniform-emission eclipse model and joins their
planet/star flux ratios into one emission spectrum.

## Configuration

Start from `examples/wasp39_eclipse_nrs1.yaml` and
`examples/wasp39_eclipse_nrs2.yaml`. The eclipse-specific settings are:

```yaml
planet:
  period: 4.05528043
  t0: 60837.375859785     # primary-transit epoch, not the eclipse midpoint
  b: 0.4498
  rprs: 0.1457
  duration: 0.11693087083333333
  ecc: 0.0
  omega: 0.0
  eclipse_depth_ppm: 800.0
  eclipse_depth_prior_width_ppm: 1000.0

stellar:
  ld_coefficients: [0.0, 0.0]

resolution:
  low: 20
  high: 300

flags:
  light_curve_model: eclipse
  fit_geometry: false
  transit_engine: jaxoplanet
  detrending_type: linear
  ld_profile: quadratic
  ld_prior: fixed
```

`planet.t0` is the primary-transit epoch even for an eclipse fit: take the
observed eclipse midpoint and subtract half a period. The star is not
limb-darkened during an eclipse, so the coefficients are fixed to zero.
NRS1 additionally masks a few misbehaving native pixels with
`wavelength_masks`.

## Run

```bash
python fit_jwst.py -c examples/wasp39_eclipse_nrs1.yaml
python fit_jwst.py -c examples/wasp39_eclipse_nrs2.yaml
```

Each run writes `*_R300_emission.csv` with the `eclipse_depth_ppm` column;
concatenate the two tables by wavelength to build the figure below.

## Result

```{image} ../_static/wasp39_g395h_eclipse_r300.png
:alt: WASP-39 b NIRSpec G395H secondary-eclipse emission spectrum at resolving power 300
:width: 820px
:align: center
```

The dayside flux ratio rises across the G395H bandpass as the planet's thermal
emission grows relative to the star. The two detectors are fitted with
independent broadband depths, so joining them does not fit a detector offset,
and every channel is conditional on the adopted eclipse timing and geometry.
