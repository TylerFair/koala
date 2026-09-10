# Synthetic thermal phase curve

A full-orbit phase curve constrains the planet's day-night brightness
contrast and the longitude of its hottest point. This tutorial fits a
synthetic NIRSpec/G395M phase curve with Koala's JAXoplanet/starry surface
model, so every recovered quantity can be compared with what was injected.
No download is needed.

## Configuration

`examples/phase_curve.yaml` describes one orbit with a transit, a secondary
eclipse, and a degree-one thermal map. The three surface quantities each have
a centre and a prior width; a positive width makes the quantity a fitted
parameter.

```yaml
planet:
  period: 2.0
  t0: 60000.0
  b: 0.25
  rprs: 0.10
  a_rs: 6.0
  dayside_flux_ppm: 1000.0
  dayside_flux_prior_width_ppm: 150.0
  nightside_flux_ppm: 300.0
  nightside_flux_prior_width_ppm: 75.0
  hotspot_offset_deg: 18.0
  hotspot_offset_prior_width_deg: 10.0

flags:
  light_curve_model: phase_curve
  fit_geometry: false
  transit_engine: jaxoplanet
  detrending_type: linear
  ld_profile: quadratic
  ld_prior: fixed
```

## Run

Generate the synthetic observation, then fit every channel:

```bash
python tools/example_phase_curves.py phase_curve
python tools/surface_publication_example.py phase_curve --output-dir results/phase_curve
```

The second command runs the same likelihood and priors as `fit_jwst.py`
for the 14 channels between 2.9 and 5.0 µm and writes the figures below.
`examples/phase_curve.yaml` can also be given to `fit_jwst.py` directly.

## Planet flux around the orbit

```{image} ../_static/phase_curve_lightcurve.png
:alt: Synthetic phase curve at 4.03 micron with the fitted model and residuals
:width: 640px
:align: center
```

The top panel is one channel's light curve against orbital phase with the
posterior model: the planet's flux rises from the nightside minimum after
transit to the dayside maximum around secondary eclipse, and the maximum
arrives after phase 0.5 because the hotspot is offset east. The residuals
bin down close to white noise.

## Dayside, nightside, and hotspot offset per wavelength

```{image} ../_static/phase_curve_recovery.png
:alt: Recovered dayside flux, nightside flux, and hotspot offset per channel against the injected values
:width: 620px
:align: center
```

Each channel recovers the injected dayside flux (rising with wavelength),
the constant nightside flux, and the hotspot offset within its 68 percent
interval. These per-channel values are the emission spectrum and the
offset spectrum, written to `spectrum_emission.csv` as
`dayside_flux_ppm`, `nightside_flux_ppm`, and `hotspot_offset_deg` with
their uncertainties.

## The thermal map

```{image} ../_static/phase_curve_map.png
:alt: Posterior median dipole brightness map and its pointwise uncertainty
:width: 640px
:align: center
```

The map is the degree-one brightness distribution implied by the posterior
at 4.03 µm: the bright spot sits east of the substellar point by the fitted
offset, and the lower panel shows where the map is least certain. The
posterior and map are saved as `posterior.npz` and `thermal_map.npz`.
