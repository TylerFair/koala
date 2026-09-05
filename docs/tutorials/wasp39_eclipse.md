# WASP-39 b secondary eclipse with NIRSpec/G395H

This worked observation fits the NRS1 and NRS2 secondary-eclipse time series
separately and joins their final $R=300$ planet/star flux ratios. It uses the
same white-light, coarse spectroscopic, and final spectroscopic stages as a
transit fit, with JAXoplanet's uniform-emission eclipse model.

```{image} ../_static/wasp39_g395h_eclipse_r300.png
:alt: WASP-39 b NIRSpec G395H secondary-eclipse emission spectrum at resolving power 300
:width: 820px
:align: center
```

The completed spectrum contains 154 finite bins: 74 from NRS1 and 80 from
NRS2. The white-light eclipse depths were $771.2^{+15.0}_{-14.8}$ ppm for NRS1
and $1288.4^{+17.1}_{-18.5}$ ppm for NRS2. These broadband values differ
because the detectors cover different wavelength ranges; they are not a
shared gray eclipse depth. The plotted channel medians span about 305--1895
ppm.

## Observation-specific setup

Start from the checked [NRS1](../../examples/wasp39_eclipse_nrs1.yaml) and
[NRS2](../../examples/wasp39_eclipse_nrs2.yaml) configurations. Point each at
its matching extracted FITS file, then run them independently:

```bash
python fit_jwst.py -c examples/wasp39_eclipse_nrs1.yaml
python fit_jwst.py -c examples/wasp39_eclipse_nrs2.yaml
```

The FITS `TIME` arrays are MJD TDB. The observed secondary-eclipse midpoint
was estimated as 60839.40350 MJD TDB. The model's `planet.t0` is still the
**primary-transit** epoch, even for `light_curve_model: eclipse`. For the fixed
circular orbit,

$$
t_0 = 60839.40350 - \frac{4.05528043}{2}
    = 60837.375859785\ \mathrm{MJD\ TDB}.
$$

Putting the eclipse midpoint directly in `planet.t0` would move the modeled
eclipse by half an orbit. The science fits use `fit_geometry: false`: the
current switch cannot fit only the eclipse time while holding $b$, $R_p/R_*$,
and $a/R_*$ fixed. Consequently, the reported depth uncertainties are
conditional on the adopted midpoint and geometry. A timing-sensitivity rerun
at roughly $t_\mathrm{ecl}\pm0.003$ day is appropriate when that uncertainty
matters.

```yaml
planet:
  period: 4.05528043
  duration: 0.11693087083333333
  t0: 60837.375859785
  b: 0.4498
  rprs: 0.1457
  ecc: 0.0
  omega: 0.0
  eclipse_depth_ppm: 1000.0
  eclipse_depth_prior_width_ppm: 1000.0

flags:
  light_curve_model: eclipse
  fit_geometry: false
  transit_engine: jaxoplanet
```

A positive prior width infers `eclipse_depth_ppm` with the implementation's
normal prior truncated at zero. This keeps the planet flux physical, but it
also means weak channels cannot express negative noise excursions. Inspect
posterior asymmetry and prior sensitivity before interpreting marginal
features.

## Limb darkening, binning, and masks

A secondary eclipse hides the planet behind the star, so stellar limb
darkening is not inferred. The fixed zero coefficients supply the uniform
stellar array required by the common model interface without adding fitted LD
parameters:

```yaml
stellar:
  ld_coefficients: [0.0, 0.0]
flags:
  ld_profile: quadratic
  ld_prior: fixed
```

`resolution.low: 20` is a coarse bridge used to stabilize the handoff to the
spectroscopic fit. It is not the published spectrum. `resolution.high: 300`
creates the final constant-resolving-power product.

Both detectors exclude the first 0.007 day of settling and use conservative
10-sigma white-light and per-channel residual clipping. NRS1 also masks four
pathological native pixels. One at 3.01838 microns and three around 3.498
microns toggled between incompatible flux states throughout the visit; without
these wavelength masks they produced two false white-light populations.

```yaml
wavelength_masks: [3.0178, 3.0190, 3.4970, 3.4990]

resolution:
  low: 20
  high: 300

flags:
  detrending_type: linear
  mask_start: jnp.min(t)
  mask_end: jnp.min(t) + 0.007

outlier_clip:
  whitelight_sigma: 10
  spectroscopic_sigma: 10
```

The wavelength mask removes 4 of 1548 finite wavelength pixels in the input
NRS1 FITS extraction and keeps all otherwise usable integrations. NRS2 does not need this mask.

## Quality checks and interpretation

Both white-light fits passed the configured quality gates with zero
divergences and eclipse-depth bulk ESS above 1000. All 80 NRS2 channels and 69
of 74 NRS1 channels completed with independent NUTS. Five NRS1 lanes that did
not pass the first attempt were replaced by the pipeline's zero-divergence
independent-HMC fallback results. The combined table has finite estimates and
positive interval widths in every retained bin.

Treat this as a single fixed-geometry reduction rather than a complete error
budget. A linear visit trend was used for both detectors. Correlated residuals,
trend choice, the fixed timing and geometry, detector-to-detector calibration,
and the nonnegative depth prior can all contribute uncertainty beyond the
reported channel intervals. Inspect the white-light residuals, wavelength-time
summaries, and noise-binning plots before assigning atmospheric significance
to spectral structure. In particular, joining the detectors by wavelength
does not fit or marginalize a detector offset.

The per-detector science products are named
`WASP-39_NIRSPEC_G395H_nrs1_R300_emission.csv` and
`WASP-39_NIRSPEC_G395H_nrs2_R300_emission.csv`. Concatenate them with a detector
label, sort by wavelength, and retain the source filenames when making the
combined table. The relevant output column is `eclipse_depth_ppm`, the
planet/star flux ratio in parts per million.
