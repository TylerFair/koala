# Rocky-planet secondary eclipse with MIRI/LRS

A secondary eclipse of a small, cool planet is a shallow dip of order 100
parts per million, and its timing is rarely known to better than an hour or
two. This tutorial fits a synthetic MIRI/LRS eclipse of a rocky planet,
"ROCKY-1 b", whose orbit and fit setup mirror the JWST/MIRI 15 µm eclipse
analysis of GJ 3929 b ([arXiv:2508.12516](https://arxiv.org/abs/2508.12516)).
The data are generated locally, so every recovered quantity can be checked
against what was injected and no download is needed.

## Configuration

`examples/eclipse.yaml` is the complete configuration. The planet block is
the part that encodes the analysis choices:

```yaml
planet:
  name: ROCKY-1
  period: {value: 2.6162644, prior: fixed}
  eclipse_time: {value: 60001.3081322, prior: uniform, low: 60001.2231322, high: 60001.3931322}
  a_rs: {value: 17.04, prior: gaussian, sigma: 0.5}
  b: {value: 0.208, prior: gaussian, sigma: 0.15, low: 0.0, high: 1.0}
  rprs: {value: 0.0318, prior: fixed}
  ecc: {value: 0.0, prior: fixed}
  omega: {value: 0.0, prior: fixed}
  eclipse_depth_ppm: {value: 100.0, prior: uniform, low: 0.0, high: 1000.0}

stellar:
  ld_coefficients: [0.0, 0.0]

flags:
  light_curve_model: eclipse
  transit_engine: jaxoplanet
  detrending_type: linear
  ld_profile: quadratic
  ld_prior: fixed
```

Every entry is a `{value, prior, ...}` mapping (see
[Parameter priors](../guides/configuration.md#parameter-priors)), so what
is fitted and what is held is visible at a glance:

- **`eclipse_time` is free, uniform in a window.** For an eclipse-only visit
  give the secondary-eclipse mid-time instead of `t0`; koala derives the
  transit epoch as `eclipse_time - period / 2` for the circular orbit and
  reports both. The uniform range spans ±0.085 d (about two hours) around
  the predicted time, as in the GJ 3929 b analysis, so the fit locates the
  eclipse rather than assuming its timing.
- **`eclipse_depth_ppm` is free, uniform from 0 to 1000 ppm.** A wide flat
  prior lets the data set the depth and keeps the posterior honest when the
  eclipse is marginal. The injected depth is 100 ppm in every channel.
- **`a_rs` and `b` are gaussian, from the literature.** The scaled semi-major
  axis is 17.04 ± 0.5 and the impact parameter 0.208 ± 0.15, the latter
  truncated to 0 to 1; these are the paper's a/R* and inclination
  (89.3 ± 0.5°, with b = a/R* cos i). They set the eclipse duration and
  shape, which an eclipse alone constrains weakly, so the prior does the work.
- **`period` and `rprs` are fixed.** The period is known to far better
  precision than one visit can improve, and the radius ratio only enters an
  eclipse through the shape of ingress and egress.

Because `b` and `a_rs` are free, the geometry is not fixed, which is why
those priors must be informative. Had every geometry key been `fixed`, koala
would hold the orbit at the given values and fit only the depth and trend.
The star is not limb-darkened during an eclipse, so the coefficients are
fixed to zero.

## Run

Generate the synthetic observation, then fit it:

```bash
python tools/example_phase_curves.py eclipse
python fit_jwst.py -c examples/eclipse.yaml
```

The generator writes `examples/data/generated/synthetic_eclipse.fits`, a
181-cadence visit spanning 0.15 d either side of the eclipse in 14 channels
between 5.5 and 11.5 µm with 120 ppm noise per point, and a
`synthetic_eclipse_truth.json` with the injected values. The white-light
stage reports the fitted eclipse time and the broadband depth; the
spectroscopic stages write `*_emission.csv` with an `eclipse_depth_ppm`
column per channel.

## Result

The white-light eclipse time lands on the injected 60001.3081 within its
uncertainty, and the broadband depth recovers 100 ppm to within a few ppm.
Each channel's depth is consistent with the flat 100 ppm injection, with
per-channel uncertainties of a few tens of ppm set by the 120 ppm noise and
the roughly 40 in-eclipse cadences. Compare `*_emission.csv` with
`synthetic_eclipse_truth.json` to confirm the recovery.
