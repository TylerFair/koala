# Rocky-planet secondary eclipse with MIRI/LRS

This tutorial fits a fully synthetic MIRI/LRS eclipse of a rocky planet,
"ROCKY-1 b". The observations are generated locally: a flat 100 ppm eclipse
in 14 channels from 5.5 to 11.5 µm, with 120 ppm Gaussian noise per point
per channel.

The workflow is **fit white light → fix geometry → fit native-channel depths**.
The spectral stage also fits a baseline and noise for each channel. No
low-resolution fit is performed.

## Full configuration

For reproducibility, here is the full `examples/eclipse.yaml`, repeated here
so it can be copied directly:

```yaml
# Synthetic rocky-planet secondary eclipse ("ROCKY-1 b") with MIRI/LRS.
# Generate the synthetic input from the repo root with:
#   python tools/example_phase_curves.py eclipse
planet:
  # Every planet parameter is {value, prior[, sigma, low, high]}; prior is fixed, uniform, log_uniform, or gaussian.
  name: ROCKY-1
  period: {value: 2.6162644, prior: fixed}
  # Secondary-eclipse mid-time (BMJD_TDB) searched over +/- 0.085 d; koala
  # derives the transit epoch t0 = eclipse_time - period / 2 (circular orbit).
  eclipse_time: {value: 60001.3081322, prior: uniform, low: 60001.2231322, high: 60001.3931322}
  a_rs: {value: 17.04, prior: gaussian, sigma: 0.5}
  b: {value: 0.208, prior: gaussian, sigma: 0.15, low: 0.0, high: 1.0}   # i = 89.3 +/- 0.5 deg
  rprs: {value: 0.0318, prior: fixed}
  ecc: {value: 0.0, prior: fixed}
  omega: {value: 0.0, prior: fixed}
  eclipse_depth_ppm: {value: 100.0, prior: uniform, low: 0.0, high: 1000.0}

stellar:
  # The star is not limb-darkened during an eclipse.
  ld_coefficients: [0.0, 0.0]

instrument: MIRI/LRS
path: .
input_dir: examples/data/generated
input_file: synthetic_eclipse.fits
output_dir: results/synthetic_eclipse

resolution:
  # The synthetic file has 14 channels (R ~ 18), so no coarse low stage.
  high: native

flags:
  light_curve_model: eclipse
  transit_engine: jaxoplanet
  detrending_type: linear
  ld_profile: quadratic
  ld_prior: fixed
  random_seed: 20260904
  spectro_chunk_size: 4
  # The recovery shown in the tutorial uses 4,000 retained draws per channel.
  highres_num_samples: 4000

outlier_clip:
  whitelight_sigma: 5
  spectroscopic_sigma: 5

host_device: cpu
```

Every entry is a `{value, prior, ...}` mapping (see
[Parameter priors](../guides/configuration.md#parameter-priors)), so what
is fitted and what is held is visible at a glance:

- **`eclipse_time` is free, uniform in a window.** For an eclipse-only visit
  give the secondary-eclipse mid-time instead of `t0`; koala derives the
  transit epoch as `eclipse_time - period / 2` for the circular orbit and
  reports both. The uniform range spans ±0.085 d (about two hours) around
  the predicted time, allowing the fit to recover the eclipse timing.
- **`eclipse_depth_ppm` is free, uniform from 0 to 1000 ppm.** A wide flat
  prior lets the data set the depth and keeps the posterior honest when the
  eclipse is marginal. The injected depth is 100 ppm in every channel.
- **`a_rs` and `b` have gaussian priors.** The scaled semi-major axis is
  17.04 ± 0.5 and the impact parameter 0.208 ± 0.15, the latter truncated
  to 0 to 1. These synthetic geometry parameters set the eclipse duration
  and shape, which a shallow eclipse alone constrains weakly.
- **`period` and `rprs` are fixed at their injected values.** The radius
  ratio enters the eclipse through the shape of ingress and egress.

In white light, `b` and `a_rs` are free, so their priors must be informative.
For spectroscopy, koala fixes the orbit to the white-light posterior-median
geometry and holds `rprs` at 0.0318. Only eclipse depth, baseline, and noise
are fitted per channel.
The star is not limb-darkened during an eclipse, so the coefficients are
fixed to zero.

## Run

Generate the synthetic observation, then fit it:

```bash
python tools/example_phase_curves.py eclipse
python fit_jwst.py -c examples/eclipse.yaml
```

The example defaults to CPU execution. For a GPU run, set `host_device: gpu`
and run the same commands in a GPU-enabled environment on an allocated GPU.
The figures below were produced on a Quadro RTX 6000; the full pipeline took
166 seconds, excluding queue time and the additional comparison plots.

The generator writes `examples/data/generated/synthetic_eclipse.fits`, a
181-cadence visit spanning 0.15 d either side of the eclipse in 14 channels
between 5.5 and 11.5 µm with 120 ppm noise per point, and a
`synthetic_eclipse_truth.json` with the injected values. The white-light
stage reports the fitted eclipse time and the broadband depth. The example
omits `resolution.low` and sets `resolution.high: native`, so it skips the
low-resolution fit and fits all 14 input channels individually. The native
spectroscopic stage writes `*_emission.csv` with an `eclipse_depth_ppm`
column per channel, conditional on the white-light geometry estimate.

## Synthetic eclipse and recovered light curve

```{image} ../_static/rocky_eclipse/eclipse_recovery.png
:alt: Synthetic white-light eclipse, injected 100 ppm model, recovered model, and residuals
:width: 760px
:align: center
```

The September 10, 2026 GPU run recovers a white-light depth of **94.6 ppm**,
with a 68.27% interval of **87.4–101.7 ppm**, containing the 100 ppm injection.
The recovered midpoint differs from the injection by **+0.13 min**, with an
interval of **−0.84 to +1.15 min**. White-light residual RMS is **32.5 ppm**.
The flux convention places the stellar baseline at zero ppm and the
out-of-eclipse planet contribution near 100 ppm.

[Light-curve PDF](../_static/rocky_eclipse/eclipse_recovery.pdf) ·
[White-light data and model CSV](../_static/rocky_eclipse/whitelight_timeseries.csv)

## Recovered native emission spectrum

```{image} ../_static/rocky_eclipse/spectrum_recovery.png
:alt: All 14 native-channel eclipse depths and 68 percent intervals compared with the injected flat 100 ppm spectrum
:width: 760px
:align: center
```

Each point is the posterior median and equal-tailed 68.27% interval for one
input channel. Eleven of the 14 intervals contain the injected 100 ppm;
individual noisy channels need not contain the truth in every 68% interval.
The median interval half-width is **25.0 ppm**. These spectral intervals are
conditional on the fixed white-light geometry; they do not propagate its
uncertainty.

[Spectrum PDF](../_static/rocky_eclipse/spectrum_recovery.pdf) ·
[Recovered emission CSV](../_static/rocky_eclipse/emission.csv) ·
[Injected properties](../_static/rocky_eclipse/truth.json)

## Posterior corner plots

```{image} ../_static/rocky_eclipse/white_light_corner.png
:alt: White-light posterior for eclipse depth, midpoint offset, scaled semi-major axis, and impact parameter, with injected values marked
:width: 760px
:align: center
```

The white-light corner shows the depth, timing offset from injection,
scaled semi-major axis, and impact parameter. Blue lines mark the injected
values. These geometry parameters are estimated here and then fixed for
the spectral fit.

```{image} ../_static/rocky_eclipse/channel_corner.png
:alt: Fixed-geometry 5.5 micron channel posterior for eclipse depth, baseline offset, and linear slope
:width: 680px
:align: center
```

The 5.5 µm channel corner shows the remaining depth/baseline/slope
correlations. The baseline axis is `(c - 1) × 10⁶`; its offset includes the
pipeline's flux normalization. Geometry and radius do not vary in this
spectral posterior. Jitter is also sampled but is omitted from this plot.

[White-light corner PDF](../_static/rocky_eclipse/white_light_corner.pdf) ·
[Channel corner PDF](../_static/rocky_eclipse/channel_corner.pdf)

## Sampling checks

White light retained 2,000 draws, extending the same warmed chain from
1,000 draws when timing ESS was 224. The extension raised it to 401,
passing the ESS ≥ 400 threshold without restarting. The white-light depth
ESS was 2,394, with zero divergences and no tree-depth saturation.

Spectroscopy retained 4,000 draws per channel. All 14 channels passed with
independent NUTS, zero divergences, and minimum depth ESS **976**; no sampler
fallback was needed. The saved samples confirm a constant radius ratio of
0.0318 in every spectral draw. Each fit uses one chain, so these checks do
not provide an independent-chain convergence comparison.

[Channel diagnostics CSV](../_static/rocky_eclipse/channel_diagnostics.csv) ·
[White-light diagnostics](../_static/rocky_eclipse/whitelight_diagnostics.json) ·
[Recovery summary](../_static/rocky_eclipse/recovery.json)
