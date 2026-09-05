# Eclipses, phase curves, and stellar spots

For an executed walkthrough with emission spectra, a thermal map and its
uncertainty, corner plots, and residual diagnostics, see
[Synthetic surface analyses](../tutorials/synthetic_surfaces.md).

JAXoplanet fits can include a secondary eclipse, a full thermal phase curve,
and rotating circular stellar spots. The choice is intentionally small:

```yaml
flags:
  light_curve_model: phase_curve  # transit, eclipse, or phase_curve
  fit_geometry: false             # keep supplied geometry fixed
```

Stellar spots are added separately under `stellar`, so the same spot model can
be used with a transit, eclipse, or phase curve. These surface models are not
available with Harmonica.

## Run a repeatable example

From the repository root, generate all three small synthetic datasets:

```bash
python tools/example_phase_curves.py
```

The generator calls the same JAXoplanet/Starry surface evaluator used during a
fit, adds a fixed random-noise realization, and writes both a FITS file and its
injection values beneath `examples/data/generated`. It does not need an
external limb-darkening grid. Run any example in the usual way:

```bash
python fit_jwst.py -c examples/eclipse.yaml
python fit_jwst.py -c examples/phase_curve.yaml
python fit_jwst.py -c examples/stellar_spots.yaml
```

The examples use `host_device: cpu` and 14 wavelength channels for portability.
Change it to `gpu` for a real observation. The normal Laplace-metric NUTS
defaults, batched wavelength channels, checkpoints, and convergence gates are
used without extra sampler controls. Surface-model validation used JAX 0.6.2,
NumPyro 0.19.0, and JAXoplanet 0.1.0.

All three published configurations have also completed the full white-light,
low-resolution, and native-resolution pipeline on CPU with the default sampler
settings. Every retained wavelength result passed the default ESS threshold
of 400 and zero-divergence gate. The one-spot example spans a rotation cycle
and a spot crossing to distinguish contrast from the instrumental baseline;
it recovered a median contrast of 0.39999 for an injected 0.4.

The generated examples set `fit_geometry: false` so they isolate recovery of
the injected emission or spot contrast and reduce the inference dimension.
Omit it or set it to `true` when the observation contains enough transit
information to infer `t0`, radius ratio, impact parameter, and `a_rs` together
with the surface.
Eclipse-only fits default to fixed geometry because they usually do not contain
that information.

## Eclipse

An eclipse fit needs one planet/star flux ratio in ppm and, if it is to be
inferred, a prior width:

```yaml
planet:
  period: 2.0
  t0: 60000.0
  b: 0.25
  rprs: 0.10
  a_rs: 6.0
  ecc: 0.0
  omega: 0.0
  eclipse_depth_ppm: 900
  eclipse_depth_prior_width_ppm: 250

flags:
  light_curve_model: eclipse
  transit_engine: jaxoplanet
```

The flux is `1 + eclipse_depth` outside secondary eclipse and returns to the
stellar baseline while the planet is hidden. JAXoplanet evaluates the exact
native occultation of a uniform-brightness planet on the Keplerian orbit. A
zero or omitted prior width fixes the configured value; a positive width
infers it with a nonnegative prior.

## Thermal phase curve

The thermal model uses the planet/star flux at the day and night extrema plus
the longitudinal offset of the brightest region:

```yaml
planet:
  dayside_flux_ppm: 1000
  dayside_flux_prior_width_ppm: 150
  nightside_flux_ppm: 300
  nightside_flux_prior_width_ppm: 75
  hotspot_offset_deg: 18
  hotspot_offset_prior_width_deg: 10

flags:
  light_curve_model: phase_curve
  transit_engine: jaxoplanet
```

Positive hotspot offset moves the phase maximum later than secondary eclipse.
The current degree-one thermal map remains nonnegative only when the larger of
the day/night fluxes is at most five times the smaller. Invalid starting values
stop immediately with a direct configuration error.

When either flux is inferred, both configured starting fluxes must be strictly
positive. The sampler enforces the ratio through a smooth conditional-quantile
parameterization rather than a hard likelihood wall. Its interval-mass
correction preserves the original independent nonnegative normal priors
conditioned on a physical map, while preventing trajectories from entering a
negative-intensity region.

The phase map is a smooth native Starry `Y00 + Y10` dipole viewed equator-on.
It rotates uniformly with the configured mean period; for an eccentric orbit,
it is a photometric brightness model rather than a tidal-spin prescription.
The configured day and night values are its exact disk-integrated extrema.

Use data that cover enough orbital phase to distinguish the day, night, and
offset terms. Do not use `cut_phase_to_transit`; the fitter rejects that option
for surface models because it would remove the needed signal.

## Physical stellar spots

A spot is described by its fixed position and angular radius. Set a positive
`contrast_prior_width` to infer its contrast between zero (photosphere) and one
(dark):

```yaml
stellar:
  rotation_period: 2.0
  spots:
    - latitude_deg: 15
      longitude_deg: 0
      radius_deg: 20
      contrast: 0.40
      contrast_prior_width: 0.15
```

JAXoplanet/Starry projects the circular spot on the rotating stellar surface
and evaluates both rotational modulation and spot crossings. In this first
surface model, latitude, longitude, radius, and rotation period are fixed;
contrast is the inferred spot quantity. Add another short mapping to `spots`
for each additional spot. Spot longitude is defined at the configured transit
epoch `t0`.

Circular spots are finite-degree smooth spherical-harmonic approximations and
their maps add linearly. Avoid overlapping high-contrast spots and very small
spots at the default degree: either can produce ringing or negative local
intensity. The moderate spot and long rotation baseline in the example are a useful
starting point.

For a self-contained synthetic fit, the examples use known quadratic
limb-darkening coefficients:

```yaml
stellar:
  ld_coefficients: [0.3, 0.2]
flags:
  ld_profile: quadratic
  ld_prior: fixed
```

For real data, remove `stellar.ld_coefficients` and configure the usual
stellar-atmosphere limb-darkening inputs instead.
With `ld_profile: power2`, surface fits use the fitter's existing polynomial
approximation to that law, as Starry consumes polynomial limb darkening.

## Bounded inference validation

Run a short recovery test through the production vectorized model and the same
Laplace-metric NUTS initialization used by the science pipeline:

```bash
python tools/validate_surface_models.py all --draws 500
```

This validation fixes the known orbital geometry and limb darkening, then
infers the eclipse depth, all three phase-curve quantities, or stellar-spot
contrast together with normalization. It writes a JSON report and the surface
CSV/PNG products under `results/surface_validation`. A successful command
requires no divergences, effective sample size of at least 50 for every tested
surface parameter, and recovery of every injected value in its reported test
interval.

In the reference CPU run with 500 posterior draws, eclipse recovery returned
1197.7 ppm for an injected 1200 ppm (ESS 478.8), and phase-curve recovery
returned 1210.5/419.9 ppm and 20.77 degrees for injected 1200/400 ppm and 20
degrees (minimum ESS 434.7). Both runs had zero divergences. The generated JSON
reports retain the exact intervals and timing breakdown for each new run.

Including basis preparation, Laplace preparation, and NUTS, eclipse runtime
fell from 23.26 to 6.29 seconds and the final smoothly constrained phase-curve
run took 19.49 seconds, compared with 66.62 seconds for the native reference.
Separate numerical tests compare the basis and native evaluator directly,
including all science-parameter gradients and eccentric geometry, to below
`4e-12` in relative flux. A change of sampler coordinates can change finite
Monte Carlo draws even though the corrected target prior is the same; the
prior tests therefore also verify the transformed density and its Jacobian.
An additional boundary run with a 1200/250 ppm day/night ratio of 4.8 used 500
draws with zero divergences, recovered all injected values, and had minimum
ESS 176.8. It is repeatable with
`--phase-nightside-ppm 250` on the validation command.

The native degree-eight stellar-spot recovery inferred 0.400027 for an
injected contrast of 0.4 (ESS 465.0, zero divergences). Its full Starry graph
required 323.77 seconds for Laplace preparation and 107.37 seconds for NUTS,
including compilation. With fixed geometry and fixed limb darkening, the exact
spot basis described below produced the same median, interval, ESS, and
divergence count to reported precision. It required 14.26 seconds to prepare
the native basis once, 5.28 seconds for Laplace preparation, and 2.44 seconds
for NUTS including compilation. The same random seeds and 500 draws were used.

For direct numerical tests of ingress/egress, eccentric eclipse timing,
automatic derivatives, phase extrema, spot rotation, and stock-JAXoplanet
parity, run:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest -q \
  tests/test_jaxoplanet_surface.py tests/test_jaxoplanet_surface_basis.py
```

## Geometry and outputs

Surface models currently support one planet and automatically use the
Keplerian `a_rs` geometry. Provide `period`, `t0`, `b`, `rprs`, and `a_rs`;
`t0` remains the transit center even for an eclipse-only dataset. `ecc` and
`omega` default to zero, and the legacy `planet.omega` value is in radians.
Fields ending in `_deg` use degrees. Flux ratios in the YAML and output tables
are ppm, so no manual unit conversion is needed. The thermal map is viewed
equator-on. Stellar spots use the configured stellar inclination, which
defaults to equator-on.

The standard white-light time series, best-fit parameters, and diagnostics are
still written. Eclipse and phase-curve runs write `*_emission.csv` and
`*_emission.png`; eclipse-only runs omit a meaningless transmission-radius
spectrum. The emission CSV contains `wavelength`,
`wavelength_err`, and `planet_index`, plus the available quantities among:

- `eclipse_depth_ppm`
- `dayside_flux_ppm`
- `nightside_flux_ppm`
- `hotspot_offset_deg`

Each quantity has matching `_err_low` and `_err_high` columns for its 68.27%
posterior interval. Spot fits write `*_stellar_spots.csv` and
`*_stellar_spots.png`, including `spot_index`, contrast, and asymmetric errors.

To compare a synthetic recovery with the exact injected values, inspect the
matching truth JSON and result table:

```bash
python - <<'PY'
import json
import pandas as pd

truth = json.load(open("examples/data/generated/synthetic_phase_curve_truth.json"))
filename = next(__import__("pathlib").Path("results/synthetic_phase_curve").glob("*_emission.csv"))
result = pd.read_csv(filename)
print("injected hotspot offset [deg]:", truth["hotspot_offset_deg"])
print(result[["wavelength", "dayside_flux_ppm", "nightside_flux_ppm", "hotspot_offset_deg"]])
PY
```

For validation, compare the injected values with the posterior intervals, then
inspect the usual effective-sample-size and divergence diagnostics. Across a
large set of noise realizations, approximately 68% coverage is expected for a
calibrated 68% interval; any one channel may fall outside it. Keep the
production sampler defaults for scientific results.

## Forward-model cost

On the validation CPU with 321 cadences, post-compilation forward/gradient
evaluations took approximately 0.12/1.00 ms for an eclipse, 0.44/2.04 ms for a
phase curve, and 5.53/14.72 ms for a degree-eight spotted star. These timings
measure the physical evaluator, not JAX compilation, data preparation, or NUTS
tree building. Phase-curve observations also contain more cadences than a
typical transit window, so total fit time scales accordingly.

Surface fits automatically use exact linear bases when geometry and limb
darkening are fixed; no extra configuration option is needed. Eclipse uses a
native dark-planet baseline and unit uniform-emission template. Phase curves
add native cosine and sine dipole templates, which preserve the configured
hotspot-offset convention. Transit-only spot fits evaluate the native
zero-contrast curve and each native unit-contrast spot curve once, then sample
contrasts as a small matrix product. These reconstructions are exact because
the light curve is linear in uniform intensity, spherical-harmonic
coefficients, and spot contrast.

For 321 cadences, eclipse and phase basis preparation took 0.74 and 1.46
seconds, while their basis value/gradient graphs compiled in 0.043 and 0.064
seconds. In a 121-cadence degree-eight spot test, basis preparation took about
14--20 seconds depending on process cache state; the resulting
contrast-plus-normalization value/gradient and Hessian compiled in 0.072 and
0.083 seconds, and a warmed value/gradient took 0.025 ms. Fits with free
geometry or limb darkening, and emission fits that also contain stellar spots,
continue to evaluate the full native Starry model because their occultation
geometry or combined surfaces change during sampling.

## Implementation references

The implementation follows the public JAXoplanet
[phase-curve tutorial](https://github.com/exoplanet-dev/jaxoplanet/blob/8f755207c93316f379632bf758ffd4ef331e6d28/docs/tutorials/phase-curve.ipynb)
and used the [Eureka JAXoplanet branch](https://github.com/kevin218/Eureka/tree/jaxoplanet)
as an additional integration reference. Those links are design references;
the numerical behavior and accepted configuration described here are covered
by this repository's local tests.
