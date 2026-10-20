# Planet and orbit parameters

Let's set up the planet's orbit for a transit fit.
We'll use the starting values from the bundled SOSS example.
Each entry gives both a value and a prior, so we can choose which parameters to fit and which to hold fixed:

```yaml
planet:
  name: WASP-39
  period: {value: 4.05528043, prior: fixed}
  t0: {value: 59787.055, prior: uniform, low: 59787.005, high: 59787.105}
  duration: {value: 0.11693087, prior: log_uniform, low: 0.01, high: 1.0}
  b: {value: 0.4498, prior: uniform, low: 0.0, high: 1.5}
  rprs: {value: 0.1457, prior: uniform, low: 0.01, high: 0.5}
```

These are starting values, not posterior measurements.
The period is fixed, while the white-light curve constrains the transit time, duration, impact parameter, and radius ratio.
You can use the scaled semimajor axis `a_rs` in place of `duration`.

## Transit and eclipse times

First, check that the ephemeris and data use the same time origin.
koala uses BMJD_TDB = BJD_TDB − 2400000.5, in days.
It converts full BJD input arrays automatically, but it does not convert epochs in the YAML file.
A time quoted as “BJD − 2400000” is 0.5 d larger than the corresponding BMJD.
Subtract that 0.5 d before entering it.
Changing the origin does not convert UTC to TDB or apply a barycentric correction.

The parameter `t0` is the primary-transit epoch.
For a circular eclipse or phase-curve fit with a fixed period, we can instead supply the eclipse time:

```yaml
planet:
  period: {value: 2.6162644, prior: fixed}
  eclipse_time: {value: 60001.3081322, prior: fixed}
  ecc: {value: 0.0, prior: fixed}
```

Give exactly one of `t0` and `eclipse_time`.
koala derives `t0 = eclipse_time - period / 2` and reports both times.
It raises an error if no allowed event falls on the observation.
The error also identifies the 0.5 d mismatch when that is the cause.

## Fixed and free geometry

For a shallow eclipse, the light curve may not contain enough information to measure the orbit.
In that case, fix `period`, `eclipse_time`, `a_rs`, `b`, and `rprs` using the transit solution.
Then fit the eclipse depth, baseline, and noise.
The [rocky-planet case study](../case_studies/rocky_eclipse.md) shows what happens when the eclipse time is poorly constrained.

For a transit, use free priors on `t0`, `b`, `rprs`, and either `duration` or `a_rs` to measure the geometry in white light.
The priors replace the old `flags.fit_geometry` switch.
A free period is fitted in white light and then held fixed for spectroscopy.
If the observation includes several transits, all epochs `t0 + n * period` contribute.

## Joint white light across observations

We can use several orders, detectors, or visits to measure one orbit.
Each observation keeps its usual koala configuration.
A separate file names the configurations to fit together:

```yaml
joint:
  members: [soss_order1.yaml, soss_order2.yaml, nrs1.yaml]
  apply_to: []
  shared: [a_rs, b]
  visits: auto
planet:
  a_rs: {value: 18.0, prior: uniform, low: 5.0, high: 40.0}
  b: {value: 0.1, prior: uniform, low: 0.0, high: 1.5}
output_dir: results/joint
host_device: cpu
```

Run `python -m koala -c joint.yaml`.
Member paths and the joint output directory are relative to this file.
The member input paths keep their ordinary meaning.
The joint host device applies to every observation.
Try `examples/joint_soss_orders.yaml` with the bundled WASP-39 data.

The joint members use jaxoplanet transit models.
The orbit must be circular and the period must be fixed and identical in every configuration.
The shared parameters are `a_rs` and `b`.
A common duration would be inappropriate when the radius changes with wavelength.
The joint planet block supplies their priors, each counted once.
Omitted priors come from the first member.
If that member fits duration, supply a joint `a_rs` prior explicitly.
A fixed duration can supply a fixed `a_rs` from its starting radius and impact parameter.

Overlapping observation ranges form a visit and share a transit epoch.
To choose visits explicitly, use `visits: [[soss_order1, soss_order2], [nrs1]]`.
Names are member file stems and every member must occur once.
The first member of each visit supplies its epoch prior.
Within each visit, spot crossings share their times and widths.
Their amplitudes remain separate.
The first member using each spot supplies its timing prior.
Limb darkening, radius, trend coefficients, and noise remain specific to each member.
Every integration contributes to the joint fit.

The `joint/` output directory contains parameter intervals, convergence diagnostics, a geometry corner plot, and a summary.
Each member directory contains its usual white-light products made from its slice of the joint posterior.
Spectroscopy then runs for each member in turn with the shared orbit and its visit epoch fixed at their posterior medians.
The usual `spectro_free_t0` option still permits channel timing offsets.
For an observation that should receive the orbit without contributing to it, put its configuration in `apply_to`.
It gets a separate white-light fit with `a_rs` and `b` fixed, followed by spectroscopy.

## Joint spectroscopic geometry

The usual channel fit holds the orbit at the white-light posterior medians.
To fit shared geometry across all channels of a stage, add:

```yaml
flags:
  spectro_joint_geometry: true
  spectro_joint_geometry_prior_inflation: 3.0
  spectro_sampler: joint_nuts
  spectro_chunk_size: auto
```

Now `t0`, `b`, and `duration` or `a_rs` are shared free parameters.
Their Gaussian priors are centered on the white-light medians.
Each prior width is the white-light standard deviation multiplied by the inflation factor.
The period stays fixed.

All channels must fit in one chunk, so memory use grows with the full spectrum.
This option requires jaxoplanet and uses the joint sampler.
It disables cadence reduction and transit-grid acceleration.
After the fit, read `*_joint_geometry.csv` to compare the shared posterior with the white-light solution.

## Reference

### Prior syntax

| `prior` | Required fields | Meaning |
|---|---|---|
| `fixed` | `value` | Hold the parameter at this value. |
| `uniform` | `value`, `low`, `high` | Start at `value` and sample uniformly between the bounds. |
| `log_uniform` | `value`, `low`, `high` | Sample uniformly in the logarithm. The lower bound must be positive. |
| `gaussian` | `value`, `sigma` | Normal prior centered at `value`. Optional `low` and `high` truncate it. |

Bare numbers are rejected, and starting values must lie within their bounds.
A uniform impact parameter with `low: 0` is sampled as `_b ~ uniform(-high, high)` with `b = |_b|`.
This preserves the prior on $b$ while removing the sampling boundary at zero.
Eccentricity and argument of periastron can only be fixed and default to zero.

Eclipse depth and the phase coefficients `phase_c1_ppm` and `phase_d1_ppm` accept all four priors.
The phase coefficients can be negative, so use `fixed`, `uniform`, or `gaussian` when you need that freedom.
For several planets, use lists of parameter mappings in the same order for each key.
A single mapping broadcasts to all planets.
The input files must already form one time series.
