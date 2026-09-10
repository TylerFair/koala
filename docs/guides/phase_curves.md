# Eclipses, phase curves, and stellar spots

JAXoplanet fits (`flags.transit_engine: jaxoplanet`) can describe three kinds
of light curve, chosen with `flags.light_curve_model`:

- `transit` (default): the planet is dark and blocks the star.
- `eclipse`: the planet has a uniform brightness given by
  `planet.eclipse_depth_ppm`; a `gaussian` (or `uniform`) prior makes it a
  fitted quantity, `fixed` holds it.
- `phase_curve`: the planet carries a smooth day--night brightness map set by
  `planet.dayside_flux_ppm`, `planet.nightside_flux_ppm`, and
  `planet.hotspot_offset_deg`, each `fixed` or `gaussian`.

Each of these is written as a `{value, prior, ...}` mapping like every other
`planet` entry; see [Parameter priors](configuration.md#parameter-priors).

All three use Keplerian geometry: give the `planet` block `a_rs`, or
`duration` from which it is derived (plus `ecc` and `omega`, which default to
zero). `planet.t0` is always the primary-transit epoch. Giving `t0`, `b`,
`rprs`, and `a_rs` (or `duration`) `prior: fixed` holds the supplied geometry
fixed, which is the usual choice for eclipse-only fits; a free prior on any
of them fits it, for example `t0: {value: ..., prior: gaussian, sigma: 0.02}`
to let the eclipse time float.

## Stellar spots

Circular spots on the rotating stellar surface are configured under
`stellar.spots` and work with any of the three models. Each spot has a fixed
position and radius; a positive `contrast_prior_width` makes its contrast
(0 is photosphere, 1 is dark) a fitted quantity.

```yaml
stellar:
  rotation_period: 25.0
  spots:
    - latitude_deg: -8.5
      longitude_deg: 2.0       # at t0
      radius_deg: 10.0
      contrast: 0.2
      contrast_prior_width: 0.15

flags:
  transit_engine: jaxoplanet
  light_curve_model: transit
```

Spots are smooth spherical-harmonic approximations, so keep them moderate in
size and non-overlapping. Surface models are not available with the Harmonica
engine.

See the [phase-curve tutorial](../tutorials/phase_curve.md) and the
[rocky-planet eclipse tutorial](../tutorials/rocky_eclipse.md) for complete
configurations; `examples/eclipse.yaml`, `examples/phase_curve.yaml`, and
`examples/stellar_spots.yaml` are small synthetic versions.
