# JAX asymmetric two-limb shape feasibility

## Status and scope

This is a geometry and implementation feasibility study. It does **not**
describe a newly implemented asymmetric `jaxoplanet` backend. The smooth,
GPU/JAX implementation is the audited Harmonica `N_c = 1` backend, now with
an opt-in physical half-area parameterization for spectroscopic fits. Existing
`delta_r` configurations and defaults are unchanged.

The implemented configuration is

```yaml
flags:
  transit_engine: harmonica
  harmonica_max_order: 1
  harmonica_spectro_parameterization: half_area
```

Internally, `rors = sqrt(S)` is the total-area-equivalent radius and the
bounded latent coordinate is

```text
q = (D_evening - D_morning) / (D_evening + D_morning).
```

The model derives deterministic `a0`, `a1`, `depth_total_area`,
`depth_evening`, and `depth_morning`. Its prior is a zero-centred truncated
normal with small-asymmetry scale `(4/pi) * odd_frac_sigma` and strict support
`|q| < 16/(9 pi)`, which enforces a globally convex limacon. The stable inverse
used by the code is

```text
x = a1/a0 = 2q / (4/pi + sqrt((4/pi)^2 - 2q^2)),
a0 = rors / sqrt(1 + x^2/2).
```

This coordinate choice preserves the represented forward shape, not the
legacy prior measure. The uniform radius prior is on `rors = sqrt(S)` instead
of `a0`, and the directly truncated `q` prior removes non-convex legacy tails.
Consequently, differences from `delta_r`/`fractional` posteriors in a
weak-asymmetry regime can be expected prior/support effects. They must not be
interpreted automatically as a kernel discrepancy.

The desired convention is

- `theta = 0`: evening/leading limb, along projected orbital velocity;
- `theta = pi`: morning/trailing limb; and
- `phi = 0`: the morning/evening axis is aligned with the orbit.

## Candidate 1: exact Catwoman geometry

Let `e = (cos(phi), sin(phi))` point toward the evening limb and let
`e_perp` be perpendicular to it. In planet-centred coordinates

```text
u = x dot e
v = x dot e_perp,
```

the Catwoman silhouette is the union of two half-discs,

```text
Omega = {u >= 0, u^2 + v^2 <= R_evening^2}
      U {u <  0, u^2 + v^2 <= R_morning^2}.
```

Away from the two joining angles, its polar radius is

```text
r(theta) = r0 + d sign(cos(theta - phi)),
r0 = (R_evening + R_morning) / 2,
d  = (R_evening - R_morning) / 2.
```

The full closed boundary also contains two radial connector segments where
the unequal semicircles meet. Thus the model has literal, independent limb
radii, but a discontinuous polar radius and corners. Its area is

```text
area = pi (R_evening^2 + R_morning^2) / 2.
```

The step has the odd-cosine Fourier expansion

```text
sign(cos(psi)) = (4/pi) [cos(psi) - cos(3 psi)/3
                         + cos(5 psi)/5 - ...].
```

Consequently, truncating Catwoman's radial function at the first Fourier term
would give `A = 4d/pi`. This is an L2/Fourier projection, not an endpoint or
half-area match.

## Candidate 2: smooth egg/limacon

The minimal smooth model is the convex limacon

```text
r(theta) = a0 + A cos(theta - phi).
```

Using

```text
cos(theta - phi) = cos(theta) cos(phi) + sin(theta) sin(phi),
```

this is exactly the Harmonica `N_c = 1` transmission string

```text
r(theta) = a0 + a1 cos(theta) + b1 sin(theta),
a1 = A cos(phi),
b1 = A sin(phi).
```

It is therefore not a new shape that could bypass Harmonica. With the axis
fixed to the projected orbital velocity, `phi = 0`, the coefficient vector is
simply `[a0, A, 0]`.

The boundary is smooth while `a0 > |A|`. The stronger condition
`|A| <= a0/2` keeps it globally convex and avoids dimpled limacons. Expected
atmospheric limb contrasts are far inside this bound.

## Catwoman-comparable half-area radii

Matching the two endpoint radii would use

```text
a0 = (R_evening + R_morning) / 2,
A  = (R_evening - R_morning) / 2.
```

This is simple, but the resulting limb areas do not equal those of Catwoman.
At first order its half-area contrast is only `2/pi` times the Catwoman
contrast. Endpoint radii should therefore not be labelled as
Catwoman-equivalent limb radii.

For the limacon, twice each half's area divided by `pi` is

```text
D_evening = a0^2 + A^2/2 + 4 a0 A/pi,
D_morning = a0^2 + A^2/2 - 4 a0 A/pi.
```

These follow directly from integrating `r(theta)^2/2` over the corresponding
angular half. They are the appropriate representative limb depths because a
Catwoman semicircle of radius `R` has `D = R^2` under the same definition.

The exact inverse mapping from desired half-area depths to the smooth shape is

```text
S = (D_evening + D_morning) / 2,
C = (D_evening - D_morning) / 2,
K = (pi C / 4)^2,

a0 = sqrt((S + sqrt(S^2 - 2K)) / 2),
A  = pi C / (4 a0).
```

The larger quadratic root is selected so that `a0 -> sqrt(S)` and `A -> 0`
as the asymmetry vanishes. Computing signed `A` as `pi*C/(4*a0)`, rather than
with `sign(C)`, keeps this transform differentiable at `C = 0`.

This mapping has three useful properties:

1. each smooth half has exactly the requested Catwoman-comparable area;
2. the total smooth silhouette area exactly equals
   `pi (D_evening + D_morning)/2`; and
3. `R_evening = sqrt(D_evening)` and
   `R_morning = sqrt(D_morning)` remain directly interpretable outputs.

Writing `q = C/S`, the inverse exists for `|q| <= sqrt(8)/pi`. The convexity
condition is more restrictive,

```text
|q| <= 16/(9 pi) ~= 0.566.
```

The existing post-processing already evaluates these exact half-area depths
in `fit_jwst.py::_harmonica_limb_product_samples`.

## Why stock circular jaxoplanet cannot evaluate either shape

The stock Agol/`jaxoplanet` occultation primitive assumes a single circular
planet boundary. Neither unequal half-discs nor a limacon is circular. Adding,
subtracting, or half-weighting two circular light curves is not exact: it
assigns the wrong stellar intensity spatially during ingress and egress, which
is precisely where the limb-asymmetry information resides.

Only `R_evening = R_morning` reduces to the stock circular solution. A shifted
circle is another mathematical exception,

```text
r(theta) = s cos(psi) + sqrt(R^2 - s^2 sin(psi)^2),
```

but its apparent `R +/- s` endpoints represent a displaced circular centre.
Along the transit chord this is a transit-time offset, not two atmospheric
limb radii, and is unsuitable as the science model.

An exact non-circular model requires a new clipped-boundary integral. Putting
that code under `models/jaxoplanet/` would not make it use the stock circular
mathematics; a limacon implementation there would duplicate Harmonica.

## Gradients, inference, and physical caveats

- The smooth limacon is differentiable in its shape parameters away from
  physical star/planet tangencies. The current pure-JAX Harmonica backend has
  regression-tested gradients through `A ~= 0`, including the formerly
  ill-conditioned circular limit.
- Exact Catwoman radius gradients can be piecewise smooth, including at equal
  radii, if the flux is implemented as the sum of two complementary half-disc
  integrals. Its corners add contact kinks. A naive hard angular `where` mask
  also gives an incorrect fitted-orientation gradient because autodiff misses
  motion of the discontinuity. Fixing `phi = 0` avoids that issue; a fitted
  `phi` would need an explicit boundary derivative.
- Exact contacts are physically non-smooth for every occultation model. They
  require the same topology and near-contact tests used in the Harmonica
  numerical audit.
- At small `A`, `a0 + A cos(theta)` is a translated circle to first order.
  Limb asymmetry is therefore strongly degenerate with `t0` (and a transverse
  first harmonic with impact parameter). A shared white-light `t0` posterior
  should be propagated into all wavelength channels; fitting an independent
  `t0` per channel or fixing an uncertain value can bias the limb spectra.
- The negative posterior correlation between morning and evening depths is
  expected. The likelihood usually constrains their sum much more strongly
  than their contrast, so samples follow approximately
  `D_evening + D_morning = constant`. Catwoman has the same statistical
  anticorrelation; it does not mean the two radii are mathematically inverses.
- For better NUTS conditioning, the internal coordinates can be total depth
  `S` and contrast `q`, with `D_evening = S(1+q)` and
  `D_morning = S(1-q)`. This is a bijection, not a loss of a limb degree of
  freedom. Priors must still be defined deliberately in the desired physical
  variables, including any Jacobian, so a convenient transform does not
  silently change the scientific prior.
- Quadratic limb darkening remains the direct fixed-coefficient law in `u1`
  and `u2`; this shape proposal requires no Kipping transformation.

## GPU speed expectation

The smooth candidate dispatches to the current Harmonica `N_c = 1` kernel, so
its speed is exactly current Harmonica speed. Rebranding it as a jaxoplanet
shape offers no circular-kernel acceleration.

The audited RTX6000 steady-state quadratic timings were

| Workload | Forward | Reverse shape gradient |
|---|---:|---:|
| 60 channels x 300 cadences | 4.114 ms | 7.132 ms |

Boundary quadrature is the dominant cost. A future exact-Catwoman specialist
could avoid Harmonica's general angular root bracketing by using analytic
circle-circle and line-circle intersections. One promising exact identity is

```text
loss_cat = loss_circle(R_morning)
         + loss_evening_halfdisc(R_evening)
         - loss_evening_halfdisc(R_morning).
```

This would reuse the stock circle and evaluate one signed half-annulus
correction with a static Green-boundary quadrature. It may provide a modest
kernel gain, plausibly tens of percent and perhaps `1.2-2x`, but this is an
unbenchmarked design estimate, not an implemented or guaranteed speedup. It
will remain materially slower than a circular model because the asymmetric
boundary integral is unavoidable.

## Recommendation

Use the existing, audited Harmonica `N_c = 1` limacon as the smooth two-limb
model. Expose and report `R_morning` and `R_evening` using the exact half-area
mapping above, fix `phi = 0`, enforce convexity, and propagate the shared
`t0` uncertainty. The current opt-in configuration is

```yaml
flags:
  transit_engine: harmonica
  harmonica_max_order: 1
  harmonica_spectro_parameterization: half_area
```

Treat an exact JAX Catwoman kernel as a separate future backend requiring its
own implementation, GPU benchmark, equal-radius parity against stock
`jaxoplanet`, random-geometry parity against Catwoman, finite-difference
gradient checks, and timing/limb-darkening injection-recovery tests.
