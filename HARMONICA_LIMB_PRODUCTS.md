# Harmonica limb products (schema v2)

This project reports Catwoman-comparable representative limb depths from the
Harmonica transmission string

\[
r(\theta)=a_0+\sum_{n\in\{1,3,5\}}a_n\cos(n\theta),
\]

where every radius is in stellar-radius units. Harmonica measures \(\theta\)
from the orbital-velocity direction. The convention used here is therefore:

- \(\theta=0\): **evening / leading** hemisphere;
- \(\theta=\pi\): **morning / trailing** hemisphere.

This mapping is recorded in every schema-v2 CSV as
`theta=0:evening/leading;theta=pi:morning/trailing`.

## Representative depths

For either hemisphere \(H\)—evening
\([-\pi/2,\pi/2]\) or morning \([\pi/2,3\pi/2]\)—its representative depth is

\[
D_H=\frac{1}{\pi}\int_H r(\theta)^2\,d\theta
    =\frac{2A_H}{\pi R_\star^2}.
\]

It is the full-circle-equivalent depth of a semicircle having the same area,
which makes it comparable to a Catwoman limb radius squared. It is a geometric
area depth, not the limb-darkened flux decrement at a particular transit time.

Define

\[
Q=a_1^2+a_3^2+a_5^2,
\qquad
C=a_1-\frac{a_3}{3}+\frac{a_5}{5}.
\]

Exact integration gives

\[
D_{\rm evening}=a_0^2+\frac{Q}{2}+\frac{4a_0C}{\pi},
\qquad
D_{\rm morning}=a_0^2+\frac{Q}{2}-\frac{4a_0C}{\pi},
\]

and

\[
D_{\rm total}=\frac{D_{\rm evening}+D_{\rm morning}}{2}
             =a_0^2+\frac{Q}{2}.
\]

There are no omitted \(a_1a_3\), \(a_1a_5\), or \(a_3a_5\) terms: distinct odd
cosine harmonics are orthogonal over either half-circle. With only \(a_1\),
the formula reduces to
\(a_0^2+a_1^2/2\pm4a_0a_1/\pi\).

Endpoint values are separate diagnostics and are not the representative
half-area depths:

\[
D_{\rm leading,endpoint}=(a_0+a_1+a_3+a_5)^2,
\qquad
D_{\rm trailing,endpoint}=(a_0-a_1-a_3-a_5)^2.
\]

For a first-order fit, the opt-in physical parameterization samples total-area
radius and half-area contrast directly:

```yaml
flags:
  transit_engine: harmonica
  harmonica_max_order: 1
  harmonica_spectro_parameterization: half_area
```

It uses \(q=(D_{\rm evening}-D_{\rm morning})/
(D_{\rm evening}+D_{\rm morning})\), stores
`rors = sqrt(D_total)`, derives `a0,a1` for the forward model, and restricts
\(|q|<16/(9\pi)\) so the smooth shape remains globally convex. Existing
`delta_r` configurations and their prior semantics are unchanged.

`half_area` is forward-model equivalent to an `a0,a1` fit after converting the
coordinates, but it is deliberately **not prior-equivalent** to `delta_r` or
`fractional`. Its uniform radius prior is placed on `sqrt(D_total)` rather than
on `a0`, and its truncated-normal prior is placed directly on `q`, excluding
the non-convex tails available to the legacy parameterizations. Posterior
differences—especially for weakly constrained asymmetry—can therefore be a
scientific prior/support effect rather than a numerical light-curve mismatch.

## Why the two spectra can look anticorrelated

At every posterior draw, the representative limbs have the form
\(D_{\rm evening}=A+\Delta\) and \(D_{\rm morning}=A-\Delta\), with
\(A=D_{\rm total}\). When the total silhouette area is tightly constrained but
ingress/egress constrain the asymmetry weakly, uncertainty in \(\Delta\) moves
the two limbs in opposite directions. Mirror-like posterior structure is
therefore expected and also occurs for two Catwoman semicircles. It does not
force the physical spectra to be inverses: wavelength dependence in \(A\)
moves both together, while wavelength dependence in \(\Delta\) separates them.

## Schema-v2 outputs

The limb-products CSV includes posterior median and 16th/84th-percentile errors
for:

- `depth_evening_*` and `depth_morning_*` (half-area-equivalent depths, ppm);
- `depth_total_area_*` (total silhouette-area depth, ppm);
- `depth_leading_endpoint_*` and `depth_trailing_endpoint_*` (ppm);
- `asymmetry_coefficient_*` (\(C\), in \(R_\star\) units);
- `endpoint_delta_r_*` (\(2[a_1+a_3+a_5]\), in \(R_\star\) units);
- `depth_a0_*` (\(a_0^2\), not total area), `a0_*`, and each active coefficient
  (`a1_*`, `a3_*`, `a5_*`);
- `limb_product_schema_version` and `limb_product_convention`.
- `planet_index` (production Harmonica fits currently require exactly one
  planet, so this is zero rather than a silently selected planet from a
  multi-planet fit).

For scientific inference, propagate the joint morning/evening posterior samples
or covariance rather than treating the marginal error bars as independent.
Every new fit therefore also writes `*_limb_posterior_samples.npz`, containing
the draw-by-wavelength arrays `depth_evening`, `depth_morning`,
`depth_total_area`, the endpoint diagnostics, `a0`, and every active odd
coefficient. Depth samples in this archive are fractional stellar area (not
ppm), and retain the exact drawwise morning/evening covariance.
State the adopted leading/trailing-to-evening/morning interpretation explicitly;
the physical terminology assumes the corresponding orbital and rotation
convention. End-to-end injection/recovery and posterior-coverage tests remain
recommended before publication.

Existing checkpoints can be converted without refitting or overwriting legacy
products with `tools/regenerate_harmonica_limb_products.py`. The newest
WASP-94 SOSS order-1 V4 run was validated from 60 contiguous R=50 chunks and
1000 draws, then regenerated beside the originals as
`WASP-94_NIRISS_SOSS_order1_R50_limb_spectra_schema_v2.csv` plus the three
`schema_v2` figures. The utility preserves deterministic `a0` in new
`half_area` checkpoints; legacy checkpoints continue to interpret `rors` as
the transmission-string mean radius.
