# Jaxoplanet quadratic optimization

## Scope and limb-darkening convention

This path uses the ordinary quadratic intensity law directly:

\[
I(\mu)=1-u_1(1-\mu)-u_2(1-\mu)^2.
\]

`u1` and `u2` are literal intensity coefficients, whether sampled or supplied
as fixed values. They are **not** Kipping `q1,q2` coordinates and no Kipping
transform is applied. Physicality restrictions, if desired, must therefore be
expressed in the priors or fixed coefficient values themselves.

## Degree-2 mathematics

For jaxoplanet's Green basis, the direct degree-2 transform is

\[
g_0=1-u_1-\frac{3}{2}u_2,\qquad
g_1=u_1+2u_2,\qquad
g_2=-\frac{1}{4}u_2,
\]

with disk normalization

\[
Z=\pi\left(g_0+\frac{2}{3}g_1\right)
 =\pi\left(1-\frac{u_1}{3}-\frac{u_2}{6}\right).
\]

The relative transit signal is evaluated as

\[
\Delta f=\frac{g_0s_0+g_1s_1+g_2s_2}{Z}-1,
\]

where

\[
s_1=-P_0-\frac{2}{3}(\kappa_1-\pi).
\]

The current stock jaxoplanet 0.1.0 calculation of `P0` uses order-10
Gauss--Legendre quadrature. Degree 2 does not enter its higher-order polynomial
loop; the fixed 10-node `P0` calculation is the main remaining light-curve
kernel work.

## Implemented parity-preserving specialization

`models/jaxoplanet/limb_dark_quadratic.py` specializes the equations above. It
keeps the stock contact branches, numerical guards, float64 behavior, and
order-10 quadrature, while removing the generic binomial coefficient transform,
the materialized three-component solution array, and its final batched dot
product.

Enable it explicitly under `flags`:

```yaml
flags:
  transit_engine: jaxoplanet
  param_method: duration
  ld_profile: quadratic
  jaxoplanet_kernel: quadratic_specialized
```

There is also an exact, GPU-oriented local-JVP route:

```yaml
flags:
  transit_engine: jaxoplanet
  param_method: duration
  ld_profile: quadratic
  ld_prior: fixed
  jaxoplanet_kernel: quadratic_local_jvp
```

`quadratic_local_jvp` retains the stock degree-two solution-vector dot for its
primal and supplies a cadence-local custom JVP. The rule recognizes symbolic
zero tangents, so fixed `u1,u2` do not construct an unused limb-darkening
Jacobian. It still computes that Jacobian when `u1,u2` are sampled: free
quadratic limb darkening is therefore mathematically supported, but the speed
measurements below apply specifically to fixed LD. Forward-mode JVP, reverse
gradients, and second derivatives were checked against stock jaxoplanet.

Routing is deliberately conservative:

- `quadratic_specialized` is used only for quadratic, degree-2,
  duration-parameterized orbits.
- `quadratic_local_jvp` has the same routing restrictions and currently
  requires quadrature order 10, matching this pipeline's production setting.
- Unsupported requests, including the Keplerian/`a_rs` path, fall back to
  stock jaxoplanet.
- `jaxoplanet_kernel: auto` still selects stock for quadratic fits. Neither
  quadratic specialization is selected automatically because their relative
  speed depends on active-window length and GPU architecture.

The unit tests compare flux and gradients with stock jaxoplanet at central,
grazing, contact, out-of-transit, and full-occultation geometries. Existing CPU
tests reach approximately machine precision in flux and sub-`1e-10` relative
gradient agreement.

## Publication/production gate

Before changing `auto`, benchmark and archive all of the following:

1. Forward parity at random physical `u1,u2` and at `b=0`, very small radius,
   inner and outer contacts on both adjacent floating-point values, grazing
   transits, `r>1`, full occultation, and out of transit.
2. Finite and stock-matching JVP/VJP or objective gradients with respect to
   `u1`, `u2`, `b`, and radius ratio, plus an end-to-end likelihood gradient.
3. Identical float64, order-10, data masks, priors, likelihood, and NUTS
   settings. Compile time must be reported separately from steady-state time.
4. Synchronized GPU forward and `value_and_grad` timing on real representative
   shapes: HAT-P-12 SOSS (226 cadences, about 89 active per window) and
   HAT-P-65 Prism (40,787 samples, about 16,515 active), with batches/chunks
   `1, 8, 40, 80, 100` on available V100, RTX 6000, and A100 hardware.
5. HLO/buffer and peak-memory inspection, followed by at least one short NUTS
   run confirming unchanged posterior summaries and no new divergences or
   non-finite gradients.

Only a reproducible end-to-end gain with all fidelity checks passing should
promote the specialized path into `auto`.

## Measured V100 performance

Synchronized steady-state timings used direct coefficients initialized at
`u1=0.3,u2=0.2` and differentiated them as a free-LD stress test, with real
saved timestamps, the exact duration window, float64, and the same synthetic
Gaussian objective for both kernels. All parity gates passed: maximum flux
differences were below `5.6e-16`, transit-gradient relative differences were
below `1.8e-15`, and the coefficients were never transformed through Kipping
coordinates.

| Real grid | Batch | Stock value+grad | Specialized value+grad | Speedup |
| --- | ---: | ---: | ---: | ---: |
| HAT-P-12 SOSS (89/226 active) | 1 | 0.152 ms | 0.148 ms | 1.02x |
| HAT-P-12 SOSS (89/226 active) | 40 | 0.253 ms | 0.314 ms | 0.81x |
| HAT-P-12 SOSS (89/226 active) | 100 | 0.242 ms | 0.421 ms | 0.58x |
| HAT-P-65 PRISM (16,515/40,787 active) | 1 | 0.267 ms | 0.248 ms | 1.08x |
| HAT-P-65 PRISM (16,515/40,787 active) | 40 | 5.109 ms | 4.309 ms | 1.19x |
| HAT-P-65 PRISM (16,515/40,787 active) | 100 | 12.744 ms | 10.548 ms | 1.21x |

The algebraic specialization improves the long Prism reverse pass modestly,
but regresses the spectroscopic SOSS reverse pass and even the Prism batch-40
forward-only call. This is an XLA fusion/buffer-scheduling effect rather than
a mathematical difference. Consequently `quadratic_specialized` remains a
research/target-specific opt-in and `auto` safely retains stock quadratic
jaxoplanet. It cannot provide a factor-of-two whole-fit improvement.

The table above differentiates `u1,u2`; that is useful as a free-LD stress
test, but it is not representative of the requested fixed-LD spectroscopy.
When fixed `u1,u2` were moved to dynamic non-gradient inputs and the objective
instead differentiated radius ratio, impact, duration, trend, and jitter, the
current specialization no longer regressed on SOSS. The local-JVP route then
removed the remaining short-window reverse-pass overhead.

## Fixed-LD complete-potential benchmarks

The decisive benchmark uses the actual vectorized NumPyro model, transformed
priors, deterministics, real HAT-P-12/HAT-P-65 timestamp grids, static transit
window, production spot/explinear detrending, time-dependent error model, and
Gaussian likelihood. At batch 40, the unconstrained states contain 200 and
240 sampled coordinates respectively. Literal `u1,u2` are fixed dynamic model
arguments and are never Kipping coordinates.

| GPU / real grid | Stock | `quadratic_specialized` | `quadratic_local_jvp` | Best speedup over stock |
| --- | ---: | ---: | ---: | ---: |
| V100, HAT-P-12 SOSS (89 active) | 0.291 ms | 0.256 ms | **0.199 ms** | **1.46x** |
| V100, HAT-P-65 PRISM (16,515 active) | 5.940 ms | **4.336 ms** | 5.728 ms | **1.37x** |
| RTX 6000, HAT-P-12 SOSS (89 active) | 0.493 ms | 0.493 ms | **0.312 ms** | **1.58x** |
| RTX 6000, HAT-P-65 PRISM (16,515 active) | **16.562 ms** | 16.854 ms | 19.902 ms | stock best |
| A100, HAT-P-12 SOSS (89 active) | 0.359 ms | 0.291 ms | **0.279 ms** | **1.29x** |
| A100, HAT-P-65 PRISM (16,515 active) | 3.054 ms | 3.259 ms | **2.901 ms** | 1.05x |

All local-JVP potential values were bitwise equal to stock. Relative L2
gradient differences were at most `8.31e-17`; every value and gradient was
finite. The V100 local-JVP reduced HAT-P-12 HLO reductions from 89 to 61 and
estimated bytes accessed from 10.51 MB to 7.88 MB. Its compiler temporary was
2.47 MB versus 2.22 MB for stock. At the much larger PRISM shape that temporary
grew to 421 MB versus 347 MB (305 MB for `quadratic_specialized`), explaining
why the algebraic specialization remains preferable there on V100.

The A100 PRISM timings were noisier than V100 (overlapping p05/p95 ranges), so
the 1.05x local-JVP result is not a reason to use it for long PRISM windows.
On RTX 6000 the long-window stock result was 1.8% faster than the algebraic
specialization and 20.2% faster than local-JVP, with non-overlapping timing
ranges. The conservative production choice is therefore explicit,
workload- and architecture-specific:

- short fixed-LD SOSS-like windows: `quadratic_local_jvp` on all three tested
  GPUs;
- very long fixed-LD PRISM-like windows: `quadratic_specialized` on V100, but
  stock on RTX 6000; profile first on other hardware;
- free LD or an unbenchmarked shape: keep `auto`/stock unless separately
  profiled.

`auto` intentionally remains unchanged. Benchmark artifacts are
`logs/jaxoplanet_quadratic_fixed_u_potential_{hatp12,hatp65}_{v100,rtx6000,a100}.json`;
the isolated-kernel artifacts use the same stem without `_potential`.

## Short sampler integration check

An RTX 6000 A/B run on the real HAT-P-12 grid used batch 8, 100 warmup steps,
200 retained draws, the same initial unconstrained state, and the same PRNG
seed. Stock and `quadratic_local_jvp` both produced finite draws and zero
divergences. The largest posterior-mean difference over all unconstrained
sites was 0.35 pooled posterior standard deviations. End-to-end elapsed times,
including compilation, were 51.5 s and 34.0 s respectively, but the adapted
median trajectory lengths also differed (255 versus 127 leapfrog steps), so
this short stochastic run is a sampler-safety check rather than a clean kernel
speed measurement. The synchronized complete-potential table above remains
the performance result. The full record is
`logs/jaxoplanet_quadratic_local_jvp_nuts_hatp12_rtx6000.json`.

## Performance ceiling

The existing V100 reference in `logs/jaxo_v100_results.json` already places a
40-curve, 1458-time quadratic forward call near the GPU launch floor
(approximately `0.087 ms`) and its full `value_and_grad` near `0.60 ms`.
The real-grid measurements confirm that whole-fit gains from the direct
degree-2 algebra alone will be smaller because orbit construction, detrending,
likelihood work, and NUTS control flow remain unchanged. A factor-of-two
end-to-end improvement is not expected from this specialization alone.

## Future exact analytic elliptic/CUDA option

The quadratic solution can instead evaluate the linear Green-basis term with
the exact piecewise elliptic-integral formula from Agol et al. The
[`exoplanet-core`](https://github.com/exoplanet-dev/exoplanet-core) C++ code
contains stable formulas, analytic derivatives with respect to separation and
radius, and CUDA FFI kernels; the derivation is described by
[Agol et al. (2020)](https://arxiv.org/abs/1908.03222).

This would be a different numerical implementation, not bitwise parity with
jaxoplanet 0.1.0's order-10 quadrature. In an audit over 11,600 physical points
with `(u1,u2)=(0.4,0.2)` and radius ratios through 0.25, the maximum normalized
flux difference was `3.52e-8`, or `0.0352 ppm`. Representative maxima were
about `0.0041 ppm` at `r=0.10` and `0.0108 ppm` at `r=0.15`. These are tiny,
sub-ppm differences, but an exact implementation still requires validation over
the complete production prior support before adoption.

The existing FFI returns solution and derivative arrays and can itself become a
fusion boundary. A purpose-built CUDA primitive that returns the contracted
quadratic flux and needed partial derivatives directly could plausibly deliver
about `1.5-3x` core `value_and_grad` speed on long Prism workloads. Small SOSS
workloads may see little or no gain because launch, branching, and FFI overhead
already dominate. This is the credible higher-ceiling route, but it should be a
separate opt-in kernel with the same fidelity and end-to-end gates above.
