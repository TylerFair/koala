# Spectroscopic acceleration results

Status date: 2026-09-01.  These results distinguish production-safe changes
from experimental prototypes.  Synthetic timings establish engineering
performance, not posterior equivalence on a real visit.

## Confirmed implementation result

Reusable independent-sampler programs eliminate equal-width recompilation.
In the CPU smoke workload (two lanes, 33 cadences, two warmup and four retained
draws), independent NUTS changed from 27.949 seconds cold to 0.206 seconds
steady with zero compilation events, a 135.7x ratio.  Independent fixed-step
HMC changed from 17.009 to 0.193 seconds, an 88.1x ratio.  The small workload
intentionally makes compilation dominant; full-fit speedup must be taken from
the GPU ESS/s sweep, not these ratios.

The cache is production-wired for ordinary chunks and difficulty-plan batches.
Regression tests compare reused and fresh NUTS/HMC draws exactly while changing
same-shape observations, priors, initial values, and random keys.

## Opt-in direct Power-2 result

`jaxoplanet_kernel: native_power2` now exposes a float64, duration-geometry
direct Power-2 model without the degree-12 projection. It uses a 16-node
contact-stable Green-contour integral; defaults are unchanged. A 167,040-point
full-prior/contact audit found `0.00271 ppm` maximum flux error against order
256, independently confirmed by adaptive radial integration.

On the complete 40,787-cadence HAT-P-65 NumPyro potential on RTX6000, native
value-plus-gradient was `1.285x` faster than the streamed default at width 40
and `1.260x` faster at width 80. Peak device memory was about `1.33x` higher.
This is a substantially more faithful Power-2 evaluator and a modest HMC
kernel improvement, not an OOM fix or a 10x end-to-end result.

## RTX6000 power-2 emulator experiment

All entries used roughly 45 MiB of interpolation tensors and batch size 64.

| Radius interpolation | Maximum flux error | Maximum radius-gradient error | Forward speedup | Value+gradient speedup |
| --- | ---: | ---: | ---: | ---: |
| Linear, 81 knots | 1.087 ppm | 651.7 ppm/unit | 20.81x | 12.81x |
| C1 cubic, 81 knots | 0.796 ppm | 790.2 ppm/unit | 20.02x | 10.77x |
| Exact-derivative Hermite, 41 knots | 1.012 ppm | 1702.3 ppm/unit | 21.41x | 11.27x |

The worst derivative locations are at second/third contact.  Cubic and Hermite
interpolation are therefore closed as an optimization path.  For the linear
grid, a 100-ppm local likelihood probe gave maximum absolute log-likelihood
difference 0.0792, aggregate gradient relative error 1.277%, and worst-lane
domain-scaled error 24.75%.  It remains experimental and is not called by
`fit_jwst.py`.

## GPU sampler measurements

The source-fingerprinted RTX6000 pilot completed after the final runner-cache
and quality-gate changes. It used 100 warmup and 100 retained draws with one
repeat, so it is a screening result rather than a production posterior.

All 18 cases executed with finite posteriors, but only 8 passed the divergence,
tree-depth, ESS, and finiteness gates. For sampled-Uniform trends at resident
width 80, independent NUTS changed steady wall time from `162.85 s` to
`140.49 s` (`1.16x`) and summed radius-ratio ESS/s from `44.91` to `53.48`
(`1.19x`) relative to joint NUTS. Cold wall time improved `1.25x`. This is the
only like-prior, same-width end-to-end comparison worth carrying forward.

Every Gaussian-marginalized independent-NUTS case failed divergence and/or
ESS gates; every fixed-step HMC case failed its ESS gate. Marginalized joint
NUTS was slower and less ESS-efficient than sampled-Uniform joint NUTS at each
tested width. The automatic width-selection manifest was invalidated because
the recorded workload/runtime/source signatures differed, so cross-width
ratios must not be extrapolated into a full-fit claim.

Submitted or completed sweeps:

- RTX6000: Slurm job `57038731` (completed; report above)
- A100: Slurm job `57038730`
- A100 power-2 grid timing: Slurm job `57027215`
- One-hour RTX6000 width-40 backfill probe: Slurm job `57040611`

The sampler sweeps cover joint NUTS, independently adapted NUTS, and
independent fixed-step HMC; sampled-Uniform and Gaussian-marginalized linear
trends; and resident widths 8, 40, and 80.  Their outputs are written under
`logs/` and are accepted for automatic width selection only after worst-channel
divergence, tree-depth saturation, ESS, finiteness, and measured-memory gates.
