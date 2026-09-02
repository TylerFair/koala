# Jaxoplanet GPU optimization

## Production flags

```yaml
flags:
  transit_window_optimization: auto
  jaxoplanet_kernel: auto
  spectro_sampler: joint_nuts
  vmap_chunk: 40
```

Since 2026-09-01 the spectroscopic jitter prior defaults to
`spectro_jitter_prior: lognormal`, i.e. `log_jitter ~ Normal(log(0.5 *
median(yerr)), 2.0)` per channel, replacing `Uniform(log 1e-6, 0)`. The
historical prior left a flat plateau down to its floor in roughly half of the
SOSS channels (skewness 3.9, excess kurtosis 22) where the jitter is not
identified. On real SOSS and G395H chunks, exact joint NUTS under the two
priors agreed on every science site (rors/depth, c, v, c1, c2, A_spot) to
within 0.13 sigma with sigma ratios 0.93-1.10, i.e. within the seed-to-seed
Monte-Carlo scatter of NUTS itself; only `log_jitter`/`total_error` change,
and only in channels where the data do not constrain them. Set
`spectro_jitter_prior: log_uniform` to recover the old behaviour. Builder
keyword arguments (including this prior) are now part of the chunk checkpoint
fingerprint, so checkpoints written under the old prior are not reused.

Opt-in accelerated spectroscopic samplers (see
`acceleration_reports/ORCHESTRATOR_SUMMARY.md` for the V100 measurements):

```yaml
flags:
  spectro_sampler: independent_hmc          # or independent_nuts
  spectro_mass_matrix: laplace              # per-lane inverse-Hessian metric
  spectro_laplace_hessian_method: finite_difference
  spectro_laplace_fd_relative_step: 0.0002
  spectro_laplace_warmup: 150
  spectro_laplace_target_accept: 0.85       # 0.95 for independent_nuts
  spectro_laplace_fuse_program: true
  spectro_hmc_num_steps: 8
  spectro_hmc_trajectory_jitter: 0.25
  # independent_nuts: spectro_laplace_max_tree_depth: 5
```

`transit_window_optimization: auto` is the default. For duration-parameterized
spectroscopic fits, `fit_jwst.py` builds the fixed union of all transit
windows, evaluates the existing jaxoplanet transit function only at those
cadences, and restores an exactly zero transit contribution elsewhere. The
detrending model and likelihood still use every cadence. White-light and
`a_rs`/Keplerian models always use the original full-time implementation.

`jaxoplanet_kernel: auto` uses the memory-streamed degree-12 kernel for the
power-2 polynomial projection and the stock kernel for other profiles. The
streamed path preserves the Green-basis transform, contact branches,
order-10 Gauss--Legendre nodes, and float64 evaluation. It advances the odd
and even quadrature powers as recurrences instead of materializing the full
cadence-by-polynomial-order tensor. Unsupported and Keplerian calls fall back
to stock. For controlled comparisons, use `stock` or `streamed` explicitly.

An additional opt-in `fused` power-2 route contracts each Green-basis
solution term with its coefficient while the recurrence is live. It avoids
both the quadrature-power tensor and the final cadence-by-degree solution
array. It is algebraically the same order-10 float64 calculation, with only
floating-point reassociation differences. On the real V100 Prism benchmark,
its maximum streamed/fused flux difference was `2.27e-12` and its transit VJP
relative difference was `2.31e-10`:

```yaml
flags:
  ld_profile: power2
  jaxoplanet_kernel: fused
```

Complete-potential cross-GPU timing is now reported below. It remains opt-in
because the streamed route is slightly faster on short SOSS windows and a
representative full NUTS posterior comparison is still required before an
automatic routing change.

The native Power-2 evaluator is a separate, accuracy-first opt-in:

```yaml
flags:
  ld_profile: power2
  param_method: duration
  jaxoplanet_kernel: native_power2
```

It evaluates `I(mu)=1-c+c*mu**alpha` directly with a 16-node, contact-stable
Green-contour integral. It does not use the degree-12 intensity projection,
an Appell-function identity, or a learned/grid emulator. Float64 is mandatory;
`a_rs`/Keplerian geometry, non-Power-2 profiles, and interpolated polynomial
LD coefficients are rejected. Free, informed, uniform, and fixed direct
Power-2 coefficients are supported. Existing defaults remain unchanged.

On an RTX6000, order 16 differed from an order-256 reference by at most
`0.00271 ppm` over 167,040 full-prior/contact-focused cases; independent
adaptive radial integration confirmed the worst case. On actual HAT-P-65
cadences at the nominal fit point, the maximum order-16/order-256 difference
was `0.000029 ppm`, the NLL difference was `-2.2e-8`, and the full-potential
gradient relative-L2 difference was `9.2e-10`. The existing degree-12 model
differed from the direct law by as much as `8.54 ppm` at that point (and more
at broad-prior extremes), so native/degree-12 differences are not errors in
the native quadrature.

This is not the OOM solution. For the complete 40,787-cadence HAT-P-65
NumPyro potential, native value-plus-gradient was `1.285x` faster than the
current streamed default at width 40 and `1.260x` faster at width 80, while
peak device memory was about `1.33x` higher. Keep it opt-in until a real
posterior comparison passes; use it for a more faithful Power-2 model, not as
an order-of-magnitude fitting claim.

The direct quadratic profile has a separate opt-in specialization:

```yaml
flags:
  ld_profile: quadratic
  jaxoplanet_kernel: quadratic_specialized
```

It evaluates the literal coefficients in
`I(mu)=1-u1(1-mu)-u2(1-mu)^2`; it never applies Kipping's `q1,q2`
transformation. It retains jaxoplanet's order-10 approximation. A second
opt-in `quadratic_local_jvp` route supplies an exact cadence-local JVP and is
faster for short fixed-LD SOSS windows:

```yaml
flags:
  ld_profile: quadratic
  ld_prior: fixed
  jaxoplanet_kernel: quadratic_local_jvp
```

For the long PRISM shape, local-JVP compiler temporaries grow with the active
window and erase its short-window advantage. The algebraic
`quadratic_specialized` route won on V100, while stock won on RTX 6000, so
profile the complete potential on the target GPU before selecting a
long-window route. Neither route is automatic. See
`JAXOPLANET_QUADRATIC_OPTIMIZATION.md` for the complete-potential GPU timings
and differentiation tests.

`spectro_sampler: joint_nuts` is the compatibility default. The opt-in
`independent_nuts` backend gives every wavelength channel its own PRNG, NUTS
tree, dual-averaged step size, and small dense mass matrix while evaluating a
fixed-width batch on the GPU:

```yaml
flags:
  transit_window_optimization: auto
  spectro_sampler: independent_nuts
  vmap_chunk: 40
```

The final short block is padded with duplicate dummy channels and discarded
after sampling, so changing the lane width does not add scientific channels.
The returned sample dictionary and checkpoint layout remain
`[draw, wavelength, ...]`.

The first spectroscopic block is gradient-checked at its actual initialization
by default and writes `*.gradient.json`. Use
`spectro_gradient_diagnostic: each` for a full per-block debugging audit; this
is intentionally not the production default because each diagnostic creates a
new differentiated potential and is especially costly at `vmap_chunk: 1`.
Strict mode validates a fingerprinted diagnostic even when loading cached
samples. Checkpoints are target/data/config/RNG fingerprinted, use global
chunk-index-derived independent streams, and are atomically replaced.
White-light and low-resolution preprocessing products also require matching
manifests before reuse, so changing a configuration in an existing output
directory cannot silently reuse stale geometry, masks, or polynomial fits.

Per-lane dense mass adaptation provides the coordinate scaling: it rescales
correlated parameters such as radius ratio, limb darkening, jitter, and trend
coefficients in each channel's NUTS geometry. It can reduce tree depth, but it
does not change `vmap_chunk`, posterior dimension, or lane memory. The chunk
setting only controls how many independent channel states are resident on the
GPU at once.

## Fidelity constraints

- Float64 remains required.
- The jaxoplanet quadrature order is unchanged.
- The default power-2 conversion remains the degree-12 projection on 300
  intensity samples. The explicit `native_power2` opt-in instead evaluates
  the direct law with its separately validated order-16 contour rule.
- No observations, priors, likelihood terms, warmup iterations, or posterior
  draws are removed.
- Physical initial values are transformed to NumPyro's unconstrained
  coordinates using each distribution's exact bijection.
- `independent_nuts` is valid only for the factorized spectroscopic model with
  white-light geometry fixed. Shared latent variables are rejected.
- The optimized source derived from jaxoplanet 0.1.0 retains its MIT notice in
  `models/jaxoplanet/JAXOPLANET_LICENSE`.

The exact window tests require array-identical flux on the reference grid and
gradient agreement at `1e-11`. The sampler tests cover bounded Uniform and
TruncatedNormal parameters, padding, deterministics, shared one-planet
geometry, current fit aliases, checkpoint concatenation, and the real
windowed power-2 model.

## Benchmark

Use the same environment as `run_fit.sbatch`:

```bash
python tools/benchmark_jaxoplanet_gpu.py \
  --platform gpu \
  --cadences 525,1458 \
  --batches 1,8,40 \
  --ld-profile both \
  --warmup 10 \
  --repeats 100 \
  --json /tmp/jaxoplanet_gpu.json
```

The harness reports cold compilation separately from synchronized steady-state
forward and value-plus-gradient timings.

## Measured GPU results

### Exact real-timestamp streamed-kernel measurements

The production comparison below uses the static interval selected by the
actual `TransitOrbit` duration mask (`abs(dt) < duration/2`), real saved
timestamps, synchronized float64 GPU execution, and the full synthetic
Gaussian objective value plus reverse gradient. These timings isolate the
light-curve objective; they are not whole-NUTS wall times.

| Real grid | GPU | Batch | Active cadences | Stock value+grad | Streamed value+grad | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HAT-P-12 SOSS | V100 | 40 | 89 / 226 | 0.451 ms | 0.338 ms | 1.33x |
| HAT-P-12 SOSS | RTX 6000 | 40 | 89 / 226 | 1.381 ms | 0.924 ms | 1.49x |
| HAT-P-12 SOSS | A100 | 40 | 89 / 226 | 0.475 ms | 0.463 ms | 1.02x |
| HAT-P-65 PRISM | V100 | 40 | 16,515 / 40,787 | 25.367 ms | 12.249 ms | 2.07x |
| HAT-P-65 PRISM | RTX 6000 | 40 | 16,515 / 40,787 | 99.389 ms | 29.824 ms | 3.33x |
| HAT-P-65 PRISM | A100 | 40 | 16,515 / 40,787 | 14.348 ms | 6.016 ms | 2.38x |

Across those real-grid runs, the maximum stock/streamed flux difference was
`2.94e-12`; the worst per-channel relative L2 difference in the direct transit
VJP was `5.86e-10`. Contact, grazing, extreme-power-2, and large-occultor
stress cases were finite; their worst relative gradient L2 difference was
`1.91e-9`. All are below the adopted float64 regression gates and far below
JWST flux precision. Computing the phase once outside the channel `vmap` was
numerically exact but changed value-plus-gradient timing by roughly 0--3%; XLA
already removes most duplicated work.

The HAT-P-65 V100 benchmark's reported peak device allocation rose from about
1.28 GB at batch 40 to 3.34 GB at batch 100 for the isolated objective. That
does **not** make 100 production-safe: full NumPyro runs have subsequently
OOMed while collecting a width-100 PRISM sample block. Keep 40 as the general
jaxoplanet default until a complete fit proves a wider value on the target
GPU. The Harmonica PRISM quadratic path is substantially heavier and uses the
validated conservative width 4 in its HAT-P-65 configuration.

### Complete differentiated-potential measurements

This comparison includes transformed priors, deterministics, the power-2
projection, transit, production detrending family, error model, and Gaussian
likelihood. It is therefore a close proxy for one NUTS leapfrog evaluation,
although it still excludes tree-building control flow, adaptation, sample
transfer, and plotting.

| Real grid / GPU | Batch | Streamed median (paired speedup) | Fused median (paired speedup) |
| --- | ---: | ---: | ---: |
| HAT-P-12 SOSS / V100 | 40 | **0.463 ms (1.40x)** | 0.469 ms (1.39x) |
| HAT-P-12 SOSS / RTX 6000 | 40 | **1.110 ms (1.41x)** | 1.243 ms (1.27x) |
| HAT-P-12 SOSS / A100 | 40 | **0.554 ms (1.03x)** | 0.559 ms (1.03x) |
| HAT-P-65 PRISM / V100 | 40 | 12.150 ms (2.02x) | **8.834 ms (2.79x)** |
| HAT-P-65 PRISM / RTX 6000 | 40 | 570.930 ms (1.79x) | **34.590 ms (2.88x)** |
| HAT-P-65 PRISM / A100 | 40 | 5.135 ms (1.96x) | **4.452 ms (2.23x)** |

Each parenthetical speedup uses the stock timing measured in the same process
as that candidate; this avoids manufacturing ratios from small between-job
timing drift. The two long RTX jobs landed on markedly different-performance
nodes (including a roughly tenfold stock-timing difference), so their absolute
candidate medians must not be compared across rows within that GPU. The paired
ratios still show that both routes accelerate the same stock graph and that
fused had the stronger relative result; a same-allocation A/B is required for
a defensible absolute RTX kernel ranking.

Every value and gradient was finite. Streamed complete-potential transit VJP
relative differences were at most `2.36e-9`; fused/streamed direct transit VJP
differences were at most `2.31e-10`. These timings supersede the earlier rough
estimate that non-transit work would dominate. On a long Prism block, the
differentiated light curve is still essentially the whole leapfrog cost, so a
roughly two- to threefold leapfrog-kernel gain is real. Whole-fit speedup still
depends on adapted tree length, compilation, I/O, and post-processing. Short
SOSS calls are near the GPU launch floor: keep `auto`/streamed there. On V100
and A100, explicit `fused` is the measured long-power-2 choice. Fused also had
the stronger paired result on RTX, but profile both routes in the same
allocation before treating the cross-job absolute timing as an RTX ranking.

### Earlier synthetic-window measurements

For 40 simultaneous power-2 light curves and a conservative window containing
about 35% of the cadences:

| Cadences | Full value+grad | Window value+grad | Speedup |
| ---: | ---: | ---: | ---: |
| 525 | 1.026 ms | 0.535 ms | 1.92x |
| 1458 | 2.718 ms | 1.034 ms | 2.63x |

The restored full-cadence flux was array-identical in both cases. Relative
gradient L2 differences were `6.6e-12` and `3.9e-12`, respectively. With a
shorter 19% active window, power-2 value+gradient speedups were 2.35x and
3.62x. The direct quadratic path was already cheaper, so its measured
value+gradient speedups were 1.75x and 1.61x at the shorter window.

The HAT-P-12 SOSS file has 254 raw cadences. After its configured early-time
mask, 89 of 226 retained cadences (39.4%) are in the exact transit window.
Its expected gain is therefore nearer the conservative 35% benchmark than the
larger 19% benchmark; longer observations benefit more because the fixed
transit interval is a smaller fraction of the full time series. A direct
power-2 check on those actual 226 timestamps produced array-identical flux and
a `1.3e-10` relative gradient difference between the original and windowed
paths.

A repeated cold 40-channel sampler comparison (300 warmup plus 300 draws) took
200.3 seconds for joint NUTS and 175.4 seconds for independent NUTS: 1.14x
wall-clock speedup. Independent NUTS reduced the mean leapfrog count from
241.6 to 42.1 with no divergences in either run, but each vmapped draw waited
for a mean lane maximum of 122.8 steps. This confirms that lane imbalance,
not the light-curve kernel, is currently the main independent-sampler limit.
Posterior radius-ratio medians agreed to at most 0.23 pooled standard
deviations in this short diagnostic run. Joint NUTS achieved 67.9
radius-ratio ESS/s versus 58.5 ESS/s for independent NUTS, so the independent
backend was 0.86x as efficient despite its shorter wall time.

A kernel speedup is not an end-to-end-fit claim. The exact window is enabled
by default for safe duration-parameterized jaxoplanet spectroscopy. Keep
`joint_nuts` for production: `independent_nuts` remains an experimental,
opt-in backend and is not recommended by the current ESS-per-second result.

Serial spectral chunks and difficulty-plan batches of the same resident width
now reuse shape-specialized joint, independent-NUTS, or independent-HMC
runners. Every call still performs fresh initialization, adaptation/warmup,
and sampling; observations, errors, time, priors, starting states, and PRNG
keys remain dynamic. Exact reused-versus-fresh regressions cover both
independent kernels after changing those numerical inputs. In a deliberately
tiny CPU diagnostic, independent NUTS changed from 27.949 seconds cold to
0.206 seconds steady with no new compilation event (135.7x); this isolates the
removed compiler tax and is not a claim of 135.7x for a production posterior.
The source-fingerprinted RTX6000/A100 ESS-per-second sweeps described in
`SPECTRO_ACCELERATION_RESULTS.md` provide the end-to-end sampler measurement.

## Real pipeline smoke test

`fit_jwst.py` completed HAT-P-12 SOSS order 2 on a V100 (Slurm job 52844385,
exit code 0) in 7:27 using 20 warmup and 20 diagnostic draws per stage. This
was an integration test, not a scientifically usable posterior. The fitted
white-light duration produced a 96/226 low-resolution window and a 96/225
reference-resolution window. The 7 low-resolution channels and 37
reference-resolution channels both ran through fixed 40-lane batches,
checkpointed, restored the expected `[20, 7]` and `[20, 37]` sample layouts,
and completed with zero divergences or non-finite samples. The wall time also
includes building three 125-point stellar limb-darkening grids in a fresh
output directory, so it is not a production ETA.

## Current scheduling limitation

Independent NUTS removes the artificial joint posterior and gives each lane
independent adaptation. JAX still executes a vmapped dynamic NUTS tree until
the deepest active lane finishes. The first V100 benchmark used only 34.3% of
the available lane-step slots (`42.1 / 122.8`), so a perfect asynchronous
refill scheduler has a 2.91x trajectory-work ceiling. Reordering the same
40-channel wave cannot change its maximum, and splitting it into several
smaller static waves loses enough GPU vectorization to be counterproductive
with the measured kernels. Exact continuous refill would require a resumable
NUTS tree or a custom persistent Pallas/CUDA work queue; NumPyro's current
monolithic `build_tree` does not expose a safe lane-level refill point.
