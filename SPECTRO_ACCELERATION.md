# Spectroscopic fitting acceleration

The production path now supports measured GPU resident widths, independent
per-channel samplers, exact Gaussian marginalization of conditionally linear
trends, and difficulty-aware channel batches.  The science-preserving kernel
and transit-window optimizations remain enabled by `auto`.  Experimental time
binning and a power-2 grid emulator are isolated behind validation gates and
are not used by `fit_jwst.py`.

Measured results and active GPU job identifiers are recorded in
`SPECTRO_ACCELERATION_RESULTS.md`.

No single sampler is assumed to be fastest for every target.  Use the supplied
benchmark on the same GPU model, cadence count, sampler, and trend treatment as
the intended fit.  A width manifest is emitted only when its divergence, ESS,
and measured-memory gates pass.

## Production defaults (2026-09-02)

With no sampler flags, spectroscopy uses per-channel Laplace-metric NUTS and
white light uses a Laplace metric with its existing adaptive fallback. SOSS,
G395H, G395M, and G140H use white-light target acceptance 0.90; PRISM uses
0.99. Spectroscopy uses 0.95 and tree depth 5, except PRISM and explinear use
0.99 and PRISM uses depth 6. The shared Laplace controls are 150 warmup steps,
finite-difference Hessians, trust radius 5, depth ESS at least 400, and zero
divergences.

If a Laplace-NUTS lane fails, only that lane is rerun with exact HMC-8 using
25% trajectory jitter and target acceptance 0.85, then with legacy adaptive
joint NUTS if needed. Selecting `spectro_sampler: independent_hmc` reverses
the first two samplers. Checkpoints, chunk diagnostics, and transmission CSVs
record `sampler_used` per wavelength channel. These are all exact MCMC paths.

Exponential-linear spectroscopic trends now fix their timescale to the
white-light posterior median by default; set
`spectro_fixed_timescale_trends: false` for the historical free-timescale
model. `spectro_ld_parameterization` and `whitelight_ld_parameterization`
default to `coefficients`. The opt-in `latent_gaussian` setting samples an
independent standard normal for each bounded free/wide LD coefficient and
maps it through that coefficient prior's inverse CDF, exactly preserving the
physical prior. The older `decorrelated` and `decorrelated_linear` experiments
remain opt-in. None is selected automatically: the real wide-Gaussian SOSS
benchmark failed the zero-divergence and per-lane LD-ESS gates.

## Recommended first configuration

Add these values under `flags:` in a copy of a science configuration:

```yaml
flags:
  # Existing science choices
  ld_profile: power2

  # Exact forward-model optimizations
  transit_window_optimization: auto
  jaxoplanet_kernel: auto

  # One independently adapted low-dimensional chain per wavelength channel
  spectro_sampler: independent_nuts
  vmap_chunk: 40                 # replace with a measured manifest below

  # White-light geometry handed coherently to the spectroscopic fits
  whitelight_geometry_estimator: posterior_median   # default since 2026-09-02; 'max_likelihood_draw' remains available
  whitelight_log_likelihood_batch_size: 64

  # Keep the established science model for the first equivalence run
  trend_inference: sampled_uniform

  # Expose pathological gradients early
  spectro_gradient_diagnostic: first
  spectro_gradient_diagnostic_strict: false
```

`max_likelihood_draw` means the finite retained white-light posterior draw with
the largest summed observed-data log likelihood.  All fixed geometry values
(`t0`, `duration`, `b`, `a_rs`, `cos_i`, inclination, and `rors`) come from that
same draw.  It is deliberately not a mixture of marginal medians and it is not
a new unconstrained continuous optimizer.  The selected indices, likelihood,
geometry, and source fingerprint are written atomically to
`*_whitelight_geometry_handoff.json`.  A stale or median-era cache is rejected.
The ordinary white-light summary CSV remains a posterior-median reporting
product; spectroscopic fixed geometry comes from the separate handoff JSON.

The fitting code requires JAX 64-bit mode.  Float32 is not enabled by any speed
configuration because the orbital geometry and ingress/egress gradients need a
separate end-to-end numerical validation before that would be scientifically
defensible.

## Measure an A100 or RTX6000 resident width

The A100 runner sweeps widths 8, 40, and 80 across joint NUTS, independent NUTS,
and fixed-step independent HMC, with sampled and marginalized trends in fresh
subprocesses so an OOM is recorded rather than killing the suite:

```bash
sbatch tools/run_spectro_sampler_benchmark_a100.sbatch
# or
sbatch tools/run_spectro_sampler_benchmark_rtx6000.sbatch
```

For another GPU or workload, run the benchmark directly in a GPU allocation and
match `--cadences`, warmup, samples, backend, trend mode, and candidate widths:

```bash
python tools/benchmark_spectro_samplers.py \
  --platform gpu \
  --backends joint_nuts,independent_nuts,independent_hmc \
  --trend-modes sampled_uniform,gaussian_marginalized \
  --resident-widths 8,24,40,56,80 \
  --cadences 513 \
  --warmup 300 \
  --samples 300 \
  --repeat-runs 2 \
  --manifest-backend independent_nuts \
  --manifest-trend-mode sampled_uniform \
  --emit-width-selection logs/spectro_width_selection.json \
  --json logs/spectro_sampler_benchmark.json
```

Consume a passing selection with either the generic or stage-specific key:

```yaml
flags:
  spectro_width_selection: logs/spectro_width_selection.json
  # or independently:
  lowres_width_selection: logs/lowres_width_selection.json
  highres_width_selection: logs/highres_width_selection.json
```

The selected width was measured, not extrapolated.  `fit_jwst.py` verifies the
manifest fingerprint and rejects it if its measured peak exceeds the current
device-memory budget.  A manifest is still workload-specific; do not reuse one
across materially different cadence counts, sampler backends, trend modes, or
GPU models without rerunning the sweep.

All candidate widths in one sweep are exact prefixes of one synthetic
realization at the largest requested width, with a nested low-discrepancy
channel order and width-independent problem and sampler seeds. Thus short
prefixes span the modeled spectral range rather than selecting one edge. The
Slurm launchers infer the allocator limit reported by JAX;
they do not assume a marketing-memory capacity. If a memory quality constraint
is requested but JAX cannot report peak device use, that candidate fails the
quality gate. Peak memory is a conservative process-lifetime high-water mark;
the harness releases posterior references between repeats, while the JAX
allocator may retain freed buffers.

The launcher defaults (`WARMUP=100`, `SAMPLES=100`, `CADENCES=513`) are a
synthetic comparison workload. Set all three to the exact production-stage
values before emitting a fit-consumable width manifest; the fitter rejects a
manifest measured under different values.

The suite reports cold compilation time, steady-state wall time, peak device
memory when exposed by JAX, divergences, leapfrog work, per-channel radius-ratio
ESS, and ESS/s.  Judge samplers by quality-gated ESS/s, not raw wall time.
The benchmark and production chunk router reuse one compiled runner for every
equal resident width.  Time, flux, errors, initial states, per-channel priors,
shared numerical arguments, and PRNG keys remain dynamic; incompatible static
configuration changes fail closed.  This also applies across separate regular
batches in a difficulty plan.

## Remove conditionally linear trend parameters

The opt-in marginalized mode integrates the linear coefficients exactly in a
small coefficient-space system and samples them afterward from their exact
conditional Gaussian.  It never constructs a cadence-by-cadence covariance
matrix.

```yaml
flags:
  trend_inference: gaussian_marginalized
  trend_prior_means:
    c: 1.0
    v: 0.0
  trend_prior_scales:
    c: 0.1
    v: 0.1
```

The mode covers polynomial, explinear (conditional on sampled `tau`), fixed
spot-template, two-spot-template, and discontinuity-template coefficients.  GP
trends and `detrending_type: none` fail closed.  It is available only for the
Jaxoplanet spectroscopic model.

This changes the prior family: the existing path uses bounded Uniform priors,
whereas exact analytic integration uses proper Gaussian priors.  Broad
Gaussians approximate the interior of the old bounds but are not mathematically
identical near their edges.  Before adopting it for production, compare depths
and uncertainties on representative real targets and record the prior scales.
Do not interpret a timing ratio between these two modes as posterior-equivalence
evidence.

Unless explicitly supplied in `trend_prior_means`, Gaussian prior means are
taken from the same white-light/low-resolution initialization used by the
sampled fit.  That is an empirical-Bayes choice.  Record it, perturb the means
and scales in sensitivity tests, and do not present marginalized and sampled
results as identical-prior analyses.

`interpolate_trend` is fail-closed unless the engine is Jaxoplanet and the
spectroscopic detrend is exactly linear.  The current low-to-high-resolution
handoff constructs only fixed `c` and `v`; other trend families require a
complete coefficient handoff before they can be enabled safely.

## Prevent one hard channel from slowing every lane

Run a short `independent_nuts` pilot with checkpointing.  Each chunk writes a
`*.diagnostics.json` containing globalizable per-channel tree depth, divergence,
and adapted-step-size information.  Aggregate diagnostics with explicit global
indices; never infer indices from filename ordering:

```bash
python tools/build_spectro_batch_plan.py \
  --pilot-chunk '0:40=OUTPUT/chunks/first.diagnostics.json' \
  --pilot-chunk '40:80=OUTPUT/chunks/second.diagnostics.json' \
  --nominal-width 40 \
  --quarantine-width 1 \
  --output logs/highres_batch_plan.json
```

Then set:

```yaml
flags:
  highres_batch_plan: logs/highres_batch_plan.json
  # lowres_batch_plan and spectro_batch_plan are also accepted.
```

The plan groups similarly difficult channels and gives quarantined channels a
resident width of one.  It is fingerprinted, must cover every channel exactly
once, participates in checkpoint fingerprints, and restores samples to the
original wavelength order.  Existing `chunk_mode: parallel` and `combine`
workflows operate on whole planned batches.  Each current pilot diagnostic is
also bound to a sampling-workload fingerprint over the full time/flux/error
arrays, model arguments, priors, sampler controls, and relevant source files.
The plan builder requires every chunk to carry the same fingerprint, and the
fit recomputes it before sampling.  Legacy plans and same-sized plans from a
different visit are rejected.

Fixed-step HMC is another opt-in response to lockstep NUTS trees:

```yaml
flags:
  spectro_sampler: independent_hmc
  spectro_hmc_num_steps: 16
  # optional stage overrides:
  lowres_hmc_num_steps: 16
  highres_hmc_num_steps: 16
```

Each channel still adapts its own step size and mass matrix, but all lanes use
the same explicit leapfrog count.  Select that count by real posterior ESS/s,
divergences, depth agreement, and uncertainty agreement; it is not a safe
drop-in default merely because its wall time is predictable.

## Experimental paths

`models/temporal_binning.py` proposes out-of-transit-only temporal compression.
It preserves all t1--t4 cadences plus a buffer, never crosses time gaps, includes
the exact within-bin Gaussian likelihood correction, and requires an ensemble
forward-model error and projected-gain validation.  Rejected proposals cannot
be consumed.  It is intentionally not wired into production fitting.

`models/jaxoplanet/experimental_power2_grid.py` implements a differentiable
trilinear power-2 grid for fixed geometry and cadence.  Artifacts include grid,
geometry, source, and package fingerprints; out-of-domain or unvalidated use
fails.  Evaluate it with:

```bash
sbatch tools/run_power2_grid_emulator_a100.sbatch
# or
sbatch tools/run_power2_grid_emulator_rtx6000.sbatch
```

The emulator remains outside `fit_jwst.py` until it clears flux and gradient
gates on representative targets and demonstrates NUTS posterior equivalence.
Multilinear gradients are discontinuous at knots, so forward accuracy alone is
not sufficient.

On an RTX6000, the equal-memory 81-knot linear prototype measured 20.81x
forward and 12.81x value-plus-gradient speedups, but its maximum radius-gradient
error was 651.7 ppm per unit.  Equal-memory cubic and exact-derivative Hermite
variants did not improve that gate: their worst errors occur at second/third
contact, where the overlap curve is not smooth enough for cheap interpolation
to control the uniform derivative error.  Those variants are a no-go.  The
linear prototype warrants at most one bounded posterior/energy equivalence
experiment; it is not production-enabled.

## Production acceptance checks

For a representative subset of targets and wavelength channels:

1. Compare the same retained-data mask and identical white-light geometry.
2. Require no material increase in divergences or tree-depth saturation.
3. Compare depth medians and credible-interval widths in units of the reference
   posterior uncertainty, including the worst channel rather than only the
   wavelength median.
4. Compare per-channel ESS/s and peak memory after compilation.
5. Repeat the resident-width sweep separately on A100 and RTX6000.
6. Promote only combinations that pass both posterior and numerical-gradient
   checks; otherwise keep the established sampled-Uniform joint-NUTS result.
