# Model grids

The [combining models tutorial](../tutorials/combining_models.md) introduces a comparison using the bundled SOSS observation.
This page describes the commands, weighting methods, and numerical checks in more detail.
The workflow starts with supported extracted-spectra FITS products.
Detector-ramp calibration and spectral extraction happen before koala reads the data.

## Fit a grid

Start from a transmission configuration that already works for a single fit:

```bash
JAX_ENABLE_X64=1 python -m koala.marginalization run visit.yaml \
  --workspace analysis/visit_grid --execution batched --method stacking
```

The default grid combines linear and quadratic trends, quadratic and direct power-2 limb darkening, and fixed and uniform coefficient priors.
Each of the eight candidates fits white light and high resolution.
There is no low-resolution stage.
Each candidate has its own geometry posterior, spectroscopic posterior, sampling adaptation, and output directory.

For a visit with a discontinuity, add the step model:

```bash
JAX_ENABLE_X64=1 python -m koala.marginalization run visit.yaml \
  --workspace analysis/visit_step_grid --execution batched \
  --trends linear,quadratic,linear_discontinuity --method pseudo_bma_plus
```

This gives twelve candidates.
The white-light step uses the ordinary linear-plus-sigmoid model and the configured `t_jump_guess` and `jump_guess`.
Its epoch and log-width have the same time coordinates and physical priors as a single fit.
Each spectroscopic candidate then uses its own fitted white-light sigmoid template and samples the jump amplitude.
Compatible polynomial fits provide starting values for the step fits.
Each candidate still estimates its own metric and adapts its own chains.

## Resume or rescore

To finish an interrupted grid or change the weighting method, use:

```bash
JAX_ENABLE_X64=1 python -m koala.marginalization fit \
  analysis/visit_grid/analysis.yaml --execution batched
JAX_ENABLE_X64=1 python -m koala.marginalization score \
  analysis/visit_grid/analysis_batched.yaml --method pseudo_bma_plus
```

koala checks the saved data and model identities before reusing fits.
Scoring caches also depend on the exact input and posterior content, model code, numerical-library versions, and precision.
Rescoring reads chunks without allocating a full model-by-draw-by-wavelength-by-time array.

The batched fitter currently supports one unspotted duration-based transit with sampled linear, quadratic, or linear-plus-sigmoid trends and fixed or uniform limb darkening.
Use ordinary grid fitting for other supported single-fit configurations.
Batched products are saved in `batched_fits/` and `batched_stage_inputs/`.
Their scoring manifest is `analysis_batched.yaml`.

CPU grid commands start with four worker cores.
Use `--cpu-cores N` to change that count or `--cpu-cores 0` to keep all available cores.
GPU configurations retain the CPU affinity of their allocation.
Library callers keep their own runtime configuration.

## Choose the weights

Predictive stacking and Bayesian-bootstrap pseudo-BMA+ follow [Yao et al. (2018)](https://arxiv.org/abs/1704.02030).
The channel hierarchy uses the predictive-mixture likelihood and partial pooling approach of [Yao et al. (2022)](https://arxiv.org/abs/2101.08954).
The reference-logit prior is specified below.

- `stacking` maximizes held-out predictive density over nonnegative weights that sum to one.
- `pseudo_bma_plus` averages normalized predictive scores over an exact Bayesian bootstrap.
- `hierarchical` samples partially pooled channel weights with a logistic-normal hierarchy and checks their convergence.
- `hierarchical_map` gives faster partial pooling at a fixed strength using a MAP approximation.

The default `--pooling channel` produces channel marginal spectra.
A row of those samples is not one coherent posterior draw across wavelengths.
Use `--pooling global` with stacking or pseudo-BMA+ for a common candidate choice and posterior row across the spectrum.
The high-resolution orbit remains conditional on each candidate's white-light fit.
Achromatic depth offsets remain in the mixture.

### The hierarchical prior

The sampled hierarchy fixes the first candidate's logit to zero.
For each other candidate, it samples `mu ~ Normal(0, 1)`, `tau ~ HalfNormal(1)`, and channel deviations `eta ~ Normal(0, 1)`.
The corresponding channel logit is `mu + tau * eta`.
Weight draws, model order, and diagnostic summaries are saved.

With more than two candidates, this prior depends on which candidate is first.
For three candidates with identical predictive scores, a prior-only calculation gives mean weights of approximately `(0.299, 0.351, 0.351)`.
With eight candidates, the first mean weight is about `0.078`, compared with `0.132` for each other candidate.
Identical scores leave the weight prior unchanged.
These numbers describe the prior, not a measured depth bias.
Record the candidate order and check sensitivity if you use this method.
Default predictive stacking does not use this prior.

## Check the comparison

All candidates must score identical times, fluxes, uncertainties, and wavelengths.
Preparation disables model-dependent sigma clipping and keeps the base configuration's common exclusions.
If you need clipping, supply common preselected exclusions.
The comparison uses direct likelihoods for each integration.
Integrated trend likelihoods need a different conditional-LOO treatment and are rejected here.

High or nonfinite Pareto $k$ values block the final spectrum.
The threshold is `min(0.7, 1 - 1/log10(posterior_draws))`.
The diagnostics remain available so you can inspect the affected observations.
The `--allow-unreliable` option marks the result as exploratory.
It does not establish reliable LOO scores.

The current scores use individual integrations under an independent Normal noise likelihood.
Time-block cross-validation and correlated-noise scoring are not implemented.
If the residuals remain correlated, consider that limitation before interpreting the weights.

## Extend the grid

The ordinary runner accepts custom axes or a YAML matrix.
For example, six supported trends, two laws, and three prior treatments give 36 candidates:

```bash
python -m koala.marginalization prepare visit.yaml --workspace analysis/grid36 \
  --trends constant,linear,quadratic,cubic,quartic,explinear \
  --treatments fixed,uniform,gaussian
JAX_ENABLE_X64=1 python -m koala.marginalization fit \
  analysis/grid36/analysis.yaml --execution sequential
```

The Gaussian treatment uses the configured Gaussian limb-darkening prescription.
For propagated stellar uncertainties or Sing calibration, specify the law and treatment in a matrix:

```yaml
dataset: visit
variants:
  - name: informative_power2_linear
    overrides:
      flags:
        detrending_type: linear
        ld_profile: power2
        ld_prior: stellarprior
```

This grid runner accepts `stellarprior` only with power-2 limb darkening.
The single-fit command also supports quadratic stellar priors.
Sing calibration requires a quadratic law.
One treatment is not silently substituted for the other.

## Reference

### Output files

| File | Contents |
|---|---|
| `marginalized/spectrum.csv` | Depths and mixture intervals. |
| `marginalized/posterior.npz` | Model names, weights, scores, Pareto diagnostics, absolute candidate depth medians, and mixture draws. |
| `marginalized/diagnostics.json` | Reliability, timing, interpretation of the mixture, and method-specific diagnostics. |
| `analysis_batched.yaml` | Scoring manifest for batched fits. |
| `batched_fits/`, `batched_stage_inputs/` | Batched posterior products and stage inputs. |

### Batched sampling

The fitter groups model variants and wavelengths into one compiled program per limb-darkening law and trend family.
Each fit gets its own MAP estimate, dense metric, dispersed chains, and adaptation.
Linear candidates have an unused quadratic coordinate, and fixed-coefficient candidates have unused limb-darkening coordinates.
These auxiliary coordinates have normalized priors and integrate out exactly.
They preserve the ordinary candidate's posterior, including its selected quadratic uniform-prior basis.
Free power-2 coefficients use the smooth Maxted transformation with its exact Jacobian.
Fixed-coefficient auxiliary variables have standard-normal priors.

Preparation, stellar objects, bandpasses, compatible starting values, and compiled programs are reused.
Starting values transfer only common parameters with matching shapes.
A previous candidate's metric or adaptation state is not reused as a new candidate's adaptation.
The sweep also omits a preliminary white-light MAP solve whose result the ordinary staged optimizer would discard.

The number of simultaneous fits bounds memory use.
Compatible chunks reuse a compiled sampler while supplying new data and starting values.
Chains with too few effective samples can continue without another MAP fit or warmup.
A well-mixed fit with a divergence first retries with a fresh Laplace metric and stricter acceptance.
Full adaptive metrics remain the fallback.
Failed posteriors are not released as a spectrum.

Acceptance requires zero divergences over the complete retained or continued history, bulk and tail ESS of at least 400, and rank-normalized R-hat below 1.01 for sampled parameters.
Configured higher ESS thresholds are retained.
Diagnostics record retries and their outcomes.

High-resolution starting values use each candidate's white-light trend and chromatic limb-darkening offset.
A union of fixed-geometry transit windows skips transit evaluation outside transit while preserving every observed integration in the likelihood.
White-light fits can also skip transit evaluation outside the full epoch and duration support when period is fixed.
Variable-period or unbounded priors retain all times.

CPU batches default to four simultaneous fits.
GPU batch widths depend on the memory budget.
Use `flags.vmap_chunk` to select a width.
Accepted posterior chunks are written to disk and checked before reuse or export.
Scoring reads one chunk at a time and keeps the full depth posterior only.

### Compilation and memory

Completed compilation caches are evicted when resident memory exceeds the grid budget.
Prepared data, stellar bandpasses, starting values, and persistent compiled programs remain available.
Set `flags.grid_cache_memory_mb` to change the budget (zero disables eviction).

GPU runs prime the actual sampling program from abstract state shapes.
NumPyro programs are traced serially, and their XLA compilations then overlap.
No disposable chains are sampled.
Use `flags.grid_prime_samplers: false` to disable this step.
Retry programs are cached by strategy and simultaneous-fit count, while observations, starts, and adaptation are fresh on every retry.

GPU scoring keeps a channel's full likelihood array on the device when it fits the memory budget.
It runs the same PSIS calculation there and transfers the reduced scores.
Larger arrays use the bounded-memory disk calculation.
The Bayesian bootstrap is also accumulated in blocks to limit memory.

### Transit calculations

Quadratic transits use closed-form occulted disk moments and analytic flux derivatives.
Complete elliptic integrals use eight Bulirsch duplication steps in float64; tests check convergence against twelve steps and independent numerical disk integrals.
Central transits, exact contacts and full occultation use their analytic limits.
For fixed shared orbits with a free channel epoch and a sampled linear trend, the Gaussian likelihood accumulates the flux and its parameter derivatives together.
The orbit calculation, priors and cadence selection are unchanged.

The tests include four derivative values at two extreme contact geometries where the previous quadrature is inaccurate.
Those values are checked against independent 70-digit boundary integrals at the exact binary64 radius and separation, rather than against the inaccurate quadrature.
The remaining contact, grazing and prior-boundary tests check both the previous implementation and a converged independent integral.

Long power-2 series can share small compressed table banks across fits and cache them on disk.
Each geometry and domain is checked for flux and derivative accuracy.
Outside a checked domain, the sampler uses direct integration at the observed times.
The full radius prior is preserved.
Near inner and outer contacts and the midpoint, 128-point physical corrections cover up to sixteen cadence samples.
Dense or coalescing contact neighborhoods retain full physical integration.
Shared routing information can narrow a checked table domain but cannot widen its radius, coefficient, exponent, or geometry support.
Checking flux alone does not establish correct derivatives at a moving contact.

The high-resolution batched power-2 sampler uses 64 integration points when all fixed exponents are within the checked range [0, 10].
White-light fits retain 128 points for timing and duration derivatives.
The high-resolution calculation is checked against a 512-point reference for central, grazing, coincident, and contact geometries.
Larger exponents and the ordinary fitter retain 128 points.
Independent annular integration also checks the coincident radius/separation derivative limit.

### Archived performance measurements

The recorded CPU injection benchmark used 257 integrations, four channels, four chains, and 1,000 retained draws per chain.
All eight candidates passed the sampling and PSIS checks.
Fitting and stacking took 38.4 minutes, compared with 124 seconds for the fiducial single model (18.6 times as long).
The ordinary eight-model sweep took 58.0 minutes, so batching reduced that time by about one third.

A separate recorded SOSS test used eight native input channels and 537 integrations.
Its eight-model workflow took 68.0 minutes, compared with 162 seconds for the single model (25.1 times as long).
These archived measurements are not timings for the current complete bundled SOSS example.
They also do not establish performance for the full native SOSS spectrum or a 36-model grid.

Reopening the accepted synthetic grid and regenerating its spectrum took 12.2 seconds in a fresh process.
All posterior and score caches were reused, and posterior archive hashes were unchanged.
This measures resuming a completed analysis.
Pseudo-BMA+, partially pooled MAP weights, and sampled hierarchical weights also passed validation on both accepted grids.

The local `analysis/model_grid_validation/validation_report.md` records the timings, source identity, diagnostic checks, logs, and single-injection recovery.
To reproduce the benchmark in a CPU or GPU allocation with 64-bit precision, run `python tools/benchmark_grid_end_to_end.py NEW_DIRECTORY --execution batched`.
After it succeeds, run `python tools/benchmark_grid_resume.py NEW_DIRECTORY`.
Wider CPU batches and short-series table sampling were measured and were slower, so they are not defaults.
Fitting a grid still requires separate posteriors for all its candidates.
