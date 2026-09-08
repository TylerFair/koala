# Gaussian processes

Koala's white-light GP trends use the exact quasiseparable Matérn-3/2 kernel
from tinygp. The serial and parallel solvers represent the same covariance
model. Solver selection changes its execution algorithm, not the statistical
model.

## Install parallel tinygp

`requirements.txt` always installs tinygp from the Koala fork at a validated
commit (branch `koala-stable`), so a fresh Koala environment on Python 3.10 or newer has the parallel
solver. The direct installation command is:

```bash
python -m pip install 'tinygp @ git+https://github.com/TylerFair/tinygp.git@96d110c8fb4350b0daf50f7f1f86a9196b82fcd4'
```

The fork (`TylerFair/tinygp`) tracks upstream `dfm/tinygp` `main`, which
merged the parallel associative-scan solver as PR 269; the only fork-side
change is allowing installation on Python 3.10. The pinned revision is
byte-identical in `src/` to the branch revision on which the measurements
below were made. An environment that still carries tinygp 0.3.x from PyPI
keeps working in serial mode. See the full
[installation guide](../install.md) before installing a CUDA-enabled JAX
wheel.

## Select a solver

Set `KOALA_GP_SOLVER` before starting Python:

| Value | Meaning |
|---|---|
| `auto` | Default. Use parallel on GPU or TPU when supported; use serial on CPU. An older tinygp falls back to serial with a warning. |
| `serial` | Always use the native serial quasiseparable solver. This also works with an older PyPI tinygp 0.3.x. |
| `parallel` | Require parallel associative scans. An unsupported tinygp raises an actionable error instead of silently changing algorithms. |

A configuration file can pin the choice for one run with the optional
top-level key `gp_solver: auto|serial|parallel`, which overrides the
environment variable. The solver is not part of the white-light artifact
fingerprint because serial and parallel modes evaluate the same model.

The Python builders expose the equivalent `gp_solver=` argument. They also
accept `gp_assume_sorted=`. The safe default is `False`, which keeps tinygp's
sortedness check. Koala validates the white-light timestamps once on the host
and then passes `gp_assume_sorted=True`; duplicates are allowed, but times must
be finite, one-dimensional, nonempty, and nondecreasing.

## Mean functions and cached fits

Every GP mean is now evaluated relative to one explicit training-set
reference time, `t_ref`. Previously, tinygp called a mean function once per
scalar timestamp; an expression such as `t - min(t)` then became zero for
every cadence. That erased polynomial time terms and made the exponential
ramp constant inside the GP likelihood. Holding `t_ref` fixed restores the
intended linear, polynomial, and exponential trends.

The mean-function revision is part of the white-light artifact fingerprint.
Consequently, a cached white-light fit created with the earlier mean semantics
is deliberately recomputed. Reusing it would mix posterior parameters from a
different likelihood with the corrected prediction.

## Linear-memory training prediction

Training-point conditional means and marginal variances are computed directly
from quasiseparable triangular solves and the diagonal of the inverse
covariance. This replaces a path that requested the training timestamps as a
test grid and could materialize a dense $N\times N$ covariance. The new path
is $O(N)$ in timestamp-space storage and preserves the output `gp_flux`,
`gp_err`, and `gp_trend` columns consumed by the spectroscopic stages.

Before Laplace preparation the white-light stage also checks the optimizer's
GP hyperparameter starts. A constrained optimizer can leave `GP_log_sigma` or
`GP_log_rho` on a Uniform prior edge, where the unconstrained coordinate is
infinite and NumPyro cannot initialize; such a start is moved one percent of
the support width inside the edge, with a printed warning, and the Laplace
MAP refinement continues from there.

Laplace-NUTS still uses its dense metric in the much smaller inferred-
parameter space. Only dense timestamp-space prediction covariances have been
eliminated; the sampler's parameter-space metric and scientific model are
unchanged.

## Correctness and performance status

CPU validation checks serial and parallel log likelihoods, derivatives, and
training-point predictions against each other, with a small dense Cholesky
oracle. The recorded CPU findings below validate correctness only:

In the focused float64 CPU audit at 17 irregular cadences, serial and parallel
log probability agreed exactly. Their maximum absolute differences were
$5.33\times10^{-15}$ for the gradient, $8.70\times10^{-14}$ for the exact
Hessian, $2.33\times10^{-15}$ for forward triangular solves,
$2.22\times10^{-16}$ for the training posterior mean, and
$4.17\times10^{-18}$ for the training marginal variance. Comparisons with
dense JAX, SciPy, and NumPy references were at approximately
$10^{-13}$ or better.

CPU timings are not evidence of GPU behaviour: on the CPU the parallel scan
is slower than the serial recursion, which is why `auto` selects serial
there. A larger CPU oracle check at 512 irregular cadences with one duplicate
timestamp, in both solver modes, agreed with a dense NumPy Cholesky reference
exactly in the log likelihood and to $2\times10^{-16}$ in the training
posterior mean.

### GPU measurements

`tools/benchmark_gp.py --require-gpu --cadences 40000` was run on one Tesla
V100-PCIE-16GB in float64 with tinygp `0.3.2.dev7+gf4e651d26` and JAX 0.6.2.
Steady-state medians of five synchronized runs, with GP construction inside
each timed call:

| N | Workload | Serial (ms) | Parallel (ms) | Ratio |
|---:|---|---:|---:|---:|
| 1,024 | log likelihood | 36.6 | 2.5 | 15 |
| 1,024 | likelihood + gradient | 123.2 | 7.3 | 17 |
| 1,024 | training mean + variance | 104.0 | 4.1 | 25 |
| 4,096 | log likelihood | 113.2 | 4.1 | 28 |
| 4,096 | likelihood + gradient | 506.6 | 7.5 | 68 |
| 4,096 | training mean + variance | 414.7 | 5.1 | 82 |
| 16,384 | log likelihood | 455.6 | 4.7 | 96 |
| 16,384 | likelihood + gradient | 1788.9 | 14.2 | 126 |
| 16,384 | training mean + variance | 1639.1 | 5.6 | 294 |
| 40,000 | log likelihood | 1388.0 | 5.4 | 256 |
| 40,000 | likelihood + gradient | 4329.6 | 22.9 | 189 |
| 40,000 | training mean + variance | 3971.2 | 8.6 | 463 |

Compilation is the price: the parallel kernels compile in 4 to 14 s per
workload against 0.4 to 1.4 s for the serial ones. Serial and parallel modes
agreed exactly in the log likelihood at every size, to at most
$6.4\times10^{-12}$ absolute in the gradients, and to at most
$1.8\times10^{-15}$ in the training prediction; both agreed with the dense
float64 reference at 128 points to $7\times10^{-14}$ or better. The GPU
correctness suites (59 tests) pass on the same device with `KOALA_GP_SOLVER`
unset, `serial`, and `parallel`. Peak device memory during the benchmark was
276 MB.

These are per-evaluation GP kernel timings. A white-light fit also spends
time in the transit model, the optimizer, and NUTS warmup, so the end-to-end
gain is smaller than the ratios above and depends on the cadence count.

Run the focused GPU correctness suite inside a GPU allocation:

```bash
JAX_PLATFORMS=gpu KOALA_GP_SOLVER=parallel python -m pytest \
  tests/test_gp_solver.py tests/test_gp_training_prediction.py \
  tests/test_gp_mean_tref.py tests/test_gp_laplace_nuts.py
```

The benchmark records compilation, first execution, synchronized steady-state
median/minimum, numerical agreement, package revisions, device identity, and
memory statistics when the backend provides them:

```bash
python tools/benchmark_gp.py --require-gpu --json gp_benchmark.json
```

Its defaults are 1,024, 4,096, and 16,384 cadences in float64. Use
`--float32`, `--sizes`, `--cadences`, `--duplicates`, `--solver`, and
`--repeats` for controlled variants.

### Upstream alternatives that were measured and not adopted

tinygp `main` merged a memory-efficient `predict(return_var=True)` path
(PR 281) after the pinned revision. Its quasiseparable diagonal uses
`factor.inv()`, a QSM product, and `gram()`, none of which take the
`parallel` flag, so the variance stays on the serial recursions even when the
solver runs in parallel mode. Measured on the same V100 in float64, both
formulas agree to $3\times10^{-21}$; the koala helper based on
`matrix.inv(parallel=True)` is the one that scales:

| N | PR 281 formula, serial | PR 281 formula, parallel solver | Koala helper, parallel |
|---:|---:|---:|---:|
| 1,024 | 121 ms | 72 ms | 3.4 ms |
| 4,096 | 485 ms | 365 ms | 3.1 ms |
| 16,384 | 1919 ms | 1101 ms | 5.6 ms |
| 40,000 | 4660 ms | 3489 ms | 7.9 ms |

The open PR 272 computes predictive means and variances at arbitrary new
timestamps from the Cholesky carry in $O(J^2)$ per point. Koala only conditions
the white-light GP at its training timestamps, so it has no call site that
would use it, and the branch is unreviewed and serial-only.

## Matched full-fit comparison

Copy one science configuration twice and give each copy a distinct
`output_dir`, for example `results/gp_serial` and `results/gp_parallel`. Keep
all data, model, seed, and sampler settings identical. From the repository
root, run the canonical command once per solver:

```bash
KOALA_GP_SOLVER=serial python fit_jwst.py -c config_serial.yaml
KOALA_GP_SOLVER=parallel python fit_jwst.py -c config_parallel.yaml
```

Compare the two `*_whitelight_bestfit_params.csv` tables and the two
`*_whitelight_GP_database.csv` tables, including `gp_flux`, `gp_err`, and
`gp_trend`. Also inspect the white-light residual plots and convergence
diagnostics. Separate output directories are essential: otherwise cached
artifacts can obscure whether both solver paths actually ran.

This comparison was run on one V100 with the bundled WASP-39 SOSS extraction
(537 white-light cadences, `linear+gp`, R5 and R12 stages, identical seed).
Both runs finished with 1,200 retained white-light draws and no divergences.
Posterior medians differed by at most 0.03 sigma in the white-light geometry
and by at most 0.04 sigma (median 0.014 sigma) in the R12 transit depths;
`gp_flux` agreed to $1.6\times10^{-6}$ and `gp_err` to $5\times10^{-8}$. The
remaining differences are chain-to-chain sampling noise: the two solvers
round differently, so the NUTS trajectories diverge after the first step.
The same configuration with `detrending_type: linear` was run on the same
device as a no-GP baseline. Stage wall times from the output timestamps:

| Stage | no GP | serial GP | parallel GP |
|---|---:|---:|---:|
| White light (optimizer, Laplace, NUTS) | 143 s | 1633 s | 2009 s |
| R5 + R12 spectroscopy | 188 s | 193 s | 190 s |
| NUTS wall per white-light draw, Laplace chain | 0.02 s | 1.1 s | 0.14 s |

The spectroscopic stages are unchanged because the GP never enters them:
they use the saved white-light trend as a fixed regressor. The parallel
white-light stage was longer than the serial one only because its Laplace
chain failed the quality gate and the adaptive fallback chain ran at the
tree-depth limit for 18 minutes; per draw, the parallel GP chain was eight
times faster than the serial one and seven times slower than the no-GP
chain at this small cadence count, where compilation overhead sets the
floor. At PRISM cadence counts the per-gradient measurements above apply:
a serial GP white-light chain would take days, a parallel one tens of
minutes.
