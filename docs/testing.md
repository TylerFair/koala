# Testing and development

If you'd like to contribute, start by running the tests in a separate environment.
The CI setup uses Linux, Python 3.10, two CPU cores, and a 7 GB memory budget per shard.
Python 3.11 is also tested on pushes to main.
Each pull request runs all eight Python 3.10 shards.

## Run the tests

First, install the test dependencies with a C++ compiler available:

```bash
python -m venv .venv
source .venv/bin/activate
export PIP_CONSTRAINT="$PWD/tests/requirements-ci.txt"
export PIP_NO_COMPILE=1
python -m pip install --upgrade pip
python -m pip install -e . pytest pytest-xdist batman-package
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false"
python tests/run_ci.py --shard 0 --shards 8 --output test-results/shard-0
```

Run shards 0 through 7 for the complete CI selection.
Use `--plan-only` to print the file assignments before running.
Every `tests/test_*.py` file is assigned once, including files without a recorded duration.
The timing manifest `tests/ci_durations.json` helps balance the shards.
Unknown files receive a default weight.

Each file runs in a fresh pytest process, releasing JAX compiled programs between files.
The runner excludes tests marked `slow` or `gpu` and saves per-file logs and JUnit reports.
It limits each file to ten minutes and caps both subprocess address space and aggregate process-group RSS at 7 GB.
A timeout, memory limit, collection error, or test failure fails the shard.
Entirely deselected slow files and optional-data skips are recorded explicitly.

For a direct local run, use `pytest -n auto -m "not slow and not gpu"`.
Automatic xdist concurrency is capped at four, and workers split Linux CPU affinity to bound the compiler thread pools.
These workers stay alive between files.
Use the shard runner when you need to reproduce CI memory behavior.
To include slow checks, use `pytest -n 8 --dist loadfile --durations=0`.
On a cluster, run these commands in a compute allocation.

## Read a failure

The runner keeps failure IDs, per-file logs, JUnit results, peak and aggregate RSS, and elapsed times even when a shard fails.
GitHub uploads them with `if: always()`.
Look at the failing file's log first.
A local pass does not establish that the corresponding GitHub job passed.

The `slow` marker identifies CPU tests measured above about 30 seconds.
The two long Harmonica derivative cases measured 230 and 287 seconds and remain in the full suite.
The `gpu` marker is reserved for tests that require CUDA.
Simulated GPU routing still runs on CPU.
The `harmonica` marker identifies C++ extension checks.
Optional atmosphere-data tests skip with a reason when their local data are unavailable.

Every CI job installs and smoke-checks Harmonica.
The constraints select the CPU numerical stack, and the runtime Git dependencies are fetched at pinned commits.
Missing Harmonica APIs fail the smoke check.
The batman package supplies an independent transit reference.
Pip bytecode generation is disabled to reduce shared-filesystem writes.
The one-core startup fixture checks that XLA flags remain unchanged when no extra host devices are needed.

## Check a numerical change

Compare the same orbit, epochs, limb-darkening law, and normalization in float64.
The reference calculation sets the useful numerical floor.
A universal $10^{-9}$ tolerance can confuse reference error with a regression.
The table below records the comparison classes used here.
It is not a reason to loosen existing assertions.

Check gradients separately against an exact calculation or converged finite differences.
The gradient threshold is `1e-7 + 1e-6 * abs(reference)`.
For parameter recovery, the criterion is model-induced bias no larger than $0.1\sigma$ of the most precise measurement.
Check this with exact-truth injections fitted through the production inference path.
A small flux residual alone does not establish that bound.

## Reference

### Numerical comparisons

| Comparison | Tolerance or reference floor | Interpretation |
|---|---|---|
| Exact koala counterparts (streamed/stock, prepared/direct orbit, contact-safe/stock) | Absolute flux error ≤ `1e-12 + 1e-10 * depth` | Exact rearrangements preserve the model. Contact handling must also preserve finite, correct derivatives. |
| External batman quadratic | Approximately 1e-9, with a measured maximum of 4.4711e-9 in the fixture below | Independent surface integration places this fixture's residual in batman. |
| External batman power-2 | Approximately 1e-6 | Check the reference's own integration error. |
| External catwoman | Approximately 1e-8 | Match asymmetric geometry and conventions. |
| Degree-12 power-2 projection and transit grids | A documented bound for the configuration | These approximations are not defaults or exact parity references. |

### The quadratic batman residual

The test `tests/test_spectro_free_t0.py::test_channel_epochs_match_independent_batman_and_keep_orbit` uses $P=4$ d, $a/R_*=9$, and $b=0.35$.
Its two channels have radius ratios 0.12 and 0.18, quadratic coefficients (0.3, 0.2) and (0.4, 0.1), and epoch offsets −20 and +30 s.
The 151 observation times span [−0.11, 0.11] d.
There is no trend.

An independent area integral evaluates

$$F=1-\frac{\iint_{\text{planet}\cap\text{star}}
 [1-u_1(1-\mu)-u_2(1-\mu)^2]\,dx\,dy}
 {\pi(1-u_1/3-u_2/6)},\qquad \mu=\sqrt{1-x^2-y^2}.$$

We rotate the planet center to $(z,0)$ and split the integral at the stellar and planetary circle intersection.
An analytic integral over each column, followed by adaptive outer quadrature, agrees with koala within $9.1\times10^{-14}$ and $2.6\times10^{-13}$ for the two channels across all times.
At each channel's largest koala–batman discrepancy, independently nested numerical 2D quadrature gives:

| Channel | $z$ | Surface-integrated flux | koala − integral | batman − integral |
|---|---|---|---|---|
| 0 | 0.8955504863074729 | 0.9877871715939092 | < 1.1e-16 | -4.4710361e-9 |
| 1 | 0.8751020753777734 | 0.9740893498586976 | < 1.1e-16 | +3.4105521e-9 |

Tightening the nested quadrature from 2e-13 to 2e-14 changes neither reported flux.
The nested and analytic-column results differ by at most 6.6e-14.
This assigns the few-e-9 residual to batman at these parameters.
It does not establish either code's accuracy for every geometry.
The existing timing-test tolerances remain unchanged.

To reproduce the calculation from the repository root on a compute node, run:

```bash
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. python tests/reference_quadratic_integration.py
```

The script writes `tests/reference_data/quadratic_surface_integral.json` and requires SciPy and batman from the test environment.
It is a separate reference calculation, outside normal pytest collection.
