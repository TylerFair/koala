# Production-stage sampler replay harness

Worker: `harness`  
Date: 2026-09-01

## Outcome

I built an opt-in, pre-sampling dump at the common spectroscopic sampling
choke point and an offline sampler driver.  Six real CPU-produced dump files
now cover low- and high-resolution SOSS order 1, G395H/NRS1, and
Jaxoplanet/Power-2 PRISM/NRS1 calls.  Loading each file rebuilds the recorded
model constructor and checks the stored initial potential with an absolute
tolerance of `1e-8`.

The dump is a no-op unless `FIT_JWST_DUMP_SAMPLER_INPUTS` is set.  The normal
NumPyro model, priors, likelihood, masks, float64 setting, and transit physics
were not changed.  No speedup is claimed in this report: the measured real
stage runs below are execution smoke tests of the harness, not like-for-like
sampler comparisons.

## Files built or changed

| File | Purpose |
|---|---|
| `fit_jwst.py` | Opt-in atomic dump before `_run_sampling_stage` dispatch; exact model-builder provenance; stored initial potential/unconstrained state; clean first-high-resolution exit; dump-generation-only CPU/low-resolution bridge controls; masked-time diagnostic fix described below. |
| `tools/spectro_stage_inputs.py` | `StageInputs`, `load_stage_inputs`, model reconstruction, initial-potential validation, and production-equivalent channel slicing. |
| `tools/run_sampler_on_stage_inputs.py` | Offline chunked CLI for `joint_nuts`, `independent_nuts`, `independent_hmc`, and dynamically imported `models.<backend>` implementations such as future `models.laplace_is`. Writes pickle/NPZ samples, raw and summarized sampler diagnostics, compile/steady timing JSON, ArviZ diagnostics, and fidelity comparisons. |
| `tests/test_stage_inputs_dump.py` | Tiny CPU dump/loader/potential/slicing test and strict no-op test. |
| `configs_accel/HAT-P-12_soss_order1_accel_dump.yaml` | SOSS CPU capture config; 50/50 white-light geometry fit. |
| `configs_accel/GJ3470_nrs1_g395h_accel_dump.yaml` | G395H CPU capture config; 50/50 white-light geometry fit. |
| `configs_accel/HAT-P-65_nrs1_prism_jaxoplanet_accel_dump.yaml` | Jaxoplanet, fixed Power-2 LD, R50/R10 PRISM derivative; 20/20 white-light geometry fit. |
| `acceleration_reports/sbatch/run_sampler_on_stage_inputs.sbatch` | Parameterized GPU template using `pi-ekempton`, `gpu`, one GPU, CUDA 12.9, and the jaxoplanet environment. It was written only and was not submitted. |

The dump payload records the importable builder identity and all builder
arguments (`detrend_type`, `ld_mode`, `trend_mode`, `ld_profile`,
`param_method`, `transit_window`, `transit_window_indices`,
`jaxoplanet_kernel`, and `n_planets` for these models), plus `t`, `yerr`,
`indiv_y`, physical `init_params`, RNG key, NUTS/MCMC options, configured and
effective chunk sizes, backend, channel-varying-key names, all runtime model
kwargs, checkpoint signature, wavelength metadata, configuration path, and
stage dimensions. JAX arrays are copied to NumPy before pickling and the file
is installed atomically with `os.replace`.

## Real dump inventory

Root: `/scratch/midway3/tfairnington/accel_stage_inputs/`

Every row below contains the requested production spectroscopic controls
(`joint_nuts`, 1000 warmup, 1000 samples, maximum tree depth 10).  “Potential
error” is the absolute difference between the value evaluated in the live
pipeline immediately before dumping and a fresh value after loader model
reconstruction.

| Dump basename | Size (bytes) | Channels | Cadences | Active | Initial potential | Potential error |
|---|---:|---:|---:|---:|---:|---:|
| `HAT-P-12_NIRISS_SOSS_order1_R20_low_resolution_inputs.pkl` | 97,506 | 24 | 226 | 95 | -24,110.6477906024 | 0 |
| `HAT-P-12_NIRISS_SOSS_order1_Rreference_high_resolution_inputs.pkl` | 451,433 | 118 | 225 | 94 | -118,851.417207802 | 0 |
| `GJ-3470_NIRSPEC_G395H_nrs1_R20_low_resolution_inputs.pkl` | 146,487 | 5 | 2,058 | 751 | -59,806.9759128088 | 0 |
| `GJ-3470_NIRSPEC_G395H_nrs1_R300_high_resolution_inputs.pkl` | 1,861,901 | 74 | 2,058 | 751 | -684,288.984168440 | 0 |
| `HAT-P-65_NIRSPEC_PRISM_nrs1_R10_low_resolution_inputs.pkl` | 14,100,052 | 21 | 40,780 | 16,457 | -3,034,602.63165955 | 0 |
| `HAT-P-65_NIRSPEC_PRISM_nrs1_R50_high_resolution_inputs.pkl` | 69,502,191 | 106 | 40,738 | 16,440 | -13,155,981.9219261 | 0 |

The paired low/high white-light handoff fingerprints agree exactly within
each mode:

| Mode | Handoff SHA-256 prefix |
|---|---|
| SOSS | `4f233e6bf521` |
| G395H | `9328f20c1261` |
| PRISM | `a3358b9e4b0c` |

The original output directories were inspected first, but their cached
science-artifact fingerprints were not current for the present pipeline
revision, so distinct `_ACCEL` output directories were used. The geometries
therefore come from short CPU white-light fits, not the production 1000/1000
fits. SOSS used 50/50. G395H used 50/50 with dump-only white-light tree depth
2. A PRISM 200/200 attempt completed sampling but exposed the diagnostic bug
below; the corrected capture used 20/20 and tree depth 2. These short
geometries are suitable for engineering replay but are not literature-fit
references.

### High-resolution handoff qualification

Running full low-resolution production NUTS on CPU was especially prohibited
for PRISM. To reach the real high-resolution call without doing that, I used
`FIT_JWST_DUMP_BRIDGE_LOWRES=1`, which changes only dump-enabled low-resolution
MCMC to 2 warmup + 2 samples and tree depth 2. The dumped high-resolution
calls themselves still record their normal 1000/1000/tree-depth-10 settings,
full model, data, and init state. The bridge chains diverged and must not be
used as scientific posterior products. In PRISM, low-resolution residual
clipping removed another 42 cadences before the high-resolution call
(40,780 to 40,738). Consequently, the high-resolution files are exact replays
of the calls that were captured, but their low-resolution-derived
initialization/mask is not guaranteed to equal a handoff from a converged
1000/1000 low-resolution fit. This is the main open fidelity risk.

## Offline sampler verification

I ran joint NUTS on high-resolution channels `0:4`, exactly 20 warmup + 20
retained draws and one four-channel chunk, on the CPU login node. Both runs
completed, returned samples shaped `[20, 4, ...]`, emitted deterministic
sites, and wrote all requested outputs beneath
`/scratch/midway3/tfairnington/accel_stage_inputs/verification/`.

| Mode | Sampler wall (s) | Recorded JAX compile (s) | Whole process (s) | Divergences | `num_steps` median / max | Minimum reported bulk ESS |
|---|---:|---:|---:|---:|---:|---:|
| SOSS high | 180.500 | 23.291 | 206.98 | 0 | 1,023 / 1,023 | 3.144 |
| G395H high | 375.994 | 24.089 | 434.93 | 0 | 511 / 1,023 | 2.167 |

There is no steady-only row for either measurement because each verification
had one chunk and that chunk compiled. Timing JSON explicitly reports `null`
for the steady median instead of pretending compile-inclusive wall is steady
wall. The chains are far too short for convergence or fidelity conclusions;
the step saturation is expected evidence of that. R-hat is recorded as
`null` with an explanation because these backends generated one chain and
ArviZ cannot honestly estimate R-hat from one chain.

I also ran the tiny fixture through `independent_nuts` (2/2, 3 lanes: 10.318 s
sampler wall, 10.099 s recorded compile) and `independent_hmc` (2/2, 3 lanes:
3.555 s sampler wall, 3.375 s recorded compile). Both completed and wrote the
same output families. The CLI self-comparison test evaluated six per-channel
site rows and returned PASS with zero median/percentile shifts and sigma ratio
1.0. This verifies comparison plumbing only, not scientific fidelity.

## Commands to reproduce

All CPU runs used the specified 16-core cap and float64 pipeline environment.
The low-resolution capture processes were interrupted immediately after the
dump message; the bridge processes exited cleanly on their first
high-resolution dump. High dump files were then copied from the temporary
bridge directories into the inventory root.

### Generate SOSS dumps

```bash
mkdir -p /scratch/midway3/tfairnington/accel_stage_inputs /tmp/jwst_soss_bridge_dumps

env FIT_JWST_DUMP_SAMPLER_INPUTS=/scratch/midway3/tfairnington/accel_stage_inputs \
  JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  taskset -c 0-15 \
  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python fit_jwst.py \
  -c configs_accel/HAT-P-12_soss_order1_accel_dump.yaml

env FIT_JWST_DUMP_SAMPLER_INPUTS=/tmp/jwst_soss_bridge_dumps \
  FIT_JWST_DUMP_SAMPLER_INPUTS_EXIT=1 \
  FIT_JWST_DUMP_BRIDGE_LOWRES=1 \
  JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  taskset -c 0-15 /usr/bin/time -p \
  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python fit_jwst.py \
  -c configs_accel/HAT-P-12_soss_order1_accel_dump.yaml

cp /tmp/jwst_soss_bridge_dumps/HAT-P-12_NIRISS_SOSS_order1_Rreference_high_resolution_inputs.pkl \
  /scratch/midway3/tfairnington/accel_stage_inputs/
```

### Generate G395H dumps

```bash
mkdir -p /scratch/midway3/tfairnington/accel_stage_inputs /tmp/jwst_g395h_bridge_dumps

env FIT_JWST_DUMP_SAMPLER_INPUTS=/scratch/midway3/tfairnington/accel_stage_inputs \
  FIT_JWST_DUMP_WHITELIGHT_MAX_TREE_DEPTH=2 \
  JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  taskset -c 0-15 \
  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python fit_jwst.py \
  -c configs_accel/GJ3470_nrs1_g395h_accel_dump.yaml

env FIT_JWST_DUMP_SAMPLER_INPUTS=/tmp/jwst_g395h_bridge_dumps \
  FIT_JWST_DUMP_SAMPLER_INPUTS_EXIT=1 \
  FIT_JWST_DUMP_BRIDGE_LOWRES=1 \
  FIT_JWST_DUMP_WHITELIGHT_MAX_TREE_DEPTH=2 \
  JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  taskset -c 0-15 /usr/bin/time -p \
  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python fit_jwst.py \
  -c configs_accel/GJ3470_nrs1_g395h_accel_dump.yaml

cp /tmp/jwst_g395h_bridge_dumps/GJ-3470_NIRSPEC_G395H_nrs1_R300_high_resolution_inputs.pkl \
  /scratch/midway3/tfairnington/accel_stage_inputs/
```

### Generate PRISM dumps

```bash
mkdir -p /scratch/midway3/tfairnington/accel_stage_inputs /tmp/jwst_prism_bridge_dumps

env FIT_JWST_DUMP_SAMPLER_INPUTS=/scratch/midway3/tfairnington/accel_stage_inputs \
  FIT_JWST_DUMP_WHITELIGHT_MAX_TREE_DEPTH=2 \
  JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  taskset -c 0-15 \
  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python fit_jwst.py \
  -c configs_accel/HAT-P-65_nrs1_prism_jaxoplanet_accel_dump.yaml

env FIT_JWST_DUMP_SAMPLER_INPUTS=/tmp/jwst_prism_bridge_dumps \
  FIT_JWST_DUMP_SAMPLER_INPUTS_EXIT=1 \
  FIT_JWST_DUMP_BRIDGE_LOWRES=1 \
  FIT_JWST_DUMP_WHITELIGHT_MAX_TREE_DEPTH=2 \
  JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  taskset -c 0-15 /usr/bin/time -p \
  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python fit_jwst.py \
  -c configs_accel/HAT-P-65_nrs1_prism_jaxoplanet_accel_dump.yaml

cp /tmp/jwst_prism_bridge_dumps/HAT-P-65_NIRSPEC_PRISM_nrs1_R50_high_resolution_inputs.pkl \
  /scratch/midway3/tfairnington/accel_stage_inputs/
```

The final PRISM bridge completed in 207.60 s process wall. It verified the
CPU backend, reused the corrected white-light handoff, ran the deliberately
tiny low-resolution bridge, dumped high-resolution inputs, and exited before
high-resolution sampling.

### Load and validate the dumps

```bash
env JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  taskset -c 0-15 \
  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -c \
  "import glob; from tools.spectro_stage_inputs import load_stage_inputs; \
[load_stage_inputs(p) for p in sorted(glob.glob('/scratch/midway3/tfairnington/accel_stage_inputs/*_inputs.pkl'))]"
```

### Required real joint-NUTS smoke runs

```bash
mkdir -p /scratch/midway3/tfairnington/accel_stage_inputs/verification

env JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  taskset -c 0-15 /usr/bin/time -p \
  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  tools/run_sampler_on_stage_inputs.py \
  /scratch/midway3/tfairnington/accel_stage_inputs/HAT-P-12_NIRISS_SOSS_order1_Rreference_high_resolution_inputs.pkl \
  --backend joint_nuts --start 0 --end 4 --warmup 20 --samples 20 \
  --chunk-size 4 --platform cpu \
  --output-prefix /scratch/midway3/tfairnington/accel_stage_inputs/verification/soss_high_joint_20x20_ch0_4

env JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  taskset -c 0-15 /usr/bin/time -p \
  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  tools/run_sampler_on_stage_inputs.py \
  /scratch/midway3/tfairnington/accel_stage_inputs/GJ-3470_NIRSPEC_G395H_nrs1_R300_high_resolution_inputs.pkl \
  --backend joint_nuts --start 0 --end 4 --warmup 20 --samples 20 \
  --chunk-size 4 --platform cpu \
  --output-prefix /scratch/midway3/tfairnington/accel_stage_inputs/verification/g395h_high_joint_20x20_ch0_4
```

To run a future compatible backend, place its implementation at, for
example, `models/laplace_is.py` exposing `get_samples_laplace_is`,
`get_samples_independent`, or `get_samples`, then pass
`--backend laplace_is`. The driver dynamically imports the module and uses
the independent-backend call contract.

### Tests

```bash
env JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  taskset -c 0-15 \
  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m py_compile \
  fit_jwst.py tools/spectro_stage_inputs.py \
  tools/run_sampler_on_stage_inputs.py tests/test_stage_inputs_dump.py

env JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 \
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  taskset -c 0-15 \
  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest \
  tests/test_stage_inputs_dump.py \
  tests/test_fit_independent_routing.py \
  tests/test_mcmc_runner_reuse.py \
  tests/test_spectro_safety_guards.py -x -q
```

Result: **19 passed, 2 warnings in 54.03 s**. The requested pre-existing
files account for 17 passing tests; the new file accounts for two. Python
compilation also passed.

## Failure found and fixed

The first PRISM run used 200/200 white-light transitions and completed that
sampling, then failed during post-fit evaluation with incompatible time axes
`(40780,)` and `(40785,)`. The precomputed Jaxoplanet phase offsets were still
bound to the pre-mask time vector after five white-light outliers were
removed. I made a targeted fix in `fit_jwst.py` to rebuild the transit
evaluation parameters for both `t_masked` and the white-light plotting time
vector after masking. The corrected 20/20 run passed that point and produced
both PRISM dumps.

The recurring CUDA plugin initialization traceback on the GPU-less login
node was non-fatal; every run subsequently printed/recorded the CPU backend.

## Open risks and limitations

- The three white-light geometries are deliberately short CPU fits. G395H
  and PRISM also used maximum tree depth 2; all retained draws saturated that
  depth. They are engineering inputs, not trustworthy geometry posteriors.
- The high-resolution handoffs used divergent 2/2 low-resolution bridge
  chains. A GPU rerun with converged low-resolution sampling should replace
  them before literature-fidelity comparisons, especially PRISM because its
  bridge changed the cadence mask by 42 rows.
- Single-chain output makes ArviZ R-hat undefined. The JSON states this and
  leaves values null rather than reporting a misleading statistic.
- Compile timing uses JAX monitoring events. It separates recorded compiler
  work and labels compile-inclusive versus no-compile chunks, but compiler
  event duration is not identical to the entire first-call overhead.
- The 20/20 real runs only prove that loading, slicing, joint sampling,
  diagnostics, and serialization work. Their very low ESS and tree-depth
  saturation do not pass any scientific fidelity gate.
- The comparison path was validated with a self-reference fixture. No
  candidate accelerated backend and no independent production reference
  were supplied, so no scientific PASS/FAIL or speedup can honestly be
  reported yet.

