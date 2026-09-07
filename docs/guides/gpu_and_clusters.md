# GPUs and clusters

A full spectrum is intended to run on one GPU. Ask the scheduler for a GPU,
set the JAX platform before Python starts, and run the same command used
locally:

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

cd /path/to/Koala
source /path/to/environment/bin/activate

export JAX_ENABLE_X64=1
export JAX_PLATFORMS=gpu
export OMP_NUM_THREADS=8
export PYTHONHASHSEED=0
export JAX_COMPILATION_CACHE_DIR=/path/to/persistent/jax_cache

python fit_jwst.py -c config.yaml
```

Site syntax varies, but the Python command and environment variables do not.
Confirm the allocation before a long run:

```bash
python -c "import jax; print(jax.config.x64_enabled, jax.devices())"
```

It should report 64-bit mode and the allocated GPU. A CPU platform is suitable
for import checks and small tests; avoid a production fit on a shared login
node.

For bit-for-bit reproduction, set the same `PYTHONHASHSEED` before every
Python process. The configured scientific random seed controls Koala's JAX
random stream, while Python's hash seed can also affect latent-parameter
ordering during white-light MAP optimization. Runs with different hash seeds
can therefore follow different, statistically equivalent NUTS trajectories.

## Memory and chunk size

GPU memory grows with cadence count, model complexity, and the number of
resident wavelength channels. The independent samplers default to 40 channels.
If the process runs out of memory, reduce only the resident width:

```yaml
flags:
  spectro_chunk_size: 20
```

Try 10 or 4 for a more expensive model. Long, ordinary power-2 PRISM transits
with the default cadence acceleration now fit 40 resident channels on a 16 GB
V100; `spectro_chunk_size: auto` selects at most 40 for that measured path and
retains the conservative four-lane cap when the path is not eligible. This
setting changes concurrency, not the wavelength grid or posterior. The legacy
key `vmap_chunk` has the same role.

For spectroscopic stages longer than 5,000 cadences, the default
`spectro_cadence_reduction: auto` compresses the out-of-transit likelihood into
exact sufficient statistics, and `spectro_transit_grid: auto` evaluates the
transit on a contact-aware 769-node grid. The latter still evaluates a distinct
model and likelihood residual at every observed cadence; it is not time
binning. Both switches remain on the original code path for shorter SOSS and
G395H series. Set either switch to `off` for a direct-kernel comparison.

For reference, the audited HAT-P-65 NRS1 run on a 16 GB V100 used 40-channel
chunks and completed white light, 42 R20 channels, and 369 native channels in
21:10 with 5.66 GiB peak resident memory. The same tree with both switches
off took 56:47 on the same V100, a 2.7x end-to-end speed-up, and the two
native spectra agree to 0.03 sigma in every channel. Compilation and white-light work are
included in both wall times, so individual spectroscopic potential evaluations
show a larger speed-up.

## Checkpoints and wall time

Every completed channel block is written under `output_dir/chunks/`. If a job
reaches its wall-time limit, resubmit the same config and output directory;
compatible blocks are loaded and the run continues. Do not edit the YAML
between submissions unless you intend to start a new checkpoint family.

JAX compilation makes the first block of a new shape slower. Put
`JAX_COMPILATION_CACHE_DIR` on persistent, node-visible scratch to reuse
compatible compilations across jobs. Cache entries depend on code, JAX/XLA
versions, and static shapes.

## Split a long run into stages

Most analyses should use the default `analysis_stage: all`. When scheduler
limits require separate jobs, first prepare the shared products:

```yaml
flags:
  analysis_stage: prep
```

Then submit the same analysis with `analysis_stage: highres` and the same
output directory. The available values are `all`, `whitelight`, `prep`, and
`highres`. Prepared artifacts are fingerprinted; do not combine products from
different configs.

## Common failures

**No GPU appears.** Check that the job requested a GPU and that the installed
JAX wheel matches the cluster CUDA environment.

**Out of memory.** Lower `spectro_chunk_size` and resume. Rebin the spectrum
only when a coarser wavelength grid is also the scientific goal.

**The restart recomputes everything.** Compare the YAML and software revision
with the original run. A changed fingerprint correctly creates new
checkpoints.

**The first block is slow.** This is normally compilation. Equal-width blocks
should reuse the runner; a smaller final block may need one additional compile.
