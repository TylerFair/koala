# GPUs and clusters

Use a Slurm script that activates the environment and requests one GPU:

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
cd /path/to/repository
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=gpu
export OMP_NUM_THREADS=8
export JAX_COMPILATION_CACHE_DIR=/scratch/$USER/jax_cache
python fit_jwst.py -c config.yaml
```

For a CPU smoke test, set `JAX_PLATFORMS=cpu` before Python starts. Do not import JAX and then change the platform. Compile-box padding and the persistent compilation cache are enabled by default. Set `flags.jax_compilation_cache_dir` to node-visible storage, or use `flags.compile_box: false` and `flags.jax_persistent_cache: false` as the respective opt-outs. Cache reuse requires compatible JAX/XLA versions and static model shapes.

GPU memory grows with cadence count, model complexity, and `vmap_chunk`. Start with 40 independent channels on V100/A100 and reduce the width after an out-of-memory error. `chunk_mode: parallel` distributes channel ranges across array tasks; run `combine` only after all checkpoint files exist.

PRISM multi-worker chunking remains opt-in. The default processes one planet on
one GPU; choose `chunk_mode: parallel` only when separate workers and a later
combine step are explicitly configured.

## Platform selection

JAX selects its platform when it first initializes. Set the platform in the job environment before Python starts.

```bash
export JAX_PLATFORMS=gpu
export JAX_ENABLE_X64=1
```

Use `cpu` for read-only import checks on a login node. Do not run the production spectroscopic sampler on a shared login node.

## Device check

```bash
python -c "import jax; print(jax.devices())"
```

Run this inside the Slurm allocation. It should report the allocated V100, A100, or other supported CUDA device.

## Memory controls

`vmap_chunk` controls resident independent chains. Lowering it reduces memory approximately with channel concurrency. It does not rebin wavelength data.

Native PRISM usually needs more care because it combines many channels with long time arrays. Harmonica can also use more memory than a symmetric transit model. Start at 40 only when that width is known to fit the device and workload.

Use 20, 10, or 4 after an allocation failure. The final partial-width chunk can compile a separate executable.

## Cache placement

```yaml
flags:
  compile_box: true
  jax_compilation_cache_dir: /scratch/account/user/jax_cache
```

Use persistent scratch visible to every node that may resume the run. Avoid a network home directory with a tight metadata quota. Cache reuse requires compatible code, JAX/XLA version, and static shapes.

Stellar power-2 prior grids use a separate fingerprinted cache at
`/scratch/midway3/tfairnington/ld_prior_cache`. Change
`stellar.ld_prior_cache_dir` when that location is not shared, or set
`stellar.ld_prior_cache: false` to disable it. An unavailable cache directory
does not stop a fit.

## Serial chunking

```yaml
flags:
  chunk_mode: serial
  vmap_chunk: 40
```

One process walks through all channel ranges. Every completed chunk is checkpointed before the next starts. This is the simplest mode for one long allocation.

## Array-parallel chunking

```yaml
flags:
  chunk_mode: parallel
  chunk_parallel_job_count: 8
  chunk_parallel_job_index: 0
```

Set a distinct zero-based index in each array task. All tasks must see the same output and checkpoint directory. They write disjoint assigned chunks.

After every task finishes, use:

```yaml
flags:
  chunk_mode: combine
```

Combine checks completeness and restores global wavelength order. It fails rather than silently omitting a missing range.

## Split stages

```yaml
flags:
  analysis_stage: prep
```

This completes data preparation and required low-resolution work. Use `analysis_stage: highres` in a later allocation for the final channels. Prepared products are fingerprinted.

## Wall-time planning

Budget for white light, low resolution, high resolution, and plotting. The first static channel width normally includes 1--1.5 minutes of compilation. Equal-width chunks then take seconds to a few minutes each on V100/A100.

Difficult channels can take longer or trigger an alternate sampler. Use the number of channels and chunk width to estimate the call count. Keep time for the final partial-width compilation and output writing.

## Cluster troubleshooting

**CUDA device unavailable:** check Slurm GPU allocation and the JAX CUDA wheel. **Out of memory:** reduce `vmap_chunk` and resume with the new checkpoint family. **Cache misses:** check JAX versions, node-visible path, and static widths.

**Array collision:** verify unique `chunk_parallel_job_index` values and shared configuration. **Combine reports missing files:** do not force it; find and rerun the missing array assignment. **Job reaches wall time:** resubmit unchanged so completed chunks load.

**Slow filesystem:** place cache and outputs on project or scratch storage suited to many checkpoint files.
