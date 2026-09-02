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

For a CPU smoke test, set `JAX_PLATFORMS=cpu` before Python starts. Do not import JAX and then change the platform.

Enable the fitter's persistent cache with `flags.compile_box: true`; set `flags.jax_compilation_cache_dir` to node-visible storage. Cache reuse requires compatible JAX/XLA versions and static model shapes.

GPU memory grows with cadence count, model complexity, and `vmap_chunk`. Start with 40 independent channels on V100/A100 and reduce the width after an out-of-memory error. `chunk_mode: parallel` distributes channel ranges across array tasks; run `combine` only after all checkpoint files exist.

