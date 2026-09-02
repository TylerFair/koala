#!/usr/bin/env bash
set -euo pipefail

python_bin=/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python
dump=/scratch/midway3/tfairnington/accel_stage_inputs/HAT-P-65_NIRSPEC_PRISM_nrs1_R10_low_resolution_inputs.pkl
output=/scratch/midway3/tfairnington/accel_gpu_results/prism/map_r5_i200_fd2e5

"$python_bin" tools/run_sampler_on_stage_inputs.py "$dump" \
  --backend independent_hmc --start 0 --end 21 --warmup 5 --samples 5 \
  --chunk-size 21 --platform gpu \
  --builder-override jitter_prior=lognormal \
  --nuts-override mass_matrix=laplace \
  --nuts-override laplace_hessian_method=finite_difference \
  --nuts-override laplace_fd_relative_step=0.00002 \
  --nuts-override laplace_fuse_program=True \
  --nuts-override laplace_map_iterations=200 \
  --nuts-override laplace_map_decrement_tolerance=0.000001 \
  --nuts-override laplace_trust_radius=5.0 \
  --nuts-override laplace_warmup=5 \
  --nuts-override laplace_target_accept=0.99 \
  --nuts-override laplace_start_at_map=True \
  --nuts-override num_steps=8 \
  --nuts-override trajectory_jitter=0.25 \
  --output-prefix "$output"
