#!/usr/bin/env bash
set -euo pipefail

backend=${1:?usage: run_prism_high.sh hmc|nuts}
python_bin=/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python
driver=tools/run_sampler_on_stage_inputs.py
dump=/scratch/midway3/tfairnington/accel_stage_inputs/HAT-P-65_NIRSPEC_PRISM_nrs1_R50_high_resolution_inputs.pkl
output_dir=/scratch/midway3/tfairnington/accel_gpu_results/prism
common=(
  "$dump" --start 0 --end 106 --warmup 1000 --samples 1000
  --chunk-size 40 --platform gpu
  --builder-override jitter_prior=lognormal
  --nuts-override mass_matrix=laplace
  --nuts-override laplace_hessian_method=finite_difference
  --nuts-override laplace_fd_relative_step=0.0002
  --nuts-override laplace_fuse_program=True
  --nuts-override laplace_map_iterations=200
  --nuts-override laplace_map_decrement_tolerance=0.0001
  --nuts-override laplace_trust_radius=5.0
  --nuts-override laplace_warmup=200
  --nuts-override laplace_target_accept=0.99
  --nuts-override laplace_start_at_map=True
)

if [[ "$backend" == hmc ]]; then
  "$python_bin" "$driver" "${common[@]}" --backend independent_hmc \
    --nuts-override num_steps=8 --nuts-override trajectory_jitter=0.25 \
    --output-prefix "$output_dir/prism_high_map200_r5_hmc8_j25_ta99"
elif [[ "$backend" == nuts ]]; then
  "$python_bin" "$driver" "${common[@]}" --backend independent_nuts \
    --nuts-override laplace_max_tree_depth=5 \
    --output-prefix "$output_dir/prism_high_map200_r5_nuts5_ta99"
else
  echo "unknown backend: $backend" >&2
  exit 2
fi
