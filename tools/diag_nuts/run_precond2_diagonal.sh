#!/usr/bin/env bash
set -euo pipefail

case_name=${1:?usage: run_precond2_diagonal.sh soss|g395h}
python_bin=/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python
driver=tools/run_sampler_on_stage_inputs.py
output_dir=/scratch/midway3/tfairnington/accel_gpu_results/precond2
mkdir -p "$output_dir"
if [[ "$case_name" == soss ]]; then
  dump=/scratch/midway3/tfairnington/accel_stage_inputs_stellar/HAT-P-12_NIRISS_SOSS_order1_Rreference_high_resolution_inputs.pkl
  reference=/scratch/midway3/tfairnington/accel_gpu_results/orch_stellar_soss/A_joint_loguniform.pkl
  stem=stellar_soss_high
elif [[ "$case_name" == g395h ]]; then
  dump=/scratch/midway3/tfairnington/accel_stage_inputs/GJ-3470_NIRSPEC_G395H_nrs1_R300_high_resolution_inputs.pkl
  reference=/scratch/midway3/tfairnington/accel_stage_inputs/references/GJ-3470_NIRSPEC_G395H_nrs1_R300_high_resolution_inputs_ch0_40_pooled_joint_nuts.pkl
  stem=g395h_high
else
  echo "unknown case: $case_name" >&2
  exit 2
fi

common=(
  "$dump" --start 0 --end 40 --warmup 1000 --samples 1000
  --chunk-size 40 --platform gpu --compare "$reference"
  --builder-override jitter_prior=lognormal
  --nuts-override mass_matrix=laplace
  --nuts-override laplace_hessian_method=finite_difference
  --nuts-override laplace_fd_relative_step=0.0002
  --nuts-override laplace_map_method=diagonal
  --nuts-override laplace_line_search_steps=4
  --nuts-override laplace_map_iterations=16
  --nuts-override laplace_warmup=150
  --nuts-override laplace_start_at_map=False
)

"$python_bin" "$driver" "${common[@]}" \
  --backend independent_nuts \
  --nuts-override laplace_target_accept=0.95 \
  --nuts-override laplace_max_tree_depth=5 \
  --output-prefix "$output_dir/${stem}_fd_diag_nuts_d5"

for jitter in 0.0 0.25; do
  suffix=${jitter/./p}
  "$python_bin" "$driver" "${common[@]}" \
    --backend independent_hmc \
    --nuts-override laplace_target_accept=0.85 \
    --nuts-override num_steps=16 \
    --nuts-override trajectory_jitter="$jitter" \
    --output-prefix "$output_dir/${stem}_fd_diag_hmc_n16_j${suffix}"
done
