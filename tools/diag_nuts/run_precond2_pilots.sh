#!/usr/bin/env bash
set -euo pipefail

case_name=${1:?usage: run_precond2_pilots.sh soss|g395h}
python_bin=/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python
driver=tools/run_sampler_on_stage_inputs.py
output_dir=/scratch/midway3/tfairnington/accel_gpu_results/precond2
mkdir -p "$output_dir"

case "$case_name" in
  soss)
    dump=/scratch/midway3/tfairnington/accel_stage_inputs_stellar/HAT-P-12_NIRISS_SOSS_order1_Rreference_high_resolution_inputs.pkl
    reference=/scratch/midway3/tfairnington/accel_gpu_results/orch_stellar_soss/A_joint_loguniform.pkl
    stem=stellar_soss_high
    ;;
  g395h)
    dump=/scratch/midway3/tfairnington/accel_stage_inputs/GJ-3470_NIRSPEC_G395H_nrs1_R300_high_resolution_inputs.pkl
    reference=/scratch/midway3/tfairnington/accel_stage_inputs/references/GJ-3470_NIRSPEC_G395H_nrs1_R300_high_resolution_inputs_ch0_40_pooled_joint_nuts.pkl
    stem=g395h_high
    ;;
  *)
    echo "unknown case: $case_name" >&2
    exit 2
    ;;
esac

common=(
  "$dump" --start 0 --end 40 --warmup 1000 --samples 1000
  --chunk-size 40 --platform gpu --compare "$reference"
  --builder-override jitter_prior=lognormal
  --nuts-override mass_matrix=laplace
  --nuts-override laplace_hessian_method=finite_difference
  --nuts-override laplace_fd_relative_step=0.0002
  --nuts-override laplace_map_iterations=16
  --nuts-override laplace_warmup=150
  --nuts-override laplace_start_at_map=False
)

run_nuts() {
  depth=$1
  "$python_bin" "$driver" "${common[@]}" \
    --backend independent_nuts \
    --nuts-override laplace_target_accept=0.95 \
    --nuts-override laplace_max_tree_depth="$depth" \
    --output-prefix "$output_dir/${stem}_fd_nuts_d${depth}"
}

run_hmc() {
  steps=$1
  jitter=$2
  suffix=n${steps}_j${jitter/./p}
  "$python_bin" "$driver" "${common[@]}" \
    --backend independent_hmc \
    --nuts-override laplace_target_accept=0.85 \
    --nuts-override num_steps="$steps" \
    --nuts-override trajectory_jitter="$jitter" \
    --output-prefix "$output_dir/${stem}_fd_hmc_${suffix}"
}

run_nuts 10
run_nuts 5
for steps in 8 16 32; do
  run_hmc "$steps" 0.0
  run_hmc "$steps" 0.25
done

