#!/usr/bin/env bash
set -euo pipefail

case_name=${1:?usage: run_precond2_full_baseline.sh soss|g395h}
python_bin=/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python
output_dir=/scratch/midway3/tfairnington/accel_gpu_results/precond2
mkdir -p "$output_dir"
if [[ "$case_name" == soss ]]; then
  dump=/scratch/midway3/tfairnington/accel_stage_inputs_stellar/HAT-P-12_NIRISS_SOSS_order1_Rreference_high_resolution_inputs.pkl
  end=118
  output=$output_dir/stellar_soss_high_joint_production_full
elif [[ "$case_name" == g395h ]]; then
  dump=/scratch/midway3/tfairnington/accel_stage_inputs/GJ-3470_NIRSPEC_G395H_nrs1_R300_high_resolution_inputs.pkl
  end=74
  output=$output_dir/g395h_high_joint_production_full
else
  echo "unknown case: $case_name" >&2
  exit 2
fi

"$python_bin" tools/run_sampler_on_stage_inputs.py "$dump" \
  --backend joint_nuts --start 0 --end "$end" --warmup 1000 --samples 1000 \
  --chunk-size 40 --platform gpu --output-prefix "$output"
