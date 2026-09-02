#!/usr/bin/env bash
set -euo pipefail

python_bin=/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python
dump=/scratch/midway3/tfairnington/accel_stage_inputs/HAT-P-65_NIRSPEC_PRISM_nrs1_R10_low_resolution_inputs.pkl
output=/scratch/midway3/tfairnington/accel_gpu_results/prism/prism_low_joint_1000_1000

"$python_bin" tools/run_sampler_on_stage_inputs.py "$dump" \
  --backend joint_nuts --start 0 --end 21 --warmup 1000 --samples 1000 \
  --chunk-size 21 --platform gpu --output-prefix "$output"
