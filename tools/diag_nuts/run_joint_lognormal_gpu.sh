#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  g395h_low)
    dump=/scratch/midway3/tfairnington/accel_stage_inputs/GJ-3470_NIRSPEC_G395H_nrs1_R20_low_resolution_inputs.pkl
    chunk_size=5
    ;;
  soss_high)
    dump=/scratch/midway3/tfairnington/accel_stage_inputs/HAT-P-12_NIRISS_SOSS_order1_Rreference_high_resolution_inputs.pkl
    chunk_size=40
    ;;
  g395h_high)
    dump=/scratch/midway3/tfairnington/accel_stage_inputs/GJ-3470_NIRSPEC_G395H_nrs1_R300_high_resolution_inputs.pkl
    chunk_size=40
    ;;
  *)
    echo "usage: $0 {g395h_low|soss_high|g395h_high}" >&2
    exit 2
    ;;
esac

export JAX_ENABLE_X64=1
result_root=/scratch/midway3/tfairnington/accel_gpu_results/precond
mkdir -p "$result_root"

python tools/run_sampler_on_stage_inputs.py "$dump" \
  --backend joint_nuts \
  --start 0 \
  --warmup 1000 \
  --samples 1000 \
  --chunk-size "$chunk_size" \
  --platform gpu \
  --seed 0 \
  --builder-override jitter_prior=lognormal \
  --output-prefix "$result_root/${1}_joint_lognormal_full"
