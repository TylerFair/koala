#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  soss_high)
    dump=/scratch/midway3/tfairnington/accel_stage_inputs/HAT-P-12_NIRISS_SOSS_order1_Rreference_high_resolution_inputs.pkl
    chunk_size=40
    ;;
  soss_low)
    dump=/scratch/midway3/tfairnington/accel_stage_inputs/HAT-P-12_NIRISS_SOSS_order1_R20_low_resolution_inputs.pkl
    chunk_size=24
    ;;
  g395h_high)
    dump=/scratch/midway3/tfairnington/accel_stage_inputs/GJ-3470_NIRSPEC_G395H_nrs1_R300_high_resolution_inputs.pkl
    chunk_size=40
    ;;
  g395h_low)
    dump=/scratch/midway3/tfairnington/accel_stage_inputs/GJ-3470_NIRSPEC_G395H_nrs1_R20_low_resolution_inputs.pkl
    chunk_size=5
    ;;
  prism_low)
    dump=/scratch/midway3/tfairnington/accel_stage_inputs/HAT-P-65_NIRSPEC_PRISM_nrs1_R10_low_resolution_inputs.pkl
    chunk_size=21
    ;;
  *)
    echo "usage: $0 {soss_high|soss_low|g395h_high|g395h_low|prism_low}" >&2
    exit 2
    ;;
esac

export JAX_ENABLE_X64=1
result_root=/scratch/midway3/tfairnington/accel_gpu_results/precond
mkdir -p "$result_root"
laplace_warmup=${PRECOND_LAPLACE_WARMUP:-150}
laplace_target_accept=${PRECOND_TARGET_ACCEPT:-0.95}
laplace_max_tree_depth=${PRECOND_MAX_TREE_DEPTH:-10}
laplace_start_at_map=${PRECOND_START_AT_MAP:-False}
laplace_map_iterations=${PRECOND_MAP_ITERATIONS:-16}
output_tag=${PRECOND_OUTPUT_TAG:-laplace_w150_d10_ta095}
range_args=()
if [[ -n "${PRECOND_END:-}" ]]; then
  range_args=(--end "$PRECOND_END")
fi

python tools/run_sampler_on_stage_inputs.py "$dump" \
  --backend independent_nuts \
  --start 0 \
  "${range_args[@]}" \
  --warmup 1000 \
  --samples 1000 \
  --chunk-size "$chunk_size" \
  --platform gpu \
  --seed 0 \
  --builder-override jitter_prior=lognormal \
  --nuts-override mass_matrix=laplace \
  --nuts-override laplace_warmup="$laplace_warmup" \
  --nuts-override laplace_target_accept="$laplace_target_accept" \
  --nuts-override laplace_max_tree_depth="$laplace_max_tree_depth" \
  --nuts-override laplace_start_at_map="$laplace_start_at_map" \
  --nuts-override laplace_map_iterations="$laplace_map_iterations" \
  --output-prefix "$result_root/${1}_${output_tag}"
