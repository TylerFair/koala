#!/bin/bash
set -euo pipefail
export JAX_ENABLE_X64=1
for mode in adaptive laplace; do
  extra=()
  if [[ "$mode" == laplace ]]; then extra+=(--map-iterations 16); fi
  python tools/diag_whitelight/run_real_whitelight.py \
    --config configs_accel/HAT-P-65_nrs1_prism_jaxoplanet_accel_dump.yaml \
    --output-dir "wl_validation/prism_paired_${mode}_seed558" \
    --seed 558 --mass-matrix "$mode" --samples 1000 "${extra[@]}"
done
