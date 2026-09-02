#!/bin/bash
set -euo pipefail
export JAX_ENABLE_X64=1
for mode in adaptive laplace; do
  python tools/diag_whitelight/run_real_whitelight.py \
    --config configs_accel/GJ3470_nrs1_g395h_accel_dump.yaml \
    --output-dir "wl_validation/gj3470_paired_${mode}_seed558" \
    --seed 558 --mass-matrix "$mode" --samples 1000
done
