#!/bin/bash
set -euo pipefail
export JAX_ENABLE_X64=1
for mode in adaptive laplace; do
  out="wl_validation/soss_paired_${mode}_seed558"
  mkdir -p "/scratch/midway3/tfairnington/${out}"
  cp -r /scratch/midway3/tfairnington/wl_validation/smoke_soss_laplace/ld_prior_cache "/scratch/midway3/tfairnington/${out}/"
  python tools/diag_whitelight/run_real_whitelight.py \
    --config configs_accel/HAT-P-12_soss_order1_stellarinformed_accel_dump.yaml \
    --output-dir "$out" --seed 558 --mass-matrix "$mode" --samples 1000
done
