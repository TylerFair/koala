#!/bin/bash
set -u
cd /project/ekempton/tfairnington/JWST
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python tools/campaign/run_parity_dataset.py \
  --config /project/ekempton/tfairnington/JWST/configs_fiducial_stellarinformed/HAT-P-12_soss_order2_config.yaml \
  --staged-reference /scratch/midway3/tfairnington/accel_parity_staged/HAT-P-12_soss_order2_config_reference.pkl \
  --run-tag 20260902a --candidates A,B,C
