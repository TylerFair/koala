#!/bin/bash
set -u
cd /project/ekempton/tfairnington/JWST
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python tools/campaign/run_parity_dataset.py \
  --config configs_fiducial_stellarinformed/Kepler-12_nrs1_prism_v1_config.yaml \
  --staged-reference /scratch/midway3/tfairnington/accel_parity_staged/Kepler-12_prism_v1_reference.pkl \
  --run-tag 20260902a --candidates A,B,C
