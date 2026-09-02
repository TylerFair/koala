#!/bin/bash
set -u
cd /project/ekempton/tfairnington/JWST
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python tools/campaign/run_parity_dataset.py \
  --config configs_fiducial_stellarinformed/WASP-52_nrs1_g395h_config.yaml \
  --staged-reference /scratch/midway3/tfairnington/accel_parity_staged/WASP-52_g395h_nrs1_reference.pkl \
  --run-tag 20260902b --candidates A,B
