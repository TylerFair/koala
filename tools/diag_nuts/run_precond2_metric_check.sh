#!/usr/bin/env bash
set -euo pipefail

python_bin=/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python
dump=/scratch/midway3/tfairnington/accel_stage_inputs_stellar/HAT-P-12_NIRISS_SOSS_order1_Rreference_high_resolution_inputs.pkl
output=/scratch/midway3/tfairnington/accel_gpu_results/precond2/stellar_soss_fd_metric_exact_check

"$python_bin" tools/run_sampler_on_stage_inputs.py "$dump" \
  --backend independent_nuts --start 0 --end 40 --warmup 0 --samples 1 \
  --chunk-size 40 --platform gpu \
  --builder-override jitter_prior=lognormal \
  --nuts-override mass_matrix=laplace \
  --nuts-override laplace_hessian_method=finite_difference \
  --nuts-override laplace_compare_exact_hessian=True \
  --nuts-override laplace_fd_relative_step=0.0002 \
  --nuts-override laplace_map_iterations=16 \
  --nuts-override laplace_warmup=0 \
  --nuts-override laplace_max_tree_depth=5 \
  --output-prefix "$output"
