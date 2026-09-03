#!/usr/bin/env bash
set -euo pipefail
start_channel=${1:?start}
end_channel=${2:?end}
output_prefix=${3:?output prefix}
python tools/run_sampler_on_stage_inputs.py /scratch/midway3/tfairnington/accel_stage_inputs/HAT-P-65_NIRSPEC_PRISM_nrs1_R50_high_resolution_inputs.pkl --backend independent_nuts --start "$start_channel" --end "$end_channel" --warmup 200 --samples 1000 --chunk-size 4 --resident-lane-width 4 --platform gpu --builder-override jitter_prior=lognormal --nuts-override mass_matrix=laplace --nuts-override laplace_hessian_method=finite_difference --nuts-override laplace_fd_relative_step=0.0002 --nuts-override laplace_map_iterations=200 --nuts-override laplace_map_decrement_tolerance=0.0001 --nuts-override laplace_trust_radius=5.0 --nuts-override laplace_warmup=150 --nuts-override laplace_target_accept=0.99 --nuts-override laplace_max_tree_depth=6 --nuts-override laplace_start_at_map=True --output-prefix "$output_prefix"
