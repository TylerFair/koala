#!/bin/bash

set -euo pipefail


has_final_outputs() {
    local output_path="$1"

    compgen -G "${output_path}/*_whitelight_bestfit_params.csv" > /dev/null || return 1
    compgen -G "${output_path}/34_*_summary.png" > /dev/null || return 1
    compgen -G "${output_path}/31_*_spectrum_00.png" > /dev/null || return 1
    compgen -G "${output_path}/36_*_noisebin.png" > /dev/null || return 1
}

# Submit each config as a separate job
for config in configs_fiducial_fixed/*.yaml; do
    config_name=$(basename "$config" .yaml)
    output_dir=$(awk '/^[[:space:]]*output_dir:/ {gsub(/["\047]/, "", $2); print $2}' "$config")

    if [ -z "$output_dir" ]; then
        echo "Skipping $config_name - could not parse output_dir from $config"
        continue
    fi

    full_output_path="/scratch/midway3/tfairnington/${output_dir}"
    done_file="${full_output_path}/.done"
    running_file="${full_output_path}/.running"
    failed_file="${full_output_path}/.failed"
    jobid_file="${full_output_path}/.slurm_job_id"
    
    if [ -f "$done_file" ]; then
        if has_final_outputs "$full_output_path"; then
        echo "Skipping $config_name - completion marker exists: $done_file"
        continue
        fi
        echo "Found stale completion marker for $config_name; final outputs are missing."
        rm -f "$done_file"
    fi

    if has_final_outputs "$full_output_path"; then
        echo "Skipping $config_name - final outputs already exist"
        touch "$done_file"
        continue
    fi

    if [ -f "$jobid_file" ]; then
        existing_job=$(tr -d '[:space:]' < "$jobid_file")
        if [ -n "$existing_job" ] && squeue -h -j "$existing_job" --format="%i" | grep -q .; then
            echo "Skipping $config_name - already queued/running as job $existing_job"
            continue
        fi
        echo "Found stale job marker for $config_name ($existing_job); resubmitting"
        rm -f "$jobid_file" "$running_file"
    fi

    existing_job=$(squeue -h --user="$USER" --format="%i|%j" | awk -F'|' -v name="$config_name" '$2 == name {print $1; exit}' || true)
    if [ -n "$existing_job" ]; then
        echo "Skipping $config_name - already queued/running as job $existing_job"
        printf '%s\n' "$existing_job" > "$jobid_file"
        continue
    fi
    
    mkdir -p "$full_output_path"
    rm -f "$running_file"

    echo "Submitting job for $config_name (output: $full_output_path)"

    if ! submit_output=$(
        sbatch \
            --job-name="$config_name" \
            --output="logs/${config_name}_%j.out" \
            --error="logs/${config_name}_%j.err" \
            --export=ALL,CONFIG="$config",CONFIG_NAME="$config_name",FULL_OUTPUT_PATH="$full_output_path",DONE_FILE="$done_file",RUNNING_FILE="$running_file",FAILED_FILE="$failed_file",JOBID_FILE="$jobid_file" <<'EOF' 2>&1
#!/bin/bash
#SBATCH --account=pi-ekempton
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=24:00:00

set -euo pipefail

nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total --format=csv

source /home/tfairnington/miniconda3/bin/activate
conda activate /home/tfairnington/miniconda3/envs/jaxoplanet
module load cuda/12.9

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export TF_GPU_ALLOCATOR=cuda_malloc_async

mkdir -p "$FULL_OUTPUT_PATH"
rm -f "$FAILED_FILE" "$DONE_FILE"
touch "$RUNNING_FILE"

has_final_outputs() {
    local output_path="$1"

    compgen -G "${output_path}/*_whitelight_bestfit_params.csv" > /dev/null || return 1
    compgen -G "${output_path}/34_*_summary.png" > /dev/null || return 1
    compgen -G "${output_path}/31_*_spectrum_00.png" > /dev/null || return 1
    compgen -G "${output_path}/36_*_noisebin.png" > /dev/null || return 1
}

cleanup() {
    status=$?
    rm -f "$RUNNING_FILE"
    rm -f "$JOBID_FILE"
    if [ "$status" -eq 0 ] && has_final_outputs "$FULL_OUTPUT_PATH"; then
        touch "$DONE_FILE"
        rm -f "$FAILED_FILE"
    else
        if [ "$status" -eq 0 ]; then
            echo "Job exited cleanly, but final output verification failed for $FULL_OUTPUT_PATH" >&2
            status=1
        fi
        touch "$FAILED_FILE"
    fi
    exit "$status"
}
trap cleanup EXIT

python - <<'PY'
import jax
print(jax.devices())
print(jax.local_device_count())
PY

python fit_jwst.py -c "$CONFIG"
EOF
    ); then
        echo "Submission failed for $config_name"
        echo "$submit_output"
        if printf '%s\n' "$submit_output" | grep -q "QOSMaxSubmitJobPerUserLimit"; then
            echo "Reached Slurm submit limit; stopping further submissions."
            break
        fi
        continue
    fi

    job_id=$(printf '%s\n' "$submit_output" | awk '/Submitted batch job/ {print $4}' | tail -n 1)
    if [ -n "$job_id" ]; then
        printf '%s\n' "$job_id" > "$jobid_file"
    fi

    echo "$submit_output"
done

