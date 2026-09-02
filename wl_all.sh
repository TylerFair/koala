#!/bin/bash

# Submit each config as a separate job
for config in configs/*.yaml; do
    config_name=$(basename "$config" .yaml)
    
    # Extract path and output_dir from the YAML config file
    base_path=$(awk '/^[[:space:]]*path:/ {gsub(/["\047\/]/, "", $2); print $2}' "$config")
    output_dir=$(awk '/^[[:space:]]*output_dir:/ {gsub(/["\047]/, "", $2); print $2}' "$config")
    
    # Construct full path
    full_output_path="/scratch/midway3/tfairnington/${output_dir}"
    
    # Check if output directory exists
    if [ -d "$full_output_path" ]; then
        echo "Skipping $config_name - output directory already exists: $full_output_path"
    fi
    
    echo "Submitting job for $config_name (output: $full_output_path)"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${config_name}
#SBATCH --account=pi-ekempton
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/${config_name}_%j.out
#SBATCH --error=logs/${config_name}_%j.err

# Load your environment
source /home/tfairnington/miniconda3/bin/activate
conda activate /home/tfairnington/miniconda3/envs/jaxoplanet
module load cuda/12.9
#export XLA_PYTHON_CLIENT_PREALLOCATE=false
#export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
#export TF_GPU_ALLOCATOR=cuda_malloc_async
# Run the analysis
python dev_fit_jwst.py -c $config
EOF

done
