#!/bin/bash

# Loop through all config files
for config in configs/*.yaml; do
    echo "=========================================="
    echo "Running: $config"
    echo "=========================================="
    
    python fit_jwst.py -c "$config"
    
    exit_code=$?
    echo "Finished $config with exit code: $exit_code"
    echo ""
done

echo "All configs processed!"
