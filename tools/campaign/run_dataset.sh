#!/usr/bin/env bash
set -euo pipefail

exec /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  tools/campaign/run_dataset.py "$@"
