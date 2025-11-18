#!/bin/bash
# Change to repository root (where this script is located)
cd "$(dirname "$0")"

# Activate mlops service venv
source services/mlops/venv/bin/activate

# Run orchestrator with train & deploy steps
python services/mlops/orchestrator.py --steps process_annotations upload_to_ei train_model evaluate_model build_and_download deploy_model --wait-training

