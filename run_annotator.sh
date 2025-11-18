#!/bin/bash
# Change to repository root (where this script is located)
cd "$(dirname "$0")"

# Activate annotator service venv
source services/annotator/venv/bin/activate

# Run annotator app
streamlit run services/annotator/app.py

