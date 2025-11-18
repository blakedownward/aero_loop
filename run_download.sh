#!/bin/bash
# Change to repository root (where this script is located)
cd "$(dirname "$0")"

# Activate remote service venv
source services/remote/venv/bin/activate

# Run download script
python services/remote/download_sessions.py

