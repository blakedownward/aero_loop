#!/usr/bin/env python3
"""
Initialization wrapper for recording service.

This script determines the recording mode and launches the appropriate service.
Supports both "monitor" and "env" modes.
"""

import os
import sys
import datetime

# Add parent directory to path for imports
COLLECTOR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(COLLECTOR_DIR, 'config')
sys.path.insert(0, CONFIG_DIR)

import session_constants as cons

MODE = cons.REC_MODE
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

# Get paths
SRC_DIR = os.path.join(COLLECTOR_DIR, 'src')
MONITOR_SCRIPT = os.path.join(SRC_DIR, 'monitor.py')

if MODE == 'env':
    print(f'{timestamp}: Initiating ENV mode')
    # Note: rec_log_environment.py would need to be implemented if env mode is needed
    # For now, we'll just print a message
    print("ENV mode not yet implemented. Use 'monitor' mode.")
    sys.exit(1)
    
elif MODE == 'monitor':
    print(f'{timestamp}: Initiating MONITOR mode')
    
    # Change to src directory for execution
    os.chdir(SRC_DIR)
    
    # Run monitor.py
    # Note: In production, this would redirect output to a log file
    # For now, we'll run it directly
    os.system(f'python3 {MONITOR_SCRIPT}')
    
else:
    print(f'Unknown recording mode: {MODE}')
    print("Valid modes: 'monitor' or 'env'")
    sys.exit(1)

