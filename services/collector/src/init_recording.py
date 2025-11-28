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
SRC_DIR = os.path.join(COLLECTOR_DIR, 'src')
MONITOR_SCRIPT = os.path.join(SRC_DIR, 'monitor.py')


timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

print(f'{timestamp}: Initiating MONITOR mode')
    
# Change to src directory for execution
os.chdir(SRC_DIR)

# Run monitor.py
# Note: In production, this redirects output to a log file
os.system(f'python3 {MONITOR_SCRIPT}')



