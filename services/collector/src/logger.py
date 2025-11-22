"""
Logging utilities for session tracking and inference logging.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add config directory to path for imports
COLLECTOR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(COLLECTOR_DIR, 'config')
sys.path.insert(0, CONFIG_DIR)

import session_constants as c


def init_session_log():
    """Initialize session log file."""
    log_file = os.path.join(c.SESSION_PATH, 'session_log.jsonl')
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write('')  # Create empty file


def update_craft_log(craft_list):
    """
    Log aircraft detected in this iteration.
    
    Args:
        craft_list: List of aircraft hex codes detected
    """
    log_file = os.path.join(c.SESSION_PATH, 'session_log.jsonl')
    timestamp = datetime.now().isoformat()
    
    # Get unique aircraft
    unique_craft = list(set(craft_list)) if craft_list else []
    
    log_entry = {
        'timestamp': timestamp,
        'aircraft_count': len(unique_craft),
        'aircraft': unique_craft
    }
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


def update_inference_log(filename, file_status, predictions, model_version, session_path):
    """
    Log inference results for a recorded file.
    
    Args:
        filename: Name of the audio file
        file_status: "saved" or "deleted"
        predictions: Array of prediction values
        model_version: Version of the model used
        session_path: Path to the session directory
    """
    log_file = os.path.join(session_path, 'inference_log.jsonl')
    timestamp = datetime.now().isoformat()
    
    import numpy as np
    log_entry = {
        'timestamp': timestamp,
        'filename': filename,
        'status': file_status,
        'model_version': model_version,
        'predictions': {
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'count': len(predictions)
        }
    }
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

