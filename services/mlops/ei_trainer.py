"""Trigger training runs on Edge Impulse using REST API."""

import os
import sys
import json
import requests
from typing import Optional, Dict
from datetime import datetime, timezone

# Add repo root to path for imports when running as standalone
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from services.mlops.env_loader import load_env
load_env()


# Get Edge Impulse API credentials
EI_API_KEY = os.getenv("EI_API_KEY")
if not EI_API_KEY:
    raise ValueError("EI_API_KEY environment variable is required (set in .env file)")

EI_PROJECT_ID = os.getenv("EI_PROJECT_ID")
if not EI_PROJECT_ID:
    raise ValueError("EI_PROJECT_ID environment variable is required (set in .env file)")


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir))


TRAIN_LOG = os.path.join(_repo_root(), 'data', 'train_log.json')


def load_train_log() -> Dict:
    """Load the training log."""
    if os.path.isfile(TRAIN_LOG):
        try:
            with open(TRAIN_LOG, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_train_log(log: Dict):
    """Save the training log."""
    os.makedirs(os.path.dirname(TRAIN_LOG), exist_ok=True)
    with open(TRAIN_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2)


def start_training(progress_callback=None) -> Dict[str, any]:
    """Start a training job on Edge Impulse using REST API.
    
    Returns: Dictionary with training job status
    """
    try:
        if progress_callback:
            progress_callback("Starting training job on Edge Impulse...")
        
        # Use Edge Impulse REST API to start training
        # Endpoint: POST /v1/api/{projectId}/jobs/retrain
        url = f"https://studio.edgeimpulse.com/v1/api/{EI_PROJECT_ID}/jobs/retrain"
        headers = {
            'x-api-key': EI_API_KEY,
            'Content-Type': 'application/json',
        }
        
        if progress_callback:
            progress_callback(f"POST {url}")
        
        response = requests.post(url, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract job ID from response
        job_id = result.get('id') or result.get('jobId') or result.get('job_id')
        if not job_id:
            # Fallback to timestamp-based ID if API doesn't return one
            job_id = f"api_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        if progress_callback:
            progress_callback(f"Training job started (ID: {job_id})")
        
        # Log the training
        train_log = load_train_log()
        train_log[job_id] = {
            'started_at': datetime.now(timezone.utc).isoformat(),
            'job_id': job_id,
            'status': 'started',
            'api_response': result,
        }
        save_train_log(train_log)
        
        return {
            'success': True,
            'error': None,
            'job_id': job_id,
            'status': 'started',
        }
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                error_msg = error_detail.get('error', error_msg)
            except:
                error_msg = e.response.text or error_msg
        if progress_callback:
            progress_callback(f"Error starting training: {error_msg}")
        return {
            'success': False,
            'error': error_msg,
            'job_id': None,
            'status': 'failed',
        }
    except Exception as e:
        error_msg = str(e)
        if progress_callback:
            progress_callback(f"Error starting training: {error_msg}")
        return {
            'success': False,
            'error': error_msg,
            'job_id': None,
            'status': 'failed',
        }


def get_training_status(job_id: str = None) -> Dict[str, any]:
    """Get the status of a training job using REST API.
    
    If job_id is None, gets the most recent training job.
    """
    train_log = load_train_log()
    
    # If no job_id provided, get the most recent one
    if job_id is None:
        if not train_log:
            return {
                'success': False,
                'error': 'No training jobs found',
            }
        # Get most recent job
        job_id = max(train_log.keys(), key=lambda k: train_log[k].get('started_at', ''))
    
    try:
        # Use REST API to check training status
        # Endpoint: GET /v1/api/{projectId}/jobs/{jobId}/status
        headers = {
            'x-api-key': EI_API_KEY,
        }
        
        # Use the correct endpoint with /status suffix
        url = f"https://studio.edgeimpulse.com/v1/api/{EI_PROJECT_ID}/jobs/{job_id}/status"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract job info from response
        job = result.get('job', {}) if isinstance(result, dict) else result
        if not job and isinstance(result, dict):
            # Response might be the job directly
            job = result
        
        # Determine status from job data
        if job.get('finished'):
            if job.get('finishedSuccessful'):
                status = 'completed'
            else:
                status = 'failed'
        elif job.get('started'):
            status = 'running'
        else:
            status = 'queued'
        
        # Update log
        if job_id in train_log:
            train_log[job_id]['status'] = status
            train_log[job_id]['last_checked'] = datetime.now(timezone.utc).isoformat()
            train_log[job_id]['api_response'] = job
            if status == 'completed':
                train_log[job_id]['completed_at'] = datetime.now(timezone.utc).isoformat()
            save_train_log(train_log)
        
        return {
            'success': True,
            'error': None,
            'job_id': job_id,
            'status': status,
            'job': job,
        }
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                error_msg = error_detail.get('error', error_msg)
            except:
                error_msg = e.response.text or error_msg
        return {
            'success': False,
            'error': error_msg,
            'job_id': job_id,
            'status': 'unknown',
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'job_id': job_id,
            'status': 'unknown',
        }


def wait_for_training(job_id: str = None, check_interval: int = 30, 
                     progress_callback=None) -> Dict[str, any]:
    """Wait for a training job to complete.
    
    Args:
        job_id: Training job ID (if None, uses most recent)
        check_interval: Seconds between status checks
        progress_callback: Optional callback for progress updates
    
    Returns: Final training status
    """
    import time
    
    if job_id is None:
        train_log = load_train_log()
        if not train_log:
            return {
                'success': False,
                'error': 'No training jobs found',
            }
        job_id = max(train_log.keys(), key=lambda k: train_log[k].get('started_at', ''))
    
    if progress_callback:
        progress_callback(f"Waiting for training job {job_id} to complete...")
    
    while True:
        status_result = get_training_status(job_id)
        
        if not status_result['success']:
            return status_result
        
        status = status_result['status']
        
        if progress_callback:
            progress_callback(f"Training status: {status}")
        
        if status in ('completed', 'failed', 'cancelled'):
            return status_result
        
        # Wait before next check
        time.sleep(check_interval)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'status':
        # Check status
        result = get_training_status()
        if result['success']:
            print(f"Training status: {result['status']}")
        else:
            print(f"Error: {result['error']}")
    else:
        # Start training
        def print_progress(msg):
            print(msg)
        
        result = start_training(progress_callback=print_progress)
        
        if result['success']:
            print(f"\nTraining started: {result['job_id']}")
            print("Use 'python services/mlops/ei_trainer.py status' to check progress")
        else:
            print(f"\nFailed to start training: {result['error']}")

