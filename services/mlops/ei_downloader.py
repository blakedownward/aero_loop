"""Download trained models from Edge Impulse."""

import os
import sys

# Add repo root to path for imports when running as standalone
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from services.mlops.env_loader import load_env
load_env()

import json
import time
import requests
import zipfile
import re
from typing import Optional, Dict, Tuple, List
from datetime import datetime, timezone


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


MODELS_PATH = os.path.join(_repo_root(), 'models')
DOWNLOAD_LOG = os.path.join(_repo_root(), 'data', 'download_log.json')


def ensure_models_dir():
    """Ensure models directory exists."""
    os.makedirs(MODELS_PATH, exist_ok=True)


def load_download_log() -> Dict[str, Dict]:
    """Load the download log tracking which models have been downloaded."""
    if os.path.isfile(DOWNLOAD_LOG):
        try:
            with open(DOWNLOAD_LOG, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_download_log(log: Dict[str, Dict]):
    """Save the download log."""
    os.makedirs(os.path.dirname(DOWNLOAD_LOG), exist_ok=True)
    with open(DOWNLOAD_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2)


def get_model_info() -> Optional[Dict]:
    """Get information about the current trained model using REST API.
    
    This may include training metrics like loss.
    """
    try:
        # Try the learn endpoint first
        url = f"https://studio.edgeimpulse.com/v1/api/{EI_PROJECT_ID}/learn"
        headers = {
            'x-api-key': EI_API_KEY,
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        
        # If that fails, try the impulse endpoint which may have validation metrics
        url = f"https://studio.edgeimpulse.com/v1/api/{EI_PROJECT_ID}/impulse"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting model info: {e}")
        return None


def get_deployment_targets() -> Optional[list]:
    """Get list of available deployment targets.
    
    Returns list of target dicts with 'format' field that can be used as 'type' parameter.
    """
    try:
        url = f"https://studio.edgeimpulse.com/v1/api/{EI_PROJECT_ID}/deployment/targets"
        headers = {
            'x-api-key': EI_API_KEY,
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        # The API returns a dict with 'success' and 'targets' keys
        if isinstance(result, dict):
            if result.get('success') and 'targets' in result:
                targets = result['targets']
                # Extract format values from target objects
                if isinstance(targets, list):
                    return [t.get('format', t) if isinstance(t, dict) else t for t in targets]
                return targets
            # If it's a dict but not the expected structure, return as-is
            return result
        # If it's already a list, return it
        return result if isinstance(result, list) else None
    except Exception as e:
        print(f"Error getting deployment targets: {e}")
        return None


def build_deployment(deployment_type: str = 'zip', model_type: str = 'float32', 
                    engine: str = 'tflite', progress_callback=None) -> Dict[str, any]:
    """Build/trigger a deployment build in Edge Impulse.
    
    Args:
        deployment_type: Deployment target type (e.g., 'zip', 'runner-linux-armv7')
        model_type: Model quantization type ('float32' or 'int8')
        engine: Model engine ('tflite', 'tflite-eon', etc.)
        progress_callback: Optional callback for progress updates
    
    Returns: Dictionary with build job status and job_id
    """
    try:
        if progress_callback:
            progress_callback(f"Starting deployment build ({deployment_type}, {model_type}, {engine})...")
        
        # Use Edge Impulse REST API to trigger build
        # Endpoint: POST /v1/api/{projectId}/jobs/build-ondevice-model
        # Reference: https://docs.edgeimpulse.com/tutorials/tools/apis/studio/deploy-model
        url = f"https://studio.edgeimpulse.com/v1/api/{EI_PROJECT_ID}/jobs/build-ondevice-model"
        headers = {
            'x-api-key': EI_API_KEY,
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }
        
        # type goes in query string, engine goes in JSON body
        querystring = {
            'type': deployment_type,
        }
        
        # Payload with engine (and optionally modelType)
        payload = {
            'engine': engine,
        }
        
        # Add modelType to payload if not default
        if model_type != 'float32':
            payload['modelType'] = model_type
        
        if progress_callback:
            progress_callback(f"POST {url}")
            progress_callback(f"Query params: {querystring}")
            progress_callback(f"Payload: {payload}")
        
        response = requests.post(url, json=payload, headers=headers, params=querystring)
        response.raise_for_status()
        
        result = response.json()
        
        if not result.get('success'):
            error_msg = result.get('error', 'Unknown error')
            raise Exception(error_msg)
        
        # Extract job ID from response
        job_id = result.get('id')
        if not job_id:
            return {
                'success': False,
                'error': 'No job ID returned from build request',
                'response': result,
            }
        
        if progress_callback:
            progress_callback(f"Build job started (ID: {job_id})")
        
        return {
            'success': True,
            'error': None,
            'job_id': job_id,
            'status': 'started',
            'response': result,
        }
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                error_msg = error_detail.get('error', error_msg)
            except:
                error_text = e.response.text[:500] if e.response.text else "No error details"
                error_msg = f"{error_msg}: {error_text}"
        
        if progress_callback:
            progress_callback(f"Build failed: {error_msg}")
        
        return {
            'success': False,
            'error': error_msg,
            'job_id': None,
            'status': 'failed',
        }
    except Exception as e:
        error_msg = str(e)
        if progress_callback:
            progress_callback(f"Build error: {error_msg}")
        return {
            'success': False,
            'error': error_msg,
            'job_id': None,
            'status': 'failed',
        }


def get_build_stdout(job_id: str, skip_line_no: int = 0) -> List[str]:
    """Get stdout output from a build job.
    
    Args:
        job_id: The build job ID
        skip_line_no: Number of lines to skip (for incremental fetching)
    
    Returns: List of stdout lines
    """
    try:
        url = f"https://studio.edgeimpulse.com/v1/api/{EI_PROJECT_ID}/jobs/{job_id}/stdout"
        headers = {
            'x-api-key': EI_API_KEY,
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        if not result.get('success'):
            error_msg = result.get('error', 'Unknown error')
            raise Exception(error_msg)
        
        stdout = result.get('stdout', [])
        # Reverse array so it's old -> new (as per docs)
        stdout = stdout[::-1]
        return [x.get('data', '') for x in stdout[skip_line_no:]]
    except Exception as e:
        return []


def wait_for_build_completion(job_id: str, progress_callback=None, 
                              poll_interval: int = 2) -> Dict[str, any]:
    """Wait for a build job to complete, polling status and showing stdout.
    
    Args:
        job_id: The build job ID
        progress_callback: Optional callback for progress updates
        poll_interval: Seconds between status checks
    
    Returns: Dictionary with final build status
    """
    skip_line_no = 0
    
    if progress_callback:
        progress_callback(f"Waiting for build job {job_id} to complete...")
    
    while True:
        # Get status
        status_result = get_build_status(job_id, progress_callback)
        if not status_result.get('success'):
            return status_result
        
        # Get and print new stdout lines
        stdout_lines = get_build_stdout(job_id, skip_line_no)
        for line in stdout_lines:
            if progress_callback:
                progress_callback(line.rstrip())
        skip_line_no += len(stdout_lines)
        
        status = status_result['status']
        job = status_result.get('job', {})
        
        if status == 'completed':
            if progress_callback:
                progress_callback(f"Build job {job_id} completed successfully!")
            return status_result
        elif status == 'failed':
            if progress_callback:
                progress_callback(f"Build job {job_id} failed!")
            return status_result
        elif status in ['running', 'queued']:
            # Continue polling
            time.sleep(poll_interval)
        else:
            # Unknown status, continue polling
            time.sleep(poll_interval)


def get_build_status(job_id: str, progress_callback=None) -> Dict[str, any]:
    """Get the status of a deployment build job.
    
    Args:
        job_id: The build job ID
        progress_callback: Optional callback for progress updates
    
    Returns: Dictionary with build status
    """
    try:
        # Use REST API to check build job status
        # Endpoint: GET /v1/api/{projectId}/jobs/{jobId}/status
        # Reference: https://docs.edgeimpulse.com/tutorials/tools/apis/studio/deploy-model
        url = f"https://studio.edgeimpulse.com/v1/api/{EI_PROJECT_ID}/jobs/{job_id}/status"
        headers = {
            'x-api-key': EI_API_KEY,
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        
        if not result.get('success'):
            error_msg = result.get('error', 'Unknown error')
            raise Exception(error_msg)
        
        # Extract job info from response
        job = result.get('job', {})
        if not job:
            job = result
        
        # Determine status
        if job.get('finished'):
            if job.get('finishedSuccessful'):
                status = 'completed'
            else:
                status = 'failed'
        elif job.get('started'):
            status = 'running'
        else:
            status = 'queued'
        
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


def get_model_metrics() -> Optional[Dict]:
    """Get model performance metrics from Edge Impulse using REST API.
    
    Returns the full evaluation result with test set metrics.
    """
    try:
        # Use REST API to get evaluation metrics
        url = f"https://studio.edgeimpulse.com/v1/api/{EI_PROJECT_ID}/deployment/evaluate"
        headers = {
            'x-api-key': EI_API_KEY,
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        # The API returns: {"success": true, "result": [...]}
        # Extract the actual metrics from the result
        if isinstance(result, dict) and result.get('success') and 'result' in result:
            results = result['result']
            if isinstance(results, list) and len(results) > 0:
                # Return the first (most recent) evaluation result
                return results[0]
            return results
        return result
    except Exception as e:
        print(f"Error getting model metrics: {e}")
        return None


def extract_version_from_zip_name(zip_filename: str) -> Optional[str]:
    """Extract version from zip filename.
    
    Examples:
    - "aeroloop-aircraft-detection-cpp-mcu-v6.zip" -> "v6"
    - "aeroloop-aircraft-detection-cpp-mcu-v6" -> "v6"
    """
    # Remove .zip extension if present
    name = zip_filename.replace('.zip', '')
    
    # Try to match version pattern (v followed by number)
    match = re.search(r'-v(\d+)$', name)
    if match:
        return f"v{match.group(1)}"
    
    # Try alternative pattern (version at end)
    match = re.search(r'v(\d+)$', name)
    if match:
        return f"v{match.group(1)}"
    
    return None


def extract_tflite_from_zip(zip_path: str, output_dir: str, progress_callback=None) -> Tuple[Optional[str], Optional[str]]:
    """Extract .tflite file from zip and return (tflite_path, version).
    
    Returns: (path_to_tflite_file, version_string) or (None, None) on error
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get version from zip filename
            zip_basename = os.path.basename(zip_path)
            version = extract_version_from_zip_name(zip_basename)
            
            # Find .tflite file in zip
            tflite_files = [f for f in zip_ref.namelist() if f.endswith('.tflite')]
            
            if not tflite_files:
                if progress_callback:
                    progress_callback(f"Warning: No .tflite file found in zip")
                return None, version
            
            # Extract the first .tflite file found
            tflite_in_zip = tflite_files[0]
            tflite_filename = os.path.basename(tflite_in_zip)
            
            # Extract to output directory
            zip_ref.extract(tflite_in_zip, output_dir)
            
            # Move from subdirectory if needed
            extracted_path = os.path.join(output_dir, tflite_in_zip)
            final_path = os.path.join(output_dir, tflite_filename)
            
            if extracted_path != final_path:
                if os.path.exists(final_path):
                    os.remove(final_path)
                os.rename(extracted_path, final_path)
            
            if progress_callback:
                progress_callback(f"Extracted {tflite_filename} from zip")
            
            return final_path, version
    
    except Exception as e:
        if progress_callback:
            progress_callback(f"Error extracting zip: {e}")
        return None, None


def download_model(deployment_type: str = 'zip', model_type: str = 'float32', 
                  engine: str = 'tflite', progress_callback=None) -> Dict[str, any]:
    """Download the trained model from Edge Impulse.
    
    Args:
        deployment_type: Deployment target type (e.g., 'zip', 'linux-armv7', etc.)
                        Use 'zip' for generic ZIP download
        model_type: Model quantization type ('float32' or 'int8')
        engine: Model engine ('tflite' or 'tflite-eon')
        progress_callback: Optional callback for progress updates
    
    Returns: Dictionary with download status and model info
    """
    ensure_models_dir()
    
    try:
        if progress_callback:
            progress_callback(f"Downloading {deployment_type} model ({model_type}, {engine}) from Edge Impulse...")
        
        # Get model download using REST API
        # The endpoint returns the ZIP file directly, not a URL
        url = f"https://studio.edgeimpulse.com/v1/api/{EI_PROJECT_ID}/deployment/download"
        headers = {
            'x-api-key': EI_API_KEY,
        }
        
        # Request parameters - start with just type, add optional params if needed
        params = {
            'type': deployment_type,  # Deployment target (required)
        }
        
        # Add optional parameters only if not using defaults
        # Some deployment types might not support all options
        if model_type != 'float32':
            params['modelType'] = model_type
        if engine != 'tflite':
            params['engine'] = engine
        
        response = requests.get(url, headers=headers, params=params, stream=True)
        
        # If we get an error, try to get more details
        if response.status_code != 200:
            try:
                error_detail = response.json()
                error_msg = error_detail.get('error', str(error_detail)) if isinstance(error_detail, dict) else str(error_detail)
                full_error = f"{response.status_code} {response.reason}: {error_msg}"
            except:
                error_text = response.text[:500] if response.text else "No error details"
                full_error = f"{response.status_code} {response.reason}: {error_text}"
            
            # Check if error says deployment doesn't exist - might need to build first
            if "No deployment exists" in full_error or "did you build" in full_error.lower():
                if progress_callback:
                    progress_callback("Deployment doesn't exist. You may need to build it first in Edge Impulse Studio.")
                    progress_callback("Alternatively, the download endpoint should auto-build, but there may be a delay.")
                    progress_callback("Try building the deployment in Edge Impulse Studio, then download again.")
            
            if progress_callback:
                progress_callback(f"API Error: {full_error}")
            raise requests.exceptions.HTTPError(full_error, response=response)
        
        # Check content type to ensure we got a file, not an error
        content_type = response.headers.get('Content-Type', '')
        if 'application/json' in content_type:
            # Got JSON instead of file - might be an error
            try:
                error_data = response.json()
                error_msg = f"Server returned JSON instead of file: {error_data}"
                raise ValueError(error_msg)
            except:
                pass
        
        response.raise_for_status()
        
        # The endpoint returns the ZIP file directly
        # Try to get filename from Content-Disposition header
        content_disposition = response.headers.get('Content-Disposition', '')
        if 'filename=' in content_disposition:
            downloaded_filename = content_disposition.split('filename=')[1].strip('"\'')
        else:
            # Default filename
            downloaded_filename = f"model_{deployment_type}_{model_type}_{engine}.zip"
        
        # The endpoint always returns a ZIP file
        # Extract version from original filename before downloading
        version = extract_version_from_zip_name(downloaded_filename)
        
        # Download as zip first (use original filename to preserve version info)
        zip_path = os.path.join(MODELS_PATH, downloaded_filename)
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total_size > 0:
                        progress = (downloaded / total_size) * 100
                        progress_callback(f"Downloaded {progress:.1f}%...")
        
        if progress_callback:
            progress_callback(f"Extracting model from zip...")
        
        # Extract .tflite from zip (version already extracted from filename)
        tflite_path, _ = extract_tflite_from_zip(zip_path, MODELS_PATH, progress_callback)
        
        if not tflite_path:
            return {
                'success': False,
                'error': 'Failed to extract .tflite file from zip',
                'model_path': None,
            }
        
        # Rename extracted file with timestamp
        tflite_basename = os.path.basename(tflite_path)
        timestamped_name = f"model_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.tflite"
        final_tflite_path = os.path.join(MODELS_PATH, timestamped_name)
        
        if tflite_path != final_tflite_path:
            if os.path.exists(final_tflite_path):
                os.remove(final_tflite_path)
            os.rename(tflite_path, final_tflite_path)
        
        model_path = final_tflite_path
        filename = timestamped_name
        
        # Clean up zip file
        try:
            os.remove(zip_path)
        except Exception:
            pass
        
        if progress_callback:
            progress_callback(f"Model saved to {model_path}")
        
        # Get model info and metrics
        model_info = get_model_info()
        model_metrics = get_model_metrics()
        
        # Log the download
        download_log = load_download_log()
        download_log[filename] = {
            'downloaded_at': datetime.now(timezone.utc).isoformat(),
            'deployment_type': deployment_type,
            'model_type': model_type,
            'engine': engine,
            'model_path': model_path,
            'model_version': version,
            'model_info': model_info,
            'model_metrics': model_metrics,
        }
        save_download_log(download_log)
        
        return {
            'success': True,
            'error': None,
            'model_path': model_path,
            'model_filename': filename,
            'model_version': version,
            'model_info': model_info,
            'model_metrics': model_metrics,
        }
    
    except Exception as e:
        error_msg = str(e)
        if progress_callback:
            progress_callback(f"Error downloading model: {error_msg}")
        return {
            'success': False,
            'error': error_msg,
            'model_path': None,
            'model_filename': None,
        }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download model from Edge Impulse or get evaluation metrics')
    parser.add_argument('--deployment-type', default='zip', 
                       help='Deployment target type (default: zip)')
    parser.add_argument('--model-type', default='float32', choices=['float32', 'int8'],
                       help='Model quantization type (default: float32)')
    parser.add_argument('--engine', default='tflite', choices=['tflite', 'tflite-eon'],
                       help='Model engine (default: tflite)')
    parser.add_argument('--list-targets', action='store_true',
                       help='List available deployment targets')
    parser.add_argument('--metrics-only', action='store_true',
                       help='Only fetch evaluation metrics, do not download model')
    parser.add_argument('--build', action='store_true',
                       help='Build deployment instead of downloading')
    parser.add_argument('--build-status', type=str, metavar='JOB_ID',
                       help='Check status of a build job')
    parser.add_argument('--wait-build', type=str, metavar='JOB_ID',
                       help='Wait for a build job to complete')
    parser.add_argument('--build-and-download', action='store_true',
                       help='Build deployment and download model (skips evaluation check)')
    parser.add_argument('--deploy-to-pi', action='store_true',
                       help='Also deploy to Raspberry Pi after download (requires --build-and-download)')
    
    args = parser.parse_args()
    
    def print_progress(msg):
        print(msg)
    
    if args.list_targets:
        print("Available deployment targets:")
        targets = get_deployment_targets()
        if targets:
            if isinstance(targets, list):
                for target in targets:
                    if isinstance(target, dict):
                        format_val = target.get('format', 'N/A')
                        name = target.get('name', 'N/A')
                        print(f"  - format: {format_val}, name: {name}")
                    else:
                        print(f"  - {target}")
            else:
                print(f"  {targets}")
        else:
            print("  Could not retrieve targets")
    elif args.wait_build:
        print(f"Waiting for build job {args.wait_build} to complete...")
        result = wait_for_build_completion(args.wait_build, print_progress)
        if result['success']:
            print(f"\nBuild Status: {result['status']}")
            if result['status'] == 'completed':
                print("Build completed successfully!")
            elif result['status'] == 'failed':
                print("Build failed!")
        else:
            print(f"Error: {result.get('error')}")
    elif args.build_status:
        print(f"Checking build status for job {args.build_status}...")
        status = get_build_status(args.build_status, print_progress)
        if status['success']:
            print(f"\nBuild Status: {status['status']}")
            if status.get('job'):
                job = status['job']
                print(f"Job ID: {job.get('id')}")
                if job.get('finished'):
                    print(f"Finished: {job.get('finished')}")
                    print(f"Success: {job.get('finishedSuccessful')}")
                elif job.get('started'):
                    print(f"Started: {job.get('started')}")
        else:
            print(f"Error: {status.get('error')}")
    elif args.build:
        result = build_deployment(
            deployment_type=args.deployment_type,
            model_type=args.model_type,
            engine=args.engine,
            progress_callback=print_progress
        )
        
        if result['success']:
            print(f"\nBuild started successfully!")
            print(f"Job ID: {result['job_id']}")
            print(f"Status: {result['status']}")
            print(f"\nMonitor build with: python services/mlops/ei_downloader.py --build-status {result['job_id']}")
        else:
            print(f"\nBuild failed: {result['error']}")
    elif args.build_and_download:
        print("Building deployment and downloading latest model...")
        print("(Skipping evaluation check - downloading latest trained model)")
        
        # Build deployment
        build_result = build_deployment(
            deployment_type=args.deployment_type,
            model_type=args.model_type,
            engine=args.engine,
            progress_callback=print_progress
        )
        
        if not build_result['success']:
            print(f"\nBuild failed: {build_result['error']}")
            sys.exit(1)
        
        job_id = build_result['job_id']
        print(f"\nBuild started. Job ID: {job_id}")
        print("Waiting for build to complete...")
        
        # Wait for build
        build_status = wait_for_build_completion(job_id, progress_callback=print_progress)
        
        if build_status['status'] != 'completed':
            print(f"\nBuild did not complete: {build_status['status']}")
            if build_status.get('error'):
                print(f"Error: {build_status['error']}")
            sys.exit(1)
        
        print("\nBuild completed successfully!")
        
        # Download model
        print("\nDownloading model...")
        download_result = download_model(
            deployment_type=args.deployment_type,
            model_type=args.model_type,
            engine=args.engine,
            progress_callback=print_progress
        )
        
        if download_result['success']:
            print(f"\n✓ Model downloaded successfully!")
            print(f"  Location: {download_result['model_filename']}")
            print(f"  Model path: {download_result.get('model_path', 'N/A')}")
            
            # Deploy to Pi if requested
            if args.deploy_to_pi:
                print("\n" + "="*60)
                print("Deploying to Raspberry Pi...")
                print("="*60)
                
                # Import deploy function
                try:
                    # Add repo root to path
                    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
                    if repo_root not in sys.path:
                        sys.path.insert(0, repo_root)
                    
                    from services.remote.deploy_model import deploy_latest_model
                    
                    deploy_result = deploy_latest_model(progress_callback=print_progress)
                    
                    if deploy_result['success']:
                        print(f"\n✓ Model deployed to Pi successfully!")
                        print(f"  Model version: {deploy_result.get('model_version', 'N/A')}")
                        print(f"  Pi will reboot shortly...")
                    else:
                        print(f"\n✗ Deployment to Pi failed: {deploy_result.get('error', 'Unknown error')}")
                        sys.exit(1)
                except ImportError as e:
                    print(f"\n✗ Could not import deploy module: {e}")
                    print("  Make sure services/remote/deploy_model.py exists")
                    sys.exit(1)
                except Exception as e:
                    print(f"\n✗ Deployment error: {e}")
                    sys.exit(1)
        else:
            print(f"\n✗ Download failed: {download_result['error']}")
            sys.exit(1)
    elif args.metrics_only:
        print("Fetching evaluation metrics from Edge Impulse...")
        metrics = get_model_metrics()
        model_info = get_model_info()
        
        if metrics:
            print("\n=== Model Evaluation Metrics ===")
            print(json.dumps(metrics, indent=2))
        else:
            print("Could not retrieve metrics")
        
        if model_info:
            print("\n=== Model Info ===")
            print(json.dumps(model_info, indent=2))
    else:
        result = download_model(
            deployment_type=args.deployment_type,
            model_type=args.model_type,
            engine=args.engine,
            progress_callback=print_progress
        )
        
        if result['success']:
            print(f"\nModel downloaded: {result['model_filename']}")
            if result.get('model_metrics'):
                print(f"\nModel metrics:")
                metrics = result['model_metrics']
                if isinstance(metrics, dict):
                    # Print test set metrics if available
                    if 'test' in metrics:
                        print(f"  Test set: {metrics['test']}")
                    if 'accuracy' in metrics:
                        print(f"  Accuracy: {metrics['accuracy']}")
                    if 'loss' in metrics:
                        print(f"  Loss: {metrics['loss']}")
                else:
                    print(f"  {metrics}")
        else:
            print(f"\nDownload failed: {result['error']}")

