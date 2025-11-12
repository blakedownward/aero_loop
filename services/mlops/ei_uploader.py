"""Upload processed samples to Edge Impulse."""

import os
import sys

# Add repo root to path for imports when running as standalone
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from services.mlops.env_loader import load_env
load_env()

import edgeimpulse as ei
import csv
import json
import requests
from typing import List, Dict, Optional
from pathlib import Path


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir))


PROC_PATH = os.path.join(_repo_root(), 'data', 'processed')
AIR_PATH = os.path.join(PROC_PATH, 'aircraft')
NEG_PATH = os.path.join(PROC_PATH, 'negative')
LABELS_CSV = os.path.join(PROC_PATH, 'labels.csv')
UPLOAD_LOG = os.path.join(PROC_PATH, 'upload_log.json')

# Get and set Edge Impulse API credentials
EI_API_KEY = os.getenv("EI_API_KEY")
if EI_API_KEY:
    ei.API_KEY = EI_API_KEY
else:
    raise ValueError("EI_API_KEY environment variable is required (set in .env file)")

EI_PROJECT_ID = os.getenv("EI_PROJECT_ID")
if not EI_PROJECT_ID:
    raise ValueError("EI_PROJECT_ID environment variable is required (set in .env file)")


def load_upload_log() -> Dict[str, Dict]:
    """Load the upload log tracking which files have been uploaded."""
    if os.path.isfile(UPLOAD_LOG):
        try:
            with open(UPLOAD_LOG, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_upload_log(log: Dict[str, Dict]):
    """Save the upload log."""
    with open(UPLOAD_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2)


def load_processed_labels() -> List[Dict]:
    """Load all processed file records from labels.csv."""
    if not os.path.isfile(LABELS_CSV):
        return []
    
    rows = []
    try:
        with open(LABELS_CSV, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception:
        pass
    
    return rows


def get_ei_manifest() -> Dict[str, Dict]:
    """Fetch the current manifest of uploaded samples from Edge Impulse."""
    try:
        # Use Edge Impulse REST API to list samples
        # API requires category parameter - fetch both training and testing
        url = f"https://studio.edgeimpulse.com/v1/api/{EI_PROJECT_ID}/raw-data"
        headers = {
            'x-api-key': EI_API_KEY,
        }
        
        manifest = {}
        
        # Fetch samples from both categories
        for category in ['training', 'testing']:
            try:
                response = requests.get(url, headers=headers, params={'category': category, 'limit': 10000})
                response.raise_for_status()
                
                result = response.json()
                
                # Extract samples from response
                if isinstance(result, dict):
                    samples = result.get('samples', [])
                    if not samples:
                        samples = result.get('data', [])
                else:
                    samples = result if isinstance(result, list) else []
                
                # Add to manifest
                for sample in samples:
                    if not isinstance(sample, dict):
                        continue
                    # Store by filename or ID for deduplication
                    # Try multiple possible field names
                    filename = (sample.get('filename') or 
                               sample.get('name') or 
                               sample.get('fileName') or
                               sample.get('file_name') or
                               '')
                    
                    if filename:
                        manifest[filename] = {
                            'id': (sample.get('id') or 
                                  sample.get('sampleId') or 
                                  sample.get('sample_id')),
                            'label': sample.get('label', ''),
                            'category': sample.get('category', category),
                        }
            except Exception as e:
                # Continue with other category if one fails
                print(f"Warning: Could not fetch {category} samples: {e}")
                continue
        
        return manifest
    except Exception as e:
        import traceback
        print(f"Warning: Could not fetch EI manifest: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return {}


def upload_sample(filepath: str, label: str, category: str = 'training', 
                 progress_callback=None) -> Optional[Dict]:
    """Upload a single sample to Edge Impulse using REST API.
    
    Args:
        filepath: Path to audio file
        label: Label for the sample ('aircraft' or 'negative')
        category: 'training' or 'testing'
        progress_callback: Optional callback for progress
    
    Returns: Response dict with sample ID, or None on failure
    """
    if not os.path.isfile(filepath):
        if progress_callback:
            progress_callback(f"File not found: {filepath}")
        return None
    
    try:
        if progress_callback:
            progress_callback(f"Uploading {os.path.basename(filepath)}...")
        
        # Use Edge Impulse Ingestion API
        url = 'https://ingestion.edgeimpulse.com/api/training/files'
        headers = {
            'x-api-key': EI_API_KEY,
            'x-label': label,
            'x-category': category,
        }
        
        # Prepare multipart form data
        with open(filepath, 'rb') as f:
            files = {
                'data': (os.path.basename(filepath), f, 'audio/wav')
            }
            
            response = requests.post(url, headers=headers, files=files)
            response.raise_for_status()
            
            result = response.json()
            
            if progress_callback:
                sample_id = result.get('id') or result.get('sampleId') or result.get('sample_id', 'unknown')
                progress_callback(f"Uploaded {os.path.basename(filepath)} (ID: {sample_id})")
            
            return result
    except requests.exceptions.RequestException as e:
        if progress_callback:
            error_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg = error_detail.get('error', error_msg)
                except:
                    error_msg = e.response.text or error_msg
            progress_callback(f"Error uploading {os.path.basename(filepath)}: {error_msg}")
        return None
    except Exception as e:
        if progress_callback:
            progress_callback(f"Error uploading {os.path.basename(filepath)}: {e}")
        return None


def upload_processed_samples(progress_callback=None) -> Dict[str, any]:
    """Upload processed samples from data/processed to Edge Impulse.
    
    All samples are uploaded to the TRAINING set. The TEST set remains fixed.
    
    Args:
        progress_callback: Optional callback for progress updates
    
    Returns: Dictionary with upload statistics
    """
    upload_log = load_upload_log()
    ei_manifest = get_ei_manifest()
    
    # Load processed labels
    labels = load_processed_labels()
    
    if not labels:
        return {
            'success': False,
            'error': 'No processed samples found in labels.csv',
            'uploaded': 0,
            'skipped': 0,
            'failed': 0,
        }
    
    if progress_callback:
        progress_callback(f"Found {len(labels)} processed samples")
    
    # Determine which samples to upload
    uploaded = 0
    skipped = 0
    failed = 0
    
    for label_row in labels:
        dst_path = label_row.get('dst', '')
        if not dst_path:
            continue
        
        # Resolve full path
        filepath = os.path.join(_repo_root(), dst_path)
        if not os.path.isfile(filepath):
            failed += 1
            continue
        
        # Check if already uploaded
        filename = os.path.basename(filepath)
        # EI manifest may not include file extension, so check both with and without
        filename_no_ext = os.path.splitext(filename)[0]
        
        # Determine label early (needed for log updates)
        class_name = label_row.get('class', '').lower()
        
        # First check if file is already in Edge Impulse manifest
        # Check both with and without extension (EI may store without .wav)
        manifest_key = None
        if filename in ei_manifest:
            manifest_key = filename
        elif filename_no_ext in ei_manifest:
            manifest_key = filename_no_ext
        
        if manifest_key:
            manifest_entry = ei_manifest[manifest_key]
            # If it's in the manifest, skip it and update our log
            if filename in upload_log:
                log_entry = upload_log[filename]
            else:
                log_entry = {}
            
            # Update log with info from manifest
            if manifest_entry.get('id'):
                log_entry['sample_id'] = manifest_entry['id']
                log_entry['label'] = class_name
                log_entry['category'] = 'training'
                log_entry['filepath'] = dst_path
                upload_log[filename] = log_entry
                save_upload_log(upload_log)
            skipped += 1
            continue
        
        # If not in manifest, check upload log
        if filename in upload_log:
            log_entry = upload_log[filename]
            # If we have a valid sample_id, trust the log and skip
            if log_entry.get('sample_id'):
                skipped += 1
                continue
        if class_name not in ('aircraft', 'negative'):
            failed += 1
            continue
        
        # All samples uploaded to training set (test set remains fixed)
        category = 'training'
        
        # Upload
        response = upload_sample(filepath, class_name, category, progress_callback)
        
        if response:
            # Log the upload
            # Handle different response formats from REST API
            sample_id = response.get('id') or response.get('sampleId') or response.get('sample_id')
            uploaded_at = response.get('created') or response.get('createdAt') or response.get('created_at', '')
            
            upload_log[filename] = {
                'uploaded_at': uploaded_at,
                'sample_id': sample_id,
                'label': class_name,
                'category': category,
                'filepath': dst_path,
            }
            save_upload_log(upload_log)
            uploaded += 1
        else:
            failed += 1
    
    return {
        'success': True,
        'error': None,
        'uploaded': uploaded,
        'skipped': skipped,
        'failed': failed,
        'total': len(labels),
    }


if __name__ == '__main__':
    def print_progress(msg):
        print(msg)
    
    result = upload_processed_samples(progress_callback=print_progress)
    
    if result['success']:
        print(f"\nUpload complete:")
        print(f"  Uploaded: {result['uploaded']}")
        print(f"  Skipped: {result['skipped']}")
        print(f"  Failed: {result['failed']}")
        print(f"  Total: {result['total']}")
    else:
        print(f"\nUpload failed: {result['error']}")
