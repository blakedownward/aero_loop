"""Deploy models to Raspberry Pi."""

import os
import sys
import json
import shutil
from datetime import datetime
from typing import Optional, Tuple, Dict, List
from pathlib import Path

# Handle both module and direct execution
if __name__ == '__main__':
    # Add parent directory to path for direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from services.remote.connection import PiConnection
    from services.remote.config import get_config
else:
    from .connection import PiConnection
    from .config import get_config


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir))


MODELS_PATH = os.path.join(_repo_root(), 'models')
DEPLOY_LOG = os.path.join(_repo_root(), 'data', 'deploy_log.json')


def load_deploy_log() -> Dict[str, Dict]:
    """Load the deployment log tracking which models have been deployed."""
    if os.path.isfile(DEPLOY_LOG):
        try:
            with open(DEPLOY_LOG, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_deploy_log(log: Dict[str, Dict]):
    """Save the deployment log."""
    os.makedirs(os.path.dirname(DEPLOY_LOG), exist_ok=True)
    with open(DEPLOY_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2)


def find_model_files(models_dir: str = None) -> List[str]:
    """Find model files in the models directory.
    
    Looks for common model file extensions: .tflite, .eim, .onnx, .pb
    """
    if models_dir is None:
        models_dir = MODELS_PATH
    
    if not os.path.isdir(models_dir):
        return []
    
    model_extensions = ['.tflite', '.eim', '.onnx', '.pb', '.h5']
    model_files = []
    
    for filename in os.listdir(models_dir):
        filepath = os.path.join(models_dir, filename)
        if os.path.isfile(filepath):
            ext = os.path.splitext(filename)[1].lower()
            if ext in model_extensions:
                model_files.append(filename)
    
    return sorted(model_files)


def get_latest_model(models_dir: str = None) -> Optional[str]:
    """Get the most recently modified model file."""
    if models_dir is None:
        models_dir = MODELS_PATH
    
    model_files = find_model_files(models_dir)
    if not model_files:
        return None
    
    # Sort by modification time, most recent first
    model_files_with_time = []
    for filename in model_files:
        filepath = os.path.join(models_dir, filename)
        mtime = os.path.getmtime(filepath)
        model_files_with_time.append((mtime, filename))
    
    model_files_with_time.sort(reverse=True)
    return model_files_with_time[0][1]


def deploy_model(connection: PiConnection, model_filename: str = None, 
                model_path: str = None, progress_callback=None) -> Tuple[bool, Optional[str]]:
    """Deploy a model file to the Raspberry Pi.
    
    Args:
        connection: PiConnection instance
        model_filename: Name of model file in models/ directory (if None, uses latest)
        model_path: Override path to model file
        progress_callback: Optional callback for progress updates
    
    Returns: (success, error_message)
    """
    config = get_config()
    
    # Determine local model file path
    if model_path:
        local_model_path = model_path
        model_filename = os.path.basename(model_path)
    elif model_filename:
        local_model_path = os.path.join(MODELS_PATH, model_filename)
    else:
        # Use latest model
        model_filename = get_latest_model()
        if not model_filename:
            return False, "No model files found in models/ directory"
        local_model_path = os.path.join(MODELS_PATH, model_filename)
    
    if not os.path.isfile(local_model_path):
        return False, f"Model file not found: {local_model_path}"
    
    # Determine remote paths
    # Application lives in /home/protopi/ten90audio/ten90audio/
    remote_model_dir = "/home/protopi/ten90audio/ten90audio"
    remote_model_path = f"{remote_model_dir}/{model_filename}"
    
    if progress_callback:
        progress_callback(f"Uploading {model_filename} to Pi...")
    
    # Upload the model file
    # Wrap progress callback to handle (transferred, total) format from upload_file
    def upload_progress(transferred, total):
        if progress_callback:
            if total > 0:
                percent = (transferred / total) * 100
                progress_callback(f"Uploading: {percent:.1f}% ({transferred}/{total} bytes)")
            else:
                progress_callback(f"Uploading: {transferred} bytes...")
    
    success, error = connection.upload_file(local_model_path, remote_model_path, upload_progress)
    if not success:
        return False, f"Upload failed: {error}"
    
    if progress_callback:
        progress_callback(f"Model uploaded successfully")
    
    # Verify deployment
    if not connection.file_exists(remote_model_path):
        return False, "Model file not found on Pi after upload"
    
    # Get file stats for verification
    remote_stat = connection.get_file_stat(remote_model_path)
    local_stat = os.stat(local_model_path)
    
    if remote_stat and remote_stat['size'] != local_stat.st_size:
        return False, f"File size mismatch: local={local_stat.st_size}, remote={remote_stat['size']}"
    
    # Get model version from download log if available
    import json
    download_log_path = os.path.join(_repo_root(), 'data', 'download_log.json')
    model_version = None
    if os.path.isfile(download_log_path):
        try:
            with open(download_log_path, 'r', encoding='utf-8') as f:
                download_log = json.load(f)
                if model_filename in download_log:
                    model_version = download_log[model_filename].get('model_version')
        except Exception:
            pass
    
    # Use model filename as version if version not available
    if not model_version:
        model_version = model_filename
    
    # Always update model_version.txt on Pi
    # Version file lives in /home/protopi/ten90audio/ten90audio/
    version_file_path = f"{remote_model_dir}/model_version.txt"
    version_content = f"{model_version}\n"
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
        tmp_file.write(version_content)
        tmp_version_path = tmp_file.name
    
    try:
        # Upload version file
        success_ver, error_ver = connection.upload_file(tmp_version_path, version_file_path)
        if not success_ver:
            if progress_callback:
                progress_callback(f"Warning: Could not upload version file: {error_ver}")
        else:
            if progress_callback:
                progress_callback(f"✓ Version file updated: {model_version}")
    finally:
        # Clean up temp file
        try:
            os.remove(tmp_version_path)
        except Exception:
            pass
    
    # Update config file on Pi with new model path
    # Config file lives at /home/protopi/ten90audio/ten90audio/config.json
    target_config_path = f"{remote_model_dir}/config.json"
    
    # Try to read existing config or create new one
    pi_config = {}
    config_updated = False
    
    if connection.file_exists(target_config_path):
        # Read existing config
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
                tmp_config_path = tmp_file.name
            
            success_dl, error_dl = connection.download_file(target_config_path, tmp_config_path, progress_callback=None)
            if success_dl:
                with open(tmp_config_path, 'r', encoding='utf-8') as f:
                    pi_config = json.load(f)
                os.remove(tmp_config_path)
        except Exception as e:
            if progress_callback:
                progress_callback(f"Warning: Could not read config from {target_config_path}: {e}")
    
    # Update config with new model path
    # Config only uses MODEL_PATH key (version is in model_version.txt)
    pi_config['MODEL_PATH'] = remote_model_path
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        json.dump(pi_config, tmp_file, indent=2)
        tmp_config_path = tmp_file.name
    
    try:
        success_cfg, error_cfg = connection.upload_file(tmp_config_path, target_config_path)
        if not success_cfg:
            if progress_callback:
                progress_callback(f"Warning: Could not update config file: {error_cfg}")
        else:
            if progress_callback:
                progress_callback(f"✓ Config file updated: {target_config_path}")
            config_updated = True
    finally:
        try:
            os.remove(tmp_config_path)
        except Exception:
            pass
    
    # Log the deployment
    deploy_log = load_deploy_log()
    deploy_log[model_filename] = {
        'deployed_at': datetime.utcnow().isoformat() + 'Z',
        'local_path': local_model_path,
        'remote_path': remote_model_path,
        'file_size': local_stat.st_size,
        'model_version': model_version,
        'config_updated': config_updated,
        'config_path': target_config_path if config_updated else None,
    }
    save_deploy_log(deploy_log)
    
    # Reboot Pi after successful deployment
    if progress_callback:
        progress_callback("Rebooting Pi...")
    
    # Use sudo reboot command - send it and immediately return success
    # The reboot will cause the connection to drop, so we don't wait for it
    try:
        # Execute reboot command without waiting for completion
        # Use nohup to detach it, or just send and catch connection drop
        if connection.client:
            stdin, stdout, stderr = connection.client.exec_command("sudo reboot")
            # Don't wait for exit status - just send the command and return
    except Exception:
        # Connection drop is expected - reboot is happening
        pass
    
    if progress_callback:
        progress_callback("✓ Reboot command sent successfully")
        progress_callback("Pi will reboot shortly...")
    
    # Return success immediately - don't wait for reboot to complete
    return True, None


def deploy_latest_model(progress_callback=None) -> Dict[str, any]:
    """Deploy the latest model to Pi.
    
    Returns dictionary with deployment status.
    """
    config = get_config()
    connection = PiConnection(config)
    
    # Connect
    success, error = connection.connect()
    if not success:
        return {
            'success': False,
            'error': error,
            'model_filename': None,
        }
    
    try:
        # Find latest model
        model_filename = get_latest_model()
        if not model_filename:
            return {
                'success': False,
                'error': 'No model files found in models/ directory',
                'model_filename': None,
            }
        
        if progress_callback:
            progress_callback(f"Found model: {model_filename}")
        
        # Deploy
        success, error = deploy_model(connection, model_filename, progress_callback=progress_callback)
        
        return {
            'success': success,
            'error': error,
            'model_filename': model_filename if success else None,
        }
    
    finally:
        connection.disconnect()


if __name__ == '__main__':
    def print_progress(msg):
        print(msg)
    
    result = deploy_latest_model(progress_callback=print_progress)
    
    if result['success']:
        print(f"\nDeployment successful: {result['model_filename']}")
    else:
        print(f"\nDeployment failed: {result['error']}")

