"""Download finished session data from Raspberry Pi."""

import os
import sys
import json
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
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


RAW_PATH = os.path.join(_repo_root(), 'data', 'raw')
DOWNLOAD_LOG = os.path.join(_repo_root(), 'data', 'download_log.json')


def load_download_log() -> Dict[str, Dict]:
    """Load the download log tracking which sessions have been downloaded."""
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


def parse_session_name(name: str) -> Optional[datetime]:
    """Parse session name to datetime. Expected format: YYYY-MM-DD_HH-MM"""
    try:
        return datetime.strptime(name, '%Y-%m-%d_%H-%M')
    except ValueError:
        return None


def has_newer_session(session_name: str, all_sessions: List[str]) -> bool:
    """Check if a newer session exists (primary detection method)."""
    session_dt = parse_session_name(session_name)
    if session_dt is None:
        return False
    
    for other_session in all_sessions:
        if other_session == session_name:
            continue
        other_dt = parse_session_name(other_session)
        if other_dt and other_dt > session_dt:
            return True
    
    return False


def has_processed_marker(connection: PiConnection, session_path: str) -> bool:
    """Check if session has .processed marker file (secondary detection method)."""
    marker_path = f"{session_path}/.processed"
    return connection.file_exists(marker_path)


def is_stale_session(connection: PiConnection, session_path: str, max_age_hours: int = 6) -> bool:
    """Check if session is stale based on age (fallback detection method)."""
    stat = connection.get_file_stat(session_path)
    if stat is None:
        return False
    
    # Get modification time
    mtime = datetime.fromtimestamp(stat['mtime'])
    age = datetime.now() - mtime
    
    # Consider stale if older than max_age_hours and no recent file activity
    if age > timedelta(hours=max_age_hours):
        # Check for any files modified in the last hour
        try:
            # List directory and check file modification times
            files = connection.list_directory(session_path)
            recent_activity = False
            for filename in files:
                if filename.startswith('.'):
                    continue
                file_path = f"{session_path}/{filename}"
                file_stat = connection.get_file_stat(file_path)
                if file_stat:
                    file_mtime = datetime.fromtimestamp(file_stat['mtime'])
                    if (datetime.now() - file_mtime) < timedelta(hours=1):
                        recent_activity = True
                        break
            
            # Stale if old and no recent activity
            return not recent_activity
        except Exception:
            pass
    
    return False


def is_session_finished(connection: PiConnection, session_name: str, session_path: str, all_sessions: List[str]) -> bool:
    """Determine if a session is finished using multiple detection methods."""
    # Primary: Newer session exists
    if has_newer_session(session_name, all_sessions):
        return True
    
    # Secondary: .processed marker exists
    if has_processed_marker(connection, session_path):
        return True
    
    # Fallback: Time-based (stale session)
    if is_stale_session(connection, session_path):
        return True
    
    return False


def get_finished_sessions(connection: PiConnection, sessions_path: str) -> List[str]:
    """Get list of finished session names."""
    all_sessions = connection.list_directory(sessions_path)
    
    # Filter to valid session names and sort
    valid_sessions = []
    for session in all_sessions:
        if parse_session_name(session) is not None:
            valid_sessions.append(session)
    
    valid_sessions.sort()
    
    # Check each session to see if it's finished
    finished = []
    for session in valid_sessions:
        session_path = f"{sessions_path}/{session}"
        if is_session_finished(connection, session, session_path, valid_sessions):
            finished.append(session)
    
    return finished


def download_session(connection: PiConnection, session_name: str, sessions_path: str, 
                    progress_callback=None) -> Tuple[bool, Optional[str], int, int]:
    """Download a single session from Pi to local data/raw directory.
    
    Returns: (success, error_message, files_downloaded, files_skipped)
    """
    remote_session_path = f"{sessions_path}/{session_name}"
    local_session_path = os.path.join(RAW_PATH, session_name)
    
    # Check if already downloaded
    download_log = load_download_log()
    if session_name in download_log:
        log_entry = download_log[session_name]
        # Verify the local directory still exists
        if os.path.isdir(local_session_path):
            # Could verify file count matches, but for now just skip
            if progress_callback:
                progress_callback(f"Skipping {session_name} (already downloaded)")
            return True, None, 0, 0
    
    # Download the session directory
    def progress_wrapper(downloaded, filename):
        if progress_callback:
            progress_callback(f"Downloading {session_name}/{filename}...")
    
    files_downloaded, files_skipped, error = connection.download_directory(
        remote_session_path, local_session_path, progress_callback=progress_wrapper
    )
    
    if error:
        return False, error, files_downloaded, files_skipped
    
    # Log the download
    download_log[session_name] = {
        'downloaded_at': datetime.now(timezone.utc).isoformat() + 'Z',
        'files_downloaded': files_downloaded,
        'files_skipped': files_skipped,
    }
    save_download_log(download_log)
    
    return True, None, files_downloaded, files_skipped


def download_finished_sessions(progress_callback=None) -> Dict[str, any]:
    """Download all finished sessions from Pi.
    
    Returns dictionary with summary statistics.
    """
    config = get_config()
    connection = PiConnection(config)
    
    # Connect
    success, error = connection.connect()
    if not success:
        return {
            'success': False,
            'error': error,
            'sessions_downloaded': 0,
            'sessions_skipped': 0,
            'total_files_downloaded': 0,
            'total_files_skipped': 0,
        }
    
    try:
        sessions_path = config.get('sessions_path')
        
        # Get finished sessions
        if progress_callback:
            progress_callback("Scanning for finished sessions...")
        
        finished_sessions = get_finished_sessions(connection, sessions_path)
        
        if not finished_sessions:
            if progress_callback:
                progress_callback("No finished sessions found.")
            return {
                'success': True,
                'error': None,
                'sessions_downloaded': 0,
                'sessions_skipped': 0,
                'total_files_downloaded': 0,
                'total_files_skipped': 0,
            }
        
        if progress_callback:
            progress_callback(f"Found {len(finished_sessions)} finished session(s)")
        
        # Download each session
        sessions_downloaded = 0
        sessions_skipped = 0
        total_files_downloaded = 0
        total_files_skipped = 0
        
        for session_name in finished_sessions:
            success, error, files_dl, files_sk = download_session(
                connection, session_name, sessions_path, progress_callback
            )
            
            if success:
                if files_dl > 0:
                    sessions_downloaded += 1
                else:
                    sessions_skipped += 1
                total_files_downloaded += files_dl
                total_files_skipped += files_sk
            else:
                if progress_callback:
                    progress_callback(f"Error downloading {session_name}: {error}")
        
        return {
            'success': True,
            'error': None,
            'sessions_downloaded': sessions_downloaded,
            'sessions_skipped': sessions_skipped,
            'total_files_downloaded': total_files_downloaded,
            'total_files_skipped': total_files_skipped,
        }
    
    finally:
        connection.disconnect()


if __name__ == '__main__':
    def print_progress(msg):
        print(msg)
    
    result = download_finished_sessions(progress_callback=print_progress)
    
    if result['success']:
        print(f"\nDownload complete:")
        print(f"  Sessions downloaded: {result['sessions_downloaded']}")
        print(f"  Sessions skipped: {result['sessions_skipped']}")
        print(f"  Files downloaded: {result['total_files_downloaded']}")
        print(f"  Files skipped: {result['total_files_skipped']}")
    else:
        print(f"\nDownload failed: {result['error']}")

