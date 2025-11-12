"""SSH connection handler for Raspberry Pi."""

import os
import paramiko
from typing import Optional, List, Tuple
from pathlib import Path

from .config import get_config


class PiConnection:
    """Manages SSH connection to Raspberry Pi."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.client: Optional[paramiko.SSHClient] = None
        self.sftp: Optional[paramiko.SFTPClient] = None
    
    def connect(self) -> Tuple[bool, Optional[str]]:
        """Establish SSH connection to Pi."""
        valid, error = self.config.validate()
        if not valid:
            return False, error
        
        try:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Prepare connection parameters
            connect_kwargs = {
                'hostname': self.config.get('host'),
                'username': self.config.get('user'),
                'port': self.config.get('ssh_port', 22),
                'timeout': self.config.get('timeout', 30),
            }
            
            # Use key file if available, otherwise password
            ssh_key_path = self.config.get('ssh_key_path')
            if ssh_key_path and os.path.isfile(ssh_key_path):
                connect_kwargs['key_filename'] = ssh_key_path
            else:
                password = self.config.get('ssh_password')
                if password:
                    connect_kwargs['password'] = password
                else:
                    return False, "No authentication method available (key or password)"
            
            self.client.connect(**connect_kwargs)
            
            # Open SFTP connection for file operations
            self.sftp = self.client.open_sftp()
            
            return True, None
        except paramiko.AuthenticationException:
            return False, "Authentication failed"
        except paramiko.SSHException as e:
            return False, f"SSH error: {str(e)}"
        except Exception as e:
            return False, f"Connection error: {str(e)}"
    
    def disconnect(self):
        """Close SSH connection."""
        if self.sftp:
            self.sftp.close()
            self.sftp = None
        if self.client:
            self.client.close()
            self.client = None
    
    def test_connection(self) -> Tuple[bool, Optional[str]]:
        """Test connection and return status."""
        if not self.client:
            success, error = self.connect()
            if not success:
                return False, error
        
        try:
            stdin, stdout, stderr = self.client.exec_command('echo "test"')
            exit_status = stdout.channel.recv_exit_status()
            if exit_status == 0:
                return True, None
            else:
                error_msg = stderr.read().decode('utf-8', errors='ignore')
                return False, f"Command failed: {error_msg}"
        except Exception as e:
            return False, f"Test failed: {str(e)}"
    
    def execute_command(self, command: str) -> Tuple[int, str, str]:
        """Execute a command on the Pi and return exit code, stdout, stderr."""
        if not self.client:
            success, error = self.connect()
            if not success:
                return -1, "", error
        
        try:
            stdin, stdout, stderr = self.client.exec_command(command)
            exit_status = stdout.channel.recv_exit_status()
            stdout_text = stdout.read().decode('utf-8', errors='ignore')
            stderr_text = stderr.read().decode('utf-8', errors='ignore')
            return exit_status, stdout_text, stderr_text
        except Exception as e:
            return -1, "", str(e)
    
    def list_directory(self, remote_path: str) -> List[str]:
        """List files and directories in remote path."""
        if not self.sftp:
            success, error = self.connect()
            if not success:
                return []
        
        try:
            return self.sftp.listdir(remote_path)
        except Exception:
            return []
    
    def file_exists(self, remote_path: str) -> bool:
        """Check if a file exists on the remote system."""
        if not self.sftp:
            success, error = self.connect()
            if not success:
                return False
        
        try:
            self.sftp.stat(remote_path)
            return True
        except FileNotFoundError:
            return False
        except Exception:
            return False
    
    def get_file_stat(self, remote_path: str) -> Optional[dict]:
        """Get file statistics (size, mtime, etc.) for remote file."""
        if not self.sftp:
            success, error = self.connect()
            if not success:
                return None
        
        try:
            stat = self.sftp.stat(remote_path)
            return {
                'size': stat.st_size,
                'mtime': stat.st_mtime,
                'mode': stat.st_mode,
            }
        except Exception:
            return None
    
    def download_file(self, remote_path: str, local_path: str, progress_callback=None) -> Tuple[bool, Optional[str]]:
        """Download a file from Pi to local system."""
        if not self.sftp:
            success, error = self.connect()
            if not success:
                return False, error
        
        try:
            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download with optional progress callback
            if progress_callback:
                # Use callback for progress tracking
                def callback(transferred, total):
                    progress_callback(transferred, total)
                self.sftp.get(remote_path, local_path, callback=callback)
            else:
                self.sftp.get(remote_path, local_path)
            
            return True, None
        except Exception as e:
            return False, str(e)
    
    def upload_file(self, local_path: str, remote_path: str, progress_callback=None) -> Tuple[bool, Optional[str]]:
        """Upload a file from local system to Pi."""
        if not self.sftp:
            success, error = self.connect()
            if not success:
                return False, error
        
        try:
            # Ensure remote directory exists
            remote_dir = os.path.dirname(remote_path)
            if remote_dir:
                self.execute_command(f'mkdir -p "{remote_dir}"')
            
            # Upload with optional progress callback
            if progress_callback:
                def callback(transferred, total):
                    progress_callback(transferred, total)
                self.sftp.put(local_path, remote_path, callback=callback)
            else:
                self.sftp.put(local_path, remote_path)
            
            return True, None
        except Exception as e:
            return False, str(e)
    
    def download_directory(self, remote_dir: str, local_dir: str, progress_callback=None, 
                          exclude_files: List[str] = None) -> Tuple[int, int, Optional[str]]:
        """Recursively download a directory from Pi to local system.
        
        Args:
            remote_dir: Remote directory path on Pi
            local_dir: Local directory path
            progress_callback: Optional callback for progress updates
            exclude_files: List of filenames to exclude from download (default: ['.processed'])
        
        Returns: (files_downloaded, files_skipped, error_message)
        """
        if exclude_files is None:
            exclude_files = ['.processed']
        
        if not self.sftp:
            success, error = self.connect()
            if not success:
                return 0, 0, error
        
        files_downloaded = 0
        files_skipped = 0
        
        try:
            os.makedirs(local_dir, exist_ok=True)
            
            def download_recursive(remote_path: str, local_path: str):
                nonlocal files_downloaded, files_skipped
                
                try:
                    items = self.sftp.listdir_attr(remote_path)
                    for item in items:
                        # Skip excluded files (e.g., .processed)
                        if item.filename in exclude_files:
                            continue
                        
                        remote_item = f"{remote_path}/{item.filename}"
                        local_item = os.path.join(local_path, item.filename)
                        
                        if item.st_mode & 0o040000:  # Directory
                            os.makedirs(local_item, exist_ok=True)
                            download_recursive(remote_item, local_item)
                        else:  # File
                            # Check if file already exists and is same size
                            if os.path.exists(local_item):
                                local_stat = os.stat(local_item)
                                if local_stat.st_size == item.st_size:
                                    files_skipped += 1
                                    continue
                            
                            try:
                                self.sftp.get(remote_item, local_item)
                                files_downloaded += 1
                                if progress_callback:
                                    progress_callback(files_downloaded, item.filename)
                            except Exception as e:
                                print(f"Warning: Failed to download {remote_item}: {e}")
                                files_skipped += 1
                except Exception as e:
                    raise Exception(f"Error in download_recursive: {e}")
            
            download_recursive(remote_dir, local_dir)
            return files_downloaded, files_skipped, None
            
        except Exception as e:
            return files_downloaded, files_skipped, str(e)
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

