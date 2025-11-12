"""Configuration management for remote Raspberry Pi connections."""

# Load .env file first
from .env_loader import load_env
load_env()

import os
import json
from typing import Optional, Dict, Any, Tuple


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir))


CONFIG_DIR = os.path.join(_repo_root(), 'config')
CONFIG_FILE = os.path.join(CONFIG_DIR, 'remote_config.json')


class RemoteConfig:
    """Manages remote connection configuration."""
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.load()
    
    def load(self):
        """Load configuration from file or environment variables."""
        # Try to load from config file
        if os.path.isfile(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
                self.config = {}
        
        # Override with environment variables if present (from .env or system env)
        self.config['host'] = os.getenv('PI_HOST', self.config.get('host', ''))
        self.config['user'] = os.getenv('PI_USER', self.config.get('user', 'pi'))
        self.config['sessions_path'] = os.getenv('PI_SESSIONS_PATH', self.config.get('sessions_path', '/home/pi/sessions'))
        self.config['model_path'] = os.getenv('PI_MODEL_PATH', self.config.get('model_path', '/home/pi/models'))
        self.config['ssh_key_path'] = os.getenv('PI_SSH_KEY_PATH', self.config.get('ssh_key_path', ''))
        self.config['ssh_password'] = os.getenv('PI_SSH_PASSWORD', self.config.get('ssh_password', ''))
        self.config['ssh_port'] = int(os.getenv('PI_SSH_PORT', self.config.get('ssh_port', 22)))
        self.config['timeout'] = int(os.getenv('PI_SSH_TIMEOUT', self.config.get('timeout', 30)))
    
    def save(self):
        """Save configuration to file."""
        os.makedirs(CONFIG_DIR, exist_ok=True)
        # Don't save password to file
        save_config = {k: v for k, v in self.config.items() if k != 'ssh_password'}
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(save_config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a configuration value."""
        self.config[key] = value
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate that required configuration is present."""
        if not self.config.get('host'):
            return False, "PI_HOST is required"
        if not self.config.get('user'):
            return False, "PI_USER is required"
        if not self.config.get('sessions_path'):
            return False, "PI_SESSIONS_PATH is required"
        if not self.config.get('model_path'):
            return False, "PI_MODEL_PATH is required"
        if not self.config.get('ssh_key_path') and not self.config.get('ssh_password'):
            return False, "Either PI_SSH_KEY_PATH or PI_SSH_PASSWORD is required"
        return True, None


# Global config instance
_config_instance: Optional[RemoteConfig] = None


def get_config() -> RemoteConfig:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = RemoteConfig()
    return _config_instance

