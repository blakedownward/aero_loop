"""Configuration management for remote Raspberry Pi connections."""

# Load .env file first
from .env_loader import load_env
load_env()

import os
from typing import Optional, Dict, Any, Tuple


class RemoteConfig:
    """Manages remote connection configuration."""
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.load()
    
    def load(self):
        """Load configuration from environment variables (from .env file or system env)."""
        self.config['host'] = os.getenv('PI_HOST', '')
        self.config['user'] = os.getenv('PI_USER', 'pi')
        self.config['sessions_path'] = os.getenv('PI_SESSIONS_PATH', '/home/pi/sessions')
        self.config['model_path'] = os.getenv('PI_MODEL_PATH', '/home/pi/models')
        self.config['ssh_key_path'] = os.getenv('PI_SSH_KEY_PATH', '')
        self.config['ssh_password'] = os.getenv('PI_SSH_PASSWORD', '')
        self.config['ssh_port'] = int(os.getenv('PI_SSH_PORT', 22))
        self.config['timeout'] = int(os.getenv('PI_SSH_TIMEOUT', 30))
    
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

