"""Load environment variables from .env file."""

import os
from pathlib import Path


def _repo_root() -> str:
    """Get the repository root directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir))


def load_env():
    """Load environment variables from .env file in repo root."""
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(_repo_root(), '.env')
        if os.path.isfile(env_path):
            load_dotenv(env_path)
    except ImportError:
        # python-dotenv not installed, skip
        pass
    except Exception:
        # Silently fail if .env file has issues
        pass


# Auto-load on import
load_env()

