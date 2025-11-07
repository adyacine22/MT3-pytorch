"""Global configuration helpers for MT3-PyTorch."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

from .project_config import PROJECT_CONFIG

_CONFIG_PATH = Path(__file__).with_name("project_config.py")


def config_path() -> Path:
    """Return the Python module path that defines the project configuration."""
    return _CONFIG_PATH


def load_project_config() -> Dict[str, Any]:
    """Return a copy of the global project configuration dictionary."""
    return copy.deepcopy(PROJECT_CONFIG)


__all__ = ["config_path", "load_project_config"]
