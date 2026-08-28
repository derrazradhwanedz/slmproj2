"""Loader for YAML config files under app/src/config."""

import os
from typing import Any, Dict

import yaml

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.normpath(os.path.join(_UTILS_DIR, "..", "config"))


def load_yaml(filename: str) -> Dict[str, Any]:
    """Load a YAML config file from app/src/config.

    Args:
        filename: File name relative to app/src/config (e.g. "nn.yaml").

    Returns:
        The parsed YAML content as a dict.

    Raises:
        FileNotFoundError: If the file does not exist in app/src/config.
    """
    path = os.path.join(CONFIG_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
