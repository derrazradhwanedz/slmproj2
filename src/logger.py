"""Application-wide logging setup, configured from app/src/config/logging.yaml."""

import logging
import os
from datetime import datetime

from utils import load_yaml

_LOG_DEFAULTS = load_yaml("logging.yaml")
_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # app/

_configured = False


def configure_logging() -> None:
    """Configure the root logger from app/src/config/logging.yaml.

    Idempotent: calls after the first are no-ops, so importing this module
    from multiple places does not attach duplicate handlers.
    """
    global _configured
    if _configured:
        return

    level_name = _LOG_DEFAULTS.get("level", "INFO")
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    fmt = _LOG_DEFAULTS.get("format", "%(asctime)s [%(levelname)s] %(message)s")
    log_dir_name = _LOG_DEFAULTS.get("log_dir")
    filename_format = _LOG_DEFAULTS.get("log_filename_format", "%Y%m%d_%H%M%S.log")

    handlers = [logging.StreamHandler()]
    if log_dir_name:
        log_dir = os.path.join(_APP_DIR, log_dir_name)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, datetime.now().strftime(filename_format))
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a module-scoped logger, configuring the root logger on first use.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A configured Logger instance.
    """
    configure_logging()
    return logging.getLogger(name)
