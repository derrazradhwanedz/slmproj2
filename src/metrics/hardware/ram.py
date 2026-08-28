"""RAM usage hardware metric."""

import logging
from typing import Tuple

import psutil


class RamUsage:
    """System RAM usage, as (used MB, percent used)."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def __call__(self) -> Tuple[float, float]:
        """Sample current RAM usage.

        Returns:
            A tuple of (used RAM in MB, percentage of RAM used).
            Returns (0.0, 0.0) if the sample fails.
        """
        try:
            mem = psutil.virtual_memory()
            return mem.used / 1024 / 1024, mem.percent
        except Exception as exc:
            self.logger.error(f"Error measuring RAM usage: {exc}")
            return 0.0, 0.0
