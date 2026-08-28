"""GPU usage hardware metric (NVIDIA only, via pynvml)."""

import logging
from typing import Optional, Tuple

import pynvml as nvml


class GpuUsage:
    """NVIDIA GPU utilization and memory usage, if available."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._available = False
        try:
            nvml.nvmlInit()
            self._available = nvml.nvmlDeviceGetCount() > 0
            self.logger.info("GPU monitoring initialized successfully.")
        except Exception as exc:
            self.logger.warning(f"Failed to initialize GPU monitoring: {exc}")

    @property
    def available(self) -> bool:
        """Whether NVIDIA GPU monitoring is available on this machine."""
        return self._available

    def __call__(self) -> Tuple[Optional[float], Optional[float]]:
        """Sample current GPU utilization and memory usage.

        Returns:
            A tuple of (GPU usage percentage, memory used in MB).
            Returns (None, None) if unavailable or the sample fails.
        """
        if not self._available:
            return None, None
        try:
            device = nvml.nvmlDeviceGetHandleByIndex(0)
            utilization = nvml.nvmlDeviceGetUtilizationRates(device)
            memory = nvml.nvmlDeviceGetMemoryInfo(device)
            return float(utilization.gpu), float(memory.used / 1024 / 1024) # type: ignore
        except Exception as exc:
            self.logger.error(f"Error measuring GPU usage: {exc}")
            return None, None

    def __del__(self) -> None:
        """Shut down NVML on destruction."""
        if self._available:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass
