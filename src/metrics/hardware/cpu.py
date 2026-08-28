"""CPU usage hardware metric."""

import logging

import psutil


class CpuUsage:
    """Average CPU usage across all cores, as a percentage.

    Uses non-blocking sampling: each call reports usage averaged over the
    time since the previous call (psutil.cpu_percent(interval=None)), not
    a blocking 1-second measurement. In a per-chunk streaming loop like
    SystemProfiler.wrap(), interval=1 would add a full second of artificial
    delay per token, swamping the real generation time.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        psutil.cpu_percent(interval=None)  # prime the internal counter

    def __call__(self) -> float:
        """Sample CPU usage since the last call, without blocking.

        Returns:
            CPU usage percentage. Returns 0.0 if the sample fails.
        """
        try:
            return psutil.cpu_percent(interval=None)
        except Exception as exc:
            self.logger.error(f"Error measuring CPU usage: {exc}")
            return 0.0
