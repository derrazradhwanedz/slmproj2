"""System resource profiling during LLM generation.

Note: the old profiler.py exposed this as a generator that returned the
averages dict via `return {...}` inside the generator function. That value
is only reachable through the StopIteration raised at the exact moment the
generator ends - a `for` loop consuming the generator swallows that
StopIteration internally, so the averages were never actually retrievable
by the caller once the loop finished (confirmed against every results CSV
in this repo: cpu_usage_avg_percent is 0.0 and gpu_memory_avg_mb is NaN in
every row). SystemProfiler.wrap()/get_averages() below replaces that
pattern with explicit state so the averages can't be lost.
"""

import time
from typing import Dict, Iterable, Iterator, List, Optional

from metrics.hardware.cpu import CpuUsage
from metrics.hardware.gpu import GpuUsage
from metrics.hardware.ram import RamUsage


class SystemProfiler:
    """Samples CPU, RAM, and GPU usage while an LLM streams a response."""

    def __init__(self, sample_interval: float = 0.5) -> None:
        """Initialize the profiler and its per-resource samplers.

        Args:
            sample_interval: Seconds to sleep between samples while wrapping
                a stream.
        """
        self.sample_interval = sample_interval
        self.cpu_usage = CpuUsage()
        self.ram_usage = RamUsage()
        self.gpu_usage = GpuUsage()
        self._cpu_samples: List[float] = []
        self._ram_mb_samples: List[float] = []
        self._ram_percent_samples: List[float] = []
        self._gpu_usage_samples: List[Optional[float]] = []
        self._gpu_memory_samples: List[Optional[float]] = []

    def wrap(self, stream_iterator: Iterable[str]) -> Iterator[str]:
        """Yield each chunk from stream_iterator while sampling usage.

        Args:
            stream_iterator: Iterable yielding chunks of the LLM response.

        Yields:
            Each chunk from stream_iterator, unchanged.
        """
        for chunk in stream_iterator:
            self._cpu_samples.append(self.cpu_usage())
            ram_mb, ram_percent = self.ram_usage()
            self._ram_mb_samples.append(ram_mb)
            self._ram_percent_samples.append(ram_percent)
            gpu_pct, gpu_mem = self.gpu_usage()
            self._gpu_usage_samples.append(gpu_pct)
            self._gpu_memory_samples.append(gpu_mem)

            yield chunk
            time.sleep(self.sample_interval)

    def get_averages(self) -> Dict[str, Optional[float]]:
        """Return the average usage sampled so far by wrap().

        Returns:
            A dict with cpu_usage_avg_percent, ram_usage_avg_mb,
            ram_usage_avg_percent, gpu_usage_avg_percent, and
            gpu_memory_avg_mb. The gpu_* keys are None if no GPU sample was
            ever available. Call this only after fully consuming the
            iterator returned by wrap().
        """
        valid_gpu_usage = [v for v in self._gpu_usage_samples if v is not None]
        valid_gpu_memory = [v for v in self._gpu_memory_samples if v is not None]
        return {
            "cpu_usage_avg_percent": (
                sum(self._cpu_samples) / len(self._cpu_samples) if self._cpu_samples else 0.0
            ),
            "ram_usage_avg_mb": (
                sum(self._ram_mb_samples) / len(self._ram_mb_samples)
                if self._ram_mb_samples
                else 0.0
            ),
            "ram_usage_avg_percent": (
                sum(self._ram_percent_samples) / len(self._ram_percent_samples)
                if self._ram_percent_samples
                else 0.0
            ),
            "gpu_usage_avg_percent": (
                sum(valid_gpu_usage) / len(valid_gpu_usage) if valid_gpu_usage else None
            ),
            "gpu_memory_avg_mb": (
                sum(valid_gpu_memory) / len(valid_gpu_memory) if valid_gpu_memory else None
            ),
        }
