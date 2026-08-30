"""GPU usage hardware metric (NVIDIA via pynvml, with Windows PDH fallback for Intel/AMD)."""

import logging
import sys
from typing import Optional, Tuple

try:
    import pynvml as nvml
except ImportError:
    nvml = None

if sys.platform == "win32":
    import ctypes

    class _PDH_FMT_COUNTERVALUE_ITEM(ctypes.Structure):
        pass

    class _ValueUnion(ctypes.Union):
        _fields_ = [
            ("longValue", ctypes.c_long),
            ("doubleValue", ctypes.c_double),
            ("stringValue", ctypes.c_wchar_p),
        ]

    class _PDH_FMT_COUNTERVALUE(ctypes.Structure):
        _fields_ = [("CStatus", ctypes.c_ulong), ("val", _ValueUnion)]

    _PDH_FMT_COUNTERVALUE_ITEM._fields_ = [
        ("szName", ctypes.c_wchar_p),
        ("FmtValue", _PDH_FMT_COUNTERVALUE),
    ]


class WindowsGpuUsage:
    """Fallback GPU tracker using Windows Performance Data Helper (PDH) API."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._available = False
        self._query = None
        self._util_counter = None
        self._mem_counter = None

        if sys.platform != "win32":
            return

        try:
            self.pdh = ctypes.windll.pdh
            self._query = ctypes.c_void_p()
            if self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)) != 0:
                return

            self._util_counter = ctypes.c_void_p()
            st1 = self.pdh.PdhAddEnglishCounterW(
                self._query,
                "\\GPU Engine(*)\\Utilization Percentage",
                0,
                ctypes.byref(self._util_counter),
            )

            self._mem_counter = ctypes.c_void_p()
            st2 = self.pdh.PdhAddEnglishCounterW(
                self._query,
                "\\GPU Process Memory(*)\\Shared Usage",
                0,
                ctypes.byref(self._mem_counter),
            )

            if st1 == 0:
                self.pdh.PdhCollectQueryData(self._query)
                self._available = True
                self.logger.info("Windows PDH GPU monitoring initialized successfully.")
        except Exception as exc:
            self.logger.warning(f"Failed to initialize Windows GPU monitoring: {exc}")

    @property
    def available(self) -> bool:
        return self._available

    def __call__(self) -> Tuple[Optional[float], Optional[float]]:
        if not self._available or not self._query:
            return None, None
        try:
            self.pdh.PdhCollectQueryData(self._query)

            # Util percentage
            util_val = 0.0
            if self._util_counter:
                buf_size = ctypes.c_ulong(0)
                item_count = ctypes.c_ulong(0)
                self.pdh.PdhGetFormattedCounterArrayW(
                    self._util_counter, 0x00000200, ctypes.byref(buf_size), ctypes.byref(item_count), None
                )
                if buf_size.value > 0:
                    buffer = (ctypes.c_byte * buf_size.value)()
                    self.pdh.PdhGetFormattedCounterArrayW(
                        self._util_counter, 0x00000200, ctypes.byref(buf_size), ctypes.byref(item_count), buffer
                    )
                    items = ctypes.cast(buffer, ctypes.POINTER(_PDH_FMT_COUNTERVALUE_ITEM))
                    util_val = sum(
                        items[i].FmtValue.val.doubleValue
                        for i in range(item_count.value)
                        if items[i].FmtValue.CStatus == 0
                    )

            # Memory used in MB
            mem_val = 0.0
            if self._mem_counter:
                buf_size = ctypes.c_ulong(0)
                item_count = ctypes.c_ulong(0)
                self.pdh.PdhGetFormattedCounterArrayW(
                    self._mem_counter, 0x00000100, ctypes.byref(buf_size), ctypes.byref(item_count), None
                )
                if buf_size.value > 0:
                    buffer = (ctypes.c_byte * buf_size.value)()
                    self.pdh.PdhGetFormattedCounterArrayW(
                        self._mem_counter, 0x00000100, ctypes.byref(buf_size), ctypes.byref(item_count), buffer
                    )
                    items = ctypes.cast(buffer, ctypes.POINTER(_PDH_FMT_COUNTERVALUE_ITEM))
                    mem_val = sum(
                        items[i].FmtValue.val.longValue
                        for i in range(item_count.value)
                        if items[i].FmtValue.CStatus == 0
                    ) / (1024 * 1024)

            return round(min(100.0, float(util_val)), 2), round(float(mem_val), 2)
        except Exception as exc:
            self.logger.error(f"Error measuring Windows GPU usage: {exc}")
            return None, None

    def __del__(self) -> None:
        if self._query:
            try:
                self.pdh.PdhCloseQuery(self._query)
            except Exception:
                pass


class GpuUsage:
    """Cross-platform GPU utilization and memory tracker (NVIDIA via NVML with Windows PDH fallback)."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._mode = None
        self._nvml_available = False
        self.win_tracker = None

        if nvml is not None:
            try:
                nvml.nvmlInit()
                if nvml.nvmlDeviceGetCount() > 0:
                    self._nvml_available = True
                    self._mode = "nvml"
                    self.logger.info("NVIDIA GPU monitoring initialized successfully.")
            except Exception:
                pass

        if not self._nvml_available and sys.platform == "win32":
            self.win_tracker = WindowsGpuUsage()
            if self.win_tracker.available:
                self._mode = "pdh"
                self.logger.info("Intel/AMD Windows GPU monitoring initialized successfully.")

        if self._mode is None:
            self.logger.warning("No compatible GPU monitoring mechanism found (neither NVML nor Windows PDH).")

    @property
    def available(self) -> bool:
        return self._mode is not None

    def __call__(self) -> Tuple[Optional[float], Optional[float]]:
        if self._mode == "nvml":
            try:
                device = nvml.nvmlDeviceGetHandleByIndex(0)
                utilization = nvml.nvmlDeviceGetUtilizationRates(device)
                memory = nvml.nvmlDeviceGetMemoryInfo(device)
                return float(utilization.gpu), float(memory.used / 1024 / 1024)
            except Exception as exc:
                self.logger.error(f"Error measuring NVIDIA GPU usage: {exc}")
                return None, None
        elif self._mode == "pdh" and self.win_tracker:
            return self.win_tracker()
        return None, None

    def __del__(self) -> None:
        if self._mode == "nvml":
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass
