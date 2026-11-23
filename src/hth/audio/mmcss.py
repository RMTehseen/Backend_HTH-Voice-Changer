"""Windows Multimedia Class Scheduler Service (MMCSS) Integration.

Elevates real-time audio callback threads to 'Pro Audio' high-priority scheduling
to eliminate buffer underruns (xruns) and OS scheduling jitter on Windows.

Strictly follows Win32 Avrt APIs:
- AvSetMmThreadCharacteristicsW (avrt.dll)
- AvSetMmThreadPriority
- AvRevertMmThreadCharacteristics

Adheres to:
- Rule A-01: Zero runtime heap allocation during audio stream.
- Rule G-04: Graceful fallback and safe handling if running without admin privileges or on non-Windows.
"""

from __future__ import annotations

import ctypes
import sys
from types import TracebackType
from typing import Optional

from hth.core.logging import get_logger

logger = get_logger("hth.audio.mmcss")

# MMCSS Thread Priorities defined by Avrt.h
AVRT_PRIORITY_LOW = -1
AVRT_PRIORITY_NORMAL = 0
AVRT_PRIORITY_HIGH = 1
AVRT_PRIORITY_CRITICAL = 2

# Global function pointers (loaded lazily on Windows)
_AVRT_AVAILABLE = False
_AvSetMmThreadCharacteristicsW = None
_AvSetMmThreadPriority = None
_AvRevertMmThreadCharacteristics = None

if sys.platform == "win32":
    try:
        from ctypes import wintypes

        _avrt = ctypes.windll.avrt

        # HANDLE AvSetMmThreadCharacteristicsW(LPCWSTR TaskName, LPDWORD TaskIndex)
        _AvSetMmThreadCharacteristicsW = _avrt.AvSetMmThreadCharacteristicsW
        _AvSetMmThreadCharacteristicsW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
        _AvSetMmThreadCharacteristicsW.restype = wintypes.HANDLE

        # BOOL AvSetMmThreadPriority(HANDLE AvrtHandle, AVRT_PRIORITY Priority)
        _AvSetMmThreadPriority = _avrt.AvSetMmThreadPriority
        _AvSetMmThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
        _AvSetMmThreadPriority.restype = wintypes.BOOL

        # BOOL AvRevertMmThreadCharacteristics(HANDLE AvrtHandle)
        _AvRevertMmThreadCharacteristics = _avrt.AvRevertMmThreadCharacteristics
        _AvRevertMmThreadCharacteristics.argtypes = [wintypes.HANDLE]
        _AvRevertMmThreadCharacteristics.restype = wintypes.BOOL

        _AVRT_AVAILABLE = True
    except Exception as exc:
        logger.warning("Failed to initialize Windows Avrt.dll: %s", exc)
        _AVRT_AVAILABLE = False


def is_mmcss_available() -> bool:
    """Returns True if Windows MMCSS APIs are accessible in the current environment."""
    return _AVRT_AVAILABLE


def enable_mmcss(
    task_name: str = "Pro Audio",
    priority: int = AVRT_PRIORITY_HIGH,
) -> Optional[int]:
    """Elevates the current calling thread to Windows MMCSS Pro Audio priority.

    Args:
        task_name: MMCSS task profile name (e.g. "Pro Audio", "Audio").
        priority: Scheduling priority offset (default AVRT_PRIORITY_HIGH = 1).

    Returns:
        Opaque MMCSS handle (int) if successful, None if unavailable or failed.
    """
    if not _AVRT_AVAILABLE or _AvSetMmThreadCharacteristicsW is None:
        return None

    try:
        from ctypes import wintypes

        task_idx = wintypes.DWORD(0)
        handle = _AvSetMmThreadCharacteristicsW(task_name, ctypes.byref(task_idx))
        if not handle:
            logger.debug(
                "AvSetMmThreadCharacteristicsW returned NULL handle for task %s", task_name
            )
            return None

        if _AvSetMmThreadPriority is not None:
            ok = _AvSetMmThreadPriority(handle, priority)
            if not ok:
                logger.debug("AvSetMmThreadPriority failed for handle %s", handle)

        logger.info(
            "MMCSS successfully enabled for thread %d (task=%s, priority=%d)",
            ctypes.windll.kernel32.GetCurrentThreadId(),
            task_name,
            priority,
        )
        return int(handle)
    except Exception as exc:
        logger.warning("Error enabling MMCSS: %s", exc)
        return None


def revert_mmcss(handle: Optional[int]) -> bool:
    """Reverts MMCSS thread priority and releases the handle.

    Args:
        handle: The handle returned by enable_mmcss(), or None.

    Returns:
        True if reverted successfully, False otherwise.
    """
    if not _AVRT_AVAILABLE or _AvRevertMmThreadCharacteristics is None or handle is None:
        return False

    try:
        from ctypes import wintypes

        ok = _AvRevertMmThreadCharacteristics(wintypes.HANDLE(handle))
        return bool(ok)
    except Exception as exc:
        logger.warning("Error reverting MMCSS handle %s: %s", handle, exc)
        return False


class MMCSSPriorityScope:
    """Context manager for elevating audio stream threads to MMCSS priority."""

    def __init__(
        self,
        task_name: str = "Pro Audio",
        priority: int = AVRT_PRIORITY_HIGH,
    ) -> None:
        self.task_name = task_name
        self.priority = priority
        self.handle: Optional[int] = None

    def __enter__(self) -> MMCSSPriorityScope:
        self.handle = enable_mmcss(self.task_name, self.priority)
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self.handle is not None:
            revert_mmcss(self.handle)
            self.handle = None
