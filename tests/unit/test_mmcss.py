"""Unit tests for Windows MMCSS priority integration."""

import sys

from hth.audio.mmcss import (
    AVRT_PRIORITY_CRITICAL,
    AVRT_PRIORITY_HIGH,
    AVRT_PRIORITY_LOW,
    AVRT_PRIORITY_NORMAL,
    MMCSSPriorityScope,
    enable_mmcss,
    is_mmcss_available,
    revert_mmcss,
)


def test_mmcss_priorities_values() -> None:
    """Verifies standard MMCSS priority constants match Avrt.h specification."""
    assert AVRT_PRIORITY_LOW == -1
    assert AVRT_PRIORITY_NORMAL == 0
    assert AVRT_PRIORITY_HIGH == 1
    assert AVRT_PRIORITY_CRITICAL == 2


def test_revert_with_none_handle_safe() -> None:
    """Verifies that reverting a None handle safely returns False without raising."""
    assert revert_mmcss(None) is False


def test_mmcss_context_manager() -> None:
    """Verifies MMCSSPriorityScope enters and exits cleanly."""
    with MMCSSPriorityScope("Pro Audio", priority=AVRT_PRIORITY_HIGH) as scope:
        if sys.platform == "win32" and is_mmcss_available():
            assert scope.handle is not None
            assert isinstance(scope.handle, int)
            assert scope.handle > 0
        else:
            assert scope.handle is None

    # Handle must be cleared upon exit
    assert scope.handle is None


def test_enable_and_revert_lifecycle() -> None:
    """Verifies manual enable_mmcss and revert_mmcss lifecycle."""
    if sys.platform != "win32" or not is_mmcss_available():
        handle = enable_mmcss("Pro Audio")
        assert handle is None
        assert revert_mmcss(handle) is False
        return

    handle = enable_mmcss("Pro Audio", priority=AVRT_PRIORITY_HIGH)
    assert handle is not None
    assert isinstance(handle, int)
    assert handle > 0

    reverted = revert_mmcss(handle)
    assert reverted is True
