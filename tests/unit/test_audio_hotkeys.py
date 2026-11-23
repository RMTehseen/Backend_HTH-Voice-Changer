"""Unit tests for NativeHotkeyManager status and callbacks."""

from hth.audio.hotkeys import NativeHotkeyManager


def test_hotkey_manager_initial_status() -> None:
    mgr = NativeHotkeyManager()
    status = mgr.get_status()
    assert status.panic_hotkey == "Ctrl+Shift+X"
    assert status.passthrough_hotkey == "Ctrl+Shift+P"
    assert status.panic_registered is False
    assert status.passthrough_registered is False
    assert len(status.conflicts) == 0


def test_hotkey_manager_callbacks() -> None:
    panic_called = []
    pass_called = []

    mgr = NativeHotkeyManager(
        on_panic=lambda: panic_called.append(True),
        on_passthrough=lambda: pass_called.append(True),
    )

    # Manually invoke callbacks
    mgr.on_panic()
    assert len(panic_called) == 1

    mgr.on_passthrough()
    assert len(pass_called) == 1


def test_hotkey_start_stop_lifecycle() -> None:
    mgr = NativeHotkeyManager()
    mgr.start()
    # On non-Windows or headless test environment, thread handles it gracefully
    mgr.stop()
    assert mgr.get_status().panic_registered is False
