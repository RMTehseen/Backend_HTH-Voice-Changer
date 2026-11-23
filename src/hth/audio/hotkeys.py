"""Native background global hotkey manager for audio panic controls.

Phase 4 requirements:
- Native background thread with Win32 message loop.
- Hotkeys:
  - Ctrl+Shift+X: Panic stop (immediately mutes output stream).
  - Ctrl+Shift+P: Deliberate natural-voice passthrough (visibly flagged, never auto-fallback).
- Conflict detection: Handle ERROR_HOTKEY_ALREADY_REGISTERED and report conflicts cleanly.
- Safe lifecycle with clean registration and unregistration.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from ctypes import wintypes
from typing import Any, Callable

from hth.core.logging import get_logger
from hth.domain.audio_contracts import HotkeyStatus

logger = get_logger("hth.audio.hotkeys")

# Win32 Constants
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012

HOTKEY_ID_PANIC = 1001
HOTKEY_ID_PASSTHROUGH = 1002

# Virtual Key Codes
VK_X = 0x58
VK_P = 0x50

ERROR_HOTKEY_ALREADY_REGISTERED = 1419


class NativeHotkeyManager:
    """Manages global hotkeys on a dedicated Windows background message thread."""

    def __init__(
        self,
        on_panic: Callable[..., Any] | None = None,
        on_passthrough: Callable[..., Any] | None = None,
    ) -> None:
        self.on_panic = on_panic
        self.on_passthrough = on_passthrough

        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()

        self._panic_registered = False
        self._passthrough_registered = False
        self._conflicts: list[str] = []

    def get_status(self) -> HotkeyStatus:
        """Return the current hotkey registration and conflict status."""
        return HotkeyStatus(
            panic_hotkey="Ctrl+Shift+X",
            passthrough_hotkey="Ctrl+Shift+P",
            panic_registered=self._panic_registered,
            passthrough_registered=self._passthrough_registered,
            conflicts=list(self._conflicts),
        )

    def start(self) -> None:
        """Start the background hotkey listener thread."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._ready_event.clear()
        self._conflicts.clear()

        self._thread = threading.Thread(
            target=self._run_hotkey_loop,
            name="HTH-GlobalHotkeys",
            daemon=True,
        )
        self._thread.start()
        self._ready_event.wait(timeout=2.0)

    def stop(self) -> None:
        """Stop the background hotkey loop and unregister hotkeys."""
        self._stop_event.set()
        if self._thread_id and sys.platform == "win32":
            # Post WM_QUIT to wake up GetMessage
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._panic_registered = False
        self._passthrough_registered = False

    def _run_hotkey_loop(self) -> None:
        if sys.platform != "win32":
            logger.info("Non-Windows environment: native hotkeys disabled")
            self._ready_event.set()
            return

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        self._thread_id = kernel32.GetCurrentThreadId()

        # 1. Register Ctrl+Shift+X (Panic Stop)
        # MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT
        mods = MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT
        res_panic = user32.RegisterHotKey(None, HOTKEY_ID_PANIC, mods, VK_X)
        if res_panic:
            self._panic_registered = True
            logger.info("Registered global Panic Stop hotkey: Ctrl+Shift+X")
        else:
            err = kernel32.GetLastError()
            msg = f"Failed to register Panic Stop (Ctrl+Shift+X) - Error {err}"
            if err == ERROR_HOTKEY_ALREADY_REGISTERED:
                msg = "Hotkey conflict: Ctrl+Shift+X is already registered by another application."
            self._conflicts.append(msg)
            logger.warning(msg)

        # 2. Register Ctrl+Shift+P (Intentional Passthrough)
        res_pass = user32.RegisterHotKey(None, HOTKEY_ID_PASSTHROUGH, mods, VK_P)
        if res_pass:
            self._passthrough_registered = True
            logger.info("Registered global Passthrough hotkey: Ctrl+Shift+P")
        else:
            err = kernel32.GetLastError()
            conflict_msg = f"Failed to register Passthrough (Ctrl+Shift+P) - Error {err}"
            if err == ERROR_HOTKEY_ALREADY_REGISTERED:
                conflict_msg = (
                    "Hotkey conflict: Ctrl+Shift+P is already registered by another application."
                )
            self._conflicts.append(conflict_msg)
            logger.warning(conflict_msg)

        self._ready_event.set()

        # Message loop
        win_msg = wintypes.MSG()
        while not self._stop_event.is_set():
            # GetMessage blocks until a message arrives
            ret = user32.GetMessageW(ctypes.byref(win_msg), None, 0, 0)
            if ret <= 0 or win_msg.message == WM_QUIT:
                break

            if win_msg.message == WM_HOTKEY:
                hotkey_id = win_msg.wParam
                if hotkey_id == HOTKEY_ID_PANIC:
                    logger.warning("HOTKEY PRESSED: Ctrl+Shift+X (Panic Stop)")
                    if self.on_panic:
                        try:
                            self.on_panic()
                        except Exception as e:
                            logger.error("Error in on_panic hotkey callback", error=str(e))
                elif hotkey_id == HOTKEY_ID_PASSTHROUGH:
                    logger.info("HOTKEY PRESSED: Ctrl+Shift+P (Natural-Voice Passthrough)")
                    if self.on_passthrough:
                        try:
                            self.on_passthrough()
                        except Exception as e:
                            logger.error("Error in on_passthrough hotkey callback", error=str(e))

            user32.TranslateMessage(ctypes.byref(win_msg))
            user32.DispatchMessageW(ctypes.byref(win_msg))

        # Cleanup registrations
        if self._panic_registered:
            user32.UnregisterHotKey(None, HOTKEY_ID_PANIC)
        if self._passthrough_registered:
            user32.UnregisterHotKey(None, HOTKEY_ID_PASSTHROUGH)
        logger.info("Native hotkey loop terminated cleanly")


# Global singleton instance
_hotkey_manager: NativeHotkeyManager | None = None


def get_hotkey_manager() -> NativeHotkeyManager:
    global _hotkey_manager
    if _hotkey_manager is None:
        from hth.audio.router import get_audio_router

        router = get_audio_router()
        _hotkey_manager = NativeHotkeyManager(
            on_panic=router.trigger_panic_stop,
            on_passthrough=router.toggle_passthrough,
        )
    return _hotkey_manager
