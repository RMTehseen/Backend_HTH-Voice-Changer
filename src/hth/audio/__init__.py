"""Windows Audio Subsystem — Phase 4 Core Audio Routing & Controls."""

from hth.audio.devices import (
    AudioDeviceProvider,
    FakeDeviceProvider,
    WindowsMMDeviceProvider,
    get_audio_device_provider,
    get_device_inventory,
    list_audio_devices,
    set_audio_device_provider,
)
from hth.audio.hotkeys import NativeHotkeyManager, get_hotkey_manager
from hth.audio.mmcss import MMCSSPriorityScope, enable_mmcss, is_mmcss_available, revert_mmcss
from hth.audio.pipeline import AudioPipeline, AudioPipelineConfig, AudioPipelineMetrics
from hth.audio.recorder import TestRecorder, get_test_recorder
from hth.audio.resampler import (
    AudioResampler,
    PolyphaseResampler16To48,
    PolyphaseResampler48To16,
)
from hth.audio.ring_buffer import AudioRingBuffer
from hth.audio.router import AudioRouter, get_audio_router

__all__ = [
    "AudioDeviceProvider",
    "AudioPipeline",
    "AudioPipelineConfig",
    "AudioPipelineMetrics",
    "AudioResampler",
    "AudioRingBuffer",
    "AudioRouter",
    "FakeDeviceProvider",
    "MMCSSPriorityScope",
    "NativeHotkeyManager",
    "PolyphaseResampler16To48",
    "PolyphaseResampler48To16",
    "TestRecorder",
    "WindowsMMDeviceProvider",
    "enable_mmcss",
    "get_audio_device_provider",
    "get_audio_router",
    "get_device_inventory",
    "get_hotkey_manager",
    "get_test_recorder",
    "is_mmcss_available",
    "list_audio_devices",
    "revert_mmcss",
    "set_audio_device_provider",
]
