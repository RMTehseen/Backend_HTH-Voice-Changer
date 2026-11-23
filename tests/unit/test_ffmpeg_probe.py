"""Unit tests for ffmpeg detection and capability probing (Phase 9)."""

from hth.adapters.offline.ffmpeg_probe import FfmpegCapabilities, probe_ffmpeg


def test_ffmpeg_capabilities_properties():
    caps_none = FfmpegCapabilities(detected=False)
    assert not caps_none.detected
    assert not caps_none.supports_mp3
    assert not caps_none.supports_ogg_opus
    assert not caps_none.supports_aac
    assert caps_none.supported_output_formats == ["wav"]

    caps_full = FfmpegCapabilities(
        detected=True,
        version="ffmpeg version 6.1.1",
        encoders={"libmp3lame", "libopus", "aac"},
    )
    assert caps_full.detected
    assert caps_full.supports_mp3
    assert caps_full.supports_ogg_opus
    assert caps_full.supports_aac
    assert set(caps_full.supported_output_formats) == {"wav", "mp3", "ogg", "m4a"}


def test_probe_ffmpeg_invalid_path():
    result = probe_ffmpeg(custom_path="C:\\nonexistent\\ffmpeg_not_found.exe")
    assert not result.detected
    assert result.version is None
    assert result.supported_output_formats == ["wav"]
