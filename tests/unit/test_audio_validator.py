"""Unit tests for audio format validation, magic bytes, and bounds enforcement (Phase 9)."""

import io
import struct
import wave

import pytest

from hth.adapters.offline.audio_validator import (
    validate_audio_file,
    verify_magic_bytes,
)
from hth.domain.api_contracts import AudioContainerFormat
from hth.domain.errors import AudioFormatError


def _make_dummy_wav(duration_sec: float = 1.0, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        nframes = int(sample_rate * duration_sec)
        samples = [0] * nframes
        wf.writeframes(struct.pack(f"<{nframes}h", *samples))
    return buf.getvalue()


def test_magic_bytes_detection():
    wav_bytes = _make_dummy_wav()
    assert verify_magic_bytes(wav_bytes) == AudioContainerFormat.WAV

    mp3_id3 = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 20
    assert verify_magic_bytes(mp3_id3) == AudioContainerFormat.MP3

    ogg_bytes = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00" + b"\x00" * 20
    assert verify_magic_bytes(ogg_bytes) == AudioContainerFormat.OGG

    flac_bytes = b"fLaC\x00\x00\x00\x22" + b"\x00" * 20
    assert verify_magic_bytes(flac_bytes) == AudioContainerFormat.FLAC

    m4a_bytes = b"\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00" + b"\x00" * 20
    assert verify_magic_bytes(m4a_bytes) == AudioContainerFormat.M4A

    assert verify_magic_bytes(b"MZ\x90\x00\x03\x00\x00\x00") is None  # EXE rejected


def test_validate_valid_wav():
    wav_data = _make_dummy_wav(duration_sec=2.0)
    meta = validate_audio_file(wav_data, "test_voice.wav")
    assert meta.container_format == AudioContainerFormat.WAV
    assert meta.magic_validated
    assert abs(meta.duration_seconds - 2.0) < 0.05
    assert meta.sample_rate == 16000


def test_reject_empty_file():
    with pytest.raises(AudioFormatError, match="Empty audio file"):
        validate_audio_file(b"", "empty.wav")


def test_reject_size_exceeded():
    wav_data = _make_dummy_wav(duration_sec=0.5)
    with pytest.raises(AudioFormatError, match="exceeds maximum allowed limit"):
        validate_audio_file(wav_data, "large.wav", max_size_bytes=100)


def test_reject_duration_exceeded():
    wav_data = _make_dummy_wav(duration_sec=10.0)
    with pytest.raises(AudioFormatError, match="exceeds maximum allowed duration"):
        validate_audio_file(wav_data, "long.wav", max_duration_seconds=5.0)


def test_reject_mismatched_container():
    # Content is WAV but named .mp3
    wav_data = _make_dummy_wav()
    with pytest.raises(AudioFormatError, match="does not match header signature"):
        validate_audio_file(wav_data, "spoofed.mp3")


def test_reject_unsafe_extension():
    with pytest.raises(AudioFormatError, match="Unsupported file extension"):
        validate_audio_file(b"RIFF\x24\x00\x00\x00WAVEfmt ", "payload.exe")
