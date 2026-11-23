"""Unit tests for ProductionOfflineVoiceConverter strict decoding and safety.

Verifies:
- Valid 16-bit PCM WAV decoding fast-path.
- Valid 24-bit PCM WAV and float32 WAV decoding via PyAV.
- Strict rejection of empty files (0 bytes) with AudioFormatError.
- Strict rejection of corrupted headers with AudioFormatError.
- Strict rejection of random garbage bytes with AudioFormatError.
- Total absence of synthetic 140 Hz sine tone fallbacks.
- Temporary file cleanup on conversion failure.
- Preservation of original input file (never overwritten or modified).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hth.adapters.offline.production_converter import ProductionOfflineVoiceConverter
from hth.domain.api_contracts import AudioContainerFormat
from hth.domain.errors import AudioFormatError


@pytest.fixture
def converter() -> ProductionOfflineVoiceConverter:
    return ProductionOfflineVoiceConverter()


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


def test_decode_valid_16bit_wav(converter: ProductionOfflineVoiceConverter, temp_dir: Path) -> None:
    """Fast-path decoding of standard 16-bit PCM WAV."""
    wav_path = temp_dir / "valid_16bit.wav"
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
    sig = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(str(wav_path), sig, sr, subtype="PCM_16", format="WAV")

    pcm, rate = converter._decode_input_audio(wav_path)
    assert rate == sr
    assert len(pcm) == len(sig)
    assert pcm.dtype == np.float32
    # Verify signal amplitude matches
    assert np.max(np.abs(pcm)) > 0.35
    assert np.max(np.abs(pcm)) <= 1.0


def test_decode_valid_24bit_wav(converter: ProductionOfflineVoiceConverter, temp_dir: Path) -> None:
    """PyAV decoding of 24-bit PCM WAV."""
    wav_path = temp_dir / "valid_24bit.wav"
    sr = 24000
    t = np.linspace(0, 0.4, int(sr * 0.4), endpoint=False, dtype=np.float32)
    sig = (0.5 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    sf.write(str(wav_path), sig, sr, subtype="PCM_24", format="WAV")

    pcm, rate = converter._decode_input_audio(wav_path)
    assert rate == sr
    assert len(pcm) == len(sig)
    assert pcm.dtype == np.float32
    assert np.max(np.abs(pcm)) > 0.45


def test_decode_valid_float32_wav(
    converter: ProductionOfflineVoiceConverter, temp_dir: Path
) -> None:
    """PyAV decoding of 32-bit floating point WAV."""
    wav_path = temp_dir / "valid_float32.wav"
    sr = 48000
    t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False, dtype=np.float32)
    sig = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(wav_path), sig, sr, subtype="FLOAT", format="WAV")

    pcm, rate = converter._decode_input_audio(wav_path)
    assert rate == sr
    assert len(pcm) == len(sig)
    assert pcm.dtype == np.float32


def test_decode_empty_file_rejected(
    converter: ProductionOfflineVoiceConverter, temp_dir: Path
) -> None:
    """Empty files (0 bytes) must raise typed AudioFormatError, never synthetic tone."""
    empty_path = temp_dir / "empty.wav"
    empty_path.write_bytes(b"")

    with pytest.raises(AudioFormatError, match="empty \\(0 bytes\\)"):
        converter._decode_input_audio(empty_path)


def test_decode_corrupted_header_rejected(
    converter: ProductionOfflineVoiceConverter, temp_dir: Path
) -> None:
    """Truncated or corrupted header must raise typed AudioFormatError."""
    corrupt_path = temp_dir / "corrupted.wav"
    corrupt_path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00")

    with pytest.raises(AudioFormatError):
        converter._decode_input_audio(corrupt_path)


def test_decode_random_garbage_rejected(
    converter: ProductionOfflineVoiceConverter, temp_dir: Path
) -> None:
    """Non-audio random bytes must raise typed AudioFormatError."""
    garbage_path = temp_dir / "garbage.bin"
    garbage_path.write_bytes(b"X" * 1024)

    with pytest.raises(AudioFormatError):
        converter._decode_input_audio(garbage_path)


def test_decode_nonexistent_file_rejected(
    converter: ProductionOfflineVoiceConverter, temp_dir: Path
) -> None:
    """Non-existent file must raise AudioFormatError."""
    missing = temp_dir / "does_not_exist.wav"
    with pytest.raises(AudioFormatError, match="Input file not found"):
        converter._decode_input_audio(missing)


@pytest.mark.asyncio
async def test_convert_end_to_end_preserves_input(
    converter: ProductionOfflineVoiceConverter, temp_dir: Path
) -> None:
    """Verify conversion succeeds, preserves input file content and reports progress."""
    input_wav = temp_dir / "input_speech.wav"
    output_wav = temp_dir / "output_speech.wav"

    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
    sig = (0.5 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    sf.write(str(input_wav), sig, sr, subtype="PCM_16", format="WAV")
    original_input_bytes = input_wav.read_bytes()

    progresses: list[float] = []

    def on_progress(p: float, cur: int, total: int) -> None:
        progresses.append(p)

    cancel_token = asyncio.Event()

    res = await converter.convert(
        job_id="job-test-01",
        input_file=input_wav,
        output_file=output_wav,
        target_format=AudioContainerFormat.WAV,
        model_id="acoustic_profile_male_to_female",
        pitch_shift=10.0,
        formant_shift=1.18,
        cancel_token=cancel_token,
        progress_callback=on_progress,
    )

    assert res == output_wav
    assert output_wav.exists()
    assert output_wav.stat().st_size > 0
    # Original input must not be modified or overwritten
    assert input_wav.read_bytes() == original_input_bytes
    # Progress must reach 1.0
    assert len(progresses) > 0
    assert progresses[-1] == 1.0


@pytest.mark.asyncio
async def test_convert_corrupt_file_raises_and_cleans_temp(
    converter: ProductionOfflineVoiceConverter, temp_dir: Path
) -> None:
    """Corrupt audio must raise AudioFormatError and ensure no temp files remain."""
    corrupt_input = temp_dir / "bad_input.wav"
    corrupt_input.write_bytes(b"NOT_A_VALID_WAV_HEADER_DATA_1234567890")
    output_wav = temp_dir / "output.wav"
    temp_wav = output_wav.with_suffix(".tmp.wav")

    cancel_token = asyncio.Event()

    with pytest.raises(AudioFormatError):
        await converter.convert(
            job_id="job-bad",
            input_file=corrupt_input,
            output_file=output_wav,
            target_format=AudioContainerFormat.WAV,
            model_id="acoustic_profile_male_to_female",
            pitch_shift=0.0,
            formant_shift=0.0,
            cancel_token=cancel_token,
            progress_callback=lambda p, c, t: None,
        )

    # Neither output nor temp file should exist
    assert not output_wav.exists()
    assert not temp_wav.exists()
    # Original corrupt file is preserved
    assert corrupt_input.exists()
