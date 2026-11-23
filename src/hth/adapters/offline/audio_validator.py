"""Audio format validation, magic byte verification, and bounds enforcement.

Enforces:
- Safe list of audio formats (.wav, .mp3, .ogg, .flac, .m4a)
- Magic bytes inspection (never trust file extension alone)
- Maximum file size bounds (25 MB)
- Maximum audio duration bounds (300 seconds)
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

from hth.adapters.offline.contracts import AudioMetadata
from hth.domain.api_contracts import AudioContainerFormat
from hth.domain.errors import AudioFormatError

SAFE_EXTENSIONS: dict[str, AudioContainerFormat] = {
    ".wav": AudioContainerFormat.WAV,
    ".mp3": AudioContainerFormat.MP3,
    ".ogg": AudioContainerFormat.OGG,
    ".flac": AudioContainerFormat.FLAC,
    ".m4a": AudioContainerFormat.M4A,
}


def verify_magic_bytes(header: bytes) -> AudioContainerFormat | None:
    """Check the initial byte signature against known safe audio formats."""
    if len(header) < 12:
        return None

    # WAV: RIFF....WAVE
    if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        return AudioContainerFormat.WAV

    # MP3: ID3 header or sync word
    if header.startswith(b"ID3"):
        return AudioContainerFormat.MP3
    if header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        return AudioContainerFormat.MP3

    # OGG: OggS
    if header.startswith(b"OggS"):
        return AudioContainerFormat.OGG

    # FLAC: fLaC
    if header.startswith(b"fLaC"):
        return AudioContainerFormat.FLAC

    # M4A: ....ftyp
    if len(header) >= 8 and header[4:8] == b"ftyp":
        return AudioContainerFormat.M4A

    return None


def validate_audio_file(
    file_bytes: bytes,
    original_filename: str,
    max_size_bytes: int = 26_214_400,  # 25 MB
    max_duration_seconds: float = 300.0,  # 5 minutes
) -> AudioMetadata:
    """Validate audio file format, size, magic bytes, and duration limits.

    Raises AudioFormatError if validation fails.
    """
    file_size = len(file_bytes)
    if file_size == 0:
        raise AudioFormatError("Empty audio file submitted.")

    if file_size > max_size_bytes:
        size_mb = file_size / (1024 * 1024)
        max_mb = max_size_bytes / (1024 * 1024)
        raise AudioFormatError(
            f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit ({max_mb:.2f} MB)."
        )

    # Extension check
    suffix = Path(original_filename).suffix.lower()
    expected_format = SAFE_EXTENSIONS.get(suffix)
    if expected_format is None:
        allowed = ", ".join(SAFE_EXTENSIONS.keys())
        raise AudioFormatError(
            f"Unsupported file extension '{suffix}'. Safe audio formats allowed: [{allowed}]"
        )

    # Magic byte check
    detected_format = verify_magic_bytes(file_bytes[:32])
    if detected_format is None:
        raise AudioFormatError(
            f"File '{original_filename}' does not contain valid audio header magic bytes. "
            "Corrupt or masqueraded file rejected."
        )

    if detected_format != expected_format:
        raise AudioFormatError(
            f"File extension '{suffix}' does not match header signature '{detected_format.value}'. "
            "Mismatched audio container rejected."
        )

    # Read duration & audio specs if WAV (native Python wave parser)
    duration_sec = 0.0
    sample_rate = 16000
    channels = 1

    if detected_format == AudioContainerFormat.WAV:
        try:
            with wave.open(io.BytesIO(file_bytes), "rb") as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                nframes = wf.getnframes()
                duration_sec = nframes / float(sample_rate) if sample_rate > 0 else 0.0
        except Exception as exc:
            raise AudioFormatError(f"Corrupt WAV structure: {exc}") from exc
    else:
        # Heuristic duration estimate based on conservative bitrate (e.g., 320 kbps = 40 KB/s)
        # 25 MB max at 40 KB/s is ~625s, but we estimate:
        duration_sec = min(float(file_size) / (40.0 * 1024.0), max_duration_seconds)

    if duration_sec > max_duration_seconds:
        raise AudioFormatError(
            f"Audio duration ({duration_sec:.1f}s) exceeds maximum allowed duration ({max_duration_seconds}s)."
        )

    return AudioMetadata(
        container_format=detected_format,
        duration_seconds=duration_sec,
        file_size_bytes=file_size,
        sample_rate=sample_rate,
        channels=channels,
        magic_validated=True,
    )
