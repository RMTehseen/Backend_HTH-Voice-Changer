"""FakeOfflineConverter — deterministic in-memory chunked offline audio conversion.

Processes audio in bounded 4096-sample chunks with live progress and cancellation.
Produces valid WAV natively without external tools; requires ffmpeg for MP3/OGG.
"""

from __future__ import annotations

import asyncio
import io
import math
import struct
import subprocess
import wave
from pathlib import Path
from typing import Callable

from hth.adapters.offline.contracts import OfflineConverterProtocol
from hth.adapters.offline.ffmpeg_probe import probe_ffmpeg
from hth.adapters.offline.model_validator import validate_model_engine_compatibility
from hth.core.logging import get_logger
from hth.domain.api_contracts import AudioContainerFormat, SupportedModelFormat
from hth.domain.errors import AudioFormatError, JobCancelledError

logger = get_logger("hth.adapters.offline.fake")

CHUNK_FRAMES = 4096


class FakeOfflineConverter(OfflineConverterProtocol):
    """In-memory offline conversion adapter for development, testing, and evaluation."""

    def __init__(self) -> None:
        self._converter_id = "fake_offline"
        self._supported_formats = {SupportedModelFormat.FAKE_MODEL}

    @property
    def converter_id(self) -> str:
        return self._converter_id

    @property
    def supported_model_formats(self) -> set[SupportedModelFormat]:
        return self._supported_formats

    def validate_model(self, model_id: str, model_format: SupportedModelFormat) -> None:
        validate_model_engine_compatibility("fake", model_id, model_format)

    async def convert(
        self,
        job_id: str,
        input_file: Path,
        output_file: Path,
        target_format: AudioContainerFormat,
        model_id: str,
        pitch_shift: float,
        formant_shift: float,
        cancel_token: asyncio.Event,
        progress_callback: Callable[[float, int, int], None],
    ) -> Path:
        """Process input file in bounded chunks, reporting progress and honoring cancellation."""
        self.validate_model(model_id, SupportedModelFormat.FAKE_MODEL)

        if not input_file.is_file():
            raise AudioFormatError(f"Input file not found: {input_file}")

        # Check ffmpeg requirement upfront if compressed format requested
        ffmpeg_caps = probe_ffmpeg()
        if target_format == AudioContainerFormat.MP3 and not ffmpeg_caps.supports_mp3:
            raise AudioFormatError(
                "Export format 'mp3' requires ffmpeg with MP3 encoder (libmp3lame), but ffmpeg was not detected.",
                codec="libmp3lame",
            )
        if target_format == AudioContainerFormat.OGG and not ffmpeg_caps.supports_ogg_opus:
            raise AudioFormatError(
                "Export format 'ogg' requires ffmpeg with OGG/Opus encoder (libopus), but ffmpeg was not detected.",
                codec="libopus",
            )

        # Read or simulate frames
        input_bytes = input_file.read_bytes()
        sample_rate = 16000
        channels = 1
        raw_pcm_samples: list[int] = []

        if input_bytes.startswith(b"RIFF") and input_bytes[8:12] == b"WAVE":
            try:
                with wave.open(io.BytesIO(input_bytes), "rb") as wf:
                    channels = wf.getnchannels()
                    sample_rate = wf.getframerate()
                    total_frames_in = wf.getnframes()
                    frames_data = wf.readframes(total_frames_in)
                    fmt = f"<{len(frames_data) // 2}h"
                    raw_pcm_samples = list(struct.unpack(fmt, frames_data))
            except Exception:
                raw_pcm_samples = []

        if not raw_pcm_samples:
            # Generate deterministic synthetic test samples (1.5 seconds sine wave)
            total_frames_in = int(sample_rate * 1.5)
            for i in range(total_frames_in):
                sample_val = int(16000 * math.sin(2 * math.pi * 440 * i / sample_rate))
                raw_pcm_samples.append(sample_val)

        total_frames = len(raw_pcm_samples) // channels if channels > 0 else len(raw_pcm_samples)
        converted_samples: list[int] = []

        processed_frames = 0
        total_chunks = max(1, math.ceil(total_frames / CHUNK_FRAMES))

        pitch_factor = 2.0 ** (pitch_shift / 12.0)

        for chunk_idx in range(total_chunks):
            # 1. Cooperative Cancellation Check on EVERY chunk
            if cancel_token.is_set():
                logger.info("Conversion job cancelled by token", job_id=job_id)
                raise JobCancelledError(
                    f"Job '{job_id}' was cancelled by user request.",
                    job_id=job_id,
                )

            # 2. Extract bounded chunk slice
            start_idx = chunk_idx * CHUNK_FRAMES * channels
            end_idx = min(len(raw_pcm_samples), start_idx + CHUNK_FRAMES * channels)
            chunk_slice = raw_pcm_samples[start_idx:end_idx]

            # 3. Apply simulated pitch/formant transformation
            for s in chunk_slice:
                val = int(s * pitch_factor)
                val = max(-32768, min(32767, val))
                converted_samples.append(val)

            chunk_len_frames = (end_idx - start_idx) // channels
            processed_frames += chunk_len_frames
            progress = min(1.0, processed_frames / total_frames) if total_frames > 0 else 1.0

            # 4. Report progress
            progress_callback(progress, processed_frames, total_frames)

            # 5. Yield execution briefly for async cooperative multi-tasking
            await asyncio.sleep(0.01)

        # Write intermediate or final WAV
        temp_wav_path = output_file.with_suffix(".tmp.wav")
        packed_frames = struct.pack(f"<{len(converted_samples)}h", *converted_samples)

        with wave.open(str(temp_wav_path), "wb") as out_wf:
            out_wf.setnchannels(channels)
            out_wf.setsampwidth(2)  # 16-bit PCM
            out_wf.setframerate(sample_rate)
            out_wf.writeframes(packed_frames)

        # If WAV requested, rename temp to final output
        if target_format == AudioContainerFormat.WAV:
            if output_file.exists():
                output_file.unlink()
            temp_wav_path.rename(output_file)
            return output_file

        # If compressed format (MP3, OGG, M4A) requested and ffmpeg is available
        if ffmpeg_caps.detected and ffmpeg_caps.binary_path:
            cmd = [
                ffmpeg_caps.binary_path,
                "-y",
                "-i",
                str(temp_wav_path),
                "-c:a",
                "libmp3lame" if target_format == AudioContainerFormat.MP3 else "libopus",
                str(output_file),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if temp_wav_path.exists():
                temp_wav_path.unlink()

            if proc.returncode != 0:
                raise AudioFormatError(
                    f"ffmpeg encoding failed with exit code {proc.returncode}: {proc.stderr}"
                )
            return output_file

        # Fallback if ffmpeg is somehow missing here
        if temp_wav_path.exists():
            temp_wav_path.unlink()
        raise AudioFormatError(
            f"Encoding to '{target_format.value}' requires ffmpeg, which is unavailable."
        )
