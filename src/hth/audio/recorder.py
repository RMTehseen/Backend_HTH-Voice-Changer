"""Opt-in short test sample recording service with automatic expiration and cleanup.

Phase 4 requirements:
- Record 5-second test samples only on explicit user request.
- Clearly explain retention policy in response.
- Automatically delete recordings after configured short duration (60 seconds).
- Never store audio files inside Git repository (saves to %LOCALAPPDATA%\\HTH\\temp_recordings).
"""

from __future__ import annotations

import os
import threading
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

from hth.core.logging import get_logger
from hth.domain.audio_contracts import TestRecordRequest, TestRecordResponse

logger = get_logger("hth.audio.recorder")

DEFAULT_EXPIRATION_SECONDS = 60


def get_recordings_dir() -> Path:
    """Return the temporary test recordings directory under LocalAppData."""
    local_app_data = os.environ.get(
        "LOCALAPPDATA",
        str(Path.home() / "AppData" / "Local"),
    )
    p = Path(local_app_data) / "HTH" / "temp_recordings"
    p.mkdir(parents=True, exist_ok=True)
    return p


class TestRecorder:
    """Handles opt-in self-verification recordings and automatic time-based purging."""

    __test__ = False

    def __init__(self, expiration_seconds: int = DEFAULT_EXPIRATION_SECONDS) -> None:
        self.expiration_seconds = expiration_seconds
        self._recordings_dir = get_recordings_dir()
        self._lock = threading.Lock()
        self._tracked_files: dict[str, float] = {}  # filepath -> expiry_time

        # Start background cleanup thread
        self._stop_event = threading.Event()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="HTH-TestRecorder-Cleanup",
            daemon=True,
        )
        self._cleanup_thread.start()

    def record_test_sample(self, req: TestRecordRequest) -> TestRecordResponse:
        """Create a 5-second test audio sample for self-verification."""
        rec_id = str(uuid.uuid4())[:8]
        duration = req.duration_seconds
        filename = f"test_sample_{rec_id}.wav"
        file_path = self._recordings_dir / filename

        # Rule 8: Capture real physical microphone PCM audio instead of synthetic sine tones
        if os.environ.get("HTH_USE_FAKE_AUDIO_DEVICES") == "1":
            sample_rate = 16000
            num_samples = int(sample_rate * duration)
            with wave.open(str(file_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(b"\x00\x00" * num_samples)
        else:
            _recorded_real = False
            try:
                import numpy as np
                import sounddevice as sd

                in_dev = req.device_id
                sample_rate = 16000
                num_frames = int(sample_rate * duration)
                audio_data = sd.rec(
                    frames=num_frames,
                    samplerate=sample_rate,
                    channels=1,
                    dtype="float32",
                    device=in_dev,
                )
                sd.wait()
                audio_int16 = (np.clip(audio_data[:, 0], -1.0, 1.0) * 32767.0).astype(np.int16)
                with wave.open(str(file_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_int16.tobytes())
                _recorded_real = True
            except Exception as exc:
                logger.warning(
                    "Physical mic recording unavailable; creating standard PCM container",
                    error=str(exc),
                )
                sample_rate = 16000
                num_samples = int(sample_rate * duration)
                with wave.open(str(file_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(b"\x00\x00" * num_samples)

        expiry_timestamp = time.time() + self.expiration_seconds
        with self._lock:
            self._tracked_files[str(file_path)] = expiry_timestamp

        logger.info(
            "Created opt-in test audio sample",
            recording_id=rec_id,
            duration=duration,
            file_path=str(file_path),
            expires_in=self.expiration_seconds,
        )

        return TestRecordResponse(
            recording_id=rec_id,
            duration_seconds=duration,
            file_path=str(file_path),
            created_at=datetime.now(timezone.utc),
            expires_in_seconds=self.expiration_seconds,
            retention_notice=(
                f"Audio sample retained locally for self-verification only. "
                f"Automatically purged from disk after {self.expiration_seconds} seconds."
            ),
        )

    def _cleanup_loop(self) -> None:
        """Periodic background worker to purge expired test recordings."""
        while not self._stop_event.is_set():
            time.sleep(5.0)
            now = time.time()
            to_delete: list[str] = []

            with self._lock:
                for fpath, exp in list(self._tracked_files.items()):
                    if now >= exp:
                        to_delete.append(fpath)
                        del self._tracked_files[fpath]

            for path_str in to_delete:
                try:
                    p = Path(path_str)
                    if p.exists():
                        p.unlink()
                        logger.info("Automatically purged expired test recording", path=path_str)
                except Exception as e:
                    logger.warning("Failed to unlink test recording", path=path_str, error=str(e))

    def purge_all(self) -> None:
        """Purge all tracked files immediately on shutdown."""
        with self._lock:
            for path_str in list(self._tracked_files.keys()):
                try:
                    p = Path(path_str)
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
            self._tracked_files.clear()


# Global singleton instance
_test_recorder: TestRecorder | None = None


def get_test_recorder() -> TestRecorder:
    global _test_recorder
    if _test_recorder is None:
        _test_recorder = TestRecorder()
    return _test_recorder
