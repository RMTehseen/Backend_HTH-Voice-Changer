"""Unit tests for TestRecorder 5-second opt-in sample recording and auto-purge."""

import time
import wave
from pathlib import Path

from hth.audio.recorder import TestRecorder
from hth.domain.audio_contracts import TestRecordRequest


def test_record_test_sample_creates_valid_wav() -> None:
    recorder = TestRecorder(expiration_seconds=2)
    req = TestRecordRequest(duration_seconds=5.0)

    resp = recorder.record_test_sample(req)
    assert resp.recording_id
    assert resp.duration_seconds == 5.0
    assert resp.expires_in_seconds == 2
    assert "retained locally" in resp.retention_notice

    fpath = Path(resp.file_path)
    assert fpath.exists()

    # Validate WAV file properties
    with wave.open(str(fpath), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        # 16000 * 5 = 80000 frames
        assert wf.getnframes() == 80000

    # Clean up
    recorder.purge_all()
    assert not fpath.exists()


def test_recorder_auto_cleanup_loop() -> None:
    # Set 1 second TTL
    recorder = TestRecorder(expiration_seconds=1)
    req = TestRecordRequest(duration_seconds=1.0)
    resp = recorder.record_test_sample(req)
    fpath = Path(resp.file_path)
    assert fpath.exists()

    # Wait for expiry and cleanup loop trigger
    time.sleep(6.0)
    assert not fpath.exists()

    recorder.purge_all()
