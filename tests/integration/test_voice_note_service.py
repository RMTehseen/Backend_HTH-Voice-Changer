"""Integration tests for VoiceNoteService (Phase 9)."""

import asyncio
import io
import struct
import wave

import pytest

from hth.adapters.fake_engine import FakeEngine
from hth.adapters.offline.fake_converter import FakeOfflineConverter
from hth.core.settings import Settings
from hth.domain.api_contracts import (
    AudioContainerFormat,
    EngineState,
    LiveConfig,
    SupportedModelFormat,
    VoiceNoteJobCreate,
    VoiceNoteJobStatusEnum,
)
from hth.domain.errors import IncompatibleModelError, ResourceConflictError
from hth.services.engine_service import EngineService
from hth.services.voice_note_service import VoiceNoteService


def _create_test_wav(duration_sec: float = 0.5, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        nframes = int(sample_rate * duration_sec)
        samples = [1000 if i % 2 == 0 else -1000 for i in range(nframes)]
        wf.writeframes(struct.pack(f"<{nframes}h", *samples))
    return buf.getvalue()


@pytest.fixture
def voice_note_env(tmp_path):
    engine = FakeEngine()
    engine_svc = EngineService(engine)
    settings = Settings(
        voice_notes_dir=str(tmp_path / "voice_notes"),
        max_file_size_mb=25,
        max_duration_seconds=300,
    )
    converter = FakeOfflineConverter()
    vn_svc = VoiceNoteService(engine_svc, converter=converter, settings=settings)
    engine_svc.set_offline_active_check(vn_svc.is_offline_active)
    return vn_svc, engine_svc, engine, tmp_path


@pytest.mark.asyncio
async def test_get_live_guide(voice_note_env):
    vn_svc, engine_svc, engine, _ = voice_note_env
    guide = await vn_svc.get_live_guide()
    assert guide.recommended_whatsapp_input == "CABLE Output (VB-Audio Virtual Cable)"
    assert len(guide.steps) == 6
    assert "voice-note bubble" in guide.whatsapp_bubble_notice
    assert "file attachments/documents" in guide.attachment_warning


@pytest.mark.asyncio
async def test_get_capabilities(voice_note_env):
    vn_svc, _, _, _ = voice_note_env
    caps = await vn_svc.get_capabilities()
    assert "wav" in caps.supported_output_formats
    assert "wav" in caps.supported_input_formats
    assert caps.max_file_size_mb == 25
    assert caps.max_duration_seconds == 300
    assert not caps.offline_job_active


@pytest.mark.asyncio
async def test_offline_conversion_chunked_success(voice_note_env):
    vn_svc, _, _, _ = voice_note_env
    wav_bytes = _create_test_wav(duration_sec=0.2)

    params = VoiceNoteJobCreate(
        target_format=AudioContainerFormat.WAV,
        model_id="test_mock",
        model_format=SupportedModelFormat.FAKE_MODEL,
        pitch_shift=2.0,
    )

    job = await vn_svc.submit_job(wav_bytes, "sample.wav", params)
    assert job.status in (VoiceNoteJobStatusEnum.PENDING, VoiceNoteJobStatusEnum.PROCESSING)
    assert job.job_id is not None

    # Wait for background worker to complete
    for _ in range(50):
        await asyncio.sleep(0.05)
        status = vn_svc.get_job_status(job.job_id)
        if status.status == VoiceNoteJobStatusEnum.COMPLETED:
            break

    final_status = vn_svc.get_job_status(job.job_id)
    assert final_status.status == VoiceNoteJobStatusEnum.COMPLETED
    assert final_status.progress == 1.0
    assert final_status.has_converted_audio

    # Verify A/B files exist on disk
    src_path = vn_svc.get_source_audio_path(job.job_id)
    out_path = vn_svc.get_converted_audio_path(job.job_id)
    assert src_path.is_file()
    assert out_path.is_file()

    # Validate output is valid WAV
    with wave.open(str(out_path), "rb") as out_wf:
        assert out_wf.getframerate() == 16000
        assert out_wf.getnchannels() == 1


@pytest.mark.asyncio
async def test_rejects_incompatible_model(voice_note_env):
    vn_svc, _, _, _ = voice_note_env
    wav_bytes = _create_test_wav(duration_sec=0.2)

    params = VoiceNoteJobCreate(
        target_format=AudioContainerFormat.WAV,
        model_id="rvc_weights",
        model_format=SupportedModelFormat.APPLIO_RVC,  # Incompatible with Fake converter
    )

    with pytest.raises(IncompatibleModelError):
        await vn_svc.submit_job(wav_bytes, "sample.wav", params)


@pytest.mark.asyncio
async def test_cooperative_cancellation(voice_note_env):
    vn_svc, _, _, tmp_path = voice_note_env
    wav_bytes = _create_test_wav(duration_sec=2.0)

    params = VoiceNoteJobCreate(
        target_format=AudioContainerFormat.WAV,
        model_id="test_mock",
        model_format=SupportedModelFormat.FAKE_MODEL,
    )

    job = await vn_svc.submit_job(wav_bytes, "long_sample.wav", params)
    # Cancel immediately
    cancelled_job = await vn_svc.cancel_job(job.job_id)
    assert cancelled_job.status == VoiceNoteJobStatusEnum.CANCELLED

    # Check that temp directory for this job was cleaned up immediately
    job_temp = tmp_path / "voice_notes" / "temp" / job.job_id
    assert not job_temp.exists()


@pytest.mark.asyncio
async def test_resource_conflict_live_engine_running(voice_note_env):
    vn_svc, engine_svc, _, _ = voice_note_env

    # Start live engine
    await engine_svc.start_conversion(LiveConfig())
    health = await engine_svc.get_health()
    assert health.state == EngineState.RUNNING

    wav_bytes = _create_test_wav(duration_sec=0.2)
    params = VoiceNoteJobCreate(
        target_format=AudioContainerFormat.WAV,
        model_id="test_mock",
        model_format=SupportedModelFormat.FAKE_MODEL,
    )

    # Offline conversion MUST be blocked while live engine is running
    with pytest.raises(ResourceConflictError, match="Cannot start offline conversion while live"):
        await vn_svc.submit_job(wav_bytes, "sample.wav", params)

    await engine_svc.stop_conversion()


@pytest.mark.asyncio
async def test_resource_conflict_start_live_while_offline_active(voice_note_env):
    vn_svc, engine_svc, _, _ = voice_note_env
    wav_bytes = _create_test_wav(duration_sec=3.0)

    params = VoiceNoteJobCreate(
        target_format=AudioContainerFormat.WAV,
        model_id="test_mock",
        model_format=SupportedModelFormat.FAKE_MODEL,
    )

    job = await vn_svc.submit_job(wav_bytes, "long.wav", params)
    assert vn_svc.is_offline_active()

    # Starting live conversion MUST be blocked while offline job is processing
    with pytest.raises(
        ResourceConflictError, match="Cannot start live conversion while an offline"
    ):
        await engine_svc.start_conversion(LiveConfig())

    await vn_svc.cancel_job(job.job_id)


@pytest.mark.asyncio
async def test_retention_cleanup(voice_note_env):
    vn_svc, _, _, tmp_path = voice_note_env
    res = vn_svc.cleanup_retention(retention_hours=0)
    assert isinstance(res.deleted_temp_files, int)
    assert isinstance(res.deleted_export_files, int)


@pytest.mark.asyncio
async def test_concurrent_offline_jobs_rejected(voice_note_env):
    vn_svc, _, _, _ = voice_note_env
    wav_bytes = _create_test_wav(duration_sec=3.0)

    params = VoiceNoteJobCreate(
        target_format=AudioContainerFormat.WAV,
        model_id="test_mock",
        model_format=SupportedModelFormat.FAKE_MODEL,
    )

    job1 = await vn_svc.submit_job(wav_bytes, "job1.wav", params)
    assert vn_svc.is_offline_active()

    # Second offline submission must be rejected with ResourceConflictError
    with pytest.raises(ResourceConflictError, match="Another offline conversion job"):
        await vn_svc.submit_job(wav_bytes, "job2.wav", params)

    await vn_svc.cancel_job(job1.job_id)


class FailingConverter(FakeOfflineConverter):
    async def convert(self, *args, **kwargs):
        raise RuntimeError("Engine DSP synthesis crash during conversion")


@pytest.mark.asyncio
async def test_engine_failure_handling(voice_note_env):
    vn_svc, _, _, tmp_path = voice_note_env
    vn_svc._converter = FailingConverter()

    wav_bytes = _create_test_wav(duration_sec=0.2)
    params = VoiceNoteJobCreate(
        target_format=AudioContainerFormat.WAV,
        model_id="test_mock",
        model_format=SupportedModelFormat.FAKE_MODEL,
    )

    job = await vn_svc.submit_job(wav_bytes, "crash.wav", params)

    for _ in range(50):
        await asyncio.sleep(0.05)
        st = vn_svc.get_job_status(job.job_id)
        if st.status == VoiceNoteJobStatusEnum.FAILED:
            break

    final_status = vn_svc.get_job_status(job.job_id)
    assert final_status.status == VoiceNoteJobStatusEnum.FAILED
    assert "Engine DSP synthesis crash" in (final_status.error or "")

    # Ensure intermediate temporary files were purged
    job_temp = tmp_path / "voice_notes" / "temp" / job.job_id
    assert not job_temp.exists()
