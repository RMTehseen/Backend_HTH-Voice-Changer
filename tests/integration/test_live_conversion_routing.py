"""Integration tests for Live Voice Conversion and VB-CABLE WhatsApp Routing."""

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from hth.audio.conversion.contracts import VoiceProfile
from hth.audio.pipeline import AudioPipeline, AudioPipelineConfig
from hth.core.app import create_app
from hth.core.settings import get_settings
from hth.domain.errors import AudioProcessingError

_HEADERS = {
    "X-HTH-Token": get_settings().session_token,
    "Origin": "http://127.0.0.1:5173",
    "Host": "127.0.0.1:8765",
}


@pytest.fixture
def test_pipeline() -> AudioPipeline:
    return AudioPipeline()


def test_male_to_female_pipeline_transformation(test_pipeline: AudioPipeline):
    """Verify Male-to-Female profile shifts pitch up and warps formants."""
    test_pipeline.set_conversion_profile(VoiceProfile.MALE_TO_FEMALE)
    assert test_pipeline.converter.config.pitch_semitones == 10.0
    assert test_pipeline.converter.config.formant_ratio == 1.18

    in_frame = np.random.uniform(-0.3, 0.3, 480).astype(np.float32)
    out_frame = np.zeros(480, dtype=np.float32)

    test_pipeline.converter.process_frame(in_frame, out_frame)
    assert not np.isnan(out_frame).any()
    assert (np.abs(out_frame) <= 1.0).all()


def test_female_to_male_pipeline_transformation(test_pipeline: AudioPipeline):
    """Verify Female-to-Male profile shifts pitch down and warps formants."""
    test_pipeline.set_conversion_profile(VoiceProfile.FEMALE_TO_MALE)
    assert test_pipeline.converter.config.pitch_semitones == -10.0
    assert test_pipeline.converter.config.formant_ratio == 0.85

    in_frame = np.random.uniform(-0.3, 0.3, 480).astype(np.float32)
    out_frame = np.zeros(480, dtype=np.float32)

    test_pipeline.converter.process_frame(in_frame, out_frame)
    assert not np.isnan(out_frame).any()
    assert (np.abs(out_frame) <= 1.0).all()


def test_vb_cable_security_guard_rejects_unconverted_mic(test_pipeline: AudioPipeline):
    """Rule 6 & 7: VB-CABLE output is strictly prohibited if conversion is not active."""
    cfg = AudioPipelineConfig(
        enable_vb_cable=True,
        enable_voice_conversion=False,  # Security violation!
    )
    with pytest.raises(AudioProcessingError) as exc_info:
        test_pipeline.start(cfg)

    assert "Security violation" in str(exc_info.value)
    assert "VB-CABLE" in str(exc_info.value)


def test_vb_cable_allowed_when_conversion_active(test_pipeline: AudioPipeline):
    """VB-CABLE transmission is permitted only when real voice conversion is verified."""
    cfg = AudioPipelineConfig(
        enable_vb_cable=True,
        enable_voice_conversion=True,
        conversion_profile=VoiceProfile.MALE_TO_FEMALE,
    )
    # In headless/test mode, stream starts cleanly
    test_pipeline.start(cfg)
    m = test_pipeline.get_metrics()
    assert m.is_running is True
    assert m.is_converting is True
    assert m.live_call_mode is True
    test_pipeline.stop()


def test_panic_mute_zeroes_live_output(test_pipeline: AudioPipeline):
    """Panic mute immediately engages fail-closed silence on live streams."""
    cfg = AudioPipelineConfig(
        enable_vb_cable=True,
        enable_voice_conversion=True,
    )
    test_pipeline.start(cfg)
    assert test_pipeline.get_metrics().is_muted is False

    test_pipeline.panic_mute()
    m = test_pipeline.get_metrics()
    assert m.is_muted is True
    test_pipeline.stop()


@pytest.mark.asyncio
async def test_live_route_guide_api():
    """Verify WhatsApp live routing guidance endpoint provides accurate instructions."""
    app = create_app()
    transport = ASGITransport(app=app)
    headers = {
        "X-HTH-Token": get_settings().session_token,
        "Origin": "http://127.0.0.1:5173",
        "Host": "127.0.0.1:8765",
    }
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
        resp = await client.get("/api/v1/voice-notes/live-guide", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "CABLE Output" in data["recommended_whatsapp_input"]
        assert len(data["steps"]) >= 5
        assert "WhatsApp" in data["whatsapp_bubble_notice"]
        assert "attachment" in data["attachment_warning"].lower()
