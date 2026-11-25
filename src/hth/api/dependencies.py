"""FastAPI dependency providers for ModelService and SwitchingService."""

from __future__ import annotations

from typing import Any

from hth.adapters.offline.contracts import OfflineConverterProtocol
from hth.core.database import get_session_factory
from hth.core.settings import get_settings
from hth.domain.api_contracts import (
    DeviceInfo,
    DeviceType,
    EngineInfo,
    EngineState,
    LiveConfig,
    LiveStatus,
    ModelManifest,
)
from hth.services.engine_service import EngineService
from hth.services.model_service import ModelService
from hth.services.switching_service import SwitchingService
from hth.services.voice_note_service import VoiceNoteService


class BeatriceVoiceEngineBridge:
    """Bridges research-sandbox BeatriceAdapter to FastAPI VoiceEngineProtocol."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def ping(self) -> bool:
        await self._inner.ping()
        return True

    async def list_devices(self) -> list[DeviceInfo]:
        devs = await self._inner.list_devices()
        return [
            DeviceInfo(
                id=str(d.device_id),
                friendly_name=d.name,
                device_type=DeviceType.INPUT if d.is_input else DeviceType.OUTPUT,
                sample_rates=d.supported_sample_rates,
                default_sample_rate=d.default_sample_rate,
                is_default=d.is_default,
            )
            for d in devs
        ]

    async def start(self, config: LiveConfig) -> None:
        from hth.domain.contracts import LiveConfig as C_LiveConfig

        c = C_LiveConfig(
            input_device_id=config.input_device_id,
            output_device_id=config.output_device_id,
            sample_rate=config.sample_rate,
            block_size=config.buffer_size,
        )
        await self._inner.start(c)

    async def stop(self) -> None:
        await self._inner.stop()

    async def set_passthrough(self, enabled: bool) -> None:
        await self._inner.set_passthrough(enabled)

    async def set_params(self, params: dict[str, Any]) -> None:
        from hth.domain.contracts import EngineParams as C_Params

        p = C_Params(
            pitch_shift=float(params.get("pitch_shift", 0.0)),
            formant_shift=float(params.get("formant_shift", 0.0)),
        )
        await self._inner.set_params(p)

    async def select_model(self, manifest: ModelManifest) -> None:
        await self._inner.select_model(manifest.model_id)

    async def panic_mute(self) -> None:
        if hasattr(self._inner, "panic_mute"):
            await self._inner.panic_mute()

    async def un_mute(self) -> None:
        if hasattr(self._inner, "un_mute"):
            await self._inner.un_mute()

    async def status(self) -> LiveStatus:
        s = await self._inner.status()
        return LiveStatus(
            state=EngineState(s.state.value if hasattr(s.state, "value") else str(s.state)),
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            underruns=s.metrics.underruns,
            frames_processed=0,
        )

    async def health(self) -> EngineInfo:
        h = await self._inner.health()
        return EngineInfo(
            engine_id="beatrice-v2-poc",
            name="Beatrice V2 (Personal POC Only)",
            version="2.0.0-rc0",
            capabilities=["pitch_shift", "formant_shift", "natural_passthrough"],
            state=EngineState.RUNNING if h.process_alive else EngineState.IDLE,
        )


_engine_service: EngineService | None = None
_model_service: ModelService | None = None
_switching_service: SwitchingService | None = None
_voice_note_service: VoiceNoteService | None = None


def init_engine_service(adapter: str = "dsp") -> None:
    """Initialise the singleton EngineService. Called once from lifespan."""
    global _engine_service
    if adapter in ("dsp", "pipeline"):
        from hth.adapters.pipeline_adapter import AudioPipelineAdapter

        _engine_service = EngineService(AudioPipelineAdapter())
    elif adapter == "fake":
        from hth.adapters.fake_engine import FakeEngine

        _engine_service = EngineService(FakeEngine())
    elif adapter == "beatrice":
        settings = get_settings()
        if settings.commercial_build:
            from hth.domain.errors import LicensingPolicyViolation

            raise LicensingPolicyViolation(
                "COMMERCIAL ISOLATION ENFORCED: Beatrice V2 is closed-source and restricted to "
                "personal non-commercial research (contributors.txt:60). It cannot be loaded "
                "in commercial release builds (Rule L-01)."
            )
        try:
            from hth.adapters.beatrice.adapter import BeatriceAdapter

            _engine_service = EngineService(BeatriceVoiceEngineBridge(BeatriceAdapter()))
        except ImportError:
            raise RuntimeError(
                "Beatrice engine adapter is quarantined and excluded from this distribution."
            )
    elif adapter == "meanvc2":
        import os

        from hth.adapters.meanvc2.adapter import MeanVC2VoiceAdapter
        from hth.adapters.meanvc2.config import MeanVC2Config
        cfg = MeanVC2Config(
            device=os.environ.get("HTH_MEANVC2_DEVICE", "cpu"),
            model_variant=os.environ.get("HTH_MEANVC2_VARIANT", "40ms"),
        )
        _engine_service = EngineService(MeanVC2VoiceAdapter(cfg))
    else:
        raise ValueError(f"Unknown engine adapter: {adapter!r}")


def get_engine_service() -> EngineService:
    """FastAPI dependency: returns the singleton EngineService."""
    global _engine_service
    if _engine_service is None:
        settings = get_settings()
        init_engine_service(settings.engine_adapter)
    assert _engine_service is not None
    return _engine_service


def get_model_service() -> ModelService:
    """FastAPI dependency: returns the singleton ModelService."""
    global _model_service
    if _model_service is None:
        factory = get_session_factory()
        _model_service = ModelService(factory)
    return _model_service


def get_switching_service() -> SwitchingService:
    """FastAPI dependency: returns the singleton SwitchingService."""
    global _switching_service
    if _switching_service is None:
        engine_svc = get_engine_service()
        model_svc = get_model_service()
        factory = get_session_factory()
        _switching_service = SwitchingService(model_svc, engine_svc, factory)
    return _switching_service


def init_voice_note_service(converter_type: str = "production") -> None:
    """Initialise the singleton VoiceNoteService. Called from lifespan."""
    global _voice_note_service
    engine_svc = get_engine_service()

    converter: OfflineConverterProtocol
    if converter_type == "fake":
        from hth.adapters.offline.fake_converter import FakeOfflineConverter

        converter = FakeOfflineConverter()
    else:
        from hth.adapters.offline.production_converter import ProductionOfflineVoiceConverter

        converter = ProductionOfflineVoiceConverter()

    _voice_note_service = VoiceNoteService(engine_svc, converter=converter)
    engine_svc.set_offline_active_check(_voice_note_service.is_offline_active)


def get_voice_note_service() -> VoiceNoteService:
    """FastAPI dependency: returns the singleton VoiceNoteService."""
    global _voice_note_service
    if _voice_note_service is None:
        init_voice_note_service()
    assert _voice_note_service is not None
    return _voice_note_service
