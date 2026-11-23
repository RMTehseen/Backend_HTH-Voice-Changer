"""Offline conversion adapters package."""

from hth.adapters.offline.contracts import OfflineConverterProtocol
from hth.adapters.offline.fake_converter import FakeOfflineConverter
from hth.adapters.offline.ffmpeg_probe import FfmpegCapabilities, probe_ffmpeg
from hth.adapters.offline.model_validator import validate_model_engine_compatibility
from hth.adapters.offline.production_converter import ProductionOfflineVoiceConverter

__all__ = [
    "OfflineConverterProtocol",
    "FakeOfflineConverter",
    "ProductionOfflineVoiceConverter",
    "FfmpegCapabilities",
    "probe_ffmpeg",
    "validate_model_engine_compatibility",
]
