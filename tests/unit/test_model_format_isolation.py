"""Unit tests for model format isolation and cross-engine rejection (Phase 9).

Enforces:
Do not assume Applio inference is compatible with Beatrice models or vice versa.
Provider adapter must reject wrong model format.
"""

import pytest

from hth.adapters.offline.model_validator import validate_model_engine_compatibility
from hth.domain.api_contracts import SupportedModelFormat
from hth.domain.errors import IncompatibleModelError


def test_applio_rejects_beatrice_model():
    """Verify that Applio adapter strictly rejects Beatrice paraphernalia models."""
    with pytest.raises(IncompatibleModelError) as exc_info:
        validate_model_engine_compatibility(
            engine_id="applio",
            model_id="beatrice_jvs_model",
            model_format=SupportedModelFormat.BEATRICE_PARAPHERNALIA,
        )
    assert "incompatible with engine 'applio'" in str(exc_info.value)
    assert exc_info.value.actual_format == SupportedModelFormat.BEATRICE_PARAPHERNALIA.value


def test_beatrice_rejects_applio_model():
    """Verify that Beatrice adapter strictly rejects Applio RVC models (.pth)."""
    with pytest.raises(IncompatibleModelError) as exc_info:
        validate_model_engine_compatibility(
            engine_id="beatrice",
            model_id="rvc_voice_weights",
            model_format=SupportedModelFormat.APPLIO_RVC,
        )
    assert "incompatible with engine 'beatrice'" in str(exc_info.value)
    assert exc_info.value.actual_format == SupportedModelFormat.APPLIO_RVC.value


def test_fake_engine_isolation():
    """Verify that Fake engine accepts only fake_model format."""
    validate_model_engine_compatibility(
        engine_id="fake",
        model_id="mock_test_model",
        model_format=SupportedModelFormat.FAKE_MODEL,
    )

    with pytest.raises(IncompatibleModelError):
        validate_model_engine_compatibility(
            engine_id="fake",
            model_id="unauthorized_rvc",
            model_format=SupportedModelFormat.APPLIO_RVC,
        )


def test_unknown_engine_rejected():
    """Verify that an unknown engine id raises IncompatibleModelError."""
    with pytest.raises(IncompatibleModelError) as exc_info:
        validate_model_engine_compatibility(
            engine_id="quantum_engine",
            model_id="any_model",
            model_format=SupportedModelFormat.FAKE_MODEL,
        )
    assert "Unknown engine adapter" in str(exc_info.value)
