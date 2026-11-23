"""Model format validation and engine isolation guards.

Enforces:
Do not assume Applio inference is compatible with Beatrice models or vice versa.
Provider adapter must reject wrong model format.
"""

from __future__ import annotations

from hth.domain.api_contracts import SupportedModelFormat
from hth.domain.errors import IncompatibleModelError

ENGINE_ALLOWED_FORMATS: dict[str, set[SupportedModelFormat]] = {
    "beatrice": {SupportedModelFormat.BEATRICE_PARAPHERNALIA},
    "applio": {SupportedModelFormat.APPLIO_RVC},
    "fake": {SupportedModelFormat.FAKE_MODEL},
    "production": {SupportedModelFormat.ACOUSTIC_PROFILE, SupportedModelFormat.FAKE_MODEL},
    "acoustic": {SupportedModelFormat.ACOUSTIC_PROFILE, SupportedModelFormat.FAKE_MODEL},
}


def validate_model_engine_compatibility(
    engine_id: str,
    model_id: str,
    model_format: SupportedModelFormat,
) -> None:
    """Validate that the requested voice model matches the engine adapter's native format.

    Raises IncompatibleModelError if there is any mismatch between engine and model format.
    """
    engine_key = engine_id.lower().strip()
    allowed_formats = ENGINE_ALLOWED_FORMATS.get(engine_key)

    if allowed_formats is None:
        raise IncompatibleModelError(
            f"Unknown engine adapter '{engine_id}'. Cannot determine model compatibility.",
            model_id=model_id,
            actual_format=model_format.value,
        )

    if model_format not in allowed_formats:
        expected_str = ", ".join(f.value for f in allowed_formats)
        raise IncompatibleModelError(
            f"Model format '{model_format.value}' is incompatible with engine '{engine_id}'. "
            f"Engine '{engine_id}' strictly requires [{expected_str}] models. "
            f"Cross-engine model conversions are strictly prohibited.",
            model_id=model_id,
            expected_format=expected_str,
            actual_format=model_format.value,
        )
