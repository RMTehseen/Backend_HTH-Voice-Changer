"""Unit tests verifying Beatrice static capabilities and parameter constraints."""

import pytest

from hth.adapters.beatrice.adapter import (
    COMMERCIAL_DISTRIBUTION_ENABLED,
    PERSONAL_POC_ONLY,
    BeatriceAdapter,
)
from hth.adapters.beatrice.capabilities import BEATRICE_CAPABILITIES
from hth.domain.contracts import EngineParams
from hth.domain.errors import CommandError


def test_beatrice_poc_markers_and_licensing() -> None:
    """Verifies that Beatrice adapter is hardcoded as personal POC only."""
    assert PERSONAL_POC_ONLY is True
    assert COMMERCIAL_DISTRIBUTION_ENABLED is False

    caps = BEATRICE_CAPABILITIES
    assert caps.is_poc_only is True
    assert caps.commercial_distribution_allowed is False
    assert (
        caps.supports_atomic_model_switching is False
    )  # Cannot hot-swap models without context recreation


def test_adapter_parameter_validation_pitch_bounds() -> None:
    """Verifies that pitch shift outside [-12, +12] semitones is rejected."""
    adapter = BeatriceAdapter()

    # Valid pitch
    adapter._validate_params(EngineParams(pitch_shift=0.0))
    adapter._validate_params(EngineParams(pitch_shift=12.0))
    adapter._validate_params(EngineParams(pitch_shift=-12.0))

    # Invalid pitch > 12
    with pytest.raises(CommandError, match="Pitch shift 15.0 out of supported range"):
        adapter._validate_params(EngineParams(pitch_shift=15.0))

    # Invalid pitch < -12
    with pytest.raises(CommandError, match="Pitch shift -13.0 out of supported range"):
        adapter._validate_params(EngineParams(pitch_shift=-13.0))


def test_adapter_parameter_validation_formant_bounds() -> None:
    """Verifies that formant shift outside [-4, +4] is rejected."""
    adapter = BeatriceAdapter()

    # Valid formant
    adapter._validate_params(EngineParams(formant_shift=4.0))
    adapter._validate_params(EngineParams(formant_shift=-4.0))

    # Invalid formant > 4
    with pytest.raises(CommandError, match="Formant shift 5.0 out of supported range"):
        adapter._validate_params(EngineParams(formant_shift=5.0))
