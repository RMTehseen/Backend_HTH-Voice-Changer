"""Beatrice V2 static and runtime capabilities definition."""

from hth.domain.contracts import EngineCapabilities

# Beatrice V2 engine capability constraints
BEATRICE_CAPABILITIES = EngineCapabilities(
    supports_pitch_shift=True,
    supports_formant_shift=True,
    supports_noise_gate=True,
    supports_atomic_model_switching=False,  # Beatrice VST requires context reload
    commercial_distribution_allowed=False,  # PERSONAL_POC_ONLY
    is_poc_only=True,
    pitch_range=(-12.0, 12.0),
    formant_range=(-4.0, 4.0),
    default_sample_rate=16000,
    supported_sample_rates=(16000,),
    default_block_size=160,
)
