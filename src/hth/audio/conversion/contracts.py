"""Domain contracts and configuration types for real voice conversion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class VoiceProfile(str, Enum):
    """Target voice transformation profiles."""

    MALE_TO_FEMALE = "male_to_female"
    FEMALE_TO_MALE = "female_to_male"
    CUSTOM = "custom"


@dataclass(frozen=True)
class VoiceConversionConfig:
    """Acoustic and spectral parameters for voice conversion."""

    profile: VoiceProfile = VoiceProfile.MALE_TO_FEMALE
    pitch_semitones: float = 10.0  # +10 semitones for male->female, -10 for female->male
    formant_ratio: float = 1.18  # Vocal tract shortening (~1.18) or lengthening (~0.85)
    chest_resonance_gain: float = 0.8  # Lower formant emphasis (150-300 Hz)
    brightness_gain: float = 1.2  # Higher harmonic emphasis (2.5-5 kHz)
    f0_min: float = 75.0  # Lowest expected pitch in Hz
    f0_max: float = 400.0  # Highest expected pitch in Hz
    sample_rate: int = 48000
    frame_size: int = 480  # 10 ms @ 48 kHz

    @classmethod
    def create_male_to_female(cls, sample_rate: int = 48000) -> VoiceConversionConfig:
        """Optimized natural male-to-female acoustic profile."""
        return cls(
            profile=VoiceProfile.MALE_TO_FEMALE,
            pitch_semitones=10.0,
            formant_ratio=1.18,
            chest_resonance_gain=0.6,
            brightness_gain=1.25,
            f0_min=80.0,
            f0_max=350.0,
            sample_rate=sample_rate,
            frame_size=sample_rate // 100,
        )

    @classmethod
    def create_female_to_male(cls, sample_rate: int = 48000) -> VoiceConversionConfig:
        """Optimized natural female-to-male acoustic profile."""
        return cls(
            profile=VoiceProfile.FEMALE_TO_MALE,
            pitch_semitones=-10.0,
            formant_ratio=0.85,
            chest_resonance_gain=1.4,
            brightness_gain=0.8,
            f0_min=120.0,
            f0_max=450.0,
            sample_rate=sample_rate,
            frame_size=sample_rate // 100,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "pitch_semitones": self.pitch_semitones,
            "formant_ratio": self.formant_ratio,
            "chest_resonance_gain": self.chest_resonance_gain,
            "brightness_gain": self.brightness_gain,
            "f0_min": self.f0_min,
            "f0_max": self.f0_max,
            "sample_rate": self.sample_rate,
            "frame_size": self.frame_size,
        }
