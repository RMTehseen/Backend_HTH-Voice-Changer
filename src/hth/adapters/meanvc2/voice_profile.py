"""MeanVC2 voice profile management."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from hth.core.logging import get_logger

logger = get_logger("hth.adapters.meanvc2.profiles")


@dataclass
class VoiceProfile:
    """A MeanVC2 target speaker voice profile."""
    profile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    display_name: str = ""
    reference_wav_path: str = ""
    reference_sha256: str = ""
    reference_duration_s: float = 0.0
    reference_sample_rate: int = 16000
    # Serialized speaker embedding (base64 or file path)
    embedding_cache_path: str = ""
    # Which model variant this embedding was computed for
    model_variant: str = "40ms"
    # MeanVC2 checkpoint version
    model_checkpoint: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # Visual and categorization metadata
    avatar_id: str = "female_eva"
    category: str = "Female"
    badge: str = "AI"
    # Voice tuning parameters
    pitch_shift: float = 0.0
    timbre_ratio: float = 1.0
    volume_gain: float = 1.0
    sound_effect: str = "none"
    favorite: bool = False
    # Consent/ownership declaration — required by Rule 4
    consent_declared: bool = False
    consent_statement: str = ""
    language_hint: str = ""
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VoiceProfile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class VoiceProfileStore:
    r"""JSON-file backed voice profile store under %LOCALAPPDATA%\HTH\MeanVC2Profiles."""

    def __init__(self, profile_dir: str):
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.profile_dir / "profiles.json"
        self._profiles: dict[str, VoiceProfile] = {}
        self._load_index()

    def _load_index(self) -> None:
        if self._index_path.exists():
            try:
                with open(self._index_path, encoding="utf-8") as f:
                    data = json.load(f)
                self._profiles = {
                    pid: VoiceProfile.from_dict(pd)
                    for pid, pd in data.items()
                }
            except Exception as e:
                logger.warning("Failed to load profile index", error=str(e))

    def _save_index(self) -> None:
        tmp = self._index_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({pid: p.to_dict() for pid, p in self._profiles.items()}, f, indent=2)
        tmp.replace(self._index_path)

    def save(self, profile: VoiceProfile) -> None:
        self._profiles[profile.profile_id] = profile
        self._save_index()
        logger.info("Voice profile saved", profile_id=profile.profile_id, name=profile.display_name)

    def get(self, profile_id: str) -> Optional[VoiceProfile]:
        return self._profiles.get(profile_id)

    def list_profiles(self) -> list[VoiceProfile]:
        return list(self._profiles.values())

    def delete(self, profile_id: str) -> bool:
        if profile_id not in self._profiles:
            return False
        profile = self._profiles.pop(profile_id)
        # Clean up embedding cache
        if profile.embedding_cache_path and Path(profile.embedding_cache_path).exists():
            try:
                Path(profile.embedding_cache_path).unlink()
            except Exception:
                pass
        self._save_index()
        logger.info("Voice profile deleted", profile_id=profile_id)
        return True

    def get_embedding_path(self, profile_id: str) -> str:
        return str(self.profile_dir / f"emb_{profile_id}.pt")


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_reference_audio(wav_path: str) -> dict:
    """Validate reference audio file. Returns dict with keys: valid, error, duration_s, sample_rate."""
    try:
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(wav_path)
        if data.ndim == 2:
            data = np.mean(data, axis=1)
        duration = len(data) / sr
        if duration < 1.0:
            return {"valid": False, "error": "Reference audio too short (< 1 second)", "duration_s": duration, "sample_rate": sr}
        if duration > 60.0:
            return {"valid": False, "error": "Reference audio too long (> 60 seconds)", "duration_s": duration, "sample_rate": sr}
        # Clipping check
        peak = float(np.max(np.abs(data)))
        if peak > 0.99:
            return {"valid": False, "error": f"Reference audio is clipped (peak={peak:.3f})", "duration_s": duration, "sample_rate": sr}
        # Silence check
        rms = float(np.sqrt(np.mean(data ** 2)))
        if rms < 0.001:
            return {"valid": False, "error": "Reference audio appears to be silence", "duration_s": duration, "sample_rate": sr}
        return {"valid": True, "error": None, "duration_s": duration, "sample_rate": sr, "peak": peak, "rms": rms}
    except Exception as e:
        return {"valid": False, "error": str(e), "duration_s": 0, "sample_rate": 0}
