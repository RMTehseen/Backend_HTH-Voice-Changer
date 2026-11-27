"""MeanVC2 voice profile management."""
from __future__ import annotations

import hashlib
import json
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

    def __init__(self, profile_dir: str, auto_seed: bool = False):
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.profile_dir / "profiles.json"
        self._profiles: dict[str, VoiceProfile] = {}
        self._load_index(auto_seed=auto_seed)

    def _load_index(self, auto_seed: bool = False) -> None:
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

        if not self._profiles and auto_seed:
            self._seed_default_profiles()
        elif auto_seed and self._profiles:
            # Merge any newly added curated profiles without overwriting user edits,
            # and attach real reference WAVs from local references/ folder if present.
            refs_dir = self.profile_dir / "references"
            changed = self._merge_missing_curated_profiles()
            if refs_dir.is_dir():
                for pid, profile in self._profiles.items():
                    if not profile.reference_wav_path:
                        cand = refs_dir / f"{pid}.wav"
                        if cand.is_file():
                            profile.reference_wav_path = str(cand)
                            changed = True
            if changed:
                self._save_index()
                logger.info("Updated voice profile index with curated profiles/references")

    def _build_default_profiles(self) -> list[VoiceProfile]:
        """Return the curated list of default HTH voice profiles."""

        def _ref(pid: str) -> str:
            """Return real reference WAV from profile_dir/references if present on disk, else ''."""
            cand = self.profile_dir / "references" / f"{pid}.wav"
            return str(cand) if cand.is_file() else ""

        # Calibrated per-profile vocal shaping (distinct ages, accents, and vocal registers):
        # Young (20s), Soft Mid (25s), Mature (30+), Deep Alto/Baritone (40+), Character.
        return [
            VoiceProfile(
                profile_id="profile-female-eva",
                display_name="Eva",
                avatar_id="female_eva",
                category="Female",
                badge="40+ ALTO",
                pitch_shift=-1.5,
                timbre_ratio=0.91,
                volume_gain=1.0,
                sound_effect="none",
                favorite=True,
                consent_declared=True,
                consent_statement="Consented internal reference voice for HTH Real Voice Changer.",
                language_hint="en-US",
                description="Mature 40+ warm authoritative alto female voice with rich chest resonance.",
                reference_wav_path=_ref("profile-female-eva"),
            ),
            VoiceProfile(
                profile_id="profile-female-ayesha",
                display_name="Ayesha",
                avatar_id="female_ayesha",
                category="Female",
                badge="25s SOFT",
                pitch_shift=1.5,
                timbre_ratio=1.01,
                volume_gain=1.0,
                sound_effect="none",
                favorite=False,
                consent_declared=True,
                consent_statement="Consented internal reference voice for HTH Real Voice Changer.",
                language_hint="ur-PK / en",
                description="Soft, natural mid-20s female voice with smooth conversational articulation.",
                reference_wav_path=_ref("profile-female-ayesha"),
            ),
            VoiceProfile(
                profile_id="profile-female-zara",
                display_name="Zara",
                avatar_id="female_zara",
                category="Female",
                badge="22s PRO",
                pitch_shift=2.5,
                timbre_ratio=1.05,
                volume_gain=1.0,
                sound_effect="none",
                favorite=True,
                consent_declared=True,
                consent_statement="Consented internal reference voice for HTH Real Voice Changer.",
                language_hint="en-US",
                description="Crisp young early-20s expressive female voice with bright modern clarity.",
                reference_wav_path=_ref("profile-female-zara"),
            ),
            VoiceProfile(
                profile_id="profile-male-bilal",
                display_name="Bilal",
                avatar_id="male_bilal",
                category="Male",
                badge="35+ DEEP",
                pitch_shift=-1.5,
                timbre_ratio=0.93,
                volume_gain=1.0,
                sound_effect="none",
                favorite=False,
                consent_declared=True,
                consent_statement="Consented internal reference voice for HTH Real Voice Changer.",
                language_hint="ur-PK / en",
                description="Deep 35+ authoritative baritone male timbre.",
                reference_wav_path=_ref("profile-male-bilal"),
            ),
            VoiceProfile(
                profile_id="profile-female-source",
                display_name="Daniel (Anime)",
                avatar_id="female_source",
                category="Character",
                badge="ANIME",
                pitch_shift=4.5,
                timbre_ratio=1.12,
                volume_gain=1.0,
                sound_effect="none",
                favorite=False,
                consent_declared=True,
                consent_statement="Consented internal reference voice for HTH Real Voice Changer.",
                language_hint="ja / en",
                description="High-spirited anime character style expressive voice.",
                reference_wav_path=_ref("profile-female-source"),
            ),
            VoiceProfile(
                profile_id="profile-character-ratke",
                display_name="Ratke (Cyber)",
                avatar_id="character_ratke",
                category="Character",
                badge="45+ BASS",
                pitch_shift=-3.5,
                timbre_ratio=0.86,
                volume_gain=1.0,
                sound_effect="none",
                favorite=False,
                consent_declared=True,
                consent_statement="Consented internal reference voice for HTH Real Voice Changer.",
                language_hint="en",
                description="Ultra-deep 45+ heavy bass cyber vocal character.",
                reference_wav_path=_ref("profile-character-ratke"),
            ),
            VoiceProfile(
                profile_id="profile-female-ananya",
                display_name="Ananya",
                avatar_id="female_ananya",
                category="Female",
                badge="35+ HINDI",
                pitch_shift=-0.5,
                timbre_ratio=0.95,
                volume_gain=1.0,
                sound_effect="none",
                favorite=True,
                consent_declared=True,
                consent_statement="Consented internal natural reference voice for HTH Voice Changer.",
                language_hint="hi-IN / ur",
                description="Grounded 35+ mature Indian Hindi/Urdu female accent, calm and confident.",
                reference_wav_path=_ref("profile-female-ananya"),
            ),
            VoiceProfile(
                profile_id="profile-female-priya",
                display_name="Priya",
                avatar_id="female_priya",
                category="Female",
                badge="24s DESI",
                pitch_shift=2.0,
                timbre_ratio=1.04,
                volume_gain=1.0,
                sound_effect="none",
                favorite=True,
                consent_declared=True,
                consent_statement="Consented internal natural reference voice for HTH Voice Changer.",
                language_hint="hi-IN / en-IN",
                description="Lively mid-20s Indian accent female voice with crisp articulation.",
                reference_wav_path=_ref("profile-female-priya"),
            ),
            VoiceProfile(
                profile_id="profile-female-sana",
                display_name="Sana",
                avatar_id="female_sana",
                category="Female",
                badge="URDU",
                pitch_shift=1.0,
                timbre_ratio=0.97,
                volume_gain=1.0,
                sound_effect="none",
                favorite=True,
                consent_declared=True,
                consent_statement="Consented internal natural reference voice for HTH Voice Changer.",
                language_hint="ur-PK",
                description="Gentle, airy Pakistani Urdu female accent with soft polite cadence.",
                reference_wav_path=_ref("profile-female-sana"),
            ),
            VoiceProfile(
                profile_id="profile-female-chloe",
                display_name="Chloe",
                avatar_id="female_chloe",
                category="Female",
                badge="MELODIC",
                pitch_shift=1.5,
                timbre_ratio=1.03,
                volume_gain=1.0,
                sound_effect="none",
                favorite=True,
                consent_declared=True,
                consent_statement="Consented internal natural reference voice for HTH Voice Changer.",
                language_hint="en-US",
                description="Melodic late-20s international female voice with smooth timbre.",
                reference_wav_path=_ref("profile-female-chloe"),
            ),
            VoiceProfile(
                profile_id="profile-female-zoya",
                display_name="Zoya (Urdu)",
                avatar_id="female_zoya",
                category="Female",
                badge="20s URDU",
                pitch_shift=3.0,
                timbre_ratio=1.08,
                volume_gain=1.0,
                sound_effect="none",
                favorite=True,
                consent_declared=True,
                consent_statement="Consented internal natural reference voice for HTH Voice Changer.",
                language_hint="ur-PK / hi",
                description="Youthful 20s Pakistani Urdu/Hindi female voice, bright and expressive.",
                reference_wav_path=_ref("profile-female-zoya"),
            ),
            VoiceProfile(
                profile_id="profile-female-mehwish",
                display_name="Mehwish",
                avatar_id="female_mehwish",
                category="Female",
                badge="30+ URDU",
                pitch_shift=0.0,
                timbre_ratio=0.96,
                volume_gain=1.0,
                sound_effect="none",
                favorite=True,
                consent_declared=True,
                consent_statement="Consented internal natural reference voice for HTH Voice Changer.",
                language_hint="ur-PK",
                description="Poised 30+ Pakistani Karachi Urdu female tone with warm relaxed flow.",
                reference_wav_path=_ref("profile-female-mehwish"),
            ),
            VoiceProfile(
                profile_id="profile-female-kavya",
                display_name="Kavya",
                avatar_id="female_kavya",
                category="Female",
                badge="42+ HINDI",
                pitch_shift=-2.0,
                timbre_ratio=0.90,
                volume_gain=1.0,
                sound_effect="none",
                favorite=True,
                consent_declared=True,
                consent_statement="Consented internal natural reference voice for HTH Voice Changer.",
                language_hint="hi-IN / ur",
                description="Rich 40+ mature Indian Hindi/Urdu female voice with grounded warmth.",
                reference_wav_path=_ref("profile-female-kavya"),
            ),
            VoiceProfile(
                profile_id="profile-female-neha",
                display_name="Neha (Desi)",
                avatar_id="female_neha",
                category="Female",
                badge="DESI",
                pitch_shift=0.5,
                timbre_ratio=1.02,
                volume_gain=1.0,
                sound_effect="none",
                favorite=False,
                consent_declared=True,
                consent_statement="Consented internal natural reference voice for HTH Voice Changer.",
                language_hint="hi-IN / ur-PK",
                description="Friendly late-20s South Asian female voice for everyday calls.",
                reference_wav_path=_ref("profile-female-neha"),
            ),
            VoiceProfile(
                profile_id="profile-male-hamza",
                display_name="Hamza",
                avatar_id="male_hamza",
                category="Male",
                badge="22s URDU",
                pitch_shift=1.0,
                timbre_ratio=1.04,
                volume_gain=1.0,
                sound_effect="none",
                favorite=False,
                consent_declared=True,
                consent_statement="Consented internal natural reference voice for HTH Voice Changer.",
                language_hint="ur-PK / hi",
                description="Young early-20s energetic Pakistani/Indian Urdu male voice.",
                reference_wav_path=_ref("profile-male-hamza"),
            ),
        ]

    def _merge_missing_curated_profiles(self) -> bool:
        """Add missing curated profiles & upgrade untouched flat-default profiles to unique vocal presets."""
        changed = False
        for p in self._build_default_profiles():
            existing = self._profiles.get(p.profile_id)
            if existing is None:
                self._profiles[p.profile_id] = p
                changed = True
            elif (
                existing.pitch_shift == 0.0
                and existing.timbre_ratio == 1.0
                and existing.sound_effect == "none"
                and (p.pitch_shift != 0.0 or p.timbre_ratio != 1.0)
            ):
                existing.pitch_shift = p.pitch_shift
                existing.timbre_ratio = p.timbre_ratio
                existing.badge = p.badge
                existing.description = p.description
                changed = True
        return changed

    def _seed_default_profiles(self) -> None:
        """Seed pre-configured curated voice profiles if store is empty (e.g. cloud or new instance)."""
        defaults = self._build_default_profiles()
        for p in defaults:
            self._profiles[p.profile_id] = p
        self._save_index()
        logger.info("Seeded curated voice profiles into store", count=len(defaults))

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
        import numpy as np
        import soundfile as sf
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
