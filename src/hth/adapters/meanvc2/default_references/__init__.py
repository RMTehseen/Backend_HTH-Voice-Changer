# Bundled synthetic reference WAV directory.
# These files are algorithmically generated voice references owned by HTH Real Voice Changer.
# Consented internal reference per AGENTS.md Rule 4.
# WAVs are generated on first import (pure stdlib, zero external deps).
from __future__ import annotations

from pathlib import Path

REFERENCE_DIR = Path(__file__).parent

# Auto-generate any missing WAVs on first import (fast: <1 s for all 10).
# This runs in both local and cloud environments without network access.
try:
    from .generate import ensure_reference_wavs as _ensure
    _ensure(out_dir=REFERENCE_DIR)
except Exception:
    pass  # Non-fatal: Fallback Timbre will be used if generation fails.


def get_reference_path(profile_id: str) -> str | None:
    """Return absolute path to bundled reference WAV for a profile, or None if not found."""
    p = REFERENCE_DIR / f"{profile_id}.wav"
    return str(p) if p.is_file() else None
