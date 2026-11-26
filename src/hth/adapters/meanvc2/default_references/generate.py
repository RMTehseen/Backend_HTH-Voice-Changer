"""
Generate synthetic formant-based reference WAV files for HTH default voice profiles.

These are algorithmically-generated speaker reference signals owned by HTH Real Voice Changer.
They are generated on first run (or when missing) using only Python stdlib — no external deps.
Consented internal reference per AGENTS.md Rule 4.

Each WAV: 16 kHz, mono, 16-bit PCM, ~4 seconds.
Distinct fundamental frequency + formant profile per voice type.
"""
from __future__ import annotations

import math
import os
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 16000
DURATION = 4.0

# (profile_id, F0_hz, formants[(F1,amp),(F2,amp),(F3,amp)], vibrato_rate, vibrato_depth, breathiness)
_PROFILES: list[tuple] = [
    # --- Female voices: F0 170–240 Hz, higher formants ---
    ("profile-female-eva",      200.0, [(800, 1.0), (2200, 0.8), (2900, 0.4)], 5.5, 0.015, 0.07),
    ("profile-female-ayesha",   185.0, [(750, 1.0), (2100, 0.8), (2800, 0.4)], 5.0, 0.015, 0.09),
    ("profile-female-zara",     210.0, [(820, 1.0), (2300, 0.9), (3000, 0.4)], 6.0, 0.015, 0.06),
    ("profile-female-ananya",   195.0, [(790, 1.0), (2150, 0.8), (2850, 0.4)], 5.2, 0.015, 0.08),
    ("profile-female-priya",    205.0, [(810, 1.0), (2250, 0.8), (2950, 0.4)], 5.8, 0.015, 0.07),
    ("profile-female-sana",     190.0, [(770, 1.0), (2080, 0.8), (2780, 0.4)], 5.3, 0.015, 0.09),
    ("profile-female-chloe",    220.0, [(850, 1.0), (2400, 0.9), (3100, 0.4)], 6.2, 0.015, 0.06),
    # --- Male voice: F0 ~115 Hz, lower formants ---
    ("profile-male-bilal",      115.0, [(650, 1.0), (1200, 0.7), (2500, 0.3)], 4.5, 0.012, 0.05),
    # --- Character voices ---
    ("profile-female-source",   240.0, [(900, 1.0), (2600, 1.0), (3200, 0.5)], 7.0, 0.015, 0.05),  # Anime
    ("profile-character-ratke",  95.0, [(600, 1.0), (1100, 0.6), (2400, 0.4)], 3.5, 0.010, 0.12),  # Cyber
]


def _generate_wav(
    profile_id: str,
    f0: float,
    formants: list[tuple[float, float]],
    vibrato_rate: float,
    vibrato_depth: float,
    breathiness: float,
    out_dir: Path,
) -> Path:
    """Synthesise a voice-like signal and write it as a 16kHz mono WAV."""
    import random
    rng = random.Random(hash(profile_id) & 0xFFFF_FFFF)

    n = int(SAMPLE_RATE * DURATION)
    buf: list[float] = []

    for i in range(n):
        t = i / SAMPLE_RATE
        # Vibrato on fundamental
        f0v = f0 * (1.0 + vibrato_depth * math.sin(2 * math.pi * vibrato_rate * t))
        s = 0.0
        # Glottal source: 8 harmonics with spectral tilt −10 dB/oct
        for h in range(1, 9):
            s += (1.0 / h ** 1.3) * math.sin(2 * math.pi * h * f0v * t)
        # Formant boost: excite nearest harmonic of each formant frequency
        for ffreq, famp in formants:
            nh = max(1, round(ffreq / f0))
            if nh <= 14:
                s += famp * math.sin(2 * math.pi * nh * f0v * t)
        # Breathiness (noise)
        noise = (rng.random() * 2.0 - 1.0) * breathiness
        s = s * (1.0 - breathiness) + noise
        buf.append(s)

    # Fade in / fade out
    att = int(0.05 * SAMPLE_RATE)
    rel = int(0.10 * SAMPLE_RATE)
    for i in range(att):
        buf[i] *= i / att
    for i in range(rel):
        buf[n - 1 - i] *= i / rel

    # Normalise to −3 dBFS
    peak = max(abs(x) for x in buf)
    if peak > 1e-6:
        scale = 0.70 / peak
        buf = [x * scale for x in buf]

    # Encode as 16-bit signed PCM
    pcm = b"".join(
        struct.pack("<h", max(-32767, min(32767, int(x * 32767))))
        for x in buf
    )

    out_path = out_dir / f"{profile_id}.wav"
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)

    return out_path


def ensure_reference_wavs(out_dir: Path | None = None, force: bool = False) -> dict[str, str]:
    """
    Generate any missing reference WAVs in *out_dir* (defaults to this package directory).

    Returns a dict mapping profile_id → absolute WAV path for every profile.
    Already-present files are skipped unless *force* is True.
    """
    if out_dir is None:
        out_dir = Path(__file__).parent

    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, str] = {}

    for row in _PROFILES:
        pid, f0, formants, vr, vd, br = row
        wav_path = out_dir / f"{pid}.wav"
        if force or not wav_path.is_file():
            _generate_wav(pid, f0, formants, vr, vd, br, out_dir)
        result[pid] = str(wav_path)

    return result


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    out = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else None
    paths = ensure_reference_wavs(out_dir=out, force=force)
    for pid, p in paths.items():
        sz = os.path.getsize(p)
        print(f"  {pid}: {p}  ({sz} bytes)")
    print(f"Done — {len(paths)} WAVs ready.")
