"""
Generate synthetic speech-like reference WAV files for HTH default voice profiles.

Improvement over previous version: Instead of pure sinusoidal tones (which produce
poor speaker embeddings — MeanVC2's speaker encoder expects speech, not sine waves),
this version synthesises pseudo-speech using:

  - Glottal pulse model: asymmetric sawtooth-like waveform that approximates the
    Liljencrants-Fant glottal source (more harmonically rich than pure sine)
  - Natural F0 contour: slowly falling pitch (declarative speech intonation)
  - Jitter: ±1.5% random F0 perturbation per period (mimics vocal cord irregularity)
  - Shimmer: ±1.5% amplitude perturbation per period
  - Syllabic amplitude modulation: 4–5 Hz AM simulates syllable-rate rhythm
  - Vowel formant transitions: 5-vowel (a→e→i→o→u) cycle over the utterance
  - Breathiness noise floor: 5–10% wideband noise mixed in
  - Proper onset/offset fades: 80ms attack, 150ms release

Result: speaker embeddings that capture distinct pitch ranges, formant structures,
and prosodic patterns — far more useful for zero-shot voice conversion than tones.

All generated audio is owned by HTH Real Voice Changer (algorithmic, no recordings).
Consented internal reference per AGENTS.md Rule 4.
"""
from __future__ import annotations

import math
import os
import random
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 16000
DURATION = 5.0  # seconds — longer = richer embedding

# English vowel formants (F1, F2, F3) in Hz — ITU-T P.501 approximate
_VOWELS: list[tuple[float, float, float]] = [
    (730, 1090, 2440),  # /a/ as in "father"
    (530, 1840, 2480),  # /e/ as in "bed"
    (270, 2290, 3010),  # /i/ as in "see"
    (570, 840,  2410),  # /o/ as in "go"
    (300, 870,  2240),  # /u/ as in "blue"
]

# (profile_id, F0_mean, F0_range, breathiness, syllable_rate, gender_weight)
# gender_weight: 1.0=female-bright, 0.5=female-soft, 0.0=male
_PROFILES: list[tuple] = [
    ("profile-female-eva",      200.0, 30.0, 0.06, 4.8, 1.00),
    ("profile-female-ayesha",   183.0, 22.0, 0.08, 4.3, 0.90),
    ("profile-female-zara",     213.0, 35.0, 0.05, 5.2, 1.00),
    ("profile-female-ananya",   194.0, 28.0, 0.07, 4.5, 0.85),
    ("profile-female-priya",    207.0, 32.0, 0.06, 4.9, 0.95),
    ("profile-female-sana",     189.0, 24.0, 0.09, 4.1, 0.80),
    ("profile-female-chloe",    221.0, 38.0, 0.05, 5.5, 1.00),
    ("profile-male-bilal",      113.0, 18.0, 0.04, 3.8, 0.00),
    ("profile-female-source",   245.0, 45.0, 0.04, 6.0, 1.00),  # Anime: high, wide range
    ("profile-character-ratke",  93.0, 12.0, 0.11, 3.2, 0.00),  # Cyber: low, raspy
]


def _glottal_pulse(phase: float) -> float:
    """
    Simplified LF-inspired glottal source waveform.
    Produces a waveform with steeper closing phase (like a real glottis),
    giving richer odd+even harmonics compared to a pure sine.
    phase in [0, 1).
    """
    if phase < 0.65:
        # Open phase: smooth rise and peak
        t = phase / 0.65
        return math.sin(math.pi * t) ** 2
    else:
        # Closing phase: sharp linear fall (glottal closure)
        t = (phase - 0.65) / 0.35
        return 1.0 - t


def _vowel_formant_boost(sample: float, f0: float, t: float,
                          vowel_idx: int, famp: float) -> float:
    """
    Add formant coloring by boosting harmonics nearest to formant frequencies.
    Rotates through 5 vowels to give spectral diversity for embedding extraction.
    """
    f1, f2, f3 = _VOWELS[vowel_idx % 5]
    boost = 0.0
    # Nearest harmonic to each formant
    for ffreq, weight in [(f1, 0.6), (f2, 0.4), (f3, 0.2)]:
        nh = max(1, round(ffreq / max(f0, 1.0)))
        if 1 <= nh <= 16:
            boost += weight * math.sin(2 * math.pi * nh * f0 * t)
    return sample + famp * boost


def _generate_wav(
    profile_id: str,
    f0_mean: float,
    f0_range: float,
    breathiness: float,
    syllable_rate: float,
    gender_weight: float,
    out_dir: Path,
) -> Path:
    """Synthesise a speech-like signal and write it as a 16kHz mono WAV."""
    rng = random.Random(hash(profile_id) & 0xFFFF_FFFF)
    n = int(SAMPLE_RATE * DURATION)

    # Pre-generate per-period jitter/shimmer tables (varying F0 and amplitude)
    # We'll apply these continuously during synthesis
    buf: list[float] = []

    # State variables
    phase = 0.0           # Glottal phase [0, 1)

    # Syllabic AM envelope state
    syl_phase = rng.random()  # randomise start of syllabic cycle

    # Vowel cycle: advance one vowel every ~300ms
    vowel_period = 0.30  # seconds per vowel
    formant_amp = 0.35   # formant contribution amplitude

    # Jitter/shimmer state
    jitter_sigma = 0.015  # ±1.5% F0 jitter
    shimmer_sigma = 0.015  # ±1.5% amplitude shimmer
    cur_jitter = 1.0
    cur_shimmer = 1.0
    # Smoothed jitter/shimmer (updated at glottal closure = period boundary)
    period_counter = 0

    for i in range(n):
        t = i / SAMPLE_RATE

        # Natural F0 contour: falling intonation over utterance with mild random walk
        # Declarative speech falls ~15% from start to end
        progress = t / DURATION
        f0_contour = f0_mean * (1.0 + 0.08 * math.sin(2 * math.pi * 0.4 * t)
                                - 0.12 * progress)
        f0_contour = max(f0_mean * 0.7, min(f0_mean * 1.3, f0_contour))

        # Apply jitter (smoothed, updated per period)
        f0_inst = f0_contour * cur_jitter

        # Advance glottal phase
        phase += f0_inst / SAMPLE_RATE
        if phase >= 1.0:
            phase -= 1.0
            period_counter += 1
            # Update jitter/shimmer at each glottal closure (pitch period boundary)
            # Smooth toward new random target
            jitter_target = 1.0 + rng.gauss(0, jitter_sigma)
            shimmer_target = 1.0 + rng.gauss(0, shimmer_sigma)
            cur_jitter = cur_jitter * 0.7 + jitter_target * 0.3
            cur_shimmer = cur_shimmer * 0.7 + shimmer_target * 0.3

        # Glottal source signal
        voiced = _glottal_pulse(phase) * cur_shimmer

        # 8 harmonics with spectral tilt −9 dB/octave (speech-like)
        harmonic_sum = 0.0
        for h in range(1, 9):
            amp = 1.0 / (h ** 1.5)
            harmonic_sum += amp * math.sin(2 * math.pi * h * f0_inst * t)

        # Mix glottal pulse with harmonic sum (blend ratio)
        source = 0.4 * voiced + 0.6 * harmonic_sum

        # Vowel formant coloring (cycles through 5 vowels)
        vowel_idx = int(t / vowel_period) % 5
        source = _vowel_formant_boost(source, f0_inst, t, vowel_idx, formant_amp)

        # Syllabic amplitude envelope: 4–5 Hz rhythm mimicking syllables
        syl_phase += syllable_rate / SAMPLE_RATE
        syl_env = 0.65 + 0.35 * (0.5 + 0.5 * math.sin(2 * math.pi * syl_phase))
        source *= syl_env

        # Breathiness: wideband noise mixed with the voiced signal
        noise = (rng.random() * 2.0 - 1.0) * breathiness
        s = source * (1.0 - breathiness * 0.5) + noise

        buf.append(s)

    # Amplitude envelope: 80ms attack, 150ms release
    att = int(0.08 * SAMPLE_RATE)
    rel = int(0.15 * SAMPLE_RATE)
    for i in range(att):
        buf[i] *= (i / att) ** 2  # quadratic onset (more natural)
    for i in range(rel):
        buf[n - 1 - i] *= (i / rel)

    # Normalise to −4 dBFS (peak = 0.63) — leave headroom for embedding processor
    peak = max(abs(x) for x in buf)
    if peak > 1e-6:
        buf = [x * (0.63 / peak) for x in buf]

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
    Returns a dict mapping profile_id → absolute WAV path.
    Already-present files are skipped unless *force* is True.
    """
    if out_dir is None:
        out_dir = Path(__file__).parent

    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, str] = {}

    for row in _PROFILES:
        pid, f0, f0r, br, sr, gw = row
        wav_path = out_dir / f"{pid}.wav"
        if force or not wav_path.is_file():
            _generate_wav(pid, f0, f0r, br, sr, gw, out_dir)
        result[pid] = str(wav_path)

    return result


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    out = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else None
    paths = ensure_reference_wavs(out_dir=out, force=force)
    for pid, p in paths.items():
        sz = os.path.getsize(p)
        print(f"  {pid}: {sz} bytes")
    print(f"Done — {len(paths)} WAVs ready.")
