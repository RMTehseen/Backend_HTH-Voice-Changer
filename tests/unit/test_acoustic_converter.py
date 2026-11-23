"""Unit tests and benchmark for AcousticVoiceConverter."""

import time

import numpy as np

from hth.audio.conversion.acoustic_converter import AcousticVoiceConverter
from hth.audio.conversion.contracts import VoiceProfile


def test_converter_initialization():
    converter = AcousticVoiceConverter()
    assert converter.sample_rate == 48000
    assert converter.frame_size == 480
    assert converter.config.profile == VoiceProfile.MALE_TO_FEMALE


def test_male_to_female_profile_config():
    converter = AcousticVoiceConverter()
    converter.set_profile(VoiceProfile.MALE_TO_FEMALE)
    assert converter.config.pitch_semitones > 0  # Pitch shifted UP
    assert converter.config.formant_ratio > 1.0  # Formants shifted UP (shorter vocal tract)


def test_female_to_male_profile_config():
    converter = AcousticVoiceConverter()
    converter.set_profile(VoiceProfile.FEMALE_TO_MALE)
    assert converter.config.pitch_semitones < 0  # Pitch shifted DOWN
    assert converter.config.formant_ratio < 1.0  # Formants shifted DOWN (longer vocal tract)
    assert converter.config.chest_resonance_gain > 1.0  # Chest resonance boosted


def test_zero_allocation_frame_processing():
    converter = AcousticVoiceConverter()
    in_frame = np.random.uniform(-0.5, 0.5, 480).astype(np.float32)
    out_frame = np.zeros(480, dtype=np.float32)

    # Process frame
    converter.process_frame(in_frame, out_frame)

    assert out_frame.shape == (480,)
    assert not np.isnan(out_frame).any()
    assert not np.isinf(out_frame).any()
    assert (np.abs(out_frame) <= 1.0).all()


def test_real_time_performance_latency_budget():
    """Verify processing 10ms frame takes < 3ms on CPU (comfortably under 10ms budget)."""
    converter = AcousticVoiceConverter()
    in_frame = np.random.uniform(-0.4, 0.4, 480).astype(np.float32)
    out_frame = np.zeros(480, dtype=np.float32)

    # Warmup
    for _ in range(10):
        converter.process_frame(in_frame, out_frame)

    durations = []
    for _ in range(100):
        t0 = time.perf_counter()
        converter.process_frame(in_frame, out_frame)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        durations.append(dt_ms)

    p50 = float(np.percentile(durations, 50))
    p95 = float(np.percentile(durations, 95))
    max_d = float(np.max(durations))

    print(f"\nAcousticVoiceConverter timing: p50={p50:.3f}ms, p95={p95:.3f}ms, max={max_d:.3f}ms")
    # Must be comfortably under the 10ms frame budget (p50 < 3ms, p95 < 8ms)
    assert p50 < 3.0, f"p50 latency too high: {p50}ms"
    assert p95 < 8.0, f"p95 latency too high: {p95}ms (exceeds frame budget)"


def test_silence_input_safety():
    converter = AcousticVoiceConverter()
    in_frame = np.zeros(480, dtype=np.float32)
    out_frame = np.ones(480, dtype=np.float32)

    converter.process_frame(in_frame, out_frame)
    # Output must be attenuated / zeroed, not amplifying noise
    assert np.max(np.abs(out_frame)) < 1e-3


def test_continuous_audio_conversion():
    converter = AcousticVoiceConverter()
    sr = 48000
    # Generate 1.0s synthetic speech-like tone (fundamental 130 Hz + harmonics)
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    signal = (
        0.3 * np.sin(2 * np.pi * 130 * t)
        + 0.15 * np.sin(2 * np.pi * 260 * t)
        + 0.08 * np.sin(2 * np.pi * 390 * t)
    )

    converted = converter.convert_audio(signal)
    assert len(converted) == len(signal)
    assert not np.isnan(converted).any()
    assert np.max(np.abs(converted)) > 0.05
    assert np.max(np.abs(converted)) <= 1.0
