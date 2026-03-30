"""Synthetic waveform generators for auditory system testing.

All functions return 1D numpy arrays at the specified sample rate.
These generators cover the feature space needed to exercise the
cochlear pipeline: pure tones, noise, harmonics, FM, AM, and silence.
"""
from __future__ import annotations

import numpy as np


def generate_pure_tone(
    frequency: float = 440.0,
    duration: float = 0.066,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Sine wave at a single frequency."""
    t = np.arange(int(duration * sample_rate)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def generate_white_noise(
    duration: float = 0.066,
    sample_rate: int = 16000,
    amplitude: float = 0.3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Uniform random noise spanning all frequencies."""
    rng = rng or np.random.default_rng(42)
    n_samples = int(duration * sample_rate)
    return (amplitude * rng.standard_normal(n_samples)).astype(np.float32)


def generate_am_noise(
    mod_freq: float = 70.0,
    duration: float = 0.066,
    sample_rate: int = 16000,
    amplitude: float = 0.3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Amplitude-modulated noise (perceived as rough/harsh).

    Roughness is strongest when mod_freq is in 15-300 Hz range (Vassilakis 2005).
    """
    rng = rng or np.random.default_rng(42)
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate
    noise = rng.standard_normal(n_samples)
    modulator = 0.5 * (1.0 + np.sin(2 * np.pi * mod_freq * t))
    return (amplitude * noise * modulator).astype(np.float32)


def generate_harmonic_stack(
    fundamental: float = 220.0,
    num_harmonics: int = 5,
    duration: float = 0.066,
    sample_rate: int = 16000,
    amplitude: float = 0.3,
) -> np.ndarray:
    """Harmonic tone (fundamental + overtones with 1/n rolloff).

    High HNR, perceived as tonal/pleasant.
    """
    t = np.arange(int(duration * sample_rate)) / sample_rate
    signal = np.zeros_like(t)
    for n in range(1, num_harmonics + 1):
        freq = fundamental * n
        if freq > sample_rate / 2:
            break
        signal += (1.0 / n) * np.sin(2 * np.pi * freq * t)
    peak = np.abs(signal).max() + 1e-8
    return (amplitude * signal / peak).astype(np.float32)


def generate_chirp(
    f_start: float = 100.0,
    f_end: float = 8000.0,
    duration: float = 0.066,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Linear frequency sweep from f_start to f_end."""
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate
    phase = 2 * np.pi * (f_start * t + (f_end - f_start) / (2 * duration) * t ** 2)
    return (amplitude * np.sin(phase)).astype(np.float32)


def generate_fm_tone(
    carrier_freq: float = 440.0,
    mod_freq: float = 5.0,
    mod_depth: float = 100.0,
    duration: float = 0.066,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> np.ndarray:
    """FM synthesis tone. Richer timbre than pure sine."""
    t = np.arange(int(duration * sample_rate)) / sample_rate
    modulator = mod_depth * np.sin(2 * np.pi * mod_freq * t)
    signal = np.sin(2 * np.pi * (carrier_freq + modulator) * t)
    return (amplitude * signal).astype(np.float32)


def generate_silence(
    duration: float = 0.066,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Zero-amplitude signal."""
    return np.zeros(int(duration * sample_rate), dtype=np.float32)
