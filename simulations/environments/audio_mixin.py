"""Audio synthesis mixin for training environments.

Provides richer-than-sine synthetic audio using FM synthesis, ADSR
envelopes, and simple convolution reverb. All implemented in pure
numpy with no extra dependencies.

Each environment that inherits from AudioMixin gets a _generate_audio()
method that produces a 16 kHz mono waveform per step, driven by the
environment state (proximity, collisions, rewards, etc.).

The audio is returned in the info dict as "audio_waveform" so the
auditory specialist can process it during training.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Synthesis primitives
# ---------------------------------------------------------------------------

def adsr_envelope(
    duration: float,
    sample_rate: int = 16000,
    attack: float = 0.01,
    decay: float = 0.02,
    sustain_level: float = 0.7,
    release: float = 0.02,
) -> np.ndarray:
    """Generate an ADSR amplitude envelope.

    Args:
        duration: total duration in seconds
        sample_rate: samples per second
        attack: attack time in seconds
        decay: decay time in seconds
        sustain_level: sustain amplitude (0-1)
        release: release time in seconds

    Returns:
        1D array of amplitude values in [0, 1]
    """
    n_samples = int(duration * sample_rate)
    n_attack = min(int(attack * sample_rate), n_samples)
    n_decay = min(int(decay * sample_rate), n_samples - n_attack)
    n_release = min(int(release * sample_rate), n_samples)
    n_sustain = max(0, n_samples - n_attack - n_decay - n_release)

    env = np.zeros(n_samples, dtype=np.float32)

    # Attack: linear ramp 0 -> 1
    if n_attack > 0:
        env[:n_attack] = np.linspace(0, 1, n_attack, dtype=np.float32)

    # Decay: linear ramp 1 -> sustain_level
    idx = n_attack
    if n_decay > 0:
        env[idx:idx + n_decay] = np.linspace(1, sustain_level, n_decay, dtype=np.float32)
    idx += n_decay

    # Sustain: constant level
    if n_sustain > 0:
        env[idx:idx + n_sustain] = sustain_level
    idx += n_sustain

    # Release: linear ramp sustain_level -> 0
    if n_release > 0:
        env[idx:idx + n_release] = np.linspace(sustain_level, 0, n_release, dtype=np.float32)

    return env


def fm_tone(
    carrier_freq: float,
    mod_freq: float = 5.0,
    mod_depth: float = 50.0,
    duration: float = 0.066,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> np.ndarray:
    """FM synthesis tone with richer timbre than pure sine.

    Frequency modulation creates sidebands that give the tone a
    bell-like or metallic quality depending on mod_freq and mod_depth.
    """
    t = np.arange(int(duration * sample_rate)) / sample_rate
    modulator = mod_depth * np.sin(2 * np.pi * mod_freq * t)
    signal = amplitude * np.sin(2 * np.pi * (carrier_freq + modulator) * t)
    return signal.astype(np.float32)


def fm_chord(
    frequencies: list[float],
    mod_freq: float = 4.0,
    mod_depth: float = 30.0,
    duration: float = 0.066,
    sample_rate: int = 16000,
    amplitude: float = 0.3,
) -> np.ndarray:
    """FM chord: sum of FM tones at given frequencies."""
    signal = np.zeros(int(duration * sample_rate), dtype=np.float32)
    per_amp = amplitude / max(len(frequencies), 1)
    for freq in frequencies:
        signal += fm_tone(freq, mod_freq, mod_depth, duration, sample_rate, per_amp)
    return signal


def noise_burst(
    duration: float = 0.015,
    sample_rate: int = 16000,
    amplitude: float = 0.4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Short noise burst with fast ADSR (impact/collision sound)."""
    rng = rng or np.random.default_rng()
    n = int(duration * sample_rate)
    noise = rng.standard_normal(n).astype(np.float32)
    env = adsr_envelope(duration, sample_rate, attack=0.001, decay=0.005,
                        sustain_level=0.2, release=0.005)
    return amplitude * noise * env


def apply_reverb(
    signal: np.ndarray,
    room_size: float = 0.3,
    sample_rate: int = 16000,
    decay: float = 0.4,
) -> np.ndarray:
    """Simple convolution reverb with exponential decay impulse response.

    Args:
        signal: input audio
        room_size: reverb duration in seconds
        decay: amplitude decay factor
    """
    ir_len = int(room_size * sample_rate)
    if ir_len < 2:
        return signal
    t = np.arange(ir_len, dtype=np.float32) / sample_rate
    ir = np.exp(-t / max(room_size * decay, 0.01)).astype(np.float32)
    ir[0] = 1.0
    ir /= np.sum(ir)
    # Convolve and truncate to original length
    convolved = np.convolve(signal, ir, mode="full")[:len(signal)]
    return convolved.astype(np.float32)


def am_roughness(
    carrier_freq: float = 200.0,
    mod_freq: float = 70.0,
    duration: float = 0.066,
    sample_rate: int = 16000,
    amplitude: float = 0.4,
    roughness_amount: float = 0.5,
) -> np.ndarray:
    """Amplitude-modulated tone perceived as rough (warning/threat sound).

    Roughness is strongest when mod_freq is 15-300 Hz (Vassilakis 2005).
    """
    t = np.arange(int(duration * sample_rate)) / sample_rate
    carrier = np.sin(2 * np.pi * carrier_freq * t)
    modulator = 1.0 - roughness_amount * 0.5 * (1.0 + np.sin(2 * np.pi * mod_freq * t))
    return (amplitude * carrier * modulator).astype(np.float32)


def ascending_sequence(
    base_freq: float = 400.0,
    num_notes: int = 3,
    note_duration: float = 0.02,
    sample_rate: int = 16000,
    amplitude: float = 0.3,
) -> np.ndarray:
    """Ascending pitch sequence (reward/success jingle)."""
    parts = []
    for i in range(num_notes):
        freq = base_freq * (2 ** (i * 4 / 12))  # major third intervals
        tone = fm_tone(freq, mod_freq=6, mod_depth=20, duration=note_duration,
                       sample_rate=sample_rate, amplitude=amplitude)
        env = adsr_envelope(note_duration, sample_rate, attack=0.002, decay=0.005,
                            sustain_level=0.6, release=0.005)
        parts.append(tone * env)
    return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------------------
# Environment mixin
# ---------------------------------------------------------------------------

class AudioMixin:
    """Mixin that adds audio synthesis to any gymnasium environment.

    Subclasses should call _generate_audio(info) in their step() method
    and store the result in info["audio_waveform"].

    Override _audio_event() to customize event detection for your environment.
    """

    _audio_sample_rate: int = 16000
    _audio_frame_duration: float = 0.066  # ~15 fps
    _audio_rng: np.random.Generator | None = None

    def _get_audio_rng(self) -> np.random.Generator:
        if self._audio_rng is None:
            self._audio_rng = np.random.default_rng()
        return self._audio_rng

    def _generate_audio(self, info: dict) -> np.ndarray:
        """Generate synthetic audio waveform based on environment state.

        The default implementation handles common events. Override
        _audio_event() to add environment-specific behavior.

        Returns:
            1D float32 array at _audio_sample_rate Hz
        """
        sr = self._audio_sample_rate
        dur = self._audio_frame_duration
        n_samples = int(dur * sr)
        rng = self._get_audio_rng()

        # Start with ambient noise floor (very quiet)
        signal = 0.02 * rng.standard_normal(n_samples).astype(np.float32)

        # Dispatch to environment-specific event handler
        event_audio = self._audio_event(info, dur, sr, rng)
        if event_audio is not None:
            # Mix event audio into the frame (pad or truncate to frame length)
            if len(event_audio) > n_samples:
                event_audio = event_audio[:n_samples]
            signal[:len(event_audio)] += event_audio

        # Soft clip to prevent clipping
        signal = np.tanh(signal)

        return signal

    def _audio_event(
        self,
        info: dict,
        duration: float,
        sample_rate: int,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        """Override in environment subclass to produce event-specific audio.

        Default: produces audio based on common info dict keys.
        """
        # Proximity-based harmonic tone (if environment provides distance info)
        distance = info.get("distance_to_target")
        if distance is not None and distance < 5.0:
            # Closer = higher pitch, louder
            proximity = max(0.0, 1.0 - distance / 5.0)
            freq = 200 + 600 * proximity  # 200-800 Hz
            tone = fm_tone(freq, mod_freq=4 + proximity * 8,
                           mod_depth=20 + proximity * 80,
                           duration=duration, sample_rate=sample_rate,
                           amplitude=0.1 + 0.3 * proximity)
            env = adsr_envelope(duration, sample_rate, attack=0.005,
                                decay=0.01, sustain_level=0.6, release=0.01)
            return tone * env

        # Reward feedback
        reward = info.get("reward", 0.0)
        if reward > 0.5:
            return apply_reverb(
                ascending_sequence(400, 3, 0.02, sample_rate, 0.3),
                room_size=0.15, sample_rate=sample_rate,
            )
        if reward < -0.3:
            return am_roughness(150, 70, duration, sample_rate, 0.3, 0.8)

        # Collision
        if info.get("collision", False):
            return noise_burst(0.02, sample_rate, 0.4, rng)

        return None


class DarkRoomAudioMixin(AudioMixin):
    """Audio synthesis specialized for the Dark Room environment."""

    def _audio_event(self, info, duration, sample_rate, rng):
        # Light proximity: FM tone that gets richer and louder as agent approaches
        distance = info.get("distance_to_light")
        if distance is not None:
            proximity = max(0.0, 1.0 - distance / 10.0)
            if proximity > 0.01:
                freq = 220 + 440 * proximity
                tone = fm_tone(freq, mod_freq=3 + proximity * 10,
                               mod_depth=10 + proximity * 100,
                               duration=duration, sample_rate=sample_rate,
                               amplitude=0.05 + 0.35 * proximity)
                env = adsr_envelope(duration, sample_rate, attack=0.005,
                                    decay=0.01, sustain_level=0.7, release=0.01)
                return tone * env

        # In the light: pleasant FM chord
        if info.get("in_light", False):
            return apply_reverb(
                fm_chord([262, 330, 392], mod_freq=3, mod_depth=15,
                         duration=duration, sample_rate=sample_rate, amplitude=0.2),
                room_size=0.2, sample_rate=sample_rate,
            )

        # Wall collision
        if info.get("collision", False):
            return noise_burst(0.015, sample_rate, 0.35, rng)

        return None


class NavigationAudioMixin(AudioMixin):
    """Audio synthesis specialized for the Navigation environment."""

    def _audio_event(self, info, duration, sample_rate, rng):
        # Room transition: FM sweep with reverb
        if info.get("room_changed", False):
            t = np.arange(int(0.03 * sample_rate)) / sample_rate
            sweep = 0.25 * np.sin(2 * np.pi * (300 + 400 * t / 0.03) * t)
            env = adsr_envelope(0.03, sample_rate, attack=0.002, decay=0.01,
                                sustain_level=0.4, release=0.01)
            return apply_reverb((sweep * env).astype(np.float32),
                                room_size=0.25, sample_rate=sample_rate)

        # Goal collection: ascending jingle
        if info.get("goal_collected", False):
            return apply_reverb(
                ascending_sequence(500, 4, 0.015, sample_rate, 0.3),
                room_size=0.15, sample_rate=sample_rate,
            )

        # Battery low warning: rough AM tone, roughness proportional to urgency
        battery = info.get("battery", 1.0)
        if battery < 0.3:
            urgency = 1.0 - battery / 0.3  # 0 at 30%, 1 at 0%
            return am_roughness(
                180, 60 + urgency * 40, duration, sample_rate,
                0.1 + 0.25 * urgency, 0.3 + 0.5 * urgency,
            )

        return None


class DMTSAudioMixin(AudioMixin):
    """Audio synthesis specialized for the DMTS environment."""

    def _audio_event(self, info, duration, sample_rate, rng):
        phase = info.get("phase", "")

        # Sample onset: bright bell tone
        if phase == "sample" and info.get("phase_step", 1) == 0:
            tone = fm_tone(800, mod_freq=12, mod_depth=200,
                           duration=0.03, sample_rate=sample_rate, amplitude=0.3)
            env = adsr_envelope(0.03, sample_rate, attack=0.001, decay=0.01,
                                sustain_level=0.3, release=0.01)
            return apply_reverb(tone * env, room_size=0.1, sample_rate=sample_rate)

        # Choice feedback
        if info.get("correct_choice", None) is True:
            return ascending_sequence(600, 3, 0.015, sample_rate, 0.25)
        if info.get("correct_choice", None) is False:
            return am_roughness(120, 80, 0.04, sample_rate, 0.3, 0.7)

        return None


class WCSTAudioMixin(AudioMixin):
    """Audio synthesis specialized for the WCST environment."""

    def _audio_event(self, info, duration, sample_rate, rng):
        # Card sort click (transient)
        if info.get("card_sorted", False):
            return noise_burst(0.008, sample_rate, 0.2, rng)

        # Correct sort: harmonic chime
        if info.get("sort_correct", None) is True:
            return apply_reverb(
                fm_chord([523, 659, 784], mod_freq=5, mod_depth=10,
                         duration=0.04, sample_rate=sample_rate, amplitude=0.2),
                room_size=0.12, sample_rate=sample_rate,
            )

        # Incorrect sort: dissonant buzz
        if info.get("sort_correct", None) is False:
            return am_roughness(100, 90, 0.04, sample_rate, 0.25, 0.8)

        # Rule change: silence (agent must infer from feedback change)
        return None
