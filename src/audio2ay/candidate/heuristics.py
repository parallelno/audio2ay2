"""Heuristic mappings from perceptual features to AY parameters.

Pure functions used by the candidate generator. See design/PLAN.md Stage 2.
"""

from __future__ import annotations

import numpy as np

from ..ay.registers import (
    TONE_PERIOD_MIN, TONE_PERIOD_MAX,
    NOISE_PERIOD_MIN, NOISE_PERIOD_MAX,
)

_CHROMA_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# MIDI note of pitch-class 0 (C) in octave 4 region used for chroma->pitch.
_C4_MIDI = 60


def hz_to_tone_period(freq_hz: float, clock_hz: int) -> int:
    """Convert a target frequency to the nearest legal 12-bit tone period."""
    if freq_hz <= 0:
        return TONE_PERIOD_MAX
    tp = int(round(clock_hz / (16.0 * freq_hz)))
    return max(TONE_PERIOD_MIN, min(TONE_PERIOD_MAX, tp))


def tone_period_to_hz(tp: int, clock_hz: int) -> float:
    return clock_hz / (16.0 * max(TONE_PERIOD_MIN, tp))


def chroma_to_hz(pitch_class: int, octave: int = 4) -> float:
    """Frequency for a chroma bin at a chosen octave (A4 = 440)."""
    midi = _C4_MIDI + pitch_class + 12 * (octave - 4)
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def loudness_to_volume(loudness: float) -> int:
    """Map normalized loudness (0..1) to a 4-bit AY volume (0..15)."""
    return int(round(np.clip(loudness, 0.0, 1.0) * 15))


def brightness_to_noise_period(brightness: float) -> int:
    """Brighter transients -> smaller noise period (higher noise pitch)."""
    b = float(np.clip(brightness, 0.0, 1.0))
    np_val = int(round(NOISE_PERIOD_MAX - b * (NOISE_PERIOD_MAX - NOISE_PERIOD_MIN)))
    return max(NOISE_PERIOD_MIN, min(NOISE_PERIOD_MAX, np_val))


def top_pitch_classes(chroma: np.ndarray, k: int) -> list[int]:
    """Indices of the k strongest chroma bins, descending."""
    return list(np.argsort(chroma)[::-1][:k])
