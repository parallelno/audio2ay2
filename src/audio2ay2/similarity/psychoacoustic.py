"""Loudness / brightness (psychoacoustic-ish) loss terms. See design/LOSS.md."""

from __future__ import annotations

import numpy as np


def rms_db(pcm: np.ndarray) -> float:
    energy = np.sqrt(np.mean(pcm ** 2) + 1e-12)
    return 20.0 * np.log10(energy + 1e-9)


def loudness_distance(a_db: float, b_db: float) -> float:
    """Normalized absolute loudness difference (~60 dB span -> 0..1)."""
    return float(np.clip(abs(a_db - b_db) / 60.0, 0.0, 1.0))


def spectral_centroid(pcm: np.ndarray, sample_rate: int) -> float:
    """Single-number spectral centroid (Hz) over the window."""
    spec = np.abs(np.fft.rfft(pcm * np.hanning(len(pcm)))) + 1e-12
    freqs = np.fft.rfftfreq(len(pcm), 1.0 / sample_rate)
    return float((freqs * spec).sum() / spec.sum())


def centroid_distance(a: float, b: float, sample_rate: int) -> float:
    nyq = sample_rate / 2.0
    return float(np.clip(abs(a - b) / nyq, 0.0, 1.0))
