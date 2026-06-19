"""Per-frame loudness, bass and brightness (acoustic features).

See design/PLAN.md Stage 1 (acoustic tier).
"""

from __future__ import annotations

import numpy as np


def rms_loudness(magnitude: np.ndarray) -> np.ndarray:
    """Perceptual-ish loudness per frame from an STFT magnitude (freq, time).

    Returns values in roughly [0, 1] via a log compression of frame energy.
    """
    energy = np.sqrt(np.mean(magnitude ** 2, axis=0) + 1e-12)
    db = 20.0 * np.log10(energy + 1e-9)
    # Map ~[-80, 0] dB to [0, 1].
    return np.clip((db + 80.0) / 80.0, 0.0, 1.0)


def band_energy(magnitude: np.ndarray, freqs: np.ndarray,
                lo: float, hi: float) -> np.ndarray:
    """Fraction of spectral energy within [lo, hi] Hz, per frame."""
    band = (freqs >= lo) & (freqs < hi)
    total = magnitude.sum(axis=0) + 1e-12
    return magnitude[band, :].sum(axis=0) / total


def bass_energy(magnitude: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    return band_energy(magnitude, freqs, 0.0, 250.0)


def brightness(magnitude: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """High-frequency energy ratio (> 2 kHz)."""
    return band_energy(magnitude, freqs, 2000.0, freqs[-1] + 1.0)
