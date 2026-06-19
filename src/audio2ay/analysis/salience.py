"""Dominant-pitch salience and confidence (musical features)."""

from __future__ import annotations

import numpy as np


def dominant_pitch(pitches: np.ndarray, magnitudes: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Pick the strongest pitch per frame from librosa.piptrack output.

    pitches/magnitudes: shape (freq_bins, n_frames). Returns (pitch_hz,
    confidence) each shape (n_frames,). Confidence is the peak magnitude
    normalized by the frame's total magnitude.
    """
    n = pitches.shape[1]
    pitch_hz = np.zeros(n, dtype=np.float64)
    confidence = np.zeros(n, dtype=np.float64)
    for t in range(n):
        col = magnitudes[:, t]
        k = int(np.argmax(col))
        peak = float(col[k])
        total = float(col.sum()) + 1e-9
        pitch_hz[t] = float(pitches[k, t])
        confidence[t] = peak / total
    return pitch_hz, np.clip(confidence, 0.0, 1.0)
