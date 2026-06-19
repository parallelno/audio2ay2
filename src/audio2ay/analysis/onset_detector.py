"""Onset / transient strength and the beat grid (musical features)."""

from __future__ import annotations

import numpy as np


def beat_positions(n_frames: int, beat_frames: np.ndarray) -> np.ndarray:
    """Phase within the current beat (0..1) for each analysis frame."""
    out = np.zeros(n_frames, dtype=np.float64)
    if beat_frames is None or len(beat_frames) < 2:
        return out
    beats = np.asarray(beat_frames, dtype=np.int64)
    for i in range(len(beats) - 1):
        a, b = beats[i], beats[i + 1]
        if b <= a:
            continue
        span = b - a
        idx = np.arange(a, min(b, n_frames))
        out[idx] = (idx - a) / span
    return out
