"""Chroma and harmonic stability (musical features).

See design/PLAN.md Stage 1 (musical tier).
"""

from __future__ import annotations

import numpy as np


def harmonic_stability(chroma: np.ndarray) -> np.ndarray:
    """Per-frame stability = similarity of chroma to its neighbour (0..1).

    chroma: shape (12, n_frames). Returns shape (n_frames,).
    """
    n = chroma.shape[1]
    out = np.zeros(n, dtype=np.float64)
    norm = np.linalg.norm(chroma, axis=0) + 1e-9
    unit = chroma / norm
    for t in range(n):
        prev = unit[:, max(0, t - 1)]
        out[t] = float(np.dot(unit[:, t], prev))
    return np.clip(out, 0.0, 1.0)
