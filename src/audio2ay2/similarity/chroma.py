"""Chroma / harmonic similarity loss terms. See design/LOSS.md."""

from __future__ import annotations

import librosa
import numpy as np


def mean_chroma(pcm: np.ndarray, sample_rate: int, n_fft: int = 2048,
                hop: int = 512) -> np.ndarray:
    """Mean 12-bin chroma vector over a window (unit-normalized)."""
    if pcm.shape[0] < n_fft:
        pcm = np.pad(pcm, (0, n_fft - pcm.shape[0]))
    chroma = librosa.feature.chroma_stft(y=pcm, sr=sample_rate, n_fft=n_fft,
                                         hop_length=hop)
    v = chroma.mean(axis=1)
    n = np.linalg.norm(v) + 1e-9
    return v / n


def chroma_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity between two unit chroma vectors (0..2 -> 0..1)."""
    cos = float(np.dot(a, b))
    return float(np.clip(1.0 - cos, 0.0, 1.0))
