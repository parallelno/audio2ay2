"""Mel / log-STFT spectral distance (the primary loss term). See design/LOSS.md."""

from __future__ import annotations

import librosa
import numpy as np

_MEL_CACHE: dict[tuple, np.ndarray] = {}


def mel_basis(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    key = (sample_rate, n_fft, n_mels)
    basis = _MEL_CACHE.get(key)
    if basis is None:
        basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
        _MEL_CACHE[key] = basis
    return basis


def log_mel(pcm: np.ndarray, sample_rate: int, n_fft: int = 1024,
            hop: int = 256, n_mels: int = 48) -> np.ndarray:
    """Compute a log-mel spectrogram (n_mels, frames)."""
    if pcm.shape[0] < n_fft:
        pcm = np.pad(pcm, (0, n_fft - pcm.shape[0]))
    stft = librosa.stft(pcm, n_fft=n_fft, hop_length=hop, center=True)
    power = (np.abs(stft) ** 2)
    mel = mel_basis(sample_rate, n_fft, n_mels) @ power
    return np.log1p(mel)


def mel_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean-absolute distance between two log-mel spectrograms, length-matched."""
    n = min(a.shape[1], b.shape[1])
    if n == 0:
        return 0.0
    diff = a[:, :n] - b[:, :n]
    denom = (np.abs(a[:, :n]).mean() + 1e-6)
    return float(np.abs(diff).mean() / denom)
