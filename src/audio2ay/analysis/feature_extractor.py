"""Feature extraction: audio -> list[FrameFeatures] at the 50 Hz control rate.

Computes one perceptual feature vector per AY frame. Organized in the three
tiers from design/PLAN.md Stage 1: acoustic (loudness/bass/brightness/centroid/
flatness/transient), musical (chroma/salience/beat/tempo/stability).
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from . import chroma as _chroma
from . import loudness as _loud
from . import onset_detector as _onset
from . import salience as _sal


@dataclass
class FrameFeatures:
    """Perceptual description of a single 20 ms frame (data contract)."""

    loudness: float
    bass_energy: float
    brightness: float
    spectral_centroid: float
    harmonic_energy: float
    transient_energy: float
    chroma: np.ndarray          # shape (12,)
    onset_strength: float
    beat_position: float
    tempo: float
    pitch_hz: float
    pitch_confidence: float
    harmonic_stability: float


def extract_features(pcm: np.ndarray, sample_rate: int, fps: int = 50,
                     n_fft: int = 2048) -> list[FrameFeatures]:
    """Extract per-frame features aligned to the ``fps`` control grid."""
    pcm = np.ascontiguousarray(pcm, dtype=np.float32)
    hop = max(1, round(sample_rate / fps))

    stft = librosa.stft(pcm, n_fft=n_fft, hop_length=hop)
    mag = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    n_frames = mag.shape[1]

    loud = _loud.rms_loudness(mag)
    bass = _loud.bass_energy(mag, freqs)
    bright = _loud.brightness(mag, freqs)
    centroid = librosa.feature.spectral_centroid(S=mag, sr=sample_rate)[0]
    flatness = librosa.feature.spectral_flatness(S=mag)[0]
    harmonic_energy = 1.0 - np.clip(flatness, 0.0, 1.0)  # tonal-ness

    onset_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(mag),
                                             sr=sample_rate, hop_length=hop)
    onset_env = _fit_length(onset_env, n_frames)
    onset_norm = onset_env / (onset_env.max() + 1e-9)

    chroma_cq = librosa.feature.chroma_stft(S=mag, sr=sample_rate)
    chroma_cq = _fit_cols(chroma_cq, n_frames)
    stability = _chroma.harmonic_stability(chroma_cq)

    try:
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sample_rate, hop_length=hop)
        tempo = float(np.atleast_1d(tempo)[0])
    except Exception:
        tempo, beat_frames = 0.0, np.array([], dtype=np.int64)
    beat_pos = _onset.beat_positions(n_frames, beat_frames)

    pitches, pmags = librosa.piptrack(S=mag, sr=sample_rate)
    pitch_hz, pitch_conf = _sal.dominant_pitch(pitches, pmags)

    centroid_norm = np.clip(centroid / (sample_rate / 2.0), 0.0, 1.0)

    out: list[FrameFeatures] = []
    for t in range(n_frames):
        out.append(FrameFeatures(
            loudness=float(loud[t]),
            bass_energy=float(bass[t]),
            brightness=float(bright[t]),
            spectral_centroid=float(centroid_norm[t]),
            harmonic_energy=float(harmonic_energy[t]),
            transient_energy=float(onset_norm[t]),
            chroma=chroma_cq[:, t].copy(),
            onset_strength=float(onset_norm[t]),
            beat_position=float(beat_pos[t]),
            tempo=float(tempo),
            pitch_hz=float(pitch_hz[t]),
            pitch_confidence=float(pitch_conf[t]),
            harmonic_stability=float(stability[t]),
        ))
    return out


def _fit_length(arr: np.ndarray, n: int) -> np.ndarray:
    if arr.shape[0] == n:
        return arr
    if arr.shape[0] > n:
        return arr[:n]
    return np.pad(arr, (0, n - arr.shape[0]))


def _fit_cols(arr: np.ndarray, n: int) -> np.ndarray:
    if arr.shape[1] == n:
        return arr
    if arr.shape[1] > n:
        return arr[:, :n]
    return np.pad(arr, ((0, 0), (0, n - arr.shape[1])))
