"""Weighted perceptual loss: compare *sound*, not registers.

The heart of the project (design/LOSS.md). For speed in the optimizer's hot loop,
each window is reduced to a single averaged power spectrum (one rFFT per analysis
block) from which every term is derived: log-mel, chroma, centroid, flatness,
plus a loudness level and energy envelope.

Loudness handling (critical): rather than normalizing every window to unit RMS
(which amplifies silence into garbage and makes the metric blind to dynamics),
the target and candidate are each divided by a *global* reference level set via
:meth:`configure` — the target's representative loud level and the AY chip's
full-scale level. This keeps silent frames silent, loud frames near 1.0, and
makes the two sources directly comparable. Silent windows are detected and their
undefined spectral-shape terms (chroma/centroid/flatness) are skipped so they do
not inject noise; the mel and loudness terms still penalize sound-vs-silence
mismatches.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

# Fixed analysis size so mel/chroma filter bases stay constant and cacheable.
_N = 2048
# A window quieter than this fraction of the reference level is treated as silent.
_SILENCE_REL = 0.02


@dataclass
class WindowFeatures:
    """Cached perceptual features for one PCM window (all 1-D / scalar)."""

    log_mel: np.ndarray
    chroma: np.ndarray
    centroid: float
    flatness: float
    level_db: float
    envelope: np.ndarray
    silent: bool
    pitch_hz: float


class PerceptualLoss:
    """Compute the weighted perceptual loss between target and candidate PCM."""

    def __init__(self, sample_rate: int, weights: dict[str, float],
                 n_mels: int = 48) -> None:
        self.sr = sample_rate
        self.w = weights
        self._window = np.hanning(_N).astype(np.float64)
        self._freqs = np.fft.rfftfreq(_N, 1.0 / sample_rate)
        self._mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=_N, n_mels=n_mels)
        self._chroma_basis = librosa.filters.chroma(sr=sample_rate, n_fft=_N)
        # Reference levels; 1.0 until configured (so identical PCM scores 0).
        self._target_ref = 1.0
        self._cand_ref = 1.0

    # -- configuration ----------------------------------------------------

    @staticmethod
    def loud_level(pcm: np.ndarray, sample_rate: int) -> float:
        """Representative *loud* level of a signal: 90th-percentile window RMS."""
        pcm = np.asarray(pcm, dtype=np.float64)
        w = max(1, sample_rate // 50)
        n = pcm.shape[0]
        if n < w:
            return float(np.sqrt(np.mean(pcm ** 2) + 1e-12))
        rms = np.array([
            np.sqrt(np.mean(pcm[i:i + w] ** 2) + 1e-12)
            for i in range(0, n - w + 1, w)
        ])
        return float(np.percentile(rms, 90))

    def configure(self, target_pcm: np.ndarray, cand_ref_rms: float) -> None:
        """Set global reference levels for target and candidate (AY) signals."""
        self._target_ref = max(self.loud_level(target_pcm, self.sr), 1e-6)
        self._cand_ref = max(float(cand_ref_rms), 1e-9)

    # -- spectrum helpers -------------------------------------------------

    def _avg_power(self, pcm: np.ndarray) -> np.ndarray:
        n = pcm.shape[0]
        if n < _N:
            pcm = np.pad(pcm, (0, _N - n))
            n = _N
        nframes = max(1, n // _N)
        acc = np.zeros(_N // 2 + 1, dtype=np.float64)
        for k in range(nframes):
            seg = pcm[k * _N:(k + 1) * _N]
            if seg.shape[0] < _N:
                seg = np.pad(seg, (0, _N - seg.shape[0]))
            acc += np.abs(np.fft.rfft(seg * self._window)) ** 2
        return acc / nframes + 1e-12

    def _envelope(self, pcm: np.ndarray, blocks: int = 16) -> np.ndarray:
        n = len(pcm)
        if n < blocks:
            return np.zeros(blocks)
        step = n // blocks
        env = np.array([
            np.sqrt(np.mean(pcm[i * step:(i + 1) * step] ** 2) + 1e-12)
            for i in range(blocks)
        ])
        return env / (env.max() + 1e-9)

    def _dominant_hz(self, power: np.ndarray) -> float:
        """Frequency of the strongest spectral peak (parabolic-interpolated)."""
        lo = int(50.0 * _N / self.sr)
        hi = int(min(5000.0, self.sr / 2) * _N / self.sr)
        band = power[lo:hi]
        if band.size == 0:
            return 0.0
        idx = lo + int(np.argmax(band))
        if 1 <= idx < power.size - 1:
            a = np.log(power[idx - 1])
            b = np.log(power[idx])
            c = np.log(power[idx + 1])
            denom = a - 2.0 * b + c
            delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
        else:
            delta = 0.0
        return float((idx + delta) * self.sr / _N)

    def _features(self, x: np.ndarray) -> WindowFeatures:
        """Features of a signal already scaled into loudness units (1.0 ~ loud)."""
        rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
        silent = rms < _SILENCE_REL
        power = self._avg_power(x)
        mel = np.log1p(self._mel_basis @ power)
        if silent:
            chroma = np.zeros(12)
            centroid = 0.0
            flatness = 1.0
            pitch_hz = 0.0
        else:
            ch = self._chroma_basis @ power
            chroma = ch / (np.linalg.norm(ch) + 1e-12)
            centroid = float((self._freqs * power).sum() / power.sum())
            gm = float(np.exp(np.mean(np.log(power))))
            flatness = gm / float(np.mean(power))
            pitch_hz = self._dominant_hz(power)
        level_db = 20.0 * np.log10(rms + 1e-4)
        return WindowFeatures(
            log_mel=mel, chroma=chroma, centroid=centroid, flatness=flatness,
            level_db=level_db, envelope=self._envelope(x), silent=silent,
            pitch_hz=pitch_hz,
        )

    # -- public API -------------------------------------------------------

    def features(self, pcm: np.ndarray) -> WindowFeatures:
        """Target-side features (scaled by the target reference level)."""
        x = np.asarray(pcm, dtype=np.float64) / (self._target_ref + 1e-12)
        return self._features(x)

    def _cand_features(self, pcm: np.ndarray) -> WindowFeatures:
        x = np.asarray(pcm, dtype=np.float64) / (self._cand_ref + 1e-12)
        return self._features(x)

    def _terms(self, target: WindowFeatures, cand: WindowFeatures) -> dict[str, float]:
        nyq = self.sr / 2.0
        both_voiced = (not target.silent) and (not cand.silent)
        if both_voiced and target.pitch_hz > 0 and cand.pitch_hz > 0:
            semis = abs(12.0 * np.log2(cand.pitch_hz / target.pitch_hz))
            pitch = float(np.clip(semis / 12.0, 0, 1))
        else:
            pitch = 0.0
        return {
            "mel": float(np.abs(target.log_mel - cand.log_mel).mean()),
            "pitch": pitch,
            "loudness": float(np.clip(abs(target.level_db - cand.level_db) / 60.0, 0, 1)),
            "centroid": (float(np.clip(abs(target.centroid - cand.centroid) / nyq, 0, 1))
                         if both_voiced else 0.0),
            "chroma": (float(np.clip(1.0 - float(np.dot(target.chroma, cand.chroma)), 0, 1))
                       if both_voiced else 0.0),
            "harmonic": (float(abs(target.flatness - cand.flatness))
                         if both_voiced else 0.0),
            "transient": float(np.abs(target.envelope - cand.envelope).mean()),
        }

    def compare(self, target: WindowFeatures, cand_pcm: np.ndarray) -> float:
        terms = self._terms(target, self._cand_features(cand_pcm))
        return sum(self.w.get(k, 0.0) * v for k, v in terms.items())

    def breakdown(self, target: WindowFeatures, cand_pcm: np.ndarray
                  ) -> dict[str, float]:
        return self._terms(target, self._cand_features(cand_pcm))
