"""Audio decode/encode to mono float32 PCM.

Decode wav/mp3/ogg/flac; encode wav/mp3 (preview). Uses soundfile (libsndfile,
which supports MP3 read/write) with a librosa fallback for exotic inputs.
See design/ROADMAP.md Phase 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Load ``path`` as mono float32 at ``sample_rate``."""
    path = Path(path)
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
    except Exception:
        import librosa  # lazy; heavier import

        mono, sr = librosa.load(str(path), sr=None, mono=True)
        mono = mono.astype(np.float32)
    if sr != sample_rate:
        import librosa

        mono = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=sample_rate)
    return np.ascontiguousarray(mono, dtype=np.float32)


def save_audio(path: str | Path, pcm: np.ndarray, sample_rate: int) -> None:
    """Write mono PCM to ``path`` (format inferred from extension)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.asarray(pcm, dtype=np.float32)
    peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
    if peak > 1.0:
        pcm = pcm / peak
    ext = path.suffix.lower().lstrip(".")
    fmt = "MP3" if ext == "mp3" else None
    sf.write(str(path), pcm, sample_rate, format=fmt)
