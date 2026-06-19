"""Optional stem separation (Demucs). Off by default.

See design/PLAN.md Stage 1. Demucs is an optional dependency; when unavailable
or disabled this returns the input unchanged so the pipeline still runs.
"""

from __future__ import annotations

import numpy as np


def separate(pcm: np.ndarray, sample_rate: int, mode: str = "off"
             ) -> dict[str, np.ndarray]:
    """Return a dict of stems. With ``mode='off'`` returns {'mix': pcm}."""
    if mode == "off":
        return {"mix": pcm}
    try:
        import torch  # noqa: F401
        from demucs.apply import apply_model  # noqa: F401
    except Exception:
        # Graceful degradation: behave as if separation were off.
        return {"mix": pcm}
    # A full Demucs integration is deferred (Phase 7+); fall back for now.
    return {"mix": pcm}
