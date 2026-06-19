"""Validation report: original-vs-AY audio, loss-over-time plot, overall score.

See design/ROADMAP.md Phase 7.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..ay.registers import AYState
from ..ay.timing import Timing
from ..audioio import save_audio
from ..preview.render import render_preview
from ..similarity.loss import PerceptualLoss
from .history import save_history_plot


def per_frame_losses(orig_pcm: np.ndarray, states: list[AYState], timing: Timing,
                     loss: PerceptualLoss, chip: str = "ay") -> np.ndarray:
    ay_pcm = render_preview(states, timing, chip)
    loss.configure(orig_pcm, PerceptualLoss.loud_level(ay_pcm, timing.sample_rate))
    hop = max(1, round(timing.sample_rate / timing.fps))
    n = min(len(states), orig_pcm.shape[0] // hop, ay_pcm.shape[0] // hop)
    out = np.zeros(n)
    for i in range(n):
        a = orig_pcm[i * hop:(i + 1) * hop]
        b = ay_pcm[i * hop:(i + 1) * hop]
        if a.shape[0] < hop:
            a = np.pad(a, (0, hop - a.shape[0]))
        if b.shape[0] < hop:
            b = np.pad(b, (0, hop - b.shape[0]))
        out[i] = loss.compare(loss.features(a), b)
    return out


def write_report(out_dir: str | Path, orig_pcm: np.ndarray,
                 states: list[AYState], timing: Timing, loss: PerceptualLoss,
                 chip: str = "ay") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ay_pcm = render_preview(states, timing, chip)
    save_audio(out_dir / "original.wav", orig_pcm, timing.sample_rate)
    save_audio(out_dir / "ay_render.wav", ay_pcm, timing.sample_rate)

    losses = per_frame_losses(orig_pcm, states, timing, loss, chip)
    save_history_plot(out_dir / "loss_over_time.png", list(losses),
                      title="Per-frame perceptual loss")

    summary = {
        "frames": len(states),
        "mean_loss": float(np.mean(losses)) if losses.size else 0.0,
        "max_loss": float(np.max(losses)) if losses.size else 0.0,
        "min_loss": float(np.min(losses)) if losses.size else 0.0,
    }
    (out_dir / "summary.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in summary.items()), encoding="utf-8")
    return summary
