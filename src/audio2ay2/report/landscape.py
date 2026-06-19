"""Loss-landscape diagnostics: sweep one field by +/-N and plot the loss.

Tells you whether the search surface is smooth or jagged *before* blaming the
optimizer (design/ROADMAP.md Phase 7, the +/-1 register probe).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..ay.registers import AYState
from ..optimizer.evaluator import RenderEvaluator
from ..similarity.loss import WindowFeatures


def sweep_tone_period(evaluator: RenderEvaluator, state: AYState,
                      target: WindowFeatures, ch: int = 0,
                      span: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Return (deltas, losses) sweeping channel ``ch`` tone period by +/-span."""
    evaluator.set_target(target)
    evaluator.set_prev(None)
    base = state.tone_period[ch]
    deltas = np.arange(-span, span + 1)
    losses = np.zeros(deltas.shape[0])
    for i, d in enumerate(deltas):
        tp = list(state.tone_period)
        tp[ch] = max(1, base + int(d))
        cand = state.with_(tone_period=(tp[0], tp[1], tp[2])).canonical()
        losses[i] = evaluator.evaluate(cand)
    return deltas, losses


def save_landscape_plot(path: str | Path, deltas: np.ndarray,
                        losses: np.ndarray,
                        title: str = "Tone-period loss landscape") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(deltas, losses, marker=".", lw=1.0)
    ax.set_xlabel("tone-period delta")
    ax.set_ylabel("loss")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
