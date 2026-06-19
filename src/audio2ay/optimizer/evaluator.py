"""The Evaluator: renders a state and scores it against the target window.

Wraps the renderer (cached) and the perceptual loss, plus an optional temporal
penalty against the previously committed state. See design/ARCHITECTURE.md s.4.
"""

from __future__ import annotations

import numpy as np

from ..ay.registers import AYState, USE_ENV
from ..ay.renderer import Renderer
from ..similarity.loss import PerceptualLoss, WindowFeatures
from .interfaces import Evaluator
from .temporal import semantic_distance


def _active_voices(state: AYState) -> int:
    """Count audible voices (tone or noise enabled with non-zero volume)."""
    count = 0
    for ch in range(3):
        tone_on = not (state.mixer >> ch) & 1
        noise_on = not (state.mixer >> (3 + ch)) & 1
        audible = state.volume[ch] == USE_ENV or state.volume[ch] > 0
        if (tone_on or noise_on) and audible:
            count += 1
    return count


class RenderEvaluator(Evaluator):
    """Score = perceptual_loss(target, render(state)) + temporal penalty."""

    def __init__(self, renderer: Renderer, loss: PerceptualLoss,
                 frames_per_eval: int = 2, temporal_weight: float = 0.0,
                 complexity_weight: float = 0.0) -> None:
        self.renderer = renderer
        self.loss = loss
        self.frames_per_eval = max(1, frames_per_eval)
        self.temporal_weight = temporal_weight
        self.complexity_weight = complexity_weight
        self.target: WindowFeatures | None = None
        self.prev: AYState | None = None
        self.evaluations = 0

    def set_target(self, target: WindowFeatures) -> None:
        self.target = target

    def set_prev(self, prev: AYState | None) -> None:
        self.prev = prev

    def render_state(self, state: AYState) -> np.ndarray:
        return self.renderer.render([state] * self.frames_per_eval)

    def evaluate(self, state: AYState) -> float:
        if self.target is None:
            raise RuntimeError("evaluator target not set")
        self.evaluations += 1
        pcm = self.render_state(state)
        score = self.loss.compare(self.target, pcm)
        if self.prev is not None and self.temporal_weight > 0.0:
            score += self.temporal_weight * semantic_distance(state, self.prev)
        if self.complexity_weight > 0.0:
            score += self.complexity_weight * max(0, _active_voices(state) - 1)
        return score
