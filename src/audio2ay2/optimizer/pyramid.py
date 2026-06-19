"""Coarse-to-fine (multi-resolution) scheduler.

Optimize at coarse time blocks first (e.g. 160 ms), then refine down to 20 ms,
carrying each block's winner into its children as the warm start. This escapes
bad local minima before fine detail is committed (design/ARCHITECTURE.md s.5).
"""

from __future__ import annotations

import numpy as np

from ..ay.registers import AYState
from ..ay.renderer import Renderer
from ..ay.timing import Timing
from ..candidate import generate_proposals
from ..similarity.loss import PerceptualLoss
from .evaluator import RenderEvaluator
from .interfaces import MoveGenerator
from .multistart import multistart


def _blocks(n: int, size: int):
    for start in range(0, n, size):
        yield start, min(start + size, n)


def run_pyramid(pcm: np.ndarray, features: list, timing: Timing,
                renderer: Renderer, loss: PerceptualLoss,
                moves: MoveGenerator, strategy_factory,
                proposals: int, pyramid_ms: tuple[int, ...],
                temporal_weight: float,
                complexity_weight: float = 0.0) -> tuple[list[AYState], list[float]]:
    """Return (per-frame states, per-block final losses at the finest level)."""
    n = len(features)
    hop = max(1, round(timing.sample_rate / timing.fps))
    frame_ms = 1000.0 / timing.fps

    scales = pyramid_ms if pyramid_ms else (int(frame_ms),)
    states: list[AYState] = [AYState() for _ in range(n)]
    finest_losses: list[float] = []

    evaluator = RenderEvaluator(renderer, loss, frames_per_eval=1,
                                temporal_weight=temporal_weight,
                                complexity_weight=complexity_weight)

    for level, scale_ms in enumerate(scales):
        block = max(1, round(scale_ms / frame_ms))
        is_finest = (level == len(scales) - 1)
        prev_state: AYState | None = None
        level_losses: list[float] = []
        for b0, b1 in _blocks(n, block):
            window = features[b0:b1]
            s0, s1 = b0 * hop, min(b1 * hop, pcm.shape[0])
            target_pcm = pcm[s0:s1]
            if target_pcm.shape[0] < hop:
                target_pcm = np.pad(target_pcm, (0, hop - target_pcm.shape[0]))
            evaluator.set_target(loss.features(target_pcm))
            evaluator.frames_per_eval = max(1, b1 - b0)
            evaluator.set_prev(prev_state)

            if level == 0:
                seeds = generate_proposals(window, timing.clock_hz, n=proposals)
            else:
                # Warm start from the parent winner AND fresh per-block proposals,
                # so a frame can escape a poor coarse compromise (e.g. recover a
                # clean monophonic lead at a note transition).
                seeds = [states[b0]] + generate_proposals(
                    window, timing.clock_hz, n=proposals)

            best, best_loss, _ = multistart(seeds, strategy_factory, moves, evaluator)
            for i in range(b0, b1):
                states[i] = best
            prev_state = best
            level_losses.append(best_loss)
        if is_finest:
            finest_losses = level_losses
    return states, finest_losses
