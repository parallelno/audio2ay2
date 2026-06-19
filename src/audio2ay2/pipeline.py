"""High-level pipeline: audio -> register stream, and stream -> preview.

Wires analysis, candidate proposals, the coarse-to-fine multi-start optimizer,
and export. See design/PLAN.md (whole pipeline) and design/ARCHITECTURE.md s.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analysis import extract_features
from .ay.registers import AYState
from .ay.renderer import Renderer
from .ay.timing import Timing
from .config import Config
from .optimizer.moves import DefaultMoves
from .optimizer.pyramid import run_pyramid
from .optimizer.stabilize import stabilize
from .optimizer.strategies import make_strategy, StrategySpec
from .similarity.loss import PerceptualLoss


@dataclass
class ConvertResult:
    states: list[AYState]
    timing: Timing
    finest_losses: list[float]


def ay_fullscale_rms(renderer: Renderer) -> float:
    """RMS of a full-volume single-tone AY render: the candidate reference level."""
    ref = AYState(tone_period=(200, 1, 1), mixer=0b111110,
                  volume=(15, 0, 0)).canonical()
    pcm = renderer.render([ref] * 4)
    return float(np.sqrt(np.mean(pcm ** 2) + 1e-12))


def convert_pcm(pcm: np.ndarray, timing: Timing, config: Config) -> ConvertResult:
    """Run the full analysis-by-synthesis conversion on mono PCM."""
    features = extract_features(pcm, timing.sample_rate, timing.fps)
    loss = PerceptualLoss(timing.sample_rate, config.resolved_weights())
    renderer = Renderer(timing=timing, chip=config.chip)
    loss.configure(pcm, ay_fullscale_rms(renderer))
    moves = DefaultMoves()

    strategy_factory = StrategySpec(config.optimizer, config.iters, seed=config.seed)

    states, finest_losses = run_pyramid(
        pcm=pcm, features=features, timing=timing, renderer=renderer, loss=loss,
        moves=moves, strategy_factory=strategy_factory,
        proposals=config.proposals, pyramid_ms=tuple(config.pyramid_ms),
        temporal_weight=config.temporal_weight,
        complexity_weight=config.complexity_weight,
        workers=config.workers,
    )
    states = stabilize(states, width=config.stabilize)
    return ConvertResult(states=states, timing=timing, finest_losses=finest_losses)
