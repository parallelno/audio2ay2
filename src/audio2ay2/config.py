"""Defaults, loss profiles and seeding. See design/LOSS.md section 3."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

# Loss weight presets (term -> weight). Terms match similarity/loss.py.
PROFILES: dict[str, dict[str, float]] = {
    "balanced": {
        "mel": 1.0, "pitch": 1.0, "loudness": 0.5, "centroid": 0.5,
        "chroma": 0.6, "harmonic": 0.4, "transient": 0.5,
    },
    "melodic": {
        "mel": 0.8, "pitch": 1.5, "loudness": 0.3, "centroid": 0.3,
        "chroma": 1.0, "harmonic": 0.6, "transient": 0.2,
    },
    "percussive": {
        "mel": 1.0, "pitch": 0.3, "loudness": 0.5, "centroid": 0.6,
        "chroma": 0.1, "harmonic": 0.1, "transient": 1.2,
    },
}

DEFAULT_PROFILE = "balanced"

# Temporal smoothness weight (penalizes register thrashing between frames).
# With the pitch-dominant semantic_distance, ~0.3 makes a 1-semitone wobble
# (~0.025 penalty) outweigh the perceptual loss's flat-minimum noise (~0.01),
# so held notes stay stable while real note changes still win.
DEFAULT_TEMPORAL_WEIGHT = 0.3

# Complexity penalty: discourages gratuitous extra audible voices (dissonance).
DEFAULT_COMPLEXITY_WEIGHT = 0.05

# Post-optimization median smoothing width (frames) to de-jitter held notes.
# 0 or 1 disables; 3 removes isolated single-frame flicker.
DEFAULT_STABILIZE = 3

# Optimizer defaults.
DEFAULT_PROPOSALS = 4
DEFAULT_PYRAMID_MS = (160, 80, 40, 20)
DEFAULT_WINDOW = 5
DEFAULT_ITERS = 200


@dataclass
class Config:
    """Run configuration threaded through the pipeline."""

    profile: str = DEFAULT_PROFILE
    optimizer: str = "annealing"
    proposals: int = DEFAULT_PROPOSALS
    pyramid_ms: tuple[int, ...] = DEFAULT_PYRAMID_MS
    window: int = DEFAULT_WINDOW
    iters: int = DEFAULT_ITERS
    temporal_weight: float = DEFAULT_TEMPORAL_WEIGHT
    complexity_weight: float = DEFAULT_COMPLEXITY_WEIGHT
    stabilize: int = DEFAULT_STABILIZE
    seed: int = 0
    chip: str = "ay"
    workers: int = 0
    weights: dict[str, float] = field(default_factory=dict)

    def resolved_weights(self) -> dict[str, float]:
        if self.weights:
            return self.weights
        return dict(PROFILES.get(self.profile, PROFILES[DEFAULT_PROFILE]))


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python and NumPy RNGs; return a NumPy Generator for the run."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
