"""Simulated annealing search strategy."""

from __future__ import annotations

import math
import random

from ...ay.registers import AYState
from ..interfaces import Evaluator, MoveGenerator, SearchStrategy


class SimulatedAnnealing(SearchStrategy):
    def __init__(self, max_iters: int = 200, t0: float = 0.5,
                 cooling: float = 0.97, seed: int = 0) -> None:
        self.max_iters = max_iters
        self.t0 = t0
        self.cooling = cooling
        self.rng = random.Random(seed)
        self.history: list[float] = []

    def optimize(self, start: AYState, moves: MoveGenerator,
                 evaluator: Evaluator) -> AYState:
        current = start.canonical()
        current_loss = evaluator.evaluate(current)
        best, best_loss = current, current_loss
        self.history = [best_loss]
        temp = self.t0
        for _ in range(self.max_iters):
            candidates = list(moves.legal(current))
            if not candidates:
                break
            move = self.rng.choice(candidates)
            cand = move.apply(current)
            loss = evaluator.evaluate(cand)
            delta = loss - current_loss
            if delta < 0 or self.rng.random() < math.exp(-delta / max(temp, 1e-6)):
                current, current_loss = cand, loss
                if loss < best_loss:
                    best, best_loss = cand, loss
            temp *= self.cooling
            self.history.append(best_loss)
        return best
