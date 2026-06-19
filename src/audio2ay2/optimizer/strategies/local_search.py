"""Greedy hill-climbing search strategy. Ships first (design/ARCHITECTURE.md)."""

from __future__ import annotations

from ...ay.registers import AYState
from ..interfaces import Evaluator, MoveGenerator, SearchStrategy


class LocalSearch(SearchStrategy):
    def __init__(self, max_iters: int = 200) -> None:
        self.max_iters = max_iters
        self.history: list[float] = []

    def optimize(self, start: AYState, moves: MoveGenerator,
                 evaluator: Evaluator) -> AYState:
        best = start.canonical()
        best_loss = evaluator.evaluate(best)
        self.history = [best_loss]
        for _ in range(self.max_iters):
            improved = False
            for move in moves.legal(best):
                cand = move.apply(best)
                loss = evaluator.evaluate(cand)
                if loss < best_loss - 1e-9:
                    best, best_loss, improved = cand, loss, True
            self.history.append(best_loss)
            if not improved:
                break
        return best
