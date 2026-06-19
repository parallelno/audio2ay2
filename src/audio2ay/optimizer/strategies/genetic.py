"""Genetic / evolutionary search strategy."""

from __future__ import annotations

import random

from ...ay.registers import AYState
from ..interfaces import Evaluator, MoveGenerator, SearchStrategy


class Genetic(SearchStrategy):
    def __init__(self, max_iters: int = 40, population: int = 12,
                 mutations: int = 3, seed: int = 0) -> None:
        self.max_iters = max_iters
        self.population = population
        self.mutations = mutations
        self.rng = random.Random(seed)
        self.history: list[float] = []

    def _mutate(self, state: AYState, moves: MoveGenerator) -> AYState:
        cand = state
        for _ in range(self.rng.randint(1, self.mutations)):
            options = list(moves.legal(cand))
            if not options:
                break
            cand = self.rng.choice(options).apply(cand)
        return cand

    def optimize(self, start: AYState, moves: MoveGenerator,
                 evaluator: Evaluator) -> AYState:
        pop = [start.canonical()]
        while len(pop) < self.population:
            pop.append(self._mutate(start, moves))
        scored = [(evaluator.evaluate(s), s) for s in pop]
        scored.sort(key=lambda x: x[0])
        best_loss, best = scored[0]
        self.history = [best_loss]
        for _ in range(self.max_iters):
            survivors = [s for _, s in scored[: max(2, self.population // 2)]]
            children: list[AYState] = []
            while len(children) < self.population - len(survivors):
                parent = self.rng.choice(survivors)
                children.append(self._mutate(parent, moves))
            pop = survivors + children
            scored = [(evaluator.evaluate(s), s) for s in pop]
            scored.sort(key=lambda x: x[0])
            if scored[0][0] < best_loss - 1e-9:
                best_loss, best = scored[0]
            self.history.append(best_loss)
        return best
