"""Beam search strategy: keep the best K states, expand their neighborhoods."""

from __future__ import annotations

from ...ay.registers import AYState
from ..interfaces import Evaluator, MoveGenerator, SearchStrategy


class BeamSearch(SearchStrategy):
    def __init__(self, max_iters: int = 40, beam_width: int = 6) -> None:
        self.max_iters = max_iters
        self.beam_width = beam_width
        self.history: list[float] = []

    def optimize(self, start: AYState, moves: MoveGenerator,
                 evaluator: Evaluator) -> AYState:
        beam = [(evaluator.evaluate(start.canonical()), start.canonical())]
        best_loss, best = beam[0]
        self.history = [best_loss]
        for _ in range(self.max_iters):
            pool: dict[bytes, tuple[float, AYState]] = {}
            for _, state in beam:
                for move in moves.legal(state):
                    cand = move.apply(state)
                    key = cand.to_registers()
                    if key in pool:
                        continue
                    pool[key] = (evaluator.evaluate(cand), cand)
            if not pool:
                break
            ranked = sorted(pool.values(), key=lambda x: x[0])[:self.beam_width]
            beam = ranked
            if ranked[0][0] < best_loss - 1e-9:
                best_loss, best = ranked[0]
            self.history.append(best_loss)
        return best
