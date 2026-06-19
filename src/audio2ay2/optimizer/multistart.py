"""Multi-start: optimize from all N proposals, keep the best basin.

Finding a good basin of attraction matters more than polishing one guess
(design/ROADMAP.md Phase 3.5).
"""

from __future__ import annotations

from ..ay.registers import AYState
from .interfaces import Evaluator, MoveGenerator, SearchStrategy


def multistart(proposals: list[AYState], strategy_factory, moves: MoveGenerator,
               evaluator: Evaluator) -> tuple[AYState, float, list[float]]:
    """Run a fresh strategy from each proposal; return (best, loss, history)."""
    best: AYState | None = None
    best_loss = float("inf")
    best_history: list[float] = []
    for seed_state in proposals:
        strategy: SearchStrategy = strategy_factory()
        result = strategy.optimize(seed_state, moves, evaluator)
        loss = evaluator.evaluate(result)
        if loss < best_loss:
            best, best_loss = result, loss
            best_history = list(getattr(strategy, "history", []))
    if best is None:
        best = proposals[0]
        best_loss = evaluator.evaluate(best)
    return best, best_loss, best_history
