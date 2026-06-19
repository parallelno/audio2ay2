"""Multi-start: optimize from all N proposals, keep the best basin.

Finding a good basin of attraction matters more than polishing one guess
(design/ROADMAP.md Phase 3.5).

When an executor (ProcessPoolExecutor) is provided, proposals are evaluated in
parallel across real OS processes, bypassing the GIL.  Each worker receives a
shallow copy of the evaluator so the Renderer LRU-cache is not shared across
process boundaries (each process builds its own warm cache within a run).

The worker function ``_run_proposal`` is module-level so it is picklable on
Windows (spawn start method).
"""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

from ..ay.registers import AYState
from .interfaces import Evaluator, MoveGenerator


# ---------------------------------------------------------------------------
# Module-level worker (must be at module scope to be picklable on Windows)
# ---------------------------------------------------------------------------

def _run_proposal(seed_state: AYState,
                  evaluator: Evaluator,
                  strategy_factory,
                  moves: MoveGenerator) -> tuple[AYState, float, list[float]]:
    ev = copy.copy(evaluator)          # fresh per-process copy
    strategy = strategy_factory()
    result = strategy.optimize(seed_state, moves, ev)
    loss = ev.evaluate(result)
    history = list(getattr(strategy, "history", []))
    return result, loss, history


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def multistart(proposals: list[AYState],
               strategy_factory,
               moves: MoveGenerator,
               evaluator: Evaluator,
               executor: Optional[ProcessPoolExecutor] = None,
               workers: int = 1) -> tuple[AYState, float, list[float]]:
    """Run a fresh strategy from each proposal; return (best, loss, history).

    Args:
        executor: a pre-created ProcessPoolExecutor to reuse (preferred — avoids
                  per-call process-spawn overhead).  If None and workers != 1,
                  a temporary pool is created for this call only.
        workers:  used only when executor is None.  0 = one worker per proposal.
    """
    n = len(proposals)

    if executor is not None and n > 1:
        futures = [
            executor.submit(_run_proposal, s, evaluator, strategy_factory, moves)
            for s in proposals
        ]
        results = [f.result() for f in as_completed(futures)]

    elif workers != 1 and n > 1:
        effective = n if workers == 0 else max(2, workers)
        with ProcessPoolExecutor(max_workers=effective) as pool:
            futures = [
                pool.submit(_run_proposal, s, evaluator, strategy_factory, moves)
                for s in proposals
            ]
            results = [f.result() for f in as_completed(futures)]

    else:
        results = [
            _run_proposal(s, evaluator, strategy_factory, moves)
            for s in proposals
        ]

    return min(results, key=lambda r: r[1])
