"""Interchangeable search strategies (plugins behind SearchStrategy)."""

from dataclasses import dataclass

from .local_search import LocalSearch
from .simulated_annealing import SimulatedAnnealing
from .beam_search import BeamSearch
from .genetic import Genetic

STRATEGIES = {
    "local": LocalSearch,
    "annealing": SimulatedAnnealing,
    "beam": BeamSearch,
    "genetic": Genetic,
}

__all__ = ["LocalSearch", "SimulatedAnnealing", "BeamSearch", "Genetic",
           "STRATEGIES", "StrategySpec"]


def make_strategy(name: str, max_iters: int, seed: int = 0):
    """Construct a strategy by name with sensible per-strategy iteration budgets."""
    if name == "local":
        return LocalSearch(max_iters=max_iters)
    if name == "annealing":
        return SimulatedAnnealing(max_iters=max_iters, seed=seed)
    if name == "beam":
        return BeamSearch(max_iters=max(10, max_iters // 5))
    if name == "genetic":
        return Genetic(max_iters=max(10, max_iters // 5), seed=seed)
    raise ValueError(f"unknown optimizer strategy: {name}")


@dataclass
class StrategySpec:
    """Picklable callable that constructs a strategy — safe to send to subprocesses."""
    name: str
    max_iters: int
    seed: int = 0

    def __call__(self):
        return make_strategy(self.name, self.max_iters, seed=self.seed)
