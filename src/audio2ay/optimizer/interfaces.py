"""The three decoupled optimizer interfaces.

The optimizer never knows about registers or about which search algorithm runs:

- ``MoveGenerator`` defines *what changes are legal*.
- ``Evaluator`` computes *how good* a state is (render + perceptual loss).
- ``SearchStrategy`` decides *which candidates to explore*.

See design/ARCHITECTURE.md section 4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable

from ..ay.registers import AYState


@dataclass(frozen=True)
class Move:
    """A single legal, semantic change to an ``AYState``."""

    name: str
    fn: Callable[[AYState], AYState]

    def apply(self, state: AYState) -> AYState:
        return self.fn(state).canonical()


class MoveGenerator(ABC):
    @abstractmethod
    def legal(self, state: AYState) -> Iterable[Move]:
        """Yield the legal moves available from ``state``."""


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, state: AYState) -> float:
        """Lower is better. Includes render + perceptual loss (+ temporal)."""


class SearchStrategy(ABC):
    @abstractmethod
    def optimize(self, start: AYState, moves: MoveGenerator,
                 evaluator: Evaluator) -> AYState:
        """Search from a single warm start; return the best state found."""
