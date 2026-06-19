"""Frame/window rendering with caching.

A stateless, cache-friendly wrapper around :class:`AYEmulator`. Because the
optimizer renders enormous numbers of short windows (often identical), we cache
PCM keyed by the exact register bytes of the window's states. For
context-sensitive renders (generators free-run across frames) the optimizer
typically evaluates short, fresh windows, so a per-window emulator is correct.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .emulator import AYEmulator
from .registers import AYState
from .timing import Timing


class Renderer:
    """Render windows of :class:`AYState` to PCM, with an LRU cache."""

    def __init__(self, timing: Timing | None = None, chip: str = "ay",
                 cache_size: int = 1 << 16) -> None:
        self.timing = timing or Timing()
        self.chip = chip
        self._cached = lru_cache(maxsize=cache_size)(self._render_key)

    def _render_key(self, key: tuple[bytes, ...]) -> np.ndarray:
        states = [AYState.from_registers(b) for b in key]
        emu = AYEmulator(timing=self.timing, chip=self.chip)
        return emu.render(states)

    def render(self, states: AYState | list[AYState]) -> np.ndarray:
        if isinstance(states, AYState):
            states = [states]
        key = tuple(st.to_registers() for st in states)
        return self._cached(key)

    def cache_info(self):
        return self._cached.cache_info()
