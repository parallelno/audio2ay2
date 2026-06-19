"""Canonical raw register-stream format (.ay): header + frames * 14 bytes.

This is the simplest replay format and the canonical one for the pipeline.
See design/AY_REFERENCE.md section 9.
"""

from __future__ import annotations

import struct
from pathlib import Path

from ..ay.registers import AYState
from ..ay.timing import DEFAULT_CLOCK_HZ, DEFAULT_FPS

MAGIC = b"A2AY"
VERSION = 1
_HEADER = struct.Struct("<4sBIHI")  # magic, version, clock, fps, frame_count


def write_raw(path: str | Path, states: list[AYState],
              clock_hz: int = DEFAULT_CLOCK_HZ, fps: int = DEFAULT_FPS) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(_HEADER.pack(MAGIC, VERSION, clock_hz, fps, len(states)))
        for st in states:
            fh.write(st.to_registers())


def read_raw(path: str | Path) -> tuple[list[AYState], int, int]:
    """Return (states, clock_hz, fps)."""
    with open(path, "rb") as fh:
        head = fh.read(_HEADER.size)
        magic, version, clock, fps, count = _HEADER.unpack(head)
        if magic != MAGIC:
            raise ValueError("not an A2AY raw stream")
        states = [AYState.from_registers(fh.read(14)) for _ in range(count)]
    return states, clock, fps
