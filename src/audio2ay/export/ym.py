"""YM6! register-stream export (interleaved). See design/AY_REFERENCE.md s.9.

Produces a YM6! file playable by StSound/AYM-style players. Big-endian fields,
16 registers per frame stored interleaved (all frames of r0, then r1, ...).
Export only; the canonical raw format is used for round-trips.
"""

from __future__ import annotations

import struct
from pathlib import Path

from ..ay.registers import AYState
from ..ay.timing import DEFAULT_CLOCK_HZ, DEFAULT_FPS


def write_ym(path: str | Path, states: list[AYState],
             clock_hz: int = DEFAULT_CLOCK_HZ, fps: int = DEFAULT_FPS,
             title: str = "audio2ay", author: str = "audio2ay",
             comment: str = "analysis-by-synthesis") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(states)
    regs = [st.to_registers() for st in states]

    out = bytearray()
    out += b"YM6!"
    out += b"LeOnArD!"
    out += struct.pack(">I", n)            # nb frames
    out += struct.pack(">I", 0x00000001)   # attributes: interleaved
    out += struct.pack(">H", 0)            # nb digidrums
    out += struct.pack(">I", clock_hz)     # master clock
    out += struct.pack(">H", fps)          # player frequency
    out += struct.pack(">I", 0)            # loop frame
    out += struct.pack(">H", 0)            # additional data size
    out += title.encode("latin-1", "replace") + b"\x00"
    out += author.encode("latin-1", "replace") + b"\x00"
    out += comment.encode("latin-1", "replace") + b"\x00"

    # Interleaved register data: 16 registers (14 used + 2 I/O = 0).
    for r in range(16):
        if r < 14:
            out += bytes(regs[f][r] for f in range(n))
        else:
            out += bytes(n)
    out += b"End!"

    with open(path, "wb") as fh:
        fh.write(out)
