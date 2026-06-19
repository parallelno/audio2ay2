"""Register-stream export/import in raw / PSG / YM formats."""

from __future__ import annotations

from pathlib import Path

from ..ay.registers import AYState
from .register_stream import write_raw, read_raw
from .psg import write_psg, read_psg
from .ym import write_ym

__all__ = ["write_raw", "read_raw", "write_psg", "read_psg", "write_ym",
           "write_stream", "read_stream"]


def write_stream(path, states: list[AYState], fmt: str, clock_hz: int,
                 fps: int) -> None:
    if fmt == "raw":
        write_raw(path, states, clock_hz, fps)
    elif fmt == "psg":
        write_psg(path, states)
    elif fmt == "ym":
        write_ym(path, states, clock_hz, fps)
    else:
        raise ValueError(f"unknown format: {fmt}")


def read_stream(path) -> tuple[list[AYState], int, int]:
    """Read a stream, detecting the format. Returns (states, clock_hz, fps)."""
    path = Path(path)
    with open(path, "rb") as fh:
        head = fh.read(4)
    if head == b"A2AY":
        return read_raw(path)
    if head[:3] == b"PSG":
        from ..ay.timing import DEFAULT_CLOCK_HZ, DEFAULT_FPS
        return read_psg(path), DEFAULT_CLOCK_HZ, DEFAULT_FPS
    raise ValueError(f"unrecognized stream format for {path}")
