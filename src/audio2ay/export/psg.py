"""PSG register-stream format (read/write). See design/AY_REFERENCE.md s.9.

Stream encoding:
- 0x00..0x0F  select register, next byte is its value
- 0xFE n      wait n*4 frames (interrupts)
- 0xFF        end of frame (wait one interrupt)
Only changed registers are written each frame for compactness.
"""

from __future__ import annotations

from pathlib import Path

from ..ay.registers import AYState

HEADER = b"PSG\x1a" + bytes(12)  # 16-byte header


def write_psg(path: str | Path, states: list[AYState]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = bytearray(HEADER)
    prev: bytes | None = None
    for st in states:
        regs = st.to_registers()
        for r in range(14):
            if prev is None or regs[r] != prev[r]:
                out.append(r)
                out.append(regs[r])
        out.append(0xFF)  # frame boundary
        prev = regs
    with open(path, "wb") as fh:
        fh.write(out)


def read_psg(path: str | Path) -> list[AYState]:
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:3] != b"PSG":
        raise ValueError("not a PSG stream")
    body = data[16:]
    states: list[AYState] = []
    regs = bytearray(14)
    i = 0
    n = len(body)
    while i < n:
        b = body[i]
        if b == 0xFF:  # commit one frame
            states.append(AYState.from_registers(bytes(regs)))
            i += 1
        elif b == 0xFE:  # wait count*4 frames
            count = body[i + 1] if i + 1 < n else 0
            for _ in range(count * 4):
                states.append(AYState.from_registers(bytes(regs)))
            i += 2
        elif b <= 0x0F:  # register write
            if i + 1 < n:
                if b < 14:
                    regs[b] = body[i + 1]
                i += 2
            else:
                break
        else:
            i += 1
    return states
