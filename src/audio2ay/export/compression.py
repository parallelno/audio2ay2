"""Optional delta/RLE compression helpers for register streams.

Most registers change rarely, so a per-register run-length encoding of the
14xN register matrix is compact. Provided as a utility; the raw/psg/ym writers
remain the canonical outputs. See design/AY_REFERENCE.md section 9.
"""

from __future__ import annotations

from ..ay.registers import AYState


def rle_encode(states: list[AYState]) -> list[tuple[int, int, int]]:
    """Encode as (register, value, run_length) triples per register lane."""
    if not states:
        return []
    matrix = [st.to_registers() for st in states]
    out: list[tuple[int, int, int]] = []
    for r in range(14):
        run_val = matrix[0][r]
        run_len = 1
        for f in range(1, len(matrix)):
            v = matrix[f][r]
            if v == run_val:
                run_len += 1
            else:
                out.append((r, run_val, run_len))
                run_val, run_len = v, 1
        out.append((r, run_val, run_len))
    return out


def rle_ratio(states: list[AYState]) -> float:
    """Compression ratio of RLE triples vs. raw 14*N bytes (lower is better)."""
    if not states:
        return 1.0
    raw = len(states) * 14
    enc = len(rle_encode(states)) * 3
    return enc / raw
