"""Legal discrete moves over the semantic AYState (the MoveGenerator).

Moves are expressed in musician's terms ("detune channel A", "swap A<->B",
"enable noise on C") and never touch raw registers, honouring the strict
boundary rule in design/ARCHITECTURE.md.
"""

from __future__ import annotations

from typing import Iterable

from ..ay.registers import (
    AYState, USE_ENV,
    MIX_TONE_A, MIX_NOISE_A,
    TONE_PERIOD_MIN, TONE_PERIOD_MAX,
    NOISE_PERIOD_MIN, NOISE_PERIOD_MAX,
)
from .interfaces import Move, MoveGenerator

# Common, musically useful envelope shapes to cycle through.
_ENV_SHAPES = [0x08, 0x0A, 0x0C, 0x0E, 0x09, 0x0D]


def _set_tone(state: AYState, ch: int, value: int) -> AYState:
    tp = list(state.tone_period)
    tp[ch] = max(TONE_PERIOD_MIN, min(TONE_PERIOD_MAX, value))
    return state.with_(tone_period=(tp[0], tp[1], tp[2]))


def _set_vol(state: AYState, ch: int, value) -> AYState:
    vol = list(state.volume)
    vol[ch] = value if value == USE_ENV else max(0, min(15, value))
    return state.with_(volume=(vol[0], vol[1], vol[2]))


def _toggle_mixer_bit(state: AYState, bit: int) -> AYState:
    return state.with_(mixer=(state.mixer ^ (1 << bit)) & 0x3F)


def _swap(state: AYState, i: int, j: int) -> AYState:
    tp = list(state.tone_period)
    vol = list(state.volume)
    tp[i], tp[j] = tp[j], tp[i]
    vol[i], vol[j] = vol[j], vol[i]
    # Swap the corresponding tone/noise mixer bits too.
    mix = state.mixer
    for base in (MIX_TONE_A, MIX_NOISE_A):
        bi, bj = base + i, base + j
        vi, vj = (mix >> bi) & 1, (mix >> bj) & 1
        mix &= ~(1 << bi) & ~(1 << bj) & 0x3F
        mix |= (vi << bj) | (vj << bi)
    return state.with_(tone_period=(tp[0], tp[1], tp[2]),
                       volume=(vol[0], vol[1], vol[2]), mixer=mix)


class DefaultMoves(MoveGenerator):
    """Neighborhood of small, legal edits around a state."""

    def __init__(self, pitch_steps=(1, -1, 12, -12, 50, -50)) -> None:
        self.pitch_steps = pitch_steps

    def legal(self, state: AYState) -> Iterable[Move]:
        moves: list[Move] = []
        # Pitch nudges per channel.
        for ch in range(3):
            for d in self.pitch_steps:
                moves.append(Move(
                    f"tone{ch}{'+' if d > 0 else ''}{d}",
                    (lambda s, c=ch, dd=d: _set_tone(s, c, s.tone_period[c] + dd)),
                ))
        # Volume nudges per channel.
        for ch in range(3):
            for d in (1, -1, 3, -3):
                moves.append(Move(
                    f"vol{ch}{'+' if d > 0 else ''}{d}",
                    (lambda s, c=ch, dd=d: _set_vol(
                        s, c, (0 if s.volume[c] == USE_ENV else s.volume[c]) + dd)),
                ))
            moves.append(Move(f"vol{ch}=env",
                              (lambda s, c=ch: _set_vol(s, c, USE_ENV))))
        # Mixer toggles: tone and noise per channel.
        for ch in range(3):
            moves.append(Move(f"tone{ch}_toggle",
                              (lambda s, c=ch: _toggle_mixer_bit(s, MIX_TONE_A + c))))
            moves.append(Move(f"noise{ch}_toggle",
                              (lambda s, c=ch: _toggle_mixer_bit(s, MIX_NOISE_A + c))))
        # Noise period.
        for d in (1, -1, 4, -4):
            moves.append(Move(
                f"noise{'+' if d > 0 else ''}{d}",
                (lambda s, dd=d: s.with_(
                    noise_period=max(NOISE_PERIOD_MIN,
                                     min(NOISE_PERIOD_MAX, s.noise_period + dd)))),
            ))
        # Envelope shape and period.
        for shp in _ENV_SHAPES:
            moves.append(Move(f"env_shape={shp:#x}",
                              (lambda s, v=shp: s.with_(env_shape=v))))
        for factor in (1.5, 0.66):
            moves.append(Move(
                f"env_period*{factor}",
                (lambda s, fc=factor: s.with_(
                    env_period=max(1, min(65535, int(s.env_period * fc) or 1000)))),
            ))
        # Channel swaps.
        for i, j in ((0, 1), (1, 2), (0, 2)):
            moves.append(Move(f"swap{i}{j}", (lambda s, a=i, b=j: _swap(s, a, b))))
        return moves
