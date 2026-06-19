"""AYState - the semantic, register-free representation of one AY frame.

This is the *only* representation the analysis, candidate, optimizer and
temporal layers ever touch. Raw registers R0..R13 exist solely at the export /
import boundary via :meth:`AYState.to_registers` / :meth:`AYState.from_registers`.
See design/ARCHITECTURE.md section 3 ("Strict boundary rule").
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# Channel indices.
CH_A, CH_B, CH_C = 0, 1, 2

# Volume sentinel: this channel's amplitude follows the envelope generator.
USE_ENV = 16

# Field limits (see design/AY_REFERENCE.md).
TONE_PERIOD_MIN, TONE_PERIOD_MAX = 1, 4095
NOISE_PERIOD_MIN, NOISE_PERIOD_MAX = 1, 31
ENV_PERIOD_MIN, ENV_PERIOD_MAX = 0, 65535
ENV_SHAPE_MIN, ENV_SHAPE_MAX = 0, 15

# Mixer (R7) bit layout, active LOW (0 = enabled).
MIX_TONE_A, MIX_TONE_B, MIX_TONE_C = 0, 1, 2
MIX_NOISE_A, MIX_NOISE_B, MIX_NOISE_C = 3, 4, 5


def _clamp(value: int, lo: int, hi: int) -> int:
    return lo if value < lo else hi if value > hi else value


@dataclass(frozen=True)
class AYState:
    """One legal AY-3-8910 frame, expressed semantically.

    Attributes:
        tone_period: 12-bit tone period for channels A, B, C (1..4095).
        noise_period: 5-bit shared noise period (1..31).
        mixer: R7 mixer bits, active low (0 = enabled).
        volume: per-channel amplitude 0..15, or ``USE_ENV`` to follow envelope.
        env_period: 16-bit shared envelope period (0..65535).
        env_shape: 4-bit envelope shape selector (0..15).
    """

    tone_period: tuple[int, int, int] = (TONE_PERIOD_MIN,) * 3
    noise_period: int = NOISE_PERIOD_MIN
    mixer: int = 0b00111111  # everything disabled (all bits high)
    volume: tuple[int, int, int] = (0, 0, 0)
    env_period: int = 0
    env_shape: int = 0

    # -- mixer helpers (semantic, no raw bit fiddling upstream) -----------

    def tone_on(self, ch: int) -> bool:
        return not (self.mixer >> ch) & 1

    def noise_on(self, ch: int) -> bool:
        return not (self.mixer >> (MIX_NOISE_A + ch)) & 1

    # -- canonical / legality --------------------------------------------

    def canonical(self) -> "AYState":
        """Return a copy with every field clamped into legal range."""
        tp = tuple(
            _clamp(int(p), TONE_PERIOD_MIN, TONE_PERIOD_MAX) for p in self.tone_period
        )
        vol = tuple(
            USE_ENV if v == USE_ENV else _clamp(int(v), 0, 15) for v in self.volume
        )
        return AYState(
            tone_period=(tp[0], tp[1], tp[2]),
            noise_period=_clamp(int(self.noise_period), NOISE_PERIOD_MIN, NOISE_PERIOD_MAX),
            mixer=int(self.mixer) & 0x3F,
            volume=(vol[0], vol[1], vol[2]),
            env_period=_clamp(int(self.env_period), ENV_PERIOD_MIN, ENV_PERIOD_MAX),
            env_shape=_clamp(int(self.env_shape), ENV_SHAPE_MIN, ENV_SHAPE_MAX),
        )

    def with_(self, **changes) -> "AYState":
        """Functional update (frozen dataclass)."""
        return replace(self, **changes)

    # -- register boundary (ONLY exporter / importer cross this) ---------

    def to_registers(self) -> bytes:
        """Encode to the 14 raw registers R0..R13."""
        s = self.canonical()
        regs = bytearray(14)
        for ch in range(3):
            tp = s.tone_period[ch]
            regs[ch * 2] = tp & 0xFF
            regs[ch * 2 + 1] = (tp >> 8) & 0x0F
        regs[6] = s.noise_period & 0x1F
        regs[7] = s.mixer & 0x3F
        for ch in range(3):
            v = s.volume[ch]
            regs[8 + ch] = 0x10 if v == USE_ENV else (v & 0x0F)
        regs[11] = s.env_period & 0xFF
        regs[12] = (s.env_period >> 8) & 0xFF
        regs[13] = s.env_shape & 0x0F
        return bytes(regs)

    @classmethod
    def from_registers(cls, b: bytes) -> "AYState":
        """Decode a 14-byte register dump back into a semantic state."""
        if len(b) < 14:
            raise ValueError(f"register dump must be >= 14 bytes, got {len(b)}")
        tone = (
            (b[1] & 0x0F) << 8 | b[0],
            (b[3] & 0x0F) << 8 | b[2],
            (b[5] & 0x0F) << 8 | b[4],
        )
        tone = tuple(max(TONE_PERIOD_MIN, t) for t in tone)
        vol = tuple(USE_ENV if (b[8 + ch] & 0x10) else (b[8 + ch] & 0x0F) for ch in range(3))
        return cls(
            tone_period=(tone[0], tone[1], tone[2]),
            noise_period=max(NOISE_PERIOD_MIN, b[6] & 0x1F),
            mixer=b[7] & 0x3F,
            volume=(vol[0], vol[1], vol[2]),
            env_period=(b[12] << 8) | b[11],
            env_shape=b[13] & 0x0F,
        )


# A full stream is one state per 50 Hz frame.
RegisterStream = list[AYState]
