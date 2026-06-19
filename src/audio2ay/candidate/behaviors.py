"""Instrument *behaviors* as a prior on proposals (not hardware knowledge).

Behaviors bias the candidate generator's proposals; the optimizer never reasons
about "what a piano is". See design/PLAN.md Stage 7. Each behavior is a small
set of biases applied when constructing an AYState from features.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Behavior:
    """A prior describing how an instrument tends to use the AY."""

    name: str
    prefer_envelope: bool = False   # sustained swells -> shared envelope
    prefer_noise: bool = False      # percussive -> noise generator
    volume_bias: int = 0            # added to mapped volume (clamped later)
    octave: int = 4                 # default octave for chroma->pitch
    env_shape: int = 0x0A           # triangle by default when envelope used


# A tiny library of behaviors that shape the diverse proposals.
LEAD = Behavior("lead", prefer_envelope=False, volume_bias=1, octave=5)
BASS = Behavior("bass", prefer_envelope=True, volume_bias=0, octave=3,
                env_shape=0x08)
ARPEGGIO = Behavior("arpeggio", prefer_envelope=False, volume_bias=0, octave=4)
PERCUSSION = Behavior("percussion", prefer_noise=True, volume_bias=2, octave=4)

DEFAULT_BEHAVIORS = {
    "lead": LEAD,
    "bass": BASS,
    "arpeggio": ARPEGGIO,
    "percussion": PERCUSSION,
}
