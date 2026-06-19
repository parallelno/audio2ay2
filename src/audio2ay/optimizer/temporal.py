"""Temporal continuity: semantic distance between successive AYStates.

Computed on the semantic fields (not registers) so the strict boundary rule
holds. Used as a smoothness penalty to discourage thrashing. See
design/PLAN.md Stage 6.

The scale matters: the penalty must be able to break ties between candidates the
perceptual loss rates as near-equal (a held note sounds the same at period 423 or
427), otherwise the optimizer picks a random nearby value every frame and the
result audibly jitters. So this returns an *un-averaged* sum dominated by pitch:
a one-semitone wobble on an audible channel costs ~0.083, an octave ~1.0.
"""

from __future__ import annotations

import math

from ..ay.registers import AYState, USE_ENV


def _audible(state: AYState, ch: int) -> bool:
    on = state.tone_on(ch) or state.noise_on(ch)
    vol = state.volume[ch]
    return on and (vol == USE_ENV or vol > 0)


def semantic_distance(a: AYState, b: AYState) -> float:
    """Smoothness cost between two states; pitch-dominant, ~0 when unchanged."""
    d = 0.0
    for ch in range(3):
        a_on = _audible(a, ch)
        b_on = _audible(b, ch)
        # Pitch change on channels audible in either state (log-pitch, capped 1 oct).
        if a_on or b_on:
            semis = abs(math.log2(max(1, a.tone_period[ch]))
                        - math.log2(max(1, b.tone_period[ch]))) * 12.0
            d += min(semis, 12.0) / 12.0
        # Structural change: a voice turning on/off (channel hopping, dropouts).
        if a_on != b_on:
            d += 0.5
        # Volume change (mild).
        va = 15.5 if a.volume[ch] == USE_ENV else a.volume[ch]
        vb = 15.5 if b.volume[ch] == USE_ENV else b.volume[ch]
        d += 0.3 * abs(va - vb) / 16.0
    # Noise period and envelope shape changes (mild).
    d += 0.2 * abs(a.noise_period - b.noise_period) / 31.0
    if a.env_shape != b.env_shape:
        d += 0.2
    return d
