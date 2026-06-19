"""Post-optimization stabilization: remove single-frame register flicker.

The perceptual loss has broad flat minima (a held note sounds the same at tone
period 423 or 427; the AY's coarse 16-step volume forces dithering between two
levels). Independently optimizing each frame therefore yields output that wobbles
by one step every frame, heard as pitch/amplitude jitter. Two complementary
passes de-jitter the stream without crossing the strict register boundary (both
operate on semantic AYState fields):

1. A short per-channel *median* filter removes isolated 1-frame deviations while
   preserving any change that lasts two or more frames (real onsets/dynamics).
2. A per-channel *hysteresis hold* then collapses the residual 2-value dither
   inside a stable note: each channel holds its current period/volume until the
   value changes by more than a small tolerance (sub-semitone for pitch, one step
   for volume), so alternating patterns the median cannot fix become constant.

Both passes are *masked by audibility*: only frames where a channel is actually
sounding are smoothed, and only using other sounding frames as neighbours. This
preserves each channel's on/off structure so silent channels are never revived
into spurious extra voices.
"""

from __future__ import annotations

import math

from ..ay.registers import AYState, USE_ENV


def _audible(s: AYState, ch: int) -> bool:
    on = s.tone_on(ch) or s.noise_on(ch)
    vol = s.volume[ch]
    return on and (vol == USE_ENV or vol > 0)


def _masked_median(values: list[int], audible: list[bool], width: int
                   ) -> list[int]:
    """Median-smooth only audible frames, using audible neighbours; keep zeros."""
    half = width // 2
    n = len(values)
    out = list(values)
    for i in range(n):
        if not audible[i]:
            continue
        window = [values[j] for j in range(max(0, i - half), min(n, i + half + 1))
                  if audible[j]]
        if len(window) >= 2:
            window.sort()
            out[i] = int(round(window[len(window) // 2]))
    return out


def _hysteresis_hold(tone: list[int], vol: list[int], audible: list[bool],
                     semis_tol: float = 0.5, vol_tol: float = 1.0
                     ) -> tuple[list[int], list[int]]:
    """Hold period/volume across an audible run until it changes beyond tolerance."""
    n = len(tone)
    t_out = list(tone)
    v_out = list(vol)
    held_tp: int | None = None
    held_v: int | None = None
    for i in range(n):
        if not audible[i]:
            held_tp = held_v = None
            continue
        if held_tp is None:
            held_tp, held_v = tone[i], vol[i]
            continue
        semis = abs(math.log2(max(1, tone[i])) - math.log2(max(1, held_tp))) * 12.0
        if semis <= semis_tol:
            t_out[i] = held_tp
        else:
            held_tp = tone[i]
        hv = 15.5 if held_v == USE_ENV else held_v
        cv = 15.5 if vol[i] == USE_ENV else vol[i]
        if abs(cv - hv) <= vol_tol:
            v_out[i] = held_v
        else:
            held_v = vol[i]
    return t_out, v_out


def stabilize(states: list[AYState], width: int = 3) -> list[AYState]:
    """De-jitter the stream: per-channel median filter then hysteresis hold."""
    if width < 3 or len(states) < width:
        return states
    audible = [[_audible(s, ch) for s in states] for ch in range(3)]
    tone = [
        _masked_median([s.tone_period[ch] for s in states], audible[ch], width)
        for ch in range(3)
    ]
    vol = [
        _masked_median([s.volume[ch] for s in states], audible[ch], width)
        for ch in range(3)
    ]
    for ch in range(3):
        tone[ch], vol[ch] = _hysteresis_hold(tone[ch], vol[ch], audible[ch])
    out: list[AYState] = []
    for i, s in enumerate(states):
        out.append(s.with_(
            tone_period=(tone[0][i], tone[1][i], tone[2][i]),
            volume=(vol[0][i], vol[1][i], vol[2][i]),
        ).canonical())
    return out
