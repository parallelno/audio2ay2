"""Candidate generator: features -> N diverse AYState proposals.

The hard part is finding a good *basin of attraction*, so we hand the optimizer
several structurally different warm starts and let multi-start keep the best.
See design/PLAN.md Stage 2 and design/ROADMAP.md Phase 3.5.
"""

from __future__ import annotations

import numpy as np

from ..ay.registers import (
    AYState, USE_ENV,
    MIX_TONE_A, MIX_NOISE_A,
)
from . import behaviors as _beh
from . import heuristics as _h


def _enable_tone(mixer: int, ch: int) -> int:
    return mixer & ~(1 << (MIX_TONE_A + ch)) & 0x3F


def _enable_noise(mixer: int, ch: int) -> int:
    return mixer & ~(1 << (MIX_NOISE_A + ch)) & 0x3F


def _aggregate(window) -> dict:
    """Collapse a window of FrameFeatures into representative scalars."""
    chroma = np.mean([f.chroma for f in window], axis=0)
    return {
        "chroma": chroma,
        "loudness": float(np.mean([f.loudness for f in window])),
        "brightness": float(np.mean([f.brightness for f in window])),
        "bass_energy": float(np.mean([f.bass_energy for f in window])),
        "transient": float(np.mean([f.transient_energy for f in window])),
        "pitch_hz": float(np.median([f.pitch_hz for f in window])),
        "pitch_confidence": float(np.mean([f.pitch_confidence for f in window])),
        "harmonic_energy": float(np.mean([f.harmonic_energy for f in window])),
    }


def _voiced_pitch_hz(agg: dict, behavior: _beh.Behavior, rank: int) -> float:
    """Pick a pitch for a voice: use detected pitch if confident, else chroma."""
    if rank == 0 and agg["pitch_confidence"] > 0.3 and agg["pitch_hz"] > 0:
        return agg["pitch_hz"]
    classes = _h.top_pitch_classes(agg["chroma"], 3)
    pc = classes[min(rank, len(classes) - 1)]
    return _h.chroma_to_hz(pc, behavior.octave)


def _voice(state_mixer: int, ch: int, hz: float, vol: int, clock_hz: int,
           use_env: bool) -> tuple[int, int, int]:
    tp = _h.hz_to_tone_period(hz, clock_hz)
    mixer = _enable_tone(state_mixer, ch)
    v = USE_ENV if use_env else vol
    return mixer, tp, v


def _melody_proposal(agg, clock_hz) -> AYState:
    beh = _beh.LEAD
    base_vol = min(15, _h.loudness_to_volume(agg["loudness"]) + beh.volume_bias)
    # Confident pitch -> a clean monophonic lead (one voice, no clutter).
    if agg["pitch_confidence"] > 0.4 and agg["pitch_hz"] > 0:
        mixer, tp, v = _voice(0x3F, 0, agg["pitch_hz"], max(1, base_vol),
                              clock_hz, False)
        return AYState(tone_period=(tp, 1, 1), mixer=mixer,
                       volume=(v, 0, 0)).canonical()
    # Otherwise spread the top chroma classes across channels.
    mixer = 0x3F
    tps = [1, 1, 1]
    vols = [0, 0, 0]
    for ch in range(3):
        hz = _voiced_pitch_hz(agg, beh, ch)
        mixer, tp, v = _voice(mixer, ch, hz, max(1, base_vol - ch * 3), clock_hz, False)
        tps[ch] = tp
        vols[ch] = v
    return AYState(tone_period=(tps[0], tps[1], tps[2]), mixer=mixer,
                   volume=(vols[0], vols[1], vols[2])).canonical()


def _bass_proposal(agg, clock_hz) -> AYState:
    beh = _beh.BASS
    vol = _h.loudness_to_volume(agg["loudness"])
    mixer = 0x3F
    # Channel A: bass with shared envelope.
    bass_hz = _voiced_pitch_hz(agg, beh, 0)
    mixer, tp_a, _ = _voice(mixer, 0, bass_hz, vol, clock_hz, True)
    # Channel B: lead on top.
    lead_hz = _voiced_pitch_hz(agg, _beh.LEAD, 0)
    mixer, tp_b, _ = _voice(mixer, 1, lead_hz, vol, clock_hz, False)
    env_period = 2000
    return AYState(tone_period=(tp_a, tp_b, 1), mixer=mixer,
                   volume=(USE_ENV, vol, 0), env_period=env_period,
                   env_shape=beh.env_shape).canonical()


def _arpeggio_proposal(agg, clock_hz) -> AYState:
    beh = _beh.ARPEGGIO
    vol = _h.loudness_to_volume(agg["loudness"])
    classes = _h.top_pitch_classes(agg["chroma"], 3)
    mixer = 0x3F
    tps = [1, 1, 1]
    vols = [0, 0, 0]
    for ch in range(3):
        pc = classes[min(ch, len(classes) - 1)]
        hz = _h.chroma_to_hz(pc, beh.octave)
        mixer, tp, v = _voice(mixer, ch, hz, vol, clock_hz, False)
        tps[ch] = tp
        vols[ch] = v
    return AYState(tone_period=(tps[0], tps[1], tps[2]), mixer=mixer,
                   volume=(vols[0], vols[1], vols[2])).canonical()


def _noise_proposal(agg, clock_hz) -> AYState:
    beh = _beh.PERCUSSION
    vol = min(15, _h.loudness_to_volume(agg["loudness"]) + beh.volume_bias)
    np_period = _h.brightness_to_noise_period(agg["brightness"])
    mixer = 0x3F
    # Channel A: noise (percussion).
    mixer = _enable_noise(mixer, 0)
    # Channel B: a tone for pitched content.
    lead_hz = _voiced_pitch_hz(agg, _beh.LEAD, 0)
    mixer, tp_b, _ = _voice(mixer, 1, lead_hz, vol, clock_hz, False)
    return AYState(tone_period=(1, tp_b, 1), noise_period=np_period,
                   mixer=mixer, volume=(vol, vol, 0)).canonical()


_BUILDERS = [_melody_proposal, _bass_proposal, _arpeggio_proposal, _noise_proposal]


def generate_proposals(window, clock_hz: int, n: int = 4) -> list[AYState]:
    """Return up to ``n`` structurally diverse proposals for a window."""
    if not window:
        return [AYState()]
    agg = _aggregate(window)
    builders = _BUILDERS[:max(1, n)]
    proposals = [b(agg, clock_hz) for b in builders]
    # If more than the canned builders are requested, add perturbed variants.
    rng = np.random.default_rng(0)
    while len(proposals) < n:
        base = proposals[len(proposals) % len(builders)]
        jitter = int(rng.integers(-2, 3))
        tp = tuple(max(1, t + jitter) for t in base.tone_period)
        proposals.append(base.with_(tone_period=(tp[0], tp[1], tp[2])).canonical())
    return proposals
