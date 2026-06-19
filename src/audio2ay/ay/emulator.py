"""Cycle-accurate(-enough) AY-3-8910 reference emulator (pure NumPy).

This is the *reference* backend described in design/ROADMAP.md Phase 1: a
vectorized, dependency-light emulator that is correct and serves as the test
oracle. A native (Rust) backend can later replace it behind the same
``render`` interface without the optimizer noticing.

Models: 3 tone generators (50% square), a shared 17-bit LFSR noise generator,
a shared envelope generator (all 16 R13 shapes), the active-low mixer, and a
logarithmic 16-step volume DAC. Registers latch on 50 Hz frame boundaries while
the generators free-run; audio is rendered at the configured sample rate.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .registers import AYState, USE_ENV
from .timing import Timing

# ---------------------------------------------------------------------------
# Volume DAC: logarithmic 16-step table (normalized 0..1), ~3 dB/step.
# Values follow the commonly used measured AY-3-8910 curve.
# ---------------------------------------------------------------------------
_AY_VOLUME_TABLE = np.array(
    [
        0.0000, 0.0137, 0.0205, 0.0291, 0.0423, 0.0618, 0.0847, 0.1369,
        0.1691, 0.2647, 0.3527, 0.4499, 0.5704, 0.6873, 0.8482, 1.0000,
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Noise: precompute the full 17-bit LFSR output sequence once. Taps at bits
# 0 and 3; period 2**17 - 1. The noise *rate* (period register) only controls
# how fast we advance through this fixed sequence.
# ---------------------------------------------------------------------------
def _build_lfsr_sequence() -> np.ndarray:
    length = (1 << 17) - 1
    seq = np.empty(length, dtype=np.float64)
    rng = 1
    for i in range(length):
        seq[i] = rng & 1
        new_bit = (rng ^ (rng >> 3)) & 1
        rng = (rng >> 1) | (new_bit << 16)
    return seq


_LFSR_SEQ = _build_lfsr_sequence()
_LFSR_LEN = _LFSR_SEQ.shape[0]

# One-pole DC-blocking high-pass (AC coupling). y[n] = x[n] - x[n-1] + R*y[n-1].
# R = 0.9995 -> ~3.5 Hz cutoff at 44.1 kHz: removes DC, preserves all musical tones.
_DC_BLOCK_R = 0.9995
_DC_BLOCK_B = np.array([1.0, -1.0])
_DC_BLOCK_A = np.array([1.0, -_DC_BLOCK_R])


# ---------------------------------------------------------------------------
# Envelope: per-shape amplitude as a function of step index (0..15 levels).
# Shapes 0-7 collapse onto 9 (\___) or 15 (/___) per the datasheet.
# ---------------------------------------------------------------------------
_RAMP_DOWN = list(range(15, -1, -1))
_RAMP_UP = list(range(0, 16))


def _envelope_amplitude(shape: int, step_index: np.ndarray) -> np.ndarray:
    """Vectorized envelope amplitude (0..15) for integer ``step_index``."""
    shape &= 0x0F
    if shape < 8:
        shape = 9 if not (shape & 0x04) else 15

    s = step_index.astype(np.int64)
    if shape == 8:  # \\\\  repeating down
        return np.array(_RAMP_DOWN)[s % 16].astype(np.float64)
    if shape == 12:  # //// repeating up
        return np.array(_RAMP_UP)[s % 16].astype(np.float64)
    if shape == 10:  # \/\/ down-up triangle
        cyc = np.array(_RAMP_DOWN + _RAMP_UP)
        return cyc[s % 32].astype(np.float64)
    if shape == 14:  # /\/\ up-down triangle
        cyc = np.array(_RAMP_UP + _RAMP_DOWN)
        return cyc[s % 32].astype(np.float64)
    # Single-ramp + hold shapes: 9 (\__0), 11 (\__15), 13 (/__15), 15 (/__0)
    if shape == 9:
        ramp, hold = _RAMP_DOWN, 0
    elif shape == 11:
        ramp, hold = _RAMP_DOWN, 15
    elif shape == 13:
        ramp, hold = _RAMP_UP, 15
    else:  # 15
        ramp, hold = _RAMP_UP, 0
    out = np.full(s.shape, float(hold), dtype=np.float64)
    in_ramp = s < 16
    out[in_ramp] = np.array(ramp)[s[in_ramp]].astype(np.float64)
    return out


class AYEmulator:
    """Stateful AY emulator. Call :meth:`render` per frame or per window.

    The instance carries generator phase so successive renders are continuous.
    Use a fresh instance (or :meth:`reset`) to render an independent stream.
    """

    def __init__(self, timing: Timing | None = None, chip: str = "ay") -> None:
        self.timing = timing or Timing()
        self.chip = chip
        self._vol_table = _AY_VOLUME_TABLE
        self.reset()

    def reset(self) -> None:
        self._tone_frac = [0.0, 0.0, 0.0]      # phase fraction in [0,1)
        self._noise_index = 0                   # absolute LFSR step
        self._noise_carry = 0.0                 # leftover master cycles
        self._env_step = 0                      # absolute envelope step
        self._env_carry = 0.0
        self._prev_shape: int | None = None
        self._sample_debt = 0.0

    # -- internal helpers -------------------------------------------------

    def _frame_samples(self) -> int:
        # Round-robin fractional samples-per-frame to avoid long-term drift.
        spf = self.timing.samples_per_frame
        n = int(spf)
        self._sample_debt += (spf - n)
        if self._sample_debt >= 1.0:
            n += 1
            self._sample_debt -= 1.0
        return n

    def _render_one_frame(self, state: AYState) -> np.ndarray:
        s = state.canonical()
        n = self._frame_samples()
        clock = self.timing.clock_hz
        dt = clock / self.timing.sample_rate  # master cycles per audio sample
        idx = np.arange(n, dtype=np.float64)

        # --- tone squares per channel ---
        gates = []
        for ch in range(3):
            tp = s.tone_period[ch]
            period = 16.0 * tp  # master cycles per full square cycle
            inc = dt / period
            phase = (self._tone_frac[ch] + inc * idx) % 1.0
            square = (phase < 0.5).astype(np.float64)
            self._tone_frac[ch] = float((self._tone_frac[ch] + inc * n) % 1.0)
            gates.append(square)

        # --- noise (shared) ---
        shift_period = 16.0 * s.noise_period
        elapsed = self._noise_carry + dt * idx
        shifts = np.floor(elapsed / shift_period).astype(np.int64)
        noise_bit = _LFSR_SEQ[(self._noise_index + shifts) % _LFSR_LEN]
        total_elapsed = self._noise_carry + dt * n
        total_shifts = int(np.floor(total_elapsed / shift_period))
        self._noise_index = (self._noise_index + total_shifts) % _LFSR_LEN
        self._noise_carry = total_elapsed - total_shifts * shift_period

        # --- envelope (shared); writing R13 resets the generator ---
        if self._prev_shape is None or s.env_shape != self._prev_shape:
            self._env_step = 0
            self._env_carry = 0.0
        self._prev_shape = s.env_shape
        ep = s.env_period if s.env_period > 0 else 1
        step_period = 256.0 * ep
        e_elapsed = self._env_carry + dt * idx
        e_steps = np.floor(e_elapsed / step_period).astype(np.int64)
        env_amp = _envelope_amplitude(s.env_shape, self._env_step + e_steps)
        e_total = self._env_carry + dt * n
        e_total_steps = int(np.floor(e_total / step_period))
        self._env_step += e_total_steps
        self._env_carry = e_total - e_total_steps * step_period

        # --- per-channel amplitude and mix ---
        out = np.zeros(n, dtype=np.float64)
        for ch in range(3):
            tone_term = gates[ch] if s.tone_on(ch) else 1.0
            noise_term = noise_bit if s.noise_on(ch) else 1.0
            gate = tone_term * noise_term  # active-low: disabled => constant 1
            if s.volume[ch] == USE_ENV:
                amp = self._vol_table[env_amp.astype(np.int64)]
            else:
                amp = self._vol_table[s.volume[ch]]
            chan = amp * gate
            # Center each channel within the frame: the unipolar DACs sit on a
            # large DC pedestal that changes with volume/duty every frame. Removing
            # it here (per channel, per frame) prevents the 50 Hz DC staircase that
            # is otherwise heard as low-frequency "spikes" on top of the tone.
            out += chan - np.mean(chan)
        return out * (1.0 / 3.0)

    # -- public API -------------------------------------------------------

    def render(self, states: AYState | list[AYState]) -> np.ndarray:
        """Render one state or a sequence of frames to mono float32 PCM."""
        if isinstance(states, AYState):
            states = [states]
        chunks = [self._render_one_frame(st) for st in states]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        pcm = np.concatenate(chunks)
        # AC-couple like the real chip's output capacitor: a one-pole DC-blocking
        # high-pass (~5 Hz) removes the large DC pedestal of the unipolar square
        # DACs. Without it, per-frame volume/duty changes leave DC steps that read
        # as a low-frequency "spikes on top" buzz at the 50 Hz frame rate.
        pcm = lfilter(_DC_BLOCK_B, _DC_BLOCK_A, pcm)
        return pcm.astype(np.float32)


def render_stream(
    states: list[AYState], timing: Timing | None = None, chip: str = "ay"
) -> np.ndarray:
    """Convenience: render a full stream from a fresh emulator instance."""
    emu = AYEmulator(timing=timing, chip=chip)
    return emu.render(states)
