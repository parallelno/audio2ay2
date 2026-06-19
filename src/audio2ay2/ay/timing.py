"""Clock, frame-rate and sample-rate timing for the AY pipeline.

The control rate (register updates) is fixed by the platform: 50 Hz (PAL) by
default. The audio render rate is independent (44.1/48 kHz). See
design/AY_REFERENCE.md section 8.
"""

from __future__ import annotations

from dataclasses import dataclass

# Default ZX-Spectrum 128 / AY-3-8912 clock in Hz.
DEFAULT_CLOCK_HZ = 1_773_400
# PAL frame rate.
DEFAULT_FPS = 50
# Default audio render sample rate.
DEFAULT_SAMPLE_RATE = 44_100


@dataclass(frozen=True)
class Timing:
    """Timing parameters shared across emulator, analysis and export."""

    clock_hz: int = DEFAULT_CLOCK_HZ
    fps: int = DEFAULT_FPS
    sample_rate: int = DEFAULT_SAMPLE_RATE

    @property
    def frame_seconds(self) -> float:
        """Duration of one control frame (e.g. 0.02 s at 50 Hz)."""
        return 1.0 / self.fps

    @property
    def samples_per_frame(self) -> float:
        """Audio samples produced per control frame (may be fractional)."""
        return self.sample_rate / self.fps

    def frame_count(self, duration_seconds: float) -> int:
        """Number of 50 Hz frames spanning ``duration_seconds``."""
        return int(round(duration_seconds * self.fps))
