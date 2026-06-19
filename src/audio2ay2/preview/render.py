"""Preview: render a register stream back to audio through the same emulator.

Closes the loop for the user exactly as the optimizer's loss closes it
internally (design/PLAN.md section 5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..ay.registers import AYState
from ..ay.timing import Timing
from ..ay.emulator import render_stream
from ..audioio import save_audio


def render_preview(states: list[AYState], timing: Timing, chip: str = "ay"
                   ) -> np.ndarray:
    return render_stream(states, timing=timing, chip=chip)


def write_preview(path: str | Path, states: list[AYState], timing: Timing,
                  chip: str = "ay") -> np.ndarray:
    pcm = render_preview(states, timing, chip)
    save_audio(path, pcm, timing.sample_rate)
    return pcm
