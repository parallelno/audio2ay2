"""Reports & diagnostics: validation, optimization history, loss landscapes."""

from .validate import write_report, per_frame_losses
from .history import save_history_plot
from .landscape import sweep_tone_period, save_landscape_plot

__all__ = [
    "write_report", "per_frame_losses", "save_history_plot",
    "sweep_tone_period", "save_landscape_plot",
]
