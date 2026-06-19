"""Optimization-history plotting (iteration vs loss). See design/ROADMAP.md P7."""

from __future__ import annotations

from pathlib import Path


def save_history_plot(path: str | Path, losses: list[float],
                      title: str = "Per-block final loss") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(losses, lw=1.0)
    ax.set_xlabel("block")
    ax.set_ylabel("loss")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
