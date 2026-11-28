"""Figure conventions for every paper figure (one figure per file, never subplots):

  figsize (5, 4); font size 15 for x/y labels and x/y ticks; grid on; tight layout; no title
  (the paper's caption carries it; `title` is accepted and ignored so callers stay readable).
  PNG always; PDF too once the pipeline is complete (FIG_PDF=1 or save(..., pdf=True)).

Figures are regenerated from the run logs by `python -m seams.analysis.make_figures` (Phase 6).
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGSIZE = (5, 4)
FONT = 15


def new_figure(title=None, xlabel=None, ylabel=None):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT)
    ax.tick_params(axis="both", labelsize=FONT)
    ax.grid(True)
    return fig, ax


def save(fig, out_dir, name, pdf=None):
    """Write <out_dir>/<name>.png (and .pdf when enabled). Returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    paths = [out_dir / f"{name}.png"]
    fig.savefig(paths[0], dpi=200)
    if pdf if pdf is not None else os.environ.get("FIG_PDF") == "1":
        paths.append(out_dir / f"{name}.pdf")
        fig.savefig(paths[1])
    plt.close(fig)
    return paths
