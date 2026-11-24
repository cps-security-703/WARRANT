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
