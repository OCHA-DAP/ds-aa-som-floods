"""Shared matplotlib style for the analysis notebooks."""

import matplotlib as mpl

from src.constants import BODY, FAINT, GRID, INK


def apply_chart_style():
    """Notebook-wide rcParams: recessive frame, bold left titles, palette cycle."""
    mpl.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#C9CFD6",
            "axes.linewidth": 0.8,
            "axes.titlelocation": "left",
            "axes.titleweight": "bold",
            "axes.titlesize": 11.5,
            "axes.titlecolor": INK,
            "axes.labelsize": 10,
            "axes.labelcolor": BODY,
            "xtick.color": FAINT,
            "ytick.color": BODY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.prop_cycle": mpl.cycler(
                color=[
                    "#2A78D6",
                    "#EB6834",
                    "#0E8A7B",
                    "#8E5FA8",
                    "#F4A93B",
                    "#B34036",
                    "#5C6B7A",
                    "#1C7293",
                ]
            ),
        }
    )


def style_ax(ax, grid="x"):
    """Recessive grid behind the marks, no tick stubs."""
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(length=0)
