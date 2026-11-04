# -*- coding: utf-8 -*-
"""House style of the figures of the Supplementary Information: figures laid out at true print width
(18 cm full / 12 cm medium / 8.5 cm half), absolute font sizes 6-9 pt, bold panel letters, no top/right spines,
600 dpi: the rcParams set on import, the sizes and colours and letter(). The script of the example neurons
(figures_v11.1/work/glm_examples/glm_src) imports it for this style.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CM = 1 / 2.54
DPI = 600

# eLife-style absolute sizes (publications/elife/fig3/fig3_composite.py)
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6,
    "axes.linewidth": 1.0, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.dpi": DPI,
})
RED = "#a50f15"
BLUES = ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c",
         "#084594", "#061a40"]
SAVE = dict(bbox_inches="tight", pad_inches=0.03)


def letter(ax, s, dx=-0.14, dy=1.08):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="left")
