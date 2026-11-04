"""Figure 2 panels C, E, F (v8) drawn from cache/fig2_v8_data.npz (see fig2_data_v8.py).

Copy of figures_v10.4/work/fig2_src/fig2_panels_v8.py for v11.1 with three more switches, E_YLABELS,
E_WSPACE and F_XMAX; at their defaults the panels are those of v11.
"""
import _paths  # noqa: F401
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

from data_loader import CACHE_DIR
from fig2c_circles_delegated import _draw_venn, IN1, OUT1, IN2, OUT2, IN_MID, OUT_MID


# Switches of the v10.4 copy; the defaults draw v10.2.
# DATA_NPZ: None = cache/fig2_v8_data.npz of v10.2, else the file of fig2_data_v104.py.
# E_LABEL_ABOVE: metrics of 2e whose "p = 0.001" goes above the axes, clear of the curves.
# F_ONLY_OBSERVED_BASELINE, F_JACCARD_YMAX: 2f right without the gray curves of stronger
# correlations and with the Jaccard axis up to this value.
DATA_NPZ = None
E_LABEL_ABOVE = ()
F_ONLY_OBSERVED_BASELINE = False
F_JACCARD_YMAX = 1
# E_YLABELS: None leaves the three plots of 2e without y-axis titles (v7-v11); else a title per metric.
# E_WSPACE: room between the plots of 2e, in axes widths; a y-axis title needs more than v11 gave.
# F_XMAX: right end of the x-axis of 2f left; the curves are drawn up to the end of the grid of the
# data file, while at least 20 cells are at the level.
E_YLABELS = None
E_WSPACE = 0.55
F_XMAX = 12


def _data():
    return dict(np.load(DATA_NPZ if DATA_NPZ is not None else CACHE_DIR / "fig2_v8_data.npz"))


def create_fig2c(fig, style, panel_size):
    d = _data()
    ax = fig.subplots(1, 1)
    lw = style.get_line_width(panel_size, "cm")
    j = float(d["venn_jaccard"])
    _draw_venn(ax, int(d["venn_intense_only"]), int(d["venn_classic_only"]), int(d["venn_both"]),
               f"all {int(d['venn_n_cells']):,} neurons; Jaccard {j:.2f}",
               fontsize=style.base_legend_fontsize, annotation_fontsize=style.base_label_size, edge_lw=lw)
    handles = [Patch(facecolor=IN1, edgecolor=OUT1, linewidth=lw, label="INTENSE only"),
               Patch(facecolor=IN_MID, edgecolor=OUT_MID, linewidth=lw, label="both"),
               Patch(facecolor=IN2, edgecolor=OUT2, linewidth=lw, label="classic only")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=style.base_legend_fontsize,
               frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.17)
    return fig, ax


def create_fig2e(fig, style, panel_size):
    d = _data()
    axs = fig.subplots(1, 3)
    lw = style.get_line_width(panel_size, "cm")
    x = d["xvals_e"]
    for ax, key, name, c in zip(axs, ("mse", "psnr", "ssim"), ("MSE", "PSNR", "SSIM"), ("b", "r", "g")):
        av, er = d[f"av_{key}"], d[f"err_{key}"]
        ax.plot(x, av, linewidth=lw, c=c, label=name)
        ax.fill_between(x, av - er, av + er, color=c, alpha=0.5)
        ax.set_xlabel(r"$-\log_{10} p$")
        if E_YLABELS is not None:
            ax.set_ylabel(E_YLABELS[name])
        ax.axvline(3, c="k", linestyle="dashed", linewidth=lw)
        if name == "MSE":
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.locator_params(axis="y", nbins=4)
        else:
            ax.locator_params(axis="y", nbins=6)
        style.apply_to_axes(ax, panel_size, "cm")
        if name in E_LABEL_ABOVE:
            ax.text(3.25, 1.01, r"$p=0.001$", transform=ax.get_xaxis_transform(),
                    fontsize=style.base_legend_fontsize, ha="left", va="bottom")
        else:
            ax.text(3.25, 0.97, r"$p=0.001$", transform=ax.get_xaxis_transform(),
                    fontsize=style.base_legend_fontsize, ha="left", va="top")
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.35, top=0.95, wspace=E_WSPACE)
    h, l = [], []
    for ax in axs:
        hh, ll = ax.get_legend_handles_labels(); h += hh; l += ll
    axs[1].legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False,
                  fontsize=style.base_legend_fontsize, columnspacing=3.0)
    return fig, list(axs)


def create_fig2f(fig, style, panel_size):
    d = _data()
    ax1, ax2 = fig.subplots(1, 2)
    lw = style.get_line_width(panel_size, "cm")
    fs = style.base_legend_fontsize
    g = d["grid"]
    ok_i, ok_c = d["n_int"] >= 20, d["n_cl"] >= 20
    ax1.plot(g[ok_i], 100 * d["conf_int"][ok_i], c="r", lw=lw, label="INTENSE cells\nconfirmed by classic")
    ax1.plot(g[ok_c], 100 * d["conf_cl"][ok_c], c=(0, 0, 192 / 255), lw=lw,
             label="classic cells\nconfirmed by INTENSE")
    ax1.axvline(3, c="k", ls="--", lw=lw * 0.7, alpha=0.7)
    ax1.set_ylabel("confirmed (%)")
    ax1.set_xlim(0, F_XMAX)
    ax1.set_ylim(0, 101)
    style.apply_to_axes(ax1, panel_size, "cm")
    ax1.set_xlabel(r"$-\log_{10} p$ (conditioning)")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=1, frameon=False, fontsize=fs)

    f = 100 * d["fracs"]
    rho = float(d["rho"])
    keys = [str(k) for k in d["base_keys"]]
    for k in ([] if F_ONLY_OBSERVED_BASELINE else keys[1:]):
        ax2.plot(f, d[f"base_{k}"], c="0.75", lw=lw * 0.7)
        i = len(f) // 3
        ax2.text(f[i], d[f"base_{k}"][i] + 0.02, rf"$\rho$={float(k):.1f}", fontsize=fs * 0.85,
                 ha="center", va="bottom", color="0.45")
    ax2.plot(f, d[f"base_{keys[0]}"], c="0.35", ls="--", lw=lw, label=rf"expected, $\rho$={rho:.2f}")
    ax2.plot(f, d["j_obs"], c="orange", lw=lw, label="INTENSE vs classic")
    ax2.set_xscale("log")
    ax2.set_ylabel("Jaccard index")
    ax2.set_ylim(0, F_JACCARD_YMAX)
    ax2.set_xlim(f[0] * 0.9, f[-1] * 1.05)
    ax2.set_xticks([1, 10, 50])
    ax2.set_xticklabels(["1", "10", "50"])
    style.apply_to_axes(ax2, panel_size, "cm")
    ax2.set_xlabel("top cells (%)")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=1, frameon=False, fontsize=fs)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.40, top=0.95, wspace=0.65)
    return fig, (ax1, ax2)
