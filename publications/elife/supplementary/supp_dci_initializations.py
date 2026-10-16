"""Supplementary figure: DCI disentanglement score D across initializations.

Reads the sweep of 12 random initializations per autoencoder variant and
draws every run, the PCA value, and the median with interquartile range.
Nothing is filtered out.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import elife_paths

# written by dci_sweep.py
DATA = os.path.join(elife_paths.CACHE_DIR, "fig5_ae_sweep.npz")

CM = 1 / 2.54
COLORS = {"PCA": "#555555", "AE": "#3b6fb6", "AE+corr": "#c8553d"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def main():
    d = np.load(DATA)
    dci = d["dci_matrix"]            # (n_seeds, n_variants)
    weights = d["corr_weights"]
    seeds = d["ae_seeds"]
    d_pca = float(d["dci_pca"])
    assert dci.shape == (len(seeds), len(weights))
    assert list(weights) == [0.0, 1.0]

    groups = [("PCA", np.array([d_pca])), ("AE", dci[:, 0]), ("AE+corr", dci[:, 1])]

    fig, ax = plt.subplots(figsize=(8.5 * CM, 6.5 * CM))
    for x, (name, vals) in enumerate(groups):
        color = COLORS[name]
        if len(vals) == 1:
            ax.plot(x, vals[0], "D", ms=5, color=color, mec="black", mew=0.5,
                    zorder=3)
            ax.annotate(f"{vals[0]:.3f}", (x, vals[0]), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=6.5)
            continue
        # fixed, evenly spread horizontal offsets ordered by value: no points overlap
        order = np.argsort(vals)
        offs = np.empty(len(vals))
        offs[order] = np.tile([-0.14, 0.0, 0.14, -0.07, 0.07], 3)[:len(vals)]
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.add_patch(plt.Rectangle((x - 0.27, q1), 0.54, q3 - q1, fc=color,
                                   alpha=0.15, ec="none", zorder=1))
        ax.plot([x - 0.27, x + 0.27], [med, med], color=color, lw=1.4, zorder=2)
        best = int(np.argmax(vals))
        ax.scatter(x + offs, vals, s=16, color=color, edgecolor="black",
                   linewidth=0.4, zorder=3)
        ax.scatter(x + offs[best], vals[best], s=46, facecolor="none",
                   edgecolor="black", linewidth=0.9, zorder=4)
        ax.annotate(f"best {vals[best]:.3f}", (x + offs[best], vals[best]),
                    xytext=(7, 0), textcoords="offset points", va="center",
                    fontsize=6.5)
        ax.annotate(f"median {med:.3f}", (x + 0.27, med), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=6.5,
                    color=color)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{n}\n(n = {len(v)})" for n, v in groups])
    ax.set_xlim(-0.5, 2.95)
    ax.set_ylim(0, 1)
    ax.set_ylabel("DCI disentanglement score $D$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.6, length=3)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(elife_paths.out_path(f"supp_dci_initializations.{ext}"), dpi=600)
    plt.close(fig)

    for name, vals in groups[1:]:
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        print(f"{name}: n={len(vals)} best={vals.max():.3f} (seed {seeds[np.argmax(vals)]}) "
              f"median={med:.3f} IQR={q1:.3f}-{q3:.3f} min={vals.min():.3f}")
    print(f"PCA: {d_pca:.3f}")


if __name__ == "__main__":
    main()
