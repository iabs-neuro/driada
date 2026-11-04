# -*- coding: utf-8 -*-
"""The four MEC figures of the SI, from given cohort and arbiter files.

Copy of the MEC part of science/intense_paper/extracted/make_appendix_figures.py in the
DRIADA repository (functions mec_counts, fig_mec, fig_concordance, fig_venn, fig_margin and
the style block), with the input files and the output folder as arguments; drawing code
unchanged. Also writes the numbers the SI text takes from these files (mec_numbers.json):
detections per method, their ratio, the share of GLM detections INTENSE recovers per
feature and the per-cell agreement categories.

Usage:
  python make_mec_figures.py --results <cohort_results_bytype.csv>
      --sets <cohort_cell_sets_bytype.jsonl> --venn <arbiter_venn.csv>
      --margin <recall_vs_margin.csv> --out <dir>
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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
SAVE = dict(bbox_inches="tight", pad_inches=0.03)
FIGS = None
# MARGIN_LAYOUT_FIX (mec_arbiter_margin): the three x-axis titles are wider than their panels and ran
# into each other, the legend covered the counts of the grid panel and the counts over the outer points
# crossed the axes. True: one x-axis title, under the middle panel; half a step of room left and right
# of the outer points; the legend below the row of counts. MARGIN_SAVE: savefig arguments that replace
# SAVE for this figure (a fixed box keeps the size of the earlier file). KEEP_MARGIN_FIG: return the
# figure instead of closing it, for the checks of the caller.
MARGIN_LAYOUT_FIX = False
# LOWERCASE_LETTERS: panel letters a, b instead of A, B (lowercase panel letters in the SI figures).
LOWERCASE_LETTERS = False
MARGIN_SAVE = None
KEEP_MARGIN_FIG = False


def letter(ax, s, dx=-0.14, dy=1.08):
    s = s.lower() if LOWERCASE_LETTERS else s
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="left")


# ======================================================================
# 4. mec_cohort_bytype: detections + coverage, 12.5 x 5 cm, panels A/B
# ======================================================================
def mec_counts(results, sets):
    cr = pd.read_csv(results)
    C = ["place", "head_direction", "speed"]
    glm = [int(cr[f"glm_{c}"].sum()) for c in C]
    bt = [int(cr[f"int_bytype_{c}"].sum()) for c in C]
    cov = {c: [0, 0] for c in C}
    cat = {"overlap": 0, "asym": 0, "conflict": 0}
    pf = {c: [0, 0, 0] for c in C}
    with open(sets) as fh:
        for line in fh:
            d = json.loads(line)
            g, b = set(d["glm"]), set(d["int_bytype"])
            for c in C:
                if c in g:
                    cov[c][1] += 1
                    if c in b:
                        cov[c][0] += 1
                if c in g and c in b:
                    pf[c][0] += 1
                elif c in b:
                    pf[c][1] += 1
                elif c in g:
                    pf[c][2] += 1
            if g or b:
                if g and b and (g & b):
                    cat["overlap"] += 1
                elif g and b:
                    cat["conflict"] += 1
                else:
                    cat["asym"] += 1
    return C, glm, bt, cov, cat, pf


def fig_mec(C, glm, bt, cov):
    RU = ["position", "head\ndirection", "speed"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5 * CM, 5 * CM))
    ax = axes[0]
    x = np.arange(3)
    w = 0.38
    ax.bar(x - w / 2, glm, w, color="#2171b5", label="GLM")
    ax.bar(x + w / 2, bt, w, color=RED, label="INTENSE")
    for xi, (g, b) in enumerate(zip(glm, bt)):
        ax.text(xi + w / 2, b + 120, f"{b / g:.2g}x", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(RU)
    ax.set_ylabel("selective cells")
    ax.set_ylim(0, 7300)
    ax.legend(frameon=False, loc="upper left")
    letter(ax, "A")
    ax = axes[1]
    vals = [100 * cov[c][0] / cov[c][1] for c in C]
    ax.bar(x, vals, 0.5, color=RED)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 2.5, f"{v:.0f}%", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(RU)
    ax.set_ylim(0, 108)
    ax.set_ylabel("GLM detections\nrecovered by INTENSE (%)")
    letter(ax, "B")
    fig.subplots_adjust(wspace=0.45)
    fig.savefig(FIGS / "mec_cohort_bytype.png", **SAVE)
    plt.close(fig)


# ======================================================================
# 5. mec_concordance_full: agreement + categories, 12.5 x 5 cm
# ======================================================================
def fig_concordance(C, cat, pf):
    RU = ["position", "head\ndirection", "speed"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5 * CM, 5 * CM))
    ax = axes[0]
    x = np.arange(3)
    both = [pf[c][0] for c in C]
    io = [pf[c][1] for c in C]
    go = [pf[c][2] for c in C]
    ax.bar(x, both, 0.5, color="#6a51a3", label="both")
    ax.bar(x, io, 0.5, bottom=both, color="#fb6a4a", label="INTENSE only")
    ax.bar(x, go, 0.5, bottom=np.array(both) + np.array(io), color="#2171b5",
           label="GLM only")
    ax.set_xticks(x)
    ax.set_xticklabels(RU)
    ax.set_ylabel("cells")
    ax.legend(frameon=False, loc="upper left")
    letter(ax, "A")
    ax = axes[1]
    tot = sum(cat.values())
    vals = [100 * cat[k] / tot for k in ("overlap", "asym", "conflict")]
    ax.bar(np.arange(3), vals, 0.5, color=["#6a51a3", "#fdae6b", "#cb181d"])
    for xi, v in zip(np.arange(3), vals):
        ax.text(xi, v + 2, f"{v:.1f}%", ha="center", fontsize=7)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["overlap", "one-sided", "conflict"])
    ax.set_ylabel("active cells (%)")
    ax.set_ylim(0, 85)
    letter(ax, "B")
    fig.subplots_adjust(wspace=0.4)
    fig.savefig(FIGS / "mec_concordance_full.png", **SAVE)
    plt.close(fig)


# ======================================================================
# 6. mec_arbiter_venn: 3 three-set diagrams, 18 x 6.5 cm
# ======================================================================
def fig_venn(venn_csv):
    vn = pd.read_csv(venn_csv)
    titles = {"grid": "grid", "border": "border", "head_direction": "head direction"}
    fig, axes = plt.subplots(1, 3, figsize=(18 * CM, 6.5 * CM))
    ccols = ["#6a51a3", RED, "#2171b5"]
    for ax, (_, v) in zip(axes, vn.iterrows()):
        centers = [(-0.35, 0.25), (0.35, 0.25), (0.0, -0.4)]
        for (cx, cy), c in zip(centers, ccols):
            ax.add_patch(Circle((cx, cy), 0.62, alpha=0.22, color=c, ec=c, lw=1.4))
        ax.text(-0.72, 0.95, f"authors\nn={v['authors']}", color=ccols[0],
                fontsize=7, ha="center", va="bottom")
        ax.text(0.72, 0.95, f"INTENSE\nn={v['intense']}", color=ccols[1],
                fontsize=7, ha="center", va="bottom")
        ax.text(0.0, -1.18, f"GLM  n={v['glm']}", color=ccols[2], fontsize=7,
                ha="center", va="top")
        pos = {"A_only": (-0.62, 0.45), "B_only": (0.62, 0.45),
               "C_only": (0.0, -0.75), "AB": (0.0, 0.52), "AC": (-0.37, -0.2),
               "BC": (0.37, -0.2), "ABC": (0.0, 0.05)}
        for k, (xx, yy) in pos.items():
            ax.text(xx, yy, str(v[k]), ha="center", va="center", fontsize=8)
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.55, 1.45)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{titles[v['type']]}\nrecall: INTENSE "
                     f"{v['intense_recall_of_authors']:.0%}, GLM "
                     f"{v['glm_recall_of_authors']:.0%}", fontsize=8, pad=2)
    fig.subplots_adjust(wspace=0.05)
    fig.savefig(FIGS / "mec_arbiter_venn.png", **SAVE)
    plt.close(fig)


# ======================================================================
# 7. mec_arbiter_margin: recall vs label confidence, 18 x 5.2 cm
# ======================================================================
def fig_margin(margin_csv):
    rm = pd.read_csv(margin_csv)
    fig, axes = plt.subplots(1, 3, figsize=(18 * CM, 5.2 * CM), sharey=True)
    for ax, t, ttl in zip(axes, ["grid", "border", "HD"],
                          ["grid", "border", "head direction"]):
        S = rm[rm.type == t]
        x = np.arange(len(S))
        ax.plot(x, S.recall_int, "o-", color=RED, lw=1.2, ms=3, label="INTENSE")
        ax.plot(x, S.recall_glm, "^-", color="#2171b5", lw=1.2, ms=3, label="GLM")
        for xi, n in zip(x, S.n):
            ax.text(xi, 1.03, f"{n}", ha="center", fontsize=6, color="#666666")
        ax.set_xticks(x)
        ax.set_xticklabels(S.margin, fontsize=6.5)
        ax.set_ylim(0, 1.12)
        if MARGIN_LAYOUT_FIX:
            ax.set_xlim(-0.5, len(S) - 0.5)
        else:
            ax.set_xlabel("score margin above authors' cutoff")
        ax.set_title(ttl, fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("recall of authors' cells")
    if MARGIN_LAYOUT_FIX:
        axes[1].set_xlabel("score margin above authors' cutoff")
        # the counts stand at y = 1.03 of 1.12, i.e. 0.92 of the axes height; the legend hangs below them
        axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.88))
    else:
        axes[0].legend(frameon=False, loc="upper left")
    fig.subplots_adjust(wspace=0.1)
    fig.savefig(FIGS / "mec_arbiter_margin.png", **(MARGIN_SAVE or SAVE))
    if KEEP_MARGIN_FIG:
        return fig, axes
    plt.close(fig)


def main():
    global FIGS
    ap = argparse.ArgumentParser()
    for k in ("results", "sets", "venn", "margin", "out"):
        ap.add_argument(f"--{k}", required=True, type=Path)
    a = ap.parse_args()
    FIGS = a.out
    FIGS.mkdir(parents=True, exist_ok=True)
    C, glm, bt, cov, cat, pf = mec_counts(a.results, a.sets)
    fig_mec(C, glm, bt, cov)
    fig_concordance(C, cat, pf)
    fig_venn(a.venn)
    fig_margin(a.margin)
    tot = sum(cat.values())
    nums = {
        "inputs": {k: str(getattr(a, k)) for k in ("results", "sets", "venn", "margin")},
        "glm": dict(zip(C, glm)), "intense": dict(zip(C, bt)),
        "ratio_intense_over_glm": {c: b / g for c, g, b in zip(C, glm, bt)},
        "glm_detections_recovered_by_intense": {
            c: {"recovered": cov[c][0], "glm": cov[c][1], "share": cov[c][0] / cov[c][1]}
            for c in C},
        "per_feature_both_intense_only_glm_only": pf,
        "cells_with_any_detection": tot,
        "categories": cat,
        "category_share": {k: v / tot for k, v in cat.items()},
    }
    with open(FIGS / "mec_numbers.json", "w") as fh:
        json.dump(nums, fh, indent=1)
    print(json.dumps(nums, indent=1))
    print("done:", sorted(p.name for p in FIGS.glob("*.png")))


if __name__ == "__main__":
    main()
