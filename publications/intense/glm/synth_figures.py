"""Supplementary Figs 1 and 2 of Section S1.

Fig. 1 (glm_ablation.png): detection F1 of the eight GLM variants and of INTENSE per cell type,
mean and 95% bootstrap interval over the 20 seeds of the t3f1 part of glm/synth_compute.py
(3,000 resamples), drawn from its parts. Fig. 2 (glm_snr_sweep.png): F1 against SNR, drawn from
tables/fig2_ci.csv of glm/synth_numbers.py. INTENSE is the configuration at p < 0.05 with Holm
correction, with the type-based representations and the removal of anti-selective pairs (the
DRIADA default).

usage: python glm/synth_figures.py  ->  <[glm] figures_dir>/glm_ablation.png, glm_snr_sweep.png
"""
import os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys  # noqa: E402

sys.dont_write_bytecode = True
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
import intense_paths as P  # noqa: E402

OUTD = P.get("glm", "synth_dir")
FIGS = P.get("glm", "figures_dir")
CM = 1 / 2.54
DPI = 600

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


def boot(v, n=3000, rng=np.random.default_rng(0)):
    bs = rng.choice(v, (n, len(v))).mean(1)
    return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def fig_ablation():
    pn = pd.concat([pd.read_csv(f) for f in sorted((OUTD / "parts").glob("f1_t3f1_*.csv"))], ignore_index=True)
    pn = pn[pn.variant != "INTENSE_holm"]
    pn["variant"] = pn.variant.replace({"INTENSE_holm_filter": "INTENSE"})
    order = ["hd_cells", "speed_cells", "event_cells", "mixed_hd_speed_lin",
             "mixed_speed_event_lin", "mixed_hd_speed_and",
             "mixed_speed_event_and", "nonselective"]
    labels = ["pure HD", "pure\nspeed", "pure\nevent", "mix\nHD+sp", "mix\nsp+ev",
              "AND\nHD+sp", "AND\nsp+ev", "non-\nselective"]
    cols = ["V0_base", "V_ttest", "V_fdr", "V_alpha10", "V_folds20",
            "V_poisson", "V_max", "V_max_perfeat", "INTENSE"]
    names = {"V0_base": "GLM baseline", "V_ttest": "+ t-test", "V_fdr": "+ FDR",
             "V_alpha10": r"+ $\alpha$=0.10", "V_folds20": "+ 20 folds",
             "V_poisson": "Poisson/spikes", "V_max": "GLM max-strength",
             "V_max_perfeat": "max, per-feature", "INTENSE": "INTENSE"}
    ps = pn.groupby(["group", "variant", "seed"]).f1.mean().reset_index()
    fig, ax = plt.subplots(figsize=(18 * CM, 8 * CM))
    x = np.arange(len(order))
    w = 0.86 / len(cols)
    for i, c in enumerate(cols):
        m, lo, hi = zip(*[boot(ps[(ps.group == g) & (ps.variant == c)].f1.values)
                          for g in order])
        color = RED if c == "INTENSE" else BLUES[i]
        ax.bar(x + (i - (len(cols) - 1) / 2) * w, m, w,
               yerr=np.vstack([np.array(m) - lo, np.array(hi) - m]),
               error_kw=dict(lw=0.7), capsize=1.2, color=color,
               label=names[c])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("detection $F_1$")
    ax.set_ylim(0, 1.02)
    ax.legend(ncol=5, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), handlelength=1.2, columnspacing=1.0)
    fig.savefig(FIGS / "glm_ablation.png", **SAVE)
    plt.close(fig)
    return ps.seed.nunique()


def fig_snr():
    ci = pd.read_csv(OUTD / "tables" / "fig2_ci.csv")
    ci = ci[ci.variant != "INTENSE_holm"]
    ci["variant"] = ci.variant.replace({"INTENSE_holm_filter": "INTENSE"})
    groups = ["hd_cells", "event_cells", "speed_cells",
              "mixed_hd_speed_lin", "mixed_speed_event_lin"]
    titles = ["pure HD", "pure event", "pure speed", "mix HD+speed", "mix speed+event"]
    variants = {"V0_base": (BLUES[1], "GLM baseline"),
                "V_alpha10": (BLUES[3], r"GLM $\alpha$=0.10"),
                "V_max": (BLUES[5], "GLM max"), "INTENSE": (RED, "INTENSE")}
    fig, axes = plt.subplots(1, 5, figsize=(18 * CM, 5.0 * CM), sharey=True)
    for k, (ax, g, t) in enumerate(zip(axes, groups, titles)):
        for v, (c, lab) in variants.items():
            s = ci[(ci.group == g) & (ci.variant == v)].sort_values("snr")
            ax.plot(s.snr, s.mean_f1, "o-", color=c, lw=1.2, ms=2.5,
                    label=lab if k == 0 else None)
            ax.fill_between(s.snr, s.ci_lo, s.ci_hi, color=c, alpha=0.15, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 8, 32, 128])
        ax.set_xticklabels(["2", "8", "32", "128"])
        ax.set_title(t, fontsize=8)
        ax.set_xlabel("SNR")
        ax.set_ylim(-0.02, 1.05)
    axes[0].set_ylabel("detection $F_1$")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, 0.97), handlelength=1.4)
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(FIGS / "glm_snr_sweep.png", **SAVE)
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    if list((OUTD / "parts").glob("f1_t3f1_*.csv")):
        n = fig_ablation()
        print(f"[OK] Fig. 1 from {n} seeds")
    if (OUTD / "tables" / "fig2_ci.csv").exists():
        fig_snr()
        print("[OK] Fig. 2")


if __name__ == "__main__":
    main()
