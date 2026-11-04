"""Draft of Supplementary Fig. X (three panels) and the data of each panel, from the S2 + S7 run on
the 16 sessions.

a  false multi-selectivity of synthetic neurons with one true variable against the information about
   that variable (excess MI of the true label, bits, log scale), per category: shares in bins of
   information (dots; bins with at least MIN_BIN neurons) and the logistic model of the expectation
   (line), fitted on the S2 + S7 cells as in 11_s3_with_s7.py and 16_significance.py; grey, the
   information of the strongest label of the selective CA1 neurons of that category.
b  label combinations of the CA1 neurons that stay multi-selective after disentanglement, next to
   the false ones expected from the model (5_composition.py, column ca1_expected_false).
c  share of multi-selective neurons under the null hypothesis, 2,000 draws of the main test
   (16_significance.py, test A_main), and the observed share in CA1.
Style of the SI figures (make_appendix_figures.py: absolute sizes 6-9 pt, no top or right spines,
600 dpi), lower-case panel letters.
v11.1 copy of s2_16mice/scripts/fig_x.py: the binary feature of speed above 5 cm/s is named "Run" /
"run" on panels a and b (the paper calls this state run). The data are read from [s2] out_dir/data
(ca1_neuron_sessions.csv, s4_composition.csv, s8_significance_draws.csv here; s2_neurons_ca1rule.csv
and s7_neurons.csv through the modules), the modules 4_calibration.py and 11_s3_with_s7.py from s2/ of
the paper folder.
Outputs (under [figures] output_dir/figure_chain): figures_v11.1/si/fig_x.png, .pdf;
         figures_v11.1/work/figx/panel_data/figx_a_bins.csv, figx_a_curves.csv, figx_b.csv,
         figx_c_hist.csv (in v11.1 byte for byte those of the draft of this figure)
usage: python -X utf8 -B figure_chain/figures_v11.1/work/figx/fig_x_v111.py
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # figure_chain/
import _paths_fig as _F  # noqa: E402  (puts the paper folder on sys.path)

HERE = _F.P.PAPER_DIR / "s2"                  # analysis modules of Supplementary Section S2
DATA = _F.P.get("s2", "out_dir") / "data"
OUT = _F.out_dir(Path(__file__).resolve().parents[2] / "si")
FIG = _F.out_dir(Path(__file__).resolve().parent / "panel_data")
CM = 1 / 2.54
MIN_BIN = 10
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 1.0, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.dpi": 600,
})
RED = "#a50f15"
BLUE = "#2171b5"
TITLE = {"place": "Place", "object": "Object", "zone": "Zone", "bodydirection": "Body direction",
         "headdirection": "Head direction", "locomotion": "Run", "speed": "Speed"}
TYPE_EN = {"place": "place", "BD": "body dir.", "HD": "head dir.", "zone": "zone", "object": "object",
           "loco": "run", "speed": "speed", "rear": "rearing", "freezing": "freezing"}


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def letter(ax, s, dx=-0.14, dy=1.08):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")


def combo_en(c):
    return " + ".join(TYPE_EN.get(p, p) for p in c.split("+"))


def main():
    C = _load("cal", "4_calibration.py")
    S = _load("s3s7", "11_s3_with_s7.py")
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        if (OUT / f"fig_x.{ext}").exists():
            raise SystemExit(f"{OUT / f'fig_x.{ext}'} exists; not overwriting")
    ca1 = pd.read_csv(DATA / "ca1_neuron_sessions.csv")
    an = ca1[ca1.category.isin(C.CATS)].reset_index(drop=True)
    obs_share = 100 * an.multi_after.sum() / len(an)
    t = pd.concat([C.s2_training("main"), S.s7_cells()], ignore_index=True)
    models = C.fit(t)

    fig = plt.figure(figsize=(18 * CM, 13.5 * CM))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.35], hspace=0.75, wspace=0.45)
    bins = np.linspace(np.log10(0.04), np.log10(1.5), 13)
    bin_rows, curve_rows = [], []
    axes_a = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(7)]
    for ax, c in zip(axes_a, C.CATS):
        d = t[t.cat == c]
        a = an[an.category == c]
        ax2 = ax.twinx()
        ax2.hist(np.log10(a.max_excess), bins=bins, color="0.88", density=True)
        ax2.set_yticks([])
        ax2.spines[["top", "right"]].set_visible(False)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        idx = np.digitize(d.x, bins)
        for k in np.unique(idx):
            y = d.y.values[idx == k]
            lo, hi = bins[max(k - 1, 0)], bins[min(k, len(bins) - 1)]
            bin_rows.append({"category": c, "bin_lo_bits": 10 ** lo, "bin_hi_bits": 10 ** hi, "n": len(y),
                             "multi": int(y.sum()), "share": y.mean()})
            if len(y) >= MIN_BIN:
                ax.plot((lo + hi) / 2, y.mean(), "o", ms=3, color=BLUE)
        xs = np.linspace(d.x.min(), d.x.max(), 60)
        ps = C.predict(models, np.array([c] * len(xs)), xs)
        curve_rows += [{"category": c, "log10_bits": x, "bits": 10 ** x, "p_multi": p} for x, p in zip(xs, ps)]
        ax.plot(xs, ps, color=BLUE, lw=1.2)
        ax.set_ylim(-0.03, 1)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xticks(np.log10([0.05, 0.2, 1]))
        ax.set_xticklabels(["0.05", "0.2", "1"])
        ax.set_xticks(np.log10([0.1, 0.5]), minor=True)
        ax.set_title(f"{TITLE[c]}, n = {len(d)}", fontsize=7, pad=3)
    for ax in axes_a[3:]:
        ax.set_xlabel("information, bits", fontsize=7)
    for ax in (axes_a[0], axes_a[4]):
        ax.set_ylabel("P(two or more\nselectivities)", fontsize=7)
    letter(axes_a[0], "a", dx=-0.42, dy=1.22)
    leg = fig.add_subplot(gs[1, 3])
    leg.axis("off")
    leg.plot([], [], "o", ms=3, color=BLUE, label=f"synthetic neurons (bins of $\\geq${MIN_BIN})")
    leg.plot([], [], color=BLUE, lw=1.2, label="logistic model")
    leg.fill_between([], [], color="0.88", label="selective CA1 neurons\n(strongest variable)")
    leg.legend(loc="center left", frameon=False, fontsize=6.5)

    comp = pd.read_csv(DATA / "s4_composition.csv").sort_values("ca1_observed", ascending=False).head(8)
    comp.to_csv(FIG / "figx_b.csv", index=False)
    axb = fig.add_subplot(gs[2, 0:2])
    yy = np.arange(len(comp))[::-1]
    axb.barh(yy + 0.19, comp.ca1_observed, height=0.38, color=RED, label="CA1")
    axb.barh(yy - 0.19, comp.ca1_expected_false, height=0.38, color="0.6",
             label="expected if each neuron\ncoded one variable")
    axb.set_yticks(yy)
    axb.set_yticklabels([combo_en(c) for c in comp.combo], fontsize=7)
    axb.set_xlabel("neurons with two or more selectivities", fontsize=7.5)
    axb.legend(frameon=False, fontsize=6.5, loc="lower right")
    letter(axb, "b", dx=-0.42, dy=1.12)

    dr = pd.read_csv(DATA / "s8_significance_draws.csv")
    null = dr[dr.test == "A_main"].null_per100.values
    axc = fig.add_subplot(gs[2, 2:4])
    hb = np.arange(np.floor(null.min() * 4) / 4, max(obs_share, null.max()) + 0.5, 0.25)
    h, e = np.histogram(null, bins=hb)
    pd.DataFrame({"bin_lo_pct": e[:-1], "bin_hi_pct": e[1:], "draws": h}).to_csv(FIG / "figx_c_hist.csv", index=False)
    axc.bar(e[:-1], h, width=np.diff(e), align="edge", color="0.6", edgecolor="white", lw=0.3)
    axc.axvline(obs_share, color=RED, lw=1.5)
    axc.text(obs_share, h.max() * 0.95, f" CA1\n {obs_share:.1f}%", color=RED, fontsize=7, va="top", ha="left")
    axc.set_xlabel("neurons with two or more selectivities, %", fontsize=7.5)
    axc.set_ylabel(f"draws (of {len(null):,})", fontsize=7.5)
    axc.set_xlim(e[0], max(obs_share + 1.8, e[-1]))
    letter(axc, "c", dx=-0.22, dy=1.12)

    pd.DataFrame(bin_rows).to_csv(FIG / "figx_a_bins.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(FIG / "figx_a_curves.csv", index=False)
    fig.savefig(OUT / "fig_x.png", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "fig_x.pdf", bbox_inches="tight", pad_inches=0.03)
    print(f"panel c: {len(null)} draws, max {null.max():.2f}, 97.5th percentile {np.percentile(null, 97.5):.2f}, "
          f"observed {obs_share:.2f}")


if __name__ == "__main__":
    main()
