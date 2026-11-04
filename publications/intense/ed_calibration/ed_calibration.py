"""Extended Data figure: false-positive calibration of INTENSE, Nature format, with Source Data.

a  false-positive rate per pair with p < 0.001 alone, by null model, delay search and variable type
b  the same with the excess-information criterion (> 0.04 bits) added
c  optimal delays selected for the false pairs of all null models (delay search on)

Inputs (written by calib_null.py / run_part.py and analyze.py): <calibration_dir>/
calibration_table.csv and false_pairs_delay.csv. Outputs: <output>/ed_calibration.pdf,
<output>/source_data/ED_calibration_source_data.xlsx.

usage: python ed_calibration/ed_calibration.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402
import nature_fig as nf  # noqa: E402
from source_data import SourceData  # noqa: E402

ALPHA = P.PVAL_THR
NULLS = [("shift", "Same session, shifted behavior", "#1f5fa8"),
         ("cross", "Other mouse", "#c2571a"),
         ("synth", "Synthetic", "#4d8f3a")]
TYPES = ["place", "speed", "direction", "discrete", "all"]
TYPE_LABELS = ["Place", "Speed", "Direction", "Discrete", "All"]


def rate_panel(ax, tab, tag, key_text):
    x = np.arange(len(TYPES))
    w = 0.13
    rows = []
    for k, (null, label, color) in enumerate(NULLS):
        for d, delays in enumerate((False, True)):
            s = tab[(tab.null == null) & (tab.delays == delays)].set_index("ftype").reindex(TYPES)
            if s["n_pairs"].isna().all():
                continue
            r = s[f"rate{tag}"].to_numpy() * 1e3
            lo = s[f"boot_lo{tag}"].to_numpy() * 1e3
            hi = s[f"boot_hi{tag}"].to_numpy() * 1e3
            xx = x + (k - 1) * 2 * w + (d - 0.5) * w
            ax.errorbar(xx, r, yerr=[r - lo, hi - r], fmt="o", ms=2.2, mew=0.6, color=color,
                        mfc=color if delays else "white", elinewidth=0.5, capsize=0,
                        label=f"{label}, {'delay search' if delays else 'no delay'}")
            for ft, kk, n, rr, l_, h_ in zip(TYPES, s[f"k{tag}"], s["n_pairs"], r, lo, hi):
                rows.append({"null model": label, "delay search": delays, "variable type": ft,
                             "pairs tested": int(n), "false pairs": int(kk), "rate per 1000 pairs": rr,
                             "95% bootstrap low": l_, "95% bootstrap high": h_})
    ax.axhline(ALPHA * 1e3, color="0.3", ls="--", lw=0.5, label="Nominal rate (p threshold)")
    ax.text(0.02, 0.98, key_text, transform=ax.transAxes, ha="left", va="top", fontsize=nf.TICK_PT)
    ax.set_xticks(x)
    ax.set_xticklabels(TYPE_LABELS)
    ax.set_xlim(-0.5, len(TYPES) - 0.5)
    ax.set_ylabel("False positives per 1,000 pairs")
    return pd.DataFrame(rows)


def main():
    nf.apply_style()
    cal = P.CALIBRATION_DIR
    tab = pd.read_csv(cal / "calibration_table.csv")
    fp = pd.read_csv(cal / "false_pairs_delay.csv")
    fig = nf.figure(58)
    sd = SourceData("ED", "Extended Data: calibration of false discoveries")
    axa = nf.axes_mm(fig, 11, 4, 58, 38)
    axb = nf.axes_mm(fig, 82, 4, 58, 38, sharey=axa)
    axc = nf.axes_mm(fig, 152, 4, 28, 38)
    da = rate_panel(axa, tab, "", "p < 0.001")
    db = rate_panel(axb, tab, "_mi04", "p < 0.001 and excess MI > 0.04 bits")
    axa.set_ylim(0, max(axa.get_ylim()[1], axb.get_ylim()[1]))
    h, lab = axa.get_legend_handles_labels()
    h, lab = h[1:] + h[:1], lab[1:] + lab[:1]      # nominal line last
    fig.legend(h, lab, loc="lower left", ncol=4, frameon=False, bbox_to_anchor=(0.04, 0.0),
               handletextpad=0.3, columnspacing=1.0)

    sec = fp.delay_frames.to_numpy() / P.FPS
    bins = np.arange(-2.125, 2.126, 0.25)
    counts, _, _ = axc.hist(sec, bins=bins, color="0.45", edgecolor="white", linewidth=0.3)
    axc.set_xlabel("Selected delay (s)")
    axc.set_ylabel("False pairs")
    axc.set_xticks([-2, -1, 0, 1, 2])
    for letter, x in (("a", 0.5), ("b", 71.5), ("c", 141.5)):
        nf.letter_mm(fig, letter, x, 1)

    sd.add("a", da, "False-positive rate of p < 0.001 per pair; error bars, 95% bootstrap over recording units")
    sd.add("b", db, "Same with excess MI above chance > 0.04 bits")
    sd.add("c", pd.DataFrame({"delay bin start (s)": bins[:-1], "delay bin end (s)": bins[1:],
                              "false pairs": counts.astype(int)}), "Histogram of the selected delays")
    sd.add("c", fp, "All false pairs with delay search (null, unit, neuron, feature, MI, excess MI, p, delay in frames)")
    sd.add("table", tab, "Full calibration table (analyze.py): Clopper-Pearson and bootstrap intervals, "
                         "expected false pairs in the cohort of 30,105 neuron-sessions")
    sd.note("a", "False-positive rate per pair at p < 0.001 under three null models, without and with the "
                 "delay search, by variable type", "calibration_table.csv")
    sd.note("b", "The same with the excess-information criterion", "calibration_table.csv")
    sd.note("c", "Delays selected for the false pairs found with the delay search (all null models pooled)",
            "false_pairs_delay.csv")
    sd.note("table", "Full calibration table behind panels a and b", "calibration_table.csv")
    sd.write(P.SOURCE_DATA_DIR / "ED_calibration_source_data.xlsx")
    return nf.save(fig, str(P.out_path("ed_calibration")))


if __name__ == "__main__":
    main()
