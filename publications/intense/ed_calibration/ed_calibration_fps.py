"""Extended Data calibration figure with the delays in seconds at the rate of each unit.

ed_calibration/ed_calibration.py of the paper converts the selected delays of
the false pairs (panel c) to seconds with the single rate of the configuration
(P.FPS). The data now have a rate per session, so this script draws the same
figure with the same panel code (rate_panel and the constants of
ed_calibration.py are imported, not copied) and only converts each false pair
with the rate of its unit: a session's rate from _metadata of the data files,
20 frames/s for the synthetic units (calib_null.run_synth generates them at
20). The Source Data sheet of panel c gets the rate and the delay in seconds of
every pair as two more columns.

usage: python ed_calibration/ed_calibration_fps.py
       -> <output_dir>/ed_calibration.pdf, <output_dir>/source_data/ED_calibration_source_data.xlsx
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER / "ed_calibration"))
sys.path.insert(0, str(PAPER))
import ed_calibration as E  # noqa: E402
import intense_paths as P  # noqa: E402
import nature_fig as nf  # noqa: E402
from source_data import SourceData  # noqa: E402

SYNTH_FPS = 20.0  # calib_null.run_synth: generate_tuned_selectivity_exp(..., fps=20)


def unit_rates(units):
    """Frame rate of every calibration unit (session name or synthXX)."""
    rates = {}
    for u in sorted(set(units)):
        if str(u).startswith("synth"):
            rates[u] = SYNTH_FPS
            continue
        with np.load(P.NOF_ALIGNED / f"{u}_aligned.npz", allow_pickle=True) as d:
            if "_metadata" not in d.files:
                raise SystemExit(f"{u}: no _metadata in {P.NOF_ALIGNED}; this script needs the v2 data")
            rates[u] = float(d["_metadata"].item()["fps"])
    return rates


def main():
    nf.apply_style()
    cal = P.CALIBRATION_DIR
    tab = pd.read_csv(cal / "calibration_table.csv")
    fp = pd.read_csv(cal / "false_pairs_delay.csv")
    rates = unit_rates(fp["unit"])
    fp["fps"] = fp["unit"].map(rates)
    fp["delay_s"] = fp["delay_frames"] / fp["fps"]

    fig = nf.figure(58)
    sd = SourceData("ED", "Extended Data: calibration of false discoveries")
    axa = nf.axes_mm(fig, 11, 4, 58, 38)
    axb = nf.axes_mm(fig, 82, 4, 58, 38, sharey=axa)
    axc = nf.axes_mm(fig, 152, 4, 28, 38)
    da = E.rate_panel(axa, tab, "", "p < 0.001")
    db = E.rate_panel(axb, tab, "_mi04", "p < 0.001 and excess MI > 0.04 bits")
    axa.set_ylim(0, max(axa.get_ylim()[1], axb.get_ylim()[1]))
    h, lab = axa.get_legend_handles_labels()
    h, lab = h[1:] + h[:1], lab[1:] + lab[:1]      # nominal line last
    fig.legend(h, lab, loc="lower left", ncol=4, frameon=False, bbox_to_anchor=(0.04, 0.0),
               handletextpad=0.3, columnspacing=1.0)

    sec = fp["delay_s"].to_numpy()
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
    sd.add("c", fp, "All false pairs with delay search (null, unit, neuron, feature, MI, excess MI, p, "
                    "delay in frames, frame rate of the unit, delay in seconds)")
    sd.add("table", tab, "Full calibration table (analyze.py): Clopper-Pearson and bootstrap intervals, "
                         "expected false pairs in the cohort of 30,105 neuron-sessions")
    sd.note("a", "False-positive rate per pair at p < 0.001 under three null models, without and with the "
                 "delay search, by variable type", "calibration_table.csv")
    sd.note("b", "The same with the excess-information criterion", "calibration_table.csv")
    sd.note("c", "Delays selected for the false pairs found with the delay search (all null models pooled), "
                 "converted to seconds at the frame rate of each unit", "false_pairs_delay.csv")
    sd.note("table", "Full calibration table behind panels a and b", "calibration_table.csv")
    sd.write(P.SOURCE_DATA_DIR / "ED_calibration_source_data.xlsx")
    print("rates of the units:", {f"{r:g}": int(sum(v == r for v in rates.values()))
                                  for r in sorted(set(rates.values()))}, flush=True)
    return nf.save(fig, str(P.out_path("ed_calibration")))


if __name__ == "__main__":
    main()
