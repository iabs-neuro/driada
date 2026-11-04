"""Source Data of Fig. 4.

usage: python -X utf8 -B source_data_builders/build_fig4.py
       -> <[figures] output_dir>/source_data/Fig4_source_data.xlsx

This builder repeats the steps of main() of figure_chain/figures_v11.1/work/fig4/make_fig4_v111.py without the
files it writes, draws the figure and compares the PNG with the one that script wrote; the values of the
workbook are read from the drawn matplotlib objects and checked against the summary table of the benchmark.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import sd_common as C

MK = C.load_module("make_fig4_v111", C.WORK / "fig4" / "make_fig4_v111.py")
m = MK.m
SCRIPT = "figure_chain/figures_v11.1/work/fig4/make_fig4_v111.py"
SNRS, SKIPS = m.SNRS, m.SKIPS
NAMES = {"c": ["corr.", "MI", "corr. + shuffles", "INTENSE"],
         "d": ["average", "MI", "average + shuffles", "INTENSE"]}
PROV = {}


def nearest_skip(v):
    i = int(np.argmin(np.abs(SKIPS - v)))
    assert np.isclose(SKIPS[i], v)
    return i


def panel_a(fig, sd, log):
    axes = fig.get_axes()[:4]
    titles = [a.get_title() for a in axes]
    assert titles[0].startswith("1.") and titles[3].startswith("4."), titles
    fs = m.f4a.SAMPLING_RATE
    l0 = C.data_lines(axes[0])
    l2 = C.data_lines(axes[1])
    l3 = C.data_lines(axes[2])
    l4 = C.data_lines(axes[3])
    assert len(l0) == 1 and len(l2) == 2 and len(l3) == 1 and len(l4) == 1
    t = np.asarray(l0[0].get_xdata(), float)
    C.assert_equal(t, np.arange(m.f4a.LENGTH) / fs, "4a time")
    for ln in l2 + l3 + l4:
        C.assert_equal(ln.get_xdata(), t, "4a time of every line")
    band = np.asarray(axes[0].collections[0].get_paths()[0].vertices, float)
    lo, hi = float(band[:, 1].min()), float(band[:, 1].max())
    before = [ln for ln in l2 if ln.get_label() == "before filtering"][0]
    after = [ln for ln in l2 if ln.get_label() == "after filtering"][0]
    sel = C.spans(axes[0])
    rnd3, rnd4 = C.spans(axes[2]), C.spans(axes[3])
    assert rnd3 == rnd4
    f0, e0 = C.runs(np.asarray(before.get_ydata()))
    f1, e1 = C.runs(np.asarray(after.get_ydata()))
    C.assert_equal(np.array(sel), np.column_stack([t[f0], t[e0 - 1] + 1 / fs]), "4a periods of row 1")
    C.assert_equal(np.array(rnd3), np.column_stack([t[f1], t[e1 - 1] + 1 / fs]), "4a periods of rows 3, 4")
    sd.add("a", pd.DataFrame({"quantity": ["range of interest, lower border (drawn, + offset 2)",
                                           "range of interest, upper border (drawn, + offset 2)",
                                           "offset added to the fBM series in row 1", "sampling rate (Hz)",
                                           "Hurst exponent", "p_skip of this example", "seed"],
                              "value": [lo, hi, 2, fs, m.f4a.HURST, m.f4a.SKIP_PROB, m.f4a.SEED]}),
           "Parameters of the example and the range of interest (green band of row 1)")
    sd.add("a", pd.DataFrame(sel, columns=["start (s)", "end (s)"]),
           "Row 1: periods the variable spends in the range of interest (gray), gaps up to 0.5 s filled")
    sd.add("a", pd.DataFrame(rnd3, columns=["start (s)", "end (s)"]),
           "Rows 3, 4: the periods kept after filtering (dark gray), i.e. the runs of the filtered binary series")
    sd.add("a", pd.DataFrame({"time (s)": t, "row 1: fBM time series + 2 (a.u.)": l0[0].get_ydata(),
                              "row 2: binary series before filtering": before.get_ydata(),
                              "row 2: binary series after filtering": after.get_ydata(),
                              "row 3: event rate, smoothed (Gaussian, sd 3 s)": l3[0].get_ydata(),
                              "row 4: pseudo-calcium signal (a.u.)": l4[0].get_ydata()}),
           f"Series as drawn, {len(t)} samples at {fs} Hz")
    sd.note("a", "Stages of generating one pseudo-calcium signal tied to a continuous variable (example, p_skip 0.6)",
            "Synthetic example generated with fixed seed 13 by DRIADA 1.2.2 (driada.experiment.synthetic) when the "
            "panel is drawn")
    PROV["Fig4a"] = ["generated at draw time with the parameters of figure_chain/figures_v11.1/work/fig4/fig4_v7src/"
                     "fig4a_generation_pipeline.py (seed 13) by DRIADA 1.2.2",
                     f"{SCRIPT}; figure_chain/figures_v11.1/work/fig4/fig4_v7src/fig4a_generation_pipeline.py"]
    log["4a"] = dict(samples=len(t), periods_row1=len(sel), periods_rows34=len(rnd3), band=[lo, hi])


def table_grid(tab, ftype, j, col):
    q = tab[(tab.ftype == ftype) & (tab.method_idx == j)]
    g = np.full((len(SNRS), len(SKIPS)), np.nan)
    for _, r in q.iterrows():
        g[SNRS.index(float(r.snr)), nearest_skip(r.skip_prob)] = r[col]
    return g, q.method.iloc[0], q.variant.iloc[0]


def panels_bc(fig, sd, log, tab):
    heat = [(ax.get_title(), np.asarray(im.get_array(), float)) for ax in fig.get_axes() for im in ax.get_images()]
    assert len(heat) == 8
    for k, ftype, key in ((0, "c", "b"), (4, "d", "c")):
        rows = []
        for j in range(4):
            title, img = heat[k + j]
            assert title == NAMES[ftype][j], (title, ftype, j)
            g, method, variant = table_grid(tab, ftype, j, "f1")
            # the panel computes F1 from the precision and recall it is given: equal to the F1 column of the
            # table up to rounding (the v11.1 check: at most 3e-16)
            both = np.isfinite(img) & np.isfinite(g)
            assert np.array_equal(np.isfinite(img), np.isfinite(g)), (key, title)
            assert np.allclose(img[both], g[both], atol=1e-12, rtol=0), (key, title)
            log.setdefault("4bc_max_abs_diff_drawn_vs_table_f1", []).append(float(np.max(np.abs(img[both] - g[both]))))
            df = pd.DataFrame(img, columns=[f"p_skip = {v:.1f}" for v in SKIPS])
            df.insert(0, "SNR", SNRS)
            df.insert(0, "method", title)
            rows.append(df)
        sd.add(key, pd.concat(rows, ignore_index=True),
               f"F1 score drawn in each heat map ({'continuous' if ftype == 'c' else 'discrete'} variables): rows SNR "
               "(drawn bottom to top), columns p_skip (left to right); mean over 5 realizations of each condition")
        q = tab[tab.ftype == ftype].copy()
        q.insert(0, "method in the figure", [NAMES[ftype][i] for i in q.method_idx])
        q = q.drop(columns=["ftype", "main"]).rename(columns={"method": "method (table)",
                                                              "variant": "criterion (table)",
                                                              "skip_prob": "p_skip"})
        sd.add(key, q.sort_values(["method_idx", "snr", "p_skip"]).drop(columns=["method_idx"]),
               "Rows of the benchmark summary table behind the panel: precision, recall, detections, true and false "
               "positives (means over the realizations), number of realizations, realizations without detections, F1")
    sd.note("b", "F1 score over the grid of SNR and p_skip for continuous variables, four methods",
            "Synthetic benchmark: 20 behavioral and 500 pseudo-calcium signals per experiment, 5 realizations of each "
            "condition, threshold p < 0.001 for all methods (summary_table.csv)")
    sd.note("c", "Same as b for discrete variables", "as b")


def panels_de(fig, sd, log, tab):
    pr_axes = [ax for ax in fig.get_axes() if ax.get_xlabel() == "precision"]
    assert len(pr_axes) == 4
    k = 0
    for ftype, key in (("c", "d"), ("d", "e")):
        rows = []
        for side, ax in (("left", pr_axes[k]), ("right", pr_axes[k + 1])):
            colls = [c for c in ax.collections if len(c.get_offsets())]
            assert len(colls) == 4
            vl = [l for l in ax.get_lines() if np.ptp(np.asarray(l.get_xdata(), float)) == 0]
            assert len(vl) == 1 and float(vl[0].get_xdata()[0]) == 0.05
            for j, coll in enumerate(colls):
                off = np.asarray(coll.get_offsets(), float)
                gp, method, variant = table_grid(tab, ftype, j, "precision")
                gr, _, _ = table_grid(tab, ftype, j, "recall")
                gf, _, _ = table_grid(tab, ftype, j, "f1")
                if side == "left":
                    idx = [(SNRS.index(s), 0) for s in SNRS[1:]]
                else:
                    idx = [(len(SNRS) - 1, i) for i in range(len(SKIPS))]
                want = np.array([[np.nan_to_num(gp[a, b], nan=0.0), np.nan_to_num(gr[a, b], nan=0.0)] for a, b in idx])
                C.assert_equal(off, want, f"4{key} {side} {NAMES[ftype][j]}")
                for (a, b), (p, r) in zip(idx, off):
                    rows.append({"plot": "left: SNR varies, p_skip = 0" if side == "left" else
                                 "right: p_skip varies, SNR = 64",
                                 "method": NAMES[ftype][j], "SNR": SNRS[a], "p_skip": round(float(SKIPS[b]), 1),
                                 "precision (drawn)": p, "recall (drawn)": r, "F1": gf[a, b],
                                 "precision undefined (no detections), drawn as 0": bool(np.isnan(gp[a, b]))})
        k += 2
        sd.add(key, pd.DataFrame(rows),
               "Precision and recall of each method as drawn (one marker per condition; marker colour = SNR on the left "
               "plot, p_skip on the right plot); vertical lines at precision 0.05, the precision of a random guess")
        log[f"4{key}"] = dict(points=len(rows))
    sd.note("d", "Precision and recall of all methods for continuous variables at different SNR (p_skip = 0, left) "
                 "and different p_skip (SNR = 64, right)", "as b (summary_table.csv)")
    sd.note("e", "Same as d for discrete variables", "as b (summary_table.csv)")


def main():
    assert Path(m.comp.__file__).resolve().is_relative_to(MK.HERE), m.comp.__file__
    tab = m.load_table()
    c, d = m.arrays(tab, "c"), m.arrays(tab, "d")
    m.pr.PR_SELECT = "slice"
    m.f4a.EXACT_SHADING = True
    m.comp.create_pr_panel = MK.pr_panel
    out = C.render_target("fig4")
    fig = m.comp.create_fig4_delegated(output_dir=out, save_formats=("png",), c_data=c, d_data=d,
                                       output_name="fig4", lowercase=True)
    log = dict(script=SCRIPT, png=C.compare_png(out / "fig4.png", C.FIG / "fig4.png"))
    sd = C.SourceData(4, "Source Data for Figure 4 | Detection of neuron-behavior dependencies in synthetic data "
                         "with known ground truth")
    panel_a(fig, sd, log)
    panels_bc(fig, sd, log, tab)
    panels_de(fig, sd, log, tab)
    PROV["Fig4b, Fig4c, Fig4d, Fig4e"] = [
        f"{C.rel(m.TABLE)} (rows with main = True for the three other methods, INTENSE at variant stage2; "
        "make_fig4_v104.load_table)",
        f"{SCRIPT}; figure_chain/figures_v10.4/work/make_fig4_v104.py; figure_chain/figures_v11.1/work/fig4/fig4_v7src/fig4bc_heatmaps.py, "
        "fig4de_pr_curves.py"]
    sd.write(C.OUT / "Fig4_source_data.xlsx")
    log["provenance"] = PROV
    C.write_log("fig4", log)


if __name__ == "__main__":
    main()
