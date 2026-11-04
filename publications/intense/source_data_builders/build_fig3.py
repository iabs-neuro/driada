"""Source Data of Fig. 3.

usage: python -X utf8 -B source_data_builders/build_fig3.py
       -> <[figures] output_dir>/source_data/Fig3_source_data.xlsx

This builder imports figure_chain/figures_v11.1/work/fig3/make_fig3_v111.py, runs its build() and compares the
PNG with the one that script wrote; the values of the workbook are read from the drawn matplotlib objects
and checked against the inputs. Panel g is a schematic and has no data.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import sd_common as C

MK = C.load_module("make_fig3_v111", C.WORK / "fig3" / "make_fig3_v111.py")
m = MK.m
SCRIPT = "figure_chain/figures_v11.1/work/fig3/make_fig3_v111.py (mode final)"
PROV = {}


def trace_panel(sf, key, sd, log, *, mouse, day, neuron, feature, feat_label, delay, fps, colour_name, prov):
    ax0, ax1 = sf.axes[:2]
    tr = C.data_lines(ax0)
    assert len(tr) == 1
    t, y = np.asarray(tr[0].get_xdata(), float), np.asarray(tr[0].get_ydata(), float)
    ds = int(round((t[1] - t[0]) * fps))
    C.assert_equal(t, np.arange(0, len(t) * ds, ds) / fps, f"3{key} time")
    sp = C.spans(ax0)
    kde = C.data_lines(ax1)
    assert len(kde) == 2, len(kde)
    sd.add(key, pd.DataFrame(sp, columns=["start (s)", "end (s)"]),
           f"Periods of {feat_label} ({colour_name}): start = first frame / fps, end = (last frame + 1) / fps")
    dens = {}
    for ln in kde:
        dens[f"{ln.get_label()}: log10(signal)"] = pd.Series(np.asarray(ln.get_xdata(), float))
        dens[f"{ln.get_label()}: density"] = pd.Series(np.asarray(ln.get_ydata(), float))
    lo, hi = ax1.get_xlim()
    sd.add(key, pd.DataFrame(dens),
           f"Right: kernel density of log10(signal) during {feat_label} and in control periods (seaborn kdeplot, "
           f"bw_adjust 0.1); the axes show log10(signal) from {lo:g} to {hi:g}")
    sd.add(key, pd.DataFrame({"time (s)": t, f"neuron {neuron}: signal (a.u.)": y}),
           f"Left: rescaled fluorescence of neuron {neuron}, mouse {mouse}, day {day[0]} (session NOF_{mouse}_{day}), "
           f"shifted by the optimal delay of the pair, {delay} frames (the signal drawn at time t is the fluorescence "
           f"at t + {delay} frames); time = frame / {fps:g} frames/s (session rate)"
           + (f", every {ds}th frame" if ds > 1 else ""))
    PROV[f"Fig3{key}"] = prov
    log[f"3{key}"] = dict(frames=len(t), ds=ds, periods=len(sp), kde=[ln.get_label() for ln in kde],
                          density_xlim=[lo, hi])


def panel_c(fig, sd, log):
    sf, axes = fig._panel_c
    csv = m.f3c._PER_SESSION_CSV
    df = pd.read_csv(csv)
    nsel_m, nsel_s, del_m, del_s, _ = m.f3c.load_nsel_delay_stats(csv)
    out = []
    for ax, cols, xs, means, sems, names in (
            (axes[0], ["pct_1", "pct_2", "pct_3"], [1, 2, 3], nsel_m, nsel_s, ["1", "2", "3+"]),
            (axes[1], ["pct_neg", "pct_zero", "pct_pos"], [0, 1, 2], del_m, del_s, ["-", "0", "+"])):
        bars = sorted([p for p in ax.patches if hasattr(p, "get_height")], key=lambda p: p.get_x())
        assert len(bars) == 3
        C.assert_equal([p.get_height() for p in bars], means, f"3c bars {cols}")
        C.assert_equal([p.get_x() + p.get_width() / 2 for p in bars], xs, f"3c bar x {cols}", atol=1e-12)
        stars = {round(float(t.get_position()[0]), 6): t.get_text() for t in ax.texts if t.get_text()}
        ann = m.f3c._pairwise_wilcoxon(df, cols, xs)
        tests = []
        for (x1, x2, p) in ann:
            i, j = xs.index(x1), xs.index(x2)
            stat, p2 = wilcoxon(df[cols[i]].values, df[cols[j]].values, alternative="two-sided")
            assert p2 == p
            drawn = stars.get(round((x1 + x2) / 2, 6), "")
            assert drawn == (m.f3c._stars(p) or ""), (cols, x1, x2, drawn, p)
            tests.append(dict(comparison=f"{names[i]} vs {names[j]}", n_sessions=len(df), W=float(stat), p=float(p),
                              **{"asterisks drawn": drawn or "none (p >= 0.05)"}))
        out.append((ax, cols, names, means, sems, tests))
    sd.add("c", pd.DataFrame({
        "plot": ["left: number of selective variables"] * 3 + ["right: optimal delay sign"] * 3,
        "bar": ["1", "2", "3+", "negative (-)", "near zero (0)", "positive (+)"],
        "column of the per-session table": ["pct_1", "pct_2", "pct_3", "pct_neg", "pct_zero", "pct_pos"],
        "mean (% of recorded neurons)": np.r_[nsel_m, del_m], "s.e.m.": np.r_[nsel_s, del_s],
        "n sessions": [len(df)] * 6}),
        "Bar heights (mean) and error bars (s.e.m. = s.d. / sqrt(n)) over the per-session percentages")
    sd.add("c", pd.DataFrame([dict(plot="left", **t) for t in out[0][5]] + [dict(plot="right", **t) for t in out[1][5]]),
           "Two-sided Wilcoxon signed-rank tests between the bars of each plot, paired by session (scipy.stats."
           "wilcoxon); asterisks: * p < 0.05, ** p < 0.01, *** p < 0.001, **** p < 0.0001")
    sd.add("c", df.rename(columns={
        "n": "recorded neurons", "pct_1": "% selective to 1 variable", "pct_2": "% selective to 2 variables",
        "pct_3": "% selective to 3 or more variables", "pct_neg": "% pairs with negative delay",
        "pct_zero": "% pairs with near-zero delay", "pct_pos": "% pairs with positive delay"}),
        f"Per-session values ({len(df)} sessions of {df.mouse.nunique()} mice), after disentanglement; all "
        "percentages are of the recorded neurons of the session (delay columns: selective neuron-variable pairs "
        "with that delay sign)")
    sd.note("c", "Percentage of recorded neurons selective to 1, 2, 3+ variables after disentanglement (left) and "
                 "selective pairs by optimal-delay sign (right); mean +- s.e.m. over 64 sessions, Wilcoxon tests",
            "Per-session counts after disentanglement (fig3c_per_session.csv)")
    PROV["Fig3c"] = [f"{C.rel(csv)}", f"{SCRIPT}; figure_chain/figures_v10.4/work/fig3_src/fig3c_bars_delegated.py "
                                      "(V7_BRACKETS = 'stacked', starbars)"]
    log["3c"] = dict(csv=C.rel(csv), tests=[t for o in out for t in o[5]])


def panel_d(fig, sd, log):
    ax = fig._subfigs["D"].axes[0]
    bars = sorted(ax.patches, key=lambda p: p.get_x())
    centre = np.array([p.get_x() + p.get_width() / 2 for p in bars])
    h = np.array([p.get_height() for p in bars])
    ang, sig = MK.load_body()
    hist, centres = MK.tuning(ang, sig)
    C.assert_equal(h, hist, "3d bars", atol=1e-12)
    C.assert_equal(centre, centres, "3d centres", atol=1e-12)
    ln = C.data_lines(ax)
    assert len(ln) == 1
    th, r = np.asarray(ln[0].get_xdata(), float), np.asarray(ln[0].get_ydata(), float)
    assert ax.get_theta_offset() == 0 and ax.get_theta_direction() == 1
    sd.add("d", pd.DataFrame({"bin centre (rad)": centre, "bin centre (deg)": np.degrees(centre),
                              "mean signal in the bin (bar length, a.u.)": h}),
           "Bars: body direction in 40 bins; bar length = mean rescaled signal of the frames in the bin "
           "(occupancy-normalised); angles from the x axis of the arena (map e), counter-clockwise, 0 deg to the right")
    sd.add("d", pd.DataFrame({"angle (rad)": th, "angle (deg)": np.degrees(th), "smoothed tuning (a.u.)": r}),
           "Green line: the binned tuning curve smoothed by a circular Gaussian (sd 0.04 of the circle); the last "
           "point repeats the first to close the curve")
    sd.note("d", "Body-direction tuning of neuron 215, mouse H02, day 4 (selective for body direction and position "
                 "after disentanglement, both at zero delay)",
            "Session NOF_H02_4D: body direction and calcium of neuron 215 (min-max rescaled)")
    PROV["Fig3d"] = ["[data] nof_aligned NOF_H02_4D_aligned.npz: bodydirection, Calcium row 215; delay 0 "
                     "(figure_chain/compute_pack_v11.1/scripts/f_candidates.py -> compute_pack_v11.1/F_fig3/candidates.json, "
                     "candidate 3)",
                     f"{SCRIPT}: load_body() + figure_chain/figures_v11.1/work/fig3/fig3d_polar_delegated.py"]
    pref = float(np.degrees(np.arctan2((h * np.sin(centre)).sum(), (h * np.cos(centre)).sum())))
    log["3d"] = dict(bars=len(h), npz=C.rel(MK.F3D._DEFAULT_ALIGNED_NPZ), neuron=MK.F3D._DEFAULT_NEURON_IDX,
                     delay=MK.F3D._DEFAULT_OPTIMAL_DELAY, preferred_deg_from_bars=pref)


def panel_e(fig, sd, log):
    ax = fig._subfigs["E"].axes[0]
    cs = ax.collections[0]
    x, y, s = m.f3e.load_place_data()
    X, Y, Z = m.f3e._build_heatmap(x, y, s)
    levels = np.asarray(cs.levels, float)
    assert levels[0] <= Z.min() and levels[-1] >= Z.max(), (levels[[0, -1]], Z.min(), Z.max())
    xg, yg = X[0], Y[:, 0]
    df = pd.DataFrame(Z, columns=[f"x = {v:.4f}" for v in xg])
    df.insert(0, "y", yg)
    sd.add("e", pd.DataFrame({"contour level (a.u.)": levels}),
           f"Filled contours: {len(levels) - 1} colour bands between these levels (jet colour map)")
    sd.add("e", df, "Activity map drawn with filled contours: mean rescaled signal of the frames in each cell of a "
                    "100 x 100 grid over the positions (rescaled to 0-1), then Gaussian smoothing (sigma 5 cells); "
                    "unvisited cells count as 0. Rows: y from the bottom of the map (y = 0) to the top; columns: x "
                    "from left to right")
    iy, ix = np.unravel_index(np.argmax(Z), Z.shape)
    sd.note("e", "Spatial activity map of the same neuron (neuron 215, mouse H02, day 4)",
            "Session NOF_H02_4D: positions x, y and calcium of neuron 215 (min-max rescaled)")
    PROV["Fig3e"] = [f"[data] nof_aligned NOF_H02_4D_aligned.npz: x, y, Calcium row 215; delay {m.f3e._DEFAULT_OPTIMAL_DELAY}",
                     f"{SCRIPT}; figure_chain/figures_v10.4/work/fig3_src/fig3e_heatmap_place_delegated.py"]
    log["3e"] = dict(npz=C.rel(m.f3e._DEFAULT_ALIGNED_NPZ), neuron=m.f3e._DEFAULT_NEURON_IDX,
                     delay=m.f3e._DEFAULT_OPTIMAL_DELAY, levels=len(levels), peak_xy=[float(xg[ix]), float(yg[iy])],
                     z_max=float(Z.max()))


def panel_f(fig, sd, log):
    ax = fig._subfigs["F"].axes[0]
    path = MK.F3F._DEFAULT_MATRIX_NPZ
    z = np.load(path, allow_pickle=True)
    names = [str(n) for n in z["names"]]
    shown = [t.get_text() for t in ax.get_yticklabels()]
    assert shown == [MK.F3F.FEATURE_NAME_CORRECTIONS.get(n, n) for n in names], shown
    arr = ax.collections[0].get_array()
    drawn = np.ma.filled(np.ma.asarray(arr, float), np.nan).reshape(len(names), len(names))
    fmi = np.array(z["mi_mean"], float)
    fmi[z["sig_frac"] < 0.5] = np.nan
    fmi[np.eye(len(names), dtype=bool)] = np.nan
    C.assert_equal(drawn, fmi, "3f matrix")
    sd.add("f", pd.DataFrame(drawn, columns=shown).set_index(pd.Index(shown, name="variable")).reset_index(),
           "Mutual information between behavioral variables (bits), mean over sessions, as drawn; empty cells are "
           "drawn gray: the diagonal and pairs significantly related (p < 0.01) in fewer than half of the sessions "
           "in which both variables exist")
    sd.add("f", pd.DataFrame(np.array(z["mi_mean"], float), columns=shown)
           .set_index(pd.Index(shown, name="variable")).reset_index(),
           "Mean MI over sessions (bits) for all pairs, including those drawn gray")
    sd.add("f", pd.DataFrame(np.array(z["sig_frac"], float), columns=shown)
           .set_index(pd.Index(shown, name="variable")).reset_index(),
           "Fraction of sessions in which the pair is significantly related (pairs below 0.5 are drawn gray)")
    sd.add("f", pd.DataFrame(np.array(z["n_sessions"], int), columns=shown)
           .set_index(pd.Index(shown, name="variable")).reset_index(),
           "Number of sessions in which both variables exist")
    sd.note("f", "Mutual information between behavioral variables averaged over sessions; pairs rarely significantly "
                 "related shown in gray", "Variable-variable MI and its significance in each of the 64 sessions "
                                          "(fig3f_matrix.npz)")
    PROV["Fig3f"] = [C.rel(path), f"{SCRIPT}; figure_chain/figures_v11.1/work/fig3/fig3f_heatmap_delegated.py"]
    log["3f"] = dict(npz=C.rel(path), names=shown)


def main():
    out = C.render_target("fig3")
    fig = MK.build(out, "fig3", ("png",))
    boxes = {k: tuple(v) for k, v in m.panel_boxes(fig).items()}
    log = dict(script=SCRIPT, png=C.compare_png(out / "fig3.png", C.FIG / "fig3.png", boxes))
    sd = C.SourceData(3, "Source Data for Figure 3 | Selectivity of hippocampal CA1 neurons and its disentanglement")
    fa = float(m.f3a.FPS)
    trace_panel(fig._subfigs["A"], "a", sd, log, mouse="H02", day="3D", neuron=312, feature="object3",
                feat_label="interaction with object 3", delay=int(m.f3a._DEFAULT_OPTIMAL_DELAY), fps=fa,
                colour_name="green",
                prov=["[v11] nof_v1 NOF_H02_3D_aligned.npz "
                      "(figure_chain/figures_v10.4/work/fig3_src/_v122.load_session, seed 42), Calcium row 312, object3; delay "
                      f"{m.f3a._DEFAULT_OPTIMAL_DELAY} frames; frame rate fps_by_session.csv",
                      f"{SCRIPT}; figure_chain/figures_v10.4/work/fig3_src/fig3a_object_delegated.py; "
                      "figure_chain/figures_v10.4/work/nfp_fps.py"])
    assert (m.f3a._MOUSE, m.f3a._SESSION, m.f3a._DEFAULT_NEURON_IDX, m.f3a._DEFAULT_FEATURE) == ("H02", "3D", 312,
                                                                                               "object3")
    f3b = MK.f3b
    assert (f3b._MOUSE, f3b._SESSION, f3b._DEFAULT_NEURON_IDX, f3b._DEFAULT_FEATURE) == ("H08", "4D", 121, "rear")
    trace_panel(fig._subfigs["B"], "b", sd, log, mouse="H08", day="4D", neuron=121, feature="rear",
                feat_label="rearing", delay=int(f3b._DEFAULT_OPTIMAL_DELAY), fps=float(f3b.FPS), colour_name="orange",
                prov=["[v11] nof_v1 NOF_H08_4D_aligned.npz "
                      "(_v122.load_session, seed 42), Calcium row 121, rear; delay "
                      f"{f3b._DEFAULT_OPTIMAL_DELAY} frames; frame rate fps_by_session.csv",
                      f"{SCRIPT}; figure_chain/figures_v11/work/fig3b_rear_v11.py (FIT_DENSITY_YLIM = True); "
                      "figure_chain/figures_v10.4/work/nfp_fps.py"])
    sd.note("a", "Object-interaction neuron 312 of mouse H02, day 3: fluorescence with periods of interaction with "
                 "object 3, and distributions of its signal during interaction and control periods",
            "Session NOF_H02_3D: calcium of neuron 312 rescaled by DRIADA 1.2.2 and the object 3 variable; "
            "densities computed when the panel is drawn")
    sd.note("b", "Rearing neuron 121 of mouse H08, day 4: fluorescence with rearing periods, and distributions of "
                 "its signal during rearing and control periods",
            "Session NOF_H08_4D: calcium of neuron 121 rescaled by DRIADA 1.2.2 and the rearing variable; "
            "densities computed when the panel is drawn")
    panel_c(fig, sd, log)
    panel_d(fig, sd, log)
    panel_e(fig, sd, log)
    panel_f(fig, sd, log)
    sd.note("g", "Schematic of the interdependencies between information measures; no data", "-")
    sd.write(C.OUT / "Fig3_source_data.xlsx")
    log["provenance"] = PROV
    C.write_log("fig3", log)


if __name__ == "__main__":
    main()
