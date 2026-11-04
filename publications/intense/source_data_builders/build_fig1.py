"""Source Data of Fig. 1.

usage: python -X utf8 -B source_data_builders/build_fig1.py
       -> <[figures] output_dir>/source_data/Fig1_source_data.xlsx

This builder imports figure_chain/figures_v11.1/work/fig1/make_fig1_v111.py, applies its settings(), draws
the figure again with the same composer and compares the PNG with the one that script wrote; the values of
the workbook are read from the drawn matplotlib objects and checked against the inputs the panel modules
read. Panel c is a schematic (raster image) and has no data.
"""
import numpy as np
import pandas as pd

import sd_common as C

MK = C.load_module("make_fig1_v111", C.WORK / "fig1" / "make_fig1_v111.py")
SCRIPT = "figure_chain/figures_v11.1/work/fig1/make_fig1_v111.py"
SRC = "figure_chain/figures_v11.1/work/fig1/fig1v8_src/fig1"
PROV = {}   # sheet -> input files and scripts, for the log of the builder (not written into the workbook)
GROUP_COLOURS = {"rejected": "#3b6fb6", "anti-selective": "#9e9e9e", "accepted": "#4daf4a"}
# display names of Fig. 3f (fig3f_heatmap_delegated.FEATURE_NAME_CORRECTIONS of figures_v11.1)
PAPER_NAME = {"bodydirection_harm2": "body direction", "headdirection_harm2": "head direction",
              "place_quad": "position", "speed_quad": "speed", "locomotion": "run",
              "object1": "object 1", "object2": "object 2", "object3": "object 3", "object4": "object 4",
              "objects": "all objects"}


def panel_a(fig, sd, log):
    axes = fig._subfigs["A"].axes
    keys = (36, 37, 38)                              # cache keys = rows of the older export
    rows = MK.f1a.NEURON_LABEL_IDX                   # -> rows of the synchronized file
    fps = float(MK.f1a.FPS)
    cache = np.load(C.SAVED / "fig1a_data.npz")
    traces, events = {}, []
    for ax, key in zip(axes, keys):
        tr = [ln for ln in C.data_lines(ax) if ln.get_marker() != "|"]
        tk = [ln for ln in ax.get_lines() if ln.get_marker() == "|"]
        assert len(tr) == 1 and len(tk) == 1, (key, len(tr), len(tk))
        t, y = np.asarray(tr[0].get_xdata(), float), np.asarray(tr[0].get_ydata(), float)
        C.assert_equal(t, np.arange(len(y)) / fps, f"1a time {key}")
        C.assert_equal(y, np.asarray(cache[f"neuron_{key}_ca"], float), f"1a trace {key}")
        if "time (s)" not in traces:
            traces["time (s)"] = t
        traces[f"neuron {rows[key]}: signal (a.u.)"] = y
        tick_x = np.sort(np.asarray(tk[0].get_xdata(), float))
        tick_y = np.asarray(tk[0].get_ydata(), float)
        first, end = C.runs(np.asarray(MK.f1a.EVENTS[key][:len(y)]))
        C.assert_equal(tick_x, (first + end) / 2 / fps, f"1a ticks {key}", atol=1e-9)
        assert np.all(tick_y == 1.12)
        for i, (a, b, x) in enumerate(zip(first, end, tick_x)):
            events.append(dict(neuron=rows[key], event=i + 1, **{"first frame": int(a), "last frame": int(b - 1),
                                                               "start (s)": a / fps, "end (s)": b / fps,
                                                               "tick position (s)": x}))
        log[f"1a_neuron_{rows[key]}"] = dict(frames=len(y), events=len(first))
    sd.add("a", pd.DataFrame(events),
           "Detected calcium events (grey ticks). The tick is drawn at the middle of the event, "
           f"(start + end) / 2; time = frame / {fps:g} frames/s (session rate). Frames are 0-based.")
    sd.add("a", pd.DataFrame(traces),
           f"Rescaled calcium fluorescence (blue) of neurons {', '.join(str(rows[k]) for k in keys)}, "
           f"session NOF_H01_1D (mouse H01, day 1), frames 0-{len(y) - 1}")
    sd.note("a", "Rescaled fluorescence traces of neurons 35, 36, 37 of mouse H01, day 1, and their detected "
                 "calcium events", "Session NOF_H01_1D: rescaled calcium of the three neurons (cache fig1a_data.npz); "
                 "events detected by the wavelet method (events_NOF_H01_1D.npz)")
    PROV["Fig1a"] = ["figure_chain/saved_inputs/fig1a_data.npz (keys neuron_36_ca..neuron_38_ca: rows of an earlier "
                     "export with one extra leading row = rows 35-37 of the synchronized file)",
                     "figure_chain/figures_v10.4/work/events_wvt.py -> figures_v10.4/work/events/events_NOF_H01_1D.npz "
                     "(n35_true_sp..n37_true_sp, fps 19.76)",
                     f"{SCRIPT}; {SRC}/fig1a_calcium_traces.py"]


def panel_b(fig, sd, log):
    axes = fig._subfigs["B"].axes
    fps = float(MK.f1b.FPS)
    cache = np.load(C.SAVED / "fig1b_data.npz")
    hd = C.data_lines(axes[0])
    xy = C.data_lines(axes[2])
    assert len(hd) == 1 and len(xy) == 2
    t = np.asarray(hd[0].get_xdata(), float)
    C.assert_equal(t, np.arange(len(t)) / fps, "1b time")
    C.assert_equal(hd[0].get_ydata(), np.asarray(cache["headdirection"], float), "1b head direction")
    C.assert_equal(xy[0].get_ydata(), np.asarray(cache["x"], float), "1b x")
    C.assert_equal(xy[1].get_ydata(), np.asarray(cache["y"], float), "1b y")
    sp = C.spans(axes[1])
    first, end = C.runs(cache["object4"])
    C.assert_equal(np.array(sp), np.column_stack([t[first], t[end - 1] + 1 / fps]), "1b periods")
    sd.add("b", pd.DataFrame(sp, columns=["start (s)", "end (s)"]),
           "Interaction with object 4 (grey periods, middle row): start = first frame / fps, "
           "end = (last frame + 1) / fps")
    sd.add("b", pd.DataFrame({"time (s)": t, "head direction (top, red)": hd[0].get_ydata(),
                              "object 4 interaction (0/1; shaded where 1)": np.asarray(cache["object4"], float),
                              "animal position x (bottom, magenta)": xy[0].get_ydata(),
                              "animal position y (bottom, cyan)": xy[1].get_ydata()}),
           f"Session NOF_H01_3D (mouse H01, day 3), frames 0-{len(t) - 1}; time = frame / {fps:g} "
           "frames/s (session rate). Values as drawn; the panel has no y scale (all three variables lie between 0 and 1)")
    sd.note("b", "Behavioral variables of mouse H01, day 3: head direction, interaction with object 4, "
                 "position x and y", "Session NOF_H01_3D: behavioral variables (cache fig1b_data.npz)")
    PROV["Fig1b"] = ["figure_chain/saved_inputs/fig1b_data.npz", "fps_by_session.csv (frame rate of NOF_H01_3D)",
                     f"{SCRIPT}; {SRC}/fig1b_behavioral_traces.py"]
    log["1b"] = dict(frames=len(t), periods=len(sp), hd_range=[float(np.min(cache["headdirection"])),
                                                              float(np.max(cache["headdirection"]))],
                     x_range=[float(np.min(cache["x"])), float(np.max(cache["x"]))])


def panel_d(fig, sd, log):
    ax0, ax1 = fig._subfigs["D"].axes[:2]
    fps = float(MK.f1c.FPS)
    tr = C.data_lines(ax0)
    assert len(tr) == 1
    t, y = np.asarray(tr[0].get_xdata(), float), np.asarray(tr[0].get_ydata(), float)
    C.assert_equal(t, np.arange(len(t)) / fps, "1d time")
    sp = C.spans(ax0)
    kde = C.data_lines(ax1)
    assert len(kde) == 2, len(kde)
    sd.add("d", pd.DataFrame(sp, columns=["start (s)", "end (s)"]),
           "Periods of interaction with object 4 (green): start = first frame / fps, end = (last frame + 1) / fps")
    dens = {}
    for ln in kde:
        name = ln.get_label()
        dens[f"{name}: log10(signal)"] = pd.Series(np.asarray(ln.get_xdata(), float))
        dens[f"{name}: density"] = pd.Series(np.asarray(ln.get_ydata(), float))
    sd.add("d", pd.DataFrame(dens),
           "Right: kernel density of log10(signal) inside (green) and outside (gray) the interaction periods "
           "(seaborn kdeplot, bw_adjust 0.1); the axes show log10(signal) from -4 to 0.5 and density up to 0.85. "
           "The densities are computed when the panel is drawn, from the calcium to which DRIADA adds uniform "
           "noise of amplitude 1e-8 without a fixed seed; between two renderings the curves inside the shown "
           "range differ by up to 0.0006 in density and 0.0003 in log10(signal)")
    sd.add("d", pd.DataFrame({"time (s)": t, "neuron 101: signal (a.u.)": y}),
           f"Left: rescaled fluorescence of neuron 101, session NOF_H01_2D (mouse H01, day 2), frames 0-{len(t) - 1}; "
           f"time = frame / {fps:g} frames/s (session rate)")
    sd.note("d", "Object-interaction cell: neuron 101 of mouse H01, day 2, its periods of interaction with "
                 "object 4 and the distributions of its signal inside and outside them",
            "Session NOF_H01_2D: calcium of neuron 101 rescaled by DRIADA 1.2.2 and the object 4 variable; "
            "densities computed when the panel is drawn")
    PROV["Fig1d"] = ["[v11] nof_v1 NOF_H01_2D_aligned.npz with figure_chain/saved_inputs/NOF_H01_2D_object4_earlier_export"
                     ".npz (row 101 of the synchronized file = row 102 of the earlier export the panel was drawn from), "
                     "built into neurons by DRIADA 1.2.2 at draw time; frame rate fps_by_session.csv",
                     f"{SCRIPT}; {SRC}/fig1c_neuron_feature_pair.py; figures_v10.4/work/nfp_fps.py"]
    log["1d"] = dict(frames=len(t), periods=len(sp), kde_points=[len(ln.get_xdata()) for ln in kde],
                     kde_labels=[ln.get_label() for ln in kde])


def panel_e(fig, sd, log):
    ax, ax_top, ax_right = fig._subfigs["E"].axes[:3]
    path = MK.NEW_PAIRS
    d = np.load(path)
    x = np.minimum(-np.log10(d["pval"]), MK.f1e.X_CLIP)
    y = np.log10(d["me_excess"])
    acc, anti = d["accepted"], d["anti_selective"]
    rej = ~acc & ~anti
    group = np.where(acc, "accepted", np.where(anti, "anti-selective", "rejected"))
    # the drawn points: one scatter, points in the shuffled order of the panel (seed 42)
    sc = [c for c in ax.collections if len(c.get_offsets()) > 0]
    assert len(sc) == 1
    off = np.asarray(sc[0].get_offsets(), float)
    order = np.random.default_rng(42).permutation(len(x))
    C.assert_equal(off[:, 0], x[order], "1e x")
    C.assert_equal(off[:, 1], y[order], "1e y")
    # marginal histograms as drawn (filled bars; the outlines repeat the same heights)
    xb = np.linspace(0, MK.f1e.X_CLIP, 41)
    yb = np.linspace(y.min(), y.max(), 41)
    from matplotlib.colors import to_rgb
    top, right = {}, {}
    for name, colour in GROUP_COLOURS.items():
        rgb = np.array(to_rgb(colour))
        bt = [p for p in ax_top.patches if hasattr(p, "get_height") and np.allclose(p.get_facecolor()[:3], rgb)
              and p.get_fill()]
        br = [p for p in ax_right.patches if hasattr(p, "get_height") and np.allclose(p.get_facecolor()[:3], rgb)
              and p.get_fill()]
        assert len(bt) == 40 and len(br) == 40, (name, len(bt), len(br))
        m = {"rejected": rej, "anti-selective": anti, "accepted": acc}[name]
        top[name] = np.array([p.get_height() for p in sorted(bt, key=lambda p: p.get_x())])
        right[name] = np.array([p.get_width() for p in sorted(br, key=lambda p: p.get_y())])
        C.assert_equal(top[name], np.histogram(x[m], bins=xb, density=True)[0], f"1e top {name}", atol=1e-12)
        C.assert_equal(right[name], np.histogram(y[m], bins=yb, density=True)[0], f"1e right {name}", atol=1e-12)
    vx = [ln for ln in ax.get_lines() if np.ptp(np.asarray(ln.get_xdata(), float)) == 0]
    hy = [ln for ln in ax.get_lines() if np.ptp(np.asarray(ln.get_ydata(), float)) == 0]
    assert len(vx) == 1 and len(hy) == 1
    x_thr, y_thr = float(vx[0].get_xdata()[0]), float(hy[0].get_ydata()[0])
    assert np.isclose(x_thr, 3.0) and np.isclose(y_thr, np.log10(0.04))
    summ = pd.DataFrame({
        "quantity": ["pairs drawn (passed Stage 1, have a Stage-2 p)", "accepted (green)",
                     "anti-selective (gray)", "rejected (blue)", "points with -log10 p clipped at 20",
                     "vertical dashed line: -log10 p", "horizontal dashed line: log10 excess MI",
                     "drawn as: p threshold", "drawn as: excess MI threshold (bits)"],
        "value": [len(x), int(acc.sum()), int(anti.sum()), int(rej.sum()), int((-np.log10(d["pval"]) > 20).sum()),
                  x_thr, y_thr, 10 ** -x_thr, 10 ** y_thr]})
    sd.add("e", summ, "Groups and thresholds. Accepted: Stage-2 significant, p < 0.001, excess MI > 0.04 bits and "
                      "activity ratio NaN or > 1; anti-selective: p < 0.001 but activity ratio <= 1 (lower activity "
                      "during the behavior, not counted as selective); rejected: the rest")
    sd.add("e", pd.DataFrame({"bin left: -log10 p": xb[:-1], "bin right: -log10 p": xb[1:],
                              **{f"density, {k}": v for k, v in top.items()}}),
           "Top marginal: histogram of -log10 p (clipped at 20) by group, density=True (each group integrates to 1)")
    sd.add("e", pd.DataFrame({"bin low: log10 excess MI": yb[:-1], "bin high: log10 excess MI": yb[1:],
                              **{f"density, {k}": v for k, v in right.items()}}),
           "Right marginal: histogram of log10(excess MI, bits) by group, density=True")
    feat = np.asarray(d["feature"]).astype(str)
    sd.add("e", pd.DataFrame({
        "session": d["session"], "neuron": d["neuron"], "feature (analysis name)": feat,
        "feature (name in the paper)": [PAPER_NAME.get(f, f) for f in feat],
        "Stage-2 p": d["pval"], "x drawn: min(-log10 p, 20)": x, "excess MI (bits)": d["me_excess"],
        "y drawn: log10 excess MI": y, "activity ratio (signal_ratio)": d["signal_ratio"],
        "Stage-2 significant": d["stage2"], "group": group}),
        "All drawn neuron-feature pairs of the 64 sessions, before disentanglement. neuron = row of the "
        "calcium matrix of the synchronized session file (0-based); the y axis is log scale (ticks 0.003-0.3 bits)")
    sd.note("e", "Neuron-behavior pairs of the hippocampal recordings in the significance-effect size plane, "
                 "with marginal histograms; thresholds p = 0.001 and 0.04 bits",
            "INTENSE results of the 64 sessions before disentanglement: pairs that passed Stage 1, with their "
            "Stage-2 p-value and excess MI (fig1e_real_pairs.npz)")
    PROV["Fig1e"] = [f"{C.rel(path)} (fig1/fig1_data.py on the INTENSE results computed at the session frame "
                     "rates)", f"{SCRIPT}; {SRC}/fig1e_scatter_marginals.py"]
    log["1e"] = dict(pairs=len(x), accepted=int(acc.sum()), anti=int(anti.sum()), rejected=int(rej.sum()))


def main():
    MK.settings()
    out = C.render_target("fig1")
    fig = MK.comp.create_fig1_delegated(output_dir=out, output_name="fig1", lowercase=True, save_formats=("png",))
    boxes = MK.panel_boxes(fig)
    log = dict(script=SCRIPT, png=C.compare_png(out / "fig1.png", C.FIG / "fig1.png", boxes))
    sd = C.SourceData(1, "Source Data for Figure 1 | Overview of INTENSE")
    panel_a(fig, sd, log)
    panel_b(fig, sd, log)
    sd.note("c", "Schematic of the INTENSE pipeline; no data", "-")
    panel_d(fig, sd, log)
    panel_e(fig, sd, log)
    sd.write(C.OUT / "Fig1_source_data.xlsx")
    log["provenance"] = PROV
    C.write_log("fig1", log)


if __name__ == "__main__":
    main()
