"""Source Data of Fig. 2.

usage: python -X utf8 -B source_data_builders/build_fig2.py
       -> <[figures] output_dir>/source_data/Fig2_source_data.xlsx

This builder imports figure_chain/figures_v11.1/work/fig2/make_fig2_v111.py, applies substitute() and
settings_v111() as its run('final') does, draws the figure again with the same composer and compares the PNG
with the one that script wrote; the values of the workbook are read from the drawn matplotlib objects and
checked against the inputs.
"""

import numpy as np
import pandas as pd

import sd_common as C

MK = C.load_module("make_fig2_v111", C.WORK / "fig2" / "make_fig2_v111.py")
SCRIPT = "figure_chain/figures_v11.1/work/fig2/make_fig2_v111.py (mode final)"
A103 = C.ART / "figures_v10.3" / "work" / "fig2_out" / "fig2a_data_v103.npz"
CELLS = C.ART / "figures_v10.4" / "work" / "cache" / "fig2_v104_cells_B.csv"
PROV = {}
CWT_STEP = 20     # frames between the columns of the CWT sheet (about 1 s)


def panel_a(fig, sd, log):
    ax1, ax2 = fig._sf_a.axes[:2]
    d = np.load(A103)
    fps = float(d["fps"])
    W = np.asarray(ax1.images[0].get_array())
    C.assert_equal(W, np.abs(d["Wx"]), "2a CWT")
    scales = np.asarray(d["wvt_scales"], float)
    freq = fps / (2 * np.pi * scales)
    trace = [ln for ln in ax2.get_lines() if ln.get_label() == "Fluorescence signal"]
    ticks = [ln for ln in ax2.get_lines() if ln.get_marker() == "|"]
    assert len(trace) == 1 and len(ticks) == 1
    x, y = np.asarray(trace[0].get_xdata(), float), np.asarray(trace[0].get_ydata(), float)
    C.assert_equal(x, np.arange(len(d["sig"])), "2a frames")
    C.assert_equal(y, d["sig"], "2a trace")
    first, end = C.runs(d["sp"])
    tx = np.sort(np.asarray(ticks[0].get_xdata(), float))
    C.assert_equal(tx, (first + end) / 2, "2a ticks", atol=1e-9)
    ytick_rows = np.asarray(ax1.get_yticks(), float)
    ytick_labels = [t.get_text() for t in ax1.get_yticklabels()]
    sd.add("a", pd.DataFrame({"event": np.arange(1, len(first) + 1), "first frame": first, "last frame": end - 1,
                              "start (s)": first / fps, "end (s)": end / fps, "tick position (s)": tx / fps}),
           f"Lower: detected calcium events (gray ticks, drawn at the middle of each event); time = frame / {fps:g} "
           "frames/s (session rate); frames are 0-based")
    sd.add("a", pd.DataFrame({"frequency label (Hz)": ytick_labels, "row of the CWT map (fractional)": ytick_rows,
                              "frequency of that row (Hz)": fps / (2 * np.pi * np.exp(np.interp(
                                  ytick_rows, np.arange(len(scales)), np.log(scales))))}),
           "Upper: frequency ticks of the CWT map and the rows they are drawn at")
    sd.add("a", pd.DataFrame({"frame": x.astype(int), "time (s)": x / fps, "signal (a.u.)": y}),
           f"Lower: fluorescence of neuron 59, mouse H31, day 1 (session NOF_H31_1D), all {len(x)} frames")
    keep = np.arange(0, W.shape[1], CWT_STEP)
    cwt = pd.DataFrame(W[:, keep], columns=[f"{k / fps:.2f} s" for k in keep])
    cwt.insert(0, "frequency (Hz)", freq)
    cwt.insert(0, "wavelet scale", scales)
    cwt.insert(0, "row (top = 0)", np.arange(len(scales)))
    sd.add("a-CWT", cwt,
           f"Upper: magnitude |W| of the continuous wavelet transform (generalized Morse wavelet) of the trace of "
           f"panel a, neuron 59 of H31 day 1. Rows: wavelet scales, top row of the image first; frequency = "
           f"{fps:g} / (2 pi scale). Columns: every {CWT_STEP}th frame (the image has {W.shape[1]} columns, one per "
           f"frame; the full map is computed from the trace in panel a)")
    sd.note("a", "CWT map (sheet Fig2a-CWT) and fluorescence trace of neuron 59, mouse H31, day 1, with detected "
                 "events", "Session NOF_H31_1D: rescaled and smoothed fluorescence of neuron 59, its CWT and "
                       "events detected on it (fig2a_data_v103.npz)")
    sd.note("a-CWT", "Magnitude of the CWT map of panel a, every 20th frame", "as panel a")
    PROV["Fig2a, Fig2a-CWT"] = [f"{C.rel(A103)} (figure_chain/figures_v10.3/work/fig2a_data_v103.py: row 59 of "
                                "[data] nof_aligned NOF_H31_1D_aligned.npz, events at 19.82 frames/s)",
                                f"{SCRIPT}; figure_chain/figures_v11.1/work/fig2/src/fig2a_wavelets_delegated.py"]
    log["2a"] = dict(frames=len(x), events=len(first), cwt_shape=list(W.shape), cwt_columns_in_book=len(keep),
                     ytick_labels=ytick_labels)


def panel_b(fig, sd, log):
    sf_b = [sf for sf in fig.subfigs[0].subfigs if sf is not fig._sf_a][0]
    ax, cax = sf_b.axes[:2]
    cache = np.load(MK.m.f2b.CACHE_FILE)
    from matplotlib import cm
    from matplotlib.colors import Normalize
    lc = [c for c in ax.collections if c.get_label() != "events"][0]
    seg = np.asarray(lc.get_segments(), float)
    C.assert_equal(seg[:, 0, 0], cache["x_ds"][:-1], "2b x")
    C.assert_equal(seg[:, 0, 1], cache["y_ds"][:-1], "2b y")
    C.assert_equal(seg[:, 1, 0], cache["x_ds"][1:], "2b x end")
    v = np.asarray(cache["neur_ds"], float)
    want = cm.get_cmap("jet")(Normalize(v.min(), v.max())(v))
    C.assert_equal(np.asarray(lc.get_colors())[:, :3], want[:, :3], "2b colours")
    st = [c for c in ax.collections if c.get_label() == "events"][0]
    off = np.asarray(st.get_offsets(), float)
    C.assert_equal(off, np.column_stack([cache["spike_x"], cache["spike_y"]]), "2b stars")
    ticks = np.asarray(cax.get_yticks(), float)
    labels = [t.get_text() for t in cax.get_yticklabels()]
    sd.add("b", pd.DataFrame({"colour bar tick: calcium value": labels, "tick position on the colour scale": ticks}),
           "Colour bar: calcium values placed where their colour lies on the scale; the colour of a segment is "
           "rescale(log(calcium + 0.05)), rescaled to 0-1 over the whole recording")
    sd.add("b", pd.DataFrame({"event": np.arange(1, len(off) + 1), "position x": off[:, 0], "position y": off[:, 1]}),
           f"Stars: animal position at the onset (first frame) of each of the {len(off)} detected events")
    sd.add("b", pd.DataFrame({"point": np.arange(len(cache["x_ds"])), "position x": cache["x_ds"],
                              "position y": cache["y_ds"],
                              "colour value of the segment from this point to the next (0-1)": np.r_[v, np.nan]}),
           "Trajectory of mouse H31, day 1 (session NOF_H31_1D), every 10th frame; positions rescaled to 0-1; "
           "segments coloured by the calcium activity of neuron 930 (jet colour map)")
    sd.note("b", "Trajectory of mouse H31, day 1, coloured by the calcium activity of place cell 930; stars, animal "
                 "position at the onset of each detected event", "Session NOF_H31_1D: positions, calcium of neuron "
                                                                 "930 and its detected events (fig2b_data_v104_onsets.npz)")
    PROV["Fig2b"] = [f"{C.rel(MK.m.f2b.CACHE_FILE)} (figure_chain/fig2_v10.4/work/fig2b_onsets.py; neuron 931 of the earlier "
                     "export = row 930 of the synchronized file)",
                     "figures_v11.1/work/fig2/cache/fig2b_cbar_ticks.json (colour bar ticks, "
                     "figure_chain/figures_v11.1/work/fig2/panel_b_ticks.py)",
                     f"{SCRIPT}; figure_chain/figures_v11.1/work/fig2/src/fig2b_trajectory_delegated.py"]
    log["2b"] = dict(points=len(cache["x_ds"]), stars=len(off), cbar_ticks=labels)


def panel_c(fig, sd, log, d, cells):
    sf_c = fig.subfigs[2].subfigs[0]
    ax = sf_c.axes[0]
    texts = [t.get_text() for t in ax.texts if t.get_text()]
    pi, pc = cells.pc_int.values, cells.pc_cl.values
    counts = dict(intense_only=int((pi & ~pc).sum()), both=int((pi & pc).sum()), classic_only=int((~pi & pc).sum()))
    assert counts == dict(intense_only=int(d["venn_intense_only"]), both=int(d["venn_both"]),
                          classic_only=int(d["venn_classic_only"])), counts
    assert len(cells) == int(d["venn_n_cells"])
    for k in ("venn_intense_only", "venn_both", "venn_classic_only"):
        assert str(int(d[k])) in texts, (k, texts)
    sd.add("c", pd.DataFrame({
        "group": ["INTENSE only", "both", "classic only", "INTENSE place cells (total)",
                  "classic place cells (total)", "all recorded neurons (neuron-sessions)", "Jaccard index",
                  "INTENSE place cells confirmed by the classic method (fraction)",
                  "classic place cells confirmed by INTENSE (fraction)"],
        "value": [int(d["venn_intense_only"]), int(d["venn_both"]), int(d["venn_classic_only"]),
                  int(d["venn_n_intense"]), int(d["venn_n_classic"]), int(d["venn_n_cells"]),
                  float(d["venn_jaccard"]), float(d["venn_frac_intense_confirmed"]),
                  float(d["venn_frac_classic_confirmed"])]}),
        "Place cells: INTENSE = position pair with p < 0.001 and excess MI > 0.04 bits; classic = event-rate map "
        "spatial information significant (Methods). Counts of neuron-sessions; per-neuron labels in sheet Fig2cef-cells")
    sd.add("c", pd.DataFrame({"text drawn on the panel": texts}), "Numbers as printed on the diagram")
    sd.note("c", "Overlap of place cells identified by INTENSE and by the classic event-based method among all "
                 "30,105 recorded neurons (neuron-sessions)", "Per-neuron place-cell labels of both methods, "
                                                             "64 sessions (sheet Fig2cef-cells)")
    log["2c"] = dict(counts=counts, texts=texts)


def panel_d(fig, sd, log):
    sf_d = fig.subfigs[2].subfigs[1]
    axs = [a for a in sf_d.axes if a.get_title() in ("Calcium", "Events")]
    assert [a.get_title() for a in axs] == ["Calcium", "Events"]
    cache = np.load(MK.m.f2d.CACHE_FILE)
    for ax, key in zip(axs, ("ca_map", "sp_map")):
        img = np.asarray(ax.images[0].get_array(), float)
        m = np.asarray(cache[key], float)
        C.assert_equal(img, (m - m.min()) / (m.max() - m.min()), f"2d {key}")
        df = pd.DataFrame(img, columns=[f"x bin {j}" for j in range(img.shape[1])])
        df.insert(0, "row", [f"y bin {img.shape[0] - 1 - i}" + (" (top row: largest y)" if i == 0 else "")
                             for i in range(img.shape[0])])
        sd.add("d", df, f"{'Left' if key == 'ca_map' else 'Right'}: 8 x 8 activity map of neuron 96, mouse H01, day 1, "
                        f"from {'the calcium signal above its 95th percentile' if key == 'ca_map' else 'detected events'}"
                        ", smoothed by one bin and normalized to [0, 1] (as drawn; top row = largest y)")
    sd.note("d", "Activity maps of neuron 96, mouse H01, day 1, from the calcium signal above its 95th percentile "
                 "(left) and from detected events (right)", "Session NOF_H01_1D: calcium, events and positions "
                                                             "(fig2d_data_v104.npz)")
    PROV["Fig2d"] = [f"{C.rel(MK.m.f2d.CACHE_FILE)} (figure_chain/fig2_v10.4/work/panel_bd_v104.py: row 96 of "
                     "[data] nof_aligned NOF_H01_1D_aligned.npz); each map normalized to [0, 1] when drawn",
                     f"{SCRIPT}; figure_chain/figures_v11.1/work/fig2/src/fig2d_activity_maps_delegated.py"]
    log["2d"] = "ok"


def panel_e(fig, sd, log, d):
    sf_e = fig.subfigs[4].subfigs[0]
    out = {"-log10 p threshold": d["xvals_e"]}
    for ax, key, name in zip(sf_e.axes[:3], ("mse", "psnr", "ssim"), ("MSE", "PSNR (dB)", "SSIM")):
        ln = [l for l in ax.get_lines() if l.get_label() in ("MSE", "PSNR", "SSIM")]
        assert len(ln) == 1
        C.assert_equal(ln[0].get_xdata(), d["xvals_e"], f"2e x {key}")
        C.assert_equal(ln[0].get_ydata(), d[f"av_{key}"], f"2e {key}")
        band = np.asarray(ax.collections[0].get_paths()[0].vertices, float)
        lo, hi = d[f"av_{key}"] - d[f"err_{key}"], d[f"av_{key}"] + d[f"err_{key}"]
        assert np.allclose(band[:, 1].min(), lo.min()) and np.allclose(band[:, 1].max(), hi.max()), key
        vl = [l for l in ax.get_lines() if np.ptp(np.asarray(l.get_xdata(), float)) == 0]
        assert len(vl) == 1 and float(vl[0].get_xdata()[0]) == 3.0
        out[f"{name}: mean"] = d[f"av_{key}"]
        out[f"{name}: s.e.m."] = d[f"err_{key}"]
    out["cells averaged (n)"] = d["n_e"]
    sd.add("e", pd.DataFrame(out),
           "Mean (line) and s.e.m. (shaded) over the neurons whose p is below the threshold under both methods; "
           "vertical dashed line at -log10 p = 3 (p = 0.001). Per-neuron values in sheet Fig2cef-cells")
    sd.note("e", "Similarity of the activity maps from the calcium signal and from events (MSE, PSNR, SSIM) as a "
                 "function of the confidence threshold", "Per-neuron map similarity and p-values of both methods "
                                                         "(sheet Fig2cef-cells)")
    log["2e"] = dict(points=len(d["xvals_e"]))


def panel_f(fig, sd, log, d):
    sf_f = fig.subfigs[4].subfigs[2]
    ax1, ax2 = sf_f.axes[:2]
    g = d["grid"]
    red = [l for l in ax1.get_lines() if l.get_label().startswith("INTENSE")][0]
    blue = [l for l in ax1.get_lines() if l.get_label().startswith("classic")][0]
    ok_i, ok_c = d["n_int"] >= 20, d["n_cl"] >= 20
    C.assert_equal(red.get_xdata(), g[ok_i], "2f red x")
    C.assert_equal(red.get_ydata(), 100 * d["conf_int"][ok_i], "2f red")
    C.assert_equal(blue.get_xdata(), g[ok_c], "2f blue x")
    C.assert_equal(blue.get_ydata(), 100 * d["conf_cl"][ok_c], "2f blue")
    left = pd.DataFrame({"-log10 p level": g,
                         "INTENSE cells at or above the level (n)": d["n_int"],
                         "INTENSE cells confirmed by classic (%) (red)": np.where(ok_i, 100 * d["conf_int"], np.nan),
                         "classic cells at or above the level (n)": d["n_cl"],
                         "classic cells confirmed by INTENSE (%) (blue)": np.where(ok_c, 100 * d["conf_cl"], np.nan)})
    sd.add("f", left, "Left: conditional confirmation. A curve is drawn only where at least 20 cells are at or above "
                      "the level (empty cells: not drawn); dashed vertical line at -log10 p = 3")
    keys = [str(k) for k in d["base_keys"]]
    exp = [l for l in ax2.get_lines() if l.get_label().startswith("expected")][0]
    obs = [l for l in ax2.get_lines() if l.get_label() == "INTENSE vs classic"][0]
    f = 100 * d["fracs"]
    C.assert_equal(obs.get_xdata(), f, "2f right x")
    C.assert_equal(obs.get_ydata(), d["j_obs"], "2f Jaccard")
    C.assert_equal(exp.get_ydata(), d[f"base_{keys[0]}"], "2f expected")
    assert len(ax2.get_lines()) == 2, "2f right: more lines than the two of v11.1"
    sd.add("f", pd.DataFrame({"top-ranked cells (%)": f, "Jaccard index, INTENSE vs classic (orange)": d["j_obs"],
                              f"expected for rank correlation {float(d['rho']):.4f} (dashed)": d[f"base_{keys[0]}"]}),
           f"Right: overlap (Jaccard index) of the top-ranked cells of the two methods; dashed, the overlap expected "
           f"for two statistics with the observed Spearman rank correlation rho = {float(d['rho']):.4f}")
    sd.note("f", "Agreement of the two methods as a function of detection confidence: conditional confirmation "
                 "(left) and Jaccard index of the top-ranked cells (right)",
            "Per-neuron p-values and place-cell labels of both methods (sheet Fig2cef-cells)")
    log["2f"] = dict(red_last_x=float(g[ok_i][-1]), red_last_pct=float(100 * d["conf_int"][ok_i][-1]),
                     blue_last_x=float(g[ok_c][-1]), blue_last_pct=float(100 * d["conf_cl"][ok_c][-1]))


def main():
    MK.substitute()
    MK.settings_v111()
    out = C.render_target("fig2")
    fig = MK.m.comp.create_fig2_delegated(output_dir=out, output_name="fig2", lowercase=True, save_formats=("png",))
    boxes = {k: (b[1], b[3], b[0], b[2]) for k, b in MK.m.BOX.items()}
    log = dict(script=SCRIPT, png=C.compare_png(out / "fig2.png", C.FIG / "fig2.png", boxes))
    d = dict(np.load(MK.m.pcef.DATA_NPZ))
    cells = pd.read_csv(CELLS)
    sd = C.SourceData(2, "Source Data for Figure 2 | Comparison of INTENSE with event-based place-cell detection")
    panel_a(fig, sd, log)
    panel_b(fig, sd, log)
    panel_c(fig, sd, log, d, cells)
    panel_d(fig, sd, log)
    panel_e(fig, sd, log, d)
    panel_f(fig, sd, log, d)
    sd.add("cef-cells", cells.rename(columns={
        "neuron": "neuron", "z_int": "INTENSE Stage-1 z (z_int)", "p_int": "INTENSE p, position (p_int)",
        "stage2": "INTENSE Stage-2 significant", "me_excess": "INTENSE excess MI, position (bits)",
        "pc_int": "INTENSE place cell", "z_cl": "classic z (z_cl)", "mse": "MSE of the two maps",
        "psnr": "PSNR of the two maps (dB)", "ssim": "SSIM of the two maps", "p_cl": "classic p (one-sided normal)",
        "pc_cl": "classic place cell"}),
        "Per neuron-session values behind panels c, e and f: session (mouse_day), neuron = row of the calcium matrix "
        "of the synchronized session file (0-based)")
    sd.note("cef-cells", "Per-neuron values behind panels c, e, f (labels, p-values, map similarity)",
            "INTENSE results and the classic event-based method on the 64 sessions (fig2_v104_cells_B.csv)")
    PROV["Fig2c, Fig2e, Fig2f"] = [f"{C.rel(MK.m.pcef.DATA_NPZ)} (figure_chain/figures_v11.1/work/fig2/panel_f_data.py: "
                                   "figures_v10.4/work/cache/fig2_v104_data_B.npz with the curves of 2f left on a grid "
                                   "to 30)", f"{SCRIPT}; figure_chain/figures_v11.1/work/fig2/src/fig2_panels_v8.py"]
    PROV["Fig2cef-cells"] = [f"{C.rel(CELLS)} (figure_chain/figures_v10.4/work/fig2_src/fig2_data_v104.py, classic variant B)"]
    sd.write(C.OUT / "Fig2_source_data.xlsx")
    log["provenance"] = PROV
    C.write_log("fig2", log)


if __name__ == "__main__":
    main()
