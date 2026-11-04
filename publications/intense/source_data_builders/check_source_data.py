"""Check of the Source Data workbooks against their inputs and, when --tex is given, against the text.

usage: python -X utf8 -B source_data_builders/check_source_data.py [--tex MANUSCRIPT.tex]
       # one line per check, [OK] / [FAILED]; exit 1 on a failure

For every panel the input files are read again (figure caches, session files, tables) and compared
with the sheet of the workbook; no figure is drawn. Where a panel computes its values while drawing
(the rescaled traces of 1d, 3a, 3b, the binned tuning of 3d, the map of 3e, the synthetic example of
4a), the same arithmetic is repeated here on the input. The kernel densities of 1d, 3a and 3b depend on
noise that DRIADA adds to the calcium when it builds the neurons, so for them only the shape of a
density is checked here. With --tex a few numbers of the workbooks are compared with the captions and
the text of the manuscript.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import warnings

import openpyxl
import pandas as pd

warnings.simplefilter("ignore", FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figure_chain"))
import _paths_fig as _F  # noqa: E402  (puts the paper folder on sys.path)
import intense_paths as P  # noqa: E402

ART = _F.D                      # [v11] figure_artifacts: what the producers of figure_chain wrote
OUT = P.SOURCE_DATA_DIR
SAVED = _F.SAVED                # figure_chain/saved_inputs
TEX = None                      # --tex
FAILED = []


def check(name, ok, detail=""):
    print(f"[{'OK' if ok else 'FAILED'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    if not ok:
        FAILED.append(name)


# The workbooks store a number with 16 significant digits (the openpyxl writer of SourceData), so a value
# read back can differ from the double it was written from in the last digit: relative 1e-15.
RTOL = 1e-15


def same(a, b, atol=0.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    ok = np.allclose(a, b, atol=atol, rtol=RTOL, equal_nan=True)
    both = np.isfinite(a) & np.isfinite(b)
    return ok, f"max |diff| {float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0:.3g}"


def blocks(book, sheet):
    """Tables of a sheet written by SourceData: caption row, header row, data rows; blank rows between."""
    rows = list(book[sheet].iter_rows(values_only=True))
    out, i = [], 0
    while i < len(rows):
        if all(v is None for v in rows[i]):
            i += 1
            continue
        cap, hdr, j = rows[i][0], rows[i + 1], i + 2
        while j < len(rows) and not all(v is None for v in rows[j]):
            j += 1
        n = max(k for k, v in enumerate(hdr) if v is not None) + 1
        df = pd.DataFrame([r[:n] for r in rows[i + 2:j]], columns=list(hdr[:n]))
        out.append(df.apply(pd.to_numeric, errors="ignore") if len(df) else df)
        i = j
    return out


def strip(row):
    """A row of cell values without the empty cells at its end."""
    row = list(row)
    while row and row[-1] is None:
        row.pop()
    return row


def num(df, col):
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float)


def fps_of(session):
    with open(_F.V.get("fps_table"), newline="") as f:
        return {r["session"]: float(r["fps"]) for r in csv.DictReader(f)}[session]


def runs(binary):
    b = (np.asarray(binary) == 1).astype(np.int8)
    e = np.flatnonzero(np.diff(np.r_[0, b, 0]) != 0)
    return e[0::2], e[1::2]


def scaled(trace):
    """neuron.ca.scdata of DRIADA 1.2.2 without its noise of 1e-8: negative values to 0, min-max scaling."""
    ca = np.asarray(trace, float).copy()
    ca[ca < 0] = 0
    return (ca - ca.min()) / (ca.max() - ca.min())


def density_shape(name, df):
    """Each kernel density: 200 points, integral close to 1 over its support (seaborn kdeplot, cut 3)."""
    cols = list(df.columns)
    for k in range(0, len(cols), 2):
        x, y = num(df, cols[k]), num(df, cols[k + 1])
        x, y = x[np.isfinite(x)], y[np.isfinite(y)]
        area = float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))
        check(f"{name} density '{cols[k + 1]}': 200 points, area 1", len(x) == 200 and abs(area - 1) < 0.01,
              f"area {area:.4f}")


def periods_check(name, df, binary, fps):
    t = np.arange(len(binary)) / fps
    a, e = runs(binary)
    ok, det = same(df.to_numpy(float), np.column_stack([t[a], t[e - 1] + 1 / fps]))
    check(f"{name} periods ({len(a)}) = runs of the input variable", ok, det)


# ---------------------------------------------------------------------------------------------- Fig. 1
def fig1():
    wb = openpyxl.load_workbook(OUT / "Fig1_source_data.xlsx", read_only=True)
    cache = SAVED
    # a
    ev = np.load(ART / "figures_v10.4" / "work" / "events" / "events_NOF_H01_1D.npz")
    fps = float(ev["fps"])
    check("1a frame rate of the events file = fps_by_session.csv", fps == fps_of("NOF_H01_1D"), f"{fps}")
    tab, tr = blocks(wb, "Fig1a")
    ca = np.load(cache / "fig1a_data.npz")
    for r in (35, 36, 37):
        f, e = runs(ev[f"n{r}_true_sp"][:5000])
        q = tab[tab.neuron == r]
        ok = (np.array_equal(num(q, "first frame"), f) and np.array_equal(num(q, "last frame"), e - 1)
              and same(num(q, "start (s)"), f / fps)[0] and same(num(q, "end (s)"), e / fps)[0]
              and same(num(q, "tick position (s)"), (f + e) / 2 / fps, 1e-9)[0])
        check(f"1a neuron {r}: {len(f)} events and ticks = events file", ok)
        check(f"1a neuron {r}: trace = fig1a_data.npz key neuron_{r + 1}_ca",
              *same(num(tr, f"neuron {r}: signal (a.u.)"), ca[f"neuron_{r + 1}_ca"]))
    check("1a time = frame / fps", *same(num(tr, "time (s)"), np.arange(5000) / fps))
    # b
    per, ser = blocks(wb, "Fig1b")
    c = np.load(cache / "fig1b_data.npz")
    fb = fps_of("NOF_H01_3D")
    check("1b time = frame / fps of NOF_H01_3D", *same(num(ser, "time (s)"), np.arange(len(c["x"])) / fb))
    for col, key in (("head direction (top, red)", "headdirection"), ("animal position x (bottom, magenta)", "x"),
                     ("animal position y (bottom, cyan)", "y"),
                     ("object 4 interaction (0/1; shaded where 1)", "object4")):
        check(f"1b {key} = fig1b_data.npz", *same(num(ser, col), c[key]))
    periods_check("1b object 4", per, c["object4"], fb)
    # d
    per, dens, tr = blocks(wb, "Fig1d")
    calcium = np.load(_F.V.get("nof_v1") / "NOF_H01_2D_aligned.npz", allow_pickle=True)["Calcium"]
    object4 = np.load(SAVED / "NOF_H01_2D_object4_earlier_export.npz")["object4"]
    fd = fps_of("NOF_H01_2D")
    check("1d time = frame / fps of NOF_H01_2D", *same(num(tr, "time (s)"), np.arange(6000) / fd))
    check("1d trace = row 101 of the synchronized file, rescaled",
          *same(num(tr, "neuron 101: signal (a.u.)"), scaled(calcium[101])[:6000], 1e-6))
    periods_check("1d object 4", per, object4[:6000], fd)
    density_shape("1d", dens)
    # e
    summ, top, right, pairs = blocks(wb, "Fig1e")
    d = np.load(P.CACHE_DIR / "fig1e_real_pairs.npz")
    x = np.minimum(-np.log10(d["pval"]), 20.0)
    y = np.log10(d["me_excess"])
    group = np.where(d["accepted"], "accepted", np.where(d["anti_selective"], "anti-selective", "rejected"))
    ok = (len(pairs) == len(x) and list(pairs["session"]) == list(d["session"])
          and np.array_equal(num(pairs, "neuron"), d["neuron"])
          and list(pairs["feature (analysis name)"]) == [str(v) for v in d["feature"]]
          and same(num(pairs, "Stage-2 p"), d["pval"])[0] and same(num(pairs, "x drawn: min(-log10 p, 20)"), x)[0]
          and same(num(pairs, "excess MI (bits)"), d["me_excess"])[0]
          and same(num(pairs, "y drawn: log10 excess MI"), y)[0]
          and same(num(pairs, "activity ratio (signal_ratio)"), d["signal_ratio"])[0]
          and list(pairs["Stage-2 significant"]) == list(d["stage2"]) and list(pairs["group"]) == list(group))
    check(f"1e all {len(x)} pairs = fig1e_real_pairs.npz", ok)
    xb, yb = np.linspace(0, 20, 41), np.linspace(y.min(), y.max(), 41)
    for g in ("rejected", "anti-selective", "accepted"):
        m = group == g
        check(f"1e top histogram, {g}", *same(num(top, f"density, {g}"), np.histogram(x[m], bins=xb, density=True)[0],
                                               1e-12))
        check(f"1e right histogram, {g}", *same(num(right, f"density, {g}"),
                                                 np.histogram(y[m], bins=yb, density=True)[0], 1e-12))
    v = dict(zip(summ["quantity"], summ["value"]))
    check("1e counts by group", (v["accepted (green)"], v["anti-selective (gray)"], v["rejected (blue)"])
          == (int(d["accepted"].sum()), int(d["anti_selective"].sum()), int((group == "rejected").sum())))
    return dict(thr_p=float(v["drawn as: p threshold"]), thr_mi=float(v["drawn as: excess MI threshold (bits)"]))


# ---------------------------------------------------------------------------------------------- Fig. 2
def fig2():
    wb = openpyxl.load_workbook(OUT / "Fig2_source_data.xlsx", read_only=True)
    # a
    d = np.load(ART / "figures_v10.3" / "work" / "fig2_out" / "fig2a_data_v103.npz")
    fps = float(d["fps"])
    check("2a frame rate 19.82 = fps_by_session.csv NOF_H31_1D", fps == fps_of("NOF_H31_1D"), f"{fps}")
    ev, fq, tr = blocks(wb, "Fig2a")
    f, e = runs(d["sp"])
    ok = (np.array_equal(num(ev, "first frame"), f) and np.array_equal(num(ev, "last frame"), e - 1)
          and same(num(ev, "tick position (s)"), (f + e) / 2 / fps)[0])
    check(f"2a {len(f)} events and ticks = fig2a_data_v103.npz", ok)
    check("2a trace", *same(num(tr, "signal (a.u.)"), d["sig"]))
    sc = np.asarray(d["wvt_scales"], float)
    # as freq_rows() of the panel, which takes the logarithm of the float32 scales of the file
    rows = np.interp(np.log(fps / (2 * np.pi * np.array([3, 1, 0.3, 0.1]))), np.log(d["wvt_scales"]), np.arange(len(sc)))
    check("2a frequency ticks 3, 1, 0.3, 0.1 Hz on their rows",
          same(num(fq, "frequency label (Hz)"), [3, 1, 0.3, 0.1])[0]
          and same(num(fq, "row of the CWT map (fractional)"), rows, 1e-9)[0])
    (cwt,) = blocks(wb, "Fig2a-CWT")
    W = np.abs(d["Wx"])[:, ::20]
    check("2a CWT |W| every 20th frame", *same(cwt.iloc[:, 3:].to_numpy(float), W))
    check("2a CWT row frequencies = fps / (2 pi scale)", *same(num(cwt, "frequency (Hz)"), fps / (2 * np.pi * sc)))
    # b
    tk, st, pts = blocks(wb, "Fig2b")
    b = np.load(SAVED / "fig2b_data_v104_onsets.npz")
    j = json.loads((ART / "figures_v11.1" / "work" / "fig2" / "cache" / "fig2b_cbar_ticks.json").read_text("utf-8"))
    check("2b trajectory and colour values", same(num(pts, "position x"), b["x_ds"])[0]
          and same(num(pts, "position y"), b["y_ds"])[0]
          and same(num(pts, "colour value of the segment from this point to the next (0-1)"), np.r_[b["neur_ds"], np.nan])[0])
    check(f"2b {len(b['spike_x'])} stars", same(num(st, "position x"), b["spike_x"])[0]
          and same(num(st, "position y"), b["spike_y"])[0])
    check("2b colour bar ticks = fig2b_cbar_ticks.json",
          [float(v) for v in tk["colour bar tick: calcium value"]] == [float(v) for v in j["ticks_calcium"]]
          and same(num(tk, "tick position on the colour scale"), j["ticks_position"])[0])
    # c, e, f
    n = np.load(ART / "figures_v11.1" / "work" / "fig2" / "cache" / "fig2_v111_data_B.npz")
    grp, txt = blocks(wb, "Fig2c")
    v = dict(zip(grp["group"], grp["value"]))
    keys = {"INTENSE only": "venn_intense_only", "both": "venn_both", "classic only": "venn_classic_only",
            "INTENSE place cells (total)": "venn_n_intense", "classic place cells (total)": "venn_n_classic",
            "all recorded neurons (neuron-sessions)": "venn_n_cells", "Jaccard index": "venn_jaccard"}
    check("2c counts = fig2_v111_data_B.npz", all(np.isclose(float(v[k]), float(n[kk]), rtol=RTOL, atol=0)
                                                   for k, kk in keys.items()),
          ", ".join(f"{k}: {v[k]}" for k in list(keys)[:3]))
    check("2c numbers printed on the diagram", sorted(str(t) for t in txt.iloc[:, 0])[:3] == sorted(
        [str(int(n["venn_intense_only"])), str(int(n["venn_both"])), str(int(n["venn_classic_only"]))]))
    (cells,) = blocks(wb, "Fig2cef-cells")
    csvc = pd.read_csv(ART / "figures_v10.4" / "work" / "cache" / "fig2_v104_cells_B.csv")
    ok = len(cells) == len(csvc) and all(
        same(num(cells, cb), csvc[cc].astype(float))[0] for cb, cc in zip(cells.columns[1:], csvc.columns[1:]))
    check(f"2cef-cells: {len(csvc)} rows = fig2_v104_cells_B.csv", ok and list(cells.iloc[:, 0]) == list(csvc.session))
    pi, pc = csvc.pc_int.values, csvc.pc_cl.values
    check("2c counts recounted from the per-neuron labels",
          ((pi & ~pc).sum(), (pi & pc).sum(), (~pi & pc).sum()) == (v["INTENSE only"], v["both"], v["classic only"]))
    (e,) = blocks(wb, "Fig2e")
    ok = same(num(e, "-log10 p threshold"), n["xvals_e"])[0] and np.array_equal(num(e, "cells averaged (n)"), n["n_e"])
    for k, name in (("mse", "MSE"), ("psnr", "PSNR (dB)"), ("ssim", "SSIM")):
        ok &= same(num(e, f"{name}: mean"), n[f"av_{k}"])[0] and same(num(e, f"{name}: s.e.m."), n[f"err_{k}"])[0]
    check("2e means, s.e.m., n = fig2_v111_data_B.npz", ok)
    left, right = blocks(wb, "Fig2f")
    ok_i, ok_c = n["n_int"] >= 20, n["n_cl"] >= 20
    check("2f left: curves drawn where >= 20 cells", same(num(left, "-log10 p level"), n["grid"])[0]
          and same(num(left, "INTENSE cells confirmed by classic (%) (red)"), np.where(ok_i, 100 * n["conf_int"], np.nan))[0]
          and same(num(left, "classic cells confirmed by INTENSE (%) (blue)"), np.where(ok_c, 100 * n["conf_cl"], np.nan))[0]
          and np.array_equal(num(left, "INTENSE cells at or above the level (n)"), n["n_int"]))
    k0 = str(n["base_keys"][0])
    check("2f right: Jaccard and its expectation", same(num(right, "top-ranked cells (%)"), 100 * n["fracs"])[0]
          and same(num(right, "Jaccard index, INTENSE vs classic (orange)"), n["j_obs"])[0]
          and same(right.iloc[:, 2].to_numpy(float), n[f"base_{k0}"])[0]
          and f"{float(n['rho']):.4f}" in right.columns[2])
    # d
    (mc, ms) = blocks(wb, "Fig2d")
    dd = np.load(ART / "figures_v10.4" / "work" / "cache" / "fig2d_data_v104.npz")
    for df, key in ((mc, "ca_map"), (ms, "sp_map")):
        m = np.asarray(dd[key], float)
        check(f"2d {key} normalized to [0, 1]", *same(df.iloc[:, 1:].to_numpy(float), (m - m.min()) / (m.max() - m.min())))
    return dict(v=v, cells=csvc)


# ---------------------------------------------------------------------------------------------- Fig. 3
def trace_check(wb, sheet, mouse, day, row, feat, delay):
    per, dens, tr = blocks(wb, sheet)
    z = np.load(_F.V.get("nof_v1") / f"NOF_{mouse}_{day}_aligned.npz", allow_pickle=True)
    fps = fps_of(f"NOF_{mouse}_{day}")
    n = z["Calcium"].shape[1] - delay
    sig = np.roll(scaled(z["Calcium"][row]), -delay)[:n]
    check(f"{sheet[3:]} time = frame / fps of NOF_{mouse}_{day}", *same(num(tr, "time (s)"), np.arange(n) / fps))
    check(f"{sheet[3:]} trace = Calcium row {row}, rescaled, shifted by {delay} frames",
          *same(num(tr, f"neuron {row}: signal (a.u.)"), sig, 1e-6))
    periods_check(f"{sheet[3:]} {feat}", per, (np.asarray(z[feat], float)[:n] > 0).astype(float), fps)
    density_shape(sheet[3:], dens)


def fig3():
    wb = openpyxl.load_workbook(OUT / "Fig3_source_data.xlsx", read_only=True)
    trace_check(wb, "Fig3a", "H02", "3D", 312, "object3", 15)
    trace_check(wb, "Fig3b", "H08", "4D", 121, "rear", 60)
    # c
    from scipy.stats import wilcoxon
    bars, tests, per = blocks(wb, "Fig3c")
    df = pd.read_csv(P.CACHE_DIR / "fig3c_per_session.csv")
    cols = ["pct_1", "pct_2", "pct_3", "pct_neg", "pct_zero", "pct_pos"]
    check("3c per-session table = fig3c_per_session.csv", same(per.iloc[:, 2:].to_numpy(float), df.iloc[:, 2:])[0]
          and list(per.mouse) == list(df.mouse))
    check("3c bar heights and s.e.m.", same(num(bars, "mean (% of recorded neurons)"), df[cols].mean())[0]
          and same(num(bars, "s.e.m."), df[cols].std() / np.sqrt(len(df)))[0])
    ok = True
    for _, t in tests.iterrows():
        side = cols[:3] if t["plot"] == "left" else cols[3:]
        names = ["1", "2", "3+"] if t["plot"] == "left" else ["-", "0", "+"]
        a, b = t["comparison"].split(" vs ")
        w, p = wilcoxon(df[side[names.index(a)]], df[side[names.index(b)]], alternative="two-sided")
        star = next((s for thr, s in ((1e-4, "****"), (1e-3, "***"), (1e-2, "**"), (0.05, "*")) if p < thr),
                    "none (p >= 0.05)")
        ok &= bool(float(t["W"]) == float(w) and np.isclose(float(t["p"]), p, rtol=RTOL, atol=0)
                   and t["asterisks drawn"] == star)
    check("3c six Wilcoxon tests (W, p, asterisks) recomputed from the per-session table", ok)
    # d, e
    z = np.load(P.NOF_ALIGNED / "NOF_H02_4D_aligned.npz", allow_pickle=True)
    cal = np.asarray(z["Calcium"][215], float)
    s = np.clip((cal - cal.min()) / (cal.max() - cal.min()), 0, None)
    ang = np.mod(np.asarray(z["bodydirection"], float) + np.pi, 2 * np.pi) - np.pi
    edges = np.linspace(-np.pi, np.pi, 41)
    idx = np.clip(np.digitize(ang, edges) - 1, 0, 39)
    hist = np.bincount(idx, weights=s, minlength=40) / np.maximum(np.bincount(idx, minlength=40), 1)
    cen = (edges[:-1] + edges[1:]) / 2
    k = np.exp(-0.5 * (np.arange(-20, 20) / (0.04 * 40)) ** 2)
    k /= k.sum()
    smooth = np.real(np.fft.ifft(np.fft.fft(hist) * np.fft.fft(np.fft.ifftshift(k))))
    bd, sm = blocks(wb, "Fig3d")
    check("3d 40 bars = body-direction tuning of neuron 215 (H02 day 4)",
          *same(np.column_stack([num(bd, "bin centre (rad)"), num(bd, "mean signal in the bin (bar length, a.u.)")]),
                np.column_stack([cen, hist]), 1e-12))
    check("3d smoothed curve", *same(num(sm, "smoothed tuning (a.u.)"), np.r_[smooth, smooth[0]], 1e-12))
    lv, mp = blocks(wb, "Fig3e")
    xr, yr = np.asarray(z["x"], float), np.asarray(z["y"], float)
    x, y = (xr - xr.min()) / (xr.max() - xr.min()), (yr - yr.min()) / (yr.max() - yr.min())
    xg, yg = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
    ix, iy = np.searchsorted(xg, x) - 1, np.searchsorted(yg, y) - 1
    ok = (ix >= 0) & (ix < 100) & (iy >= 0) & (iy < 100)
    zs, zc = np.zeros((100, 100)), np.zeros((100, 100))
    np.add.at(zs, (iy[ok], ix[ok]), s[ok])
    np.add.at(zc, (iy[ok], ix[ok]), 1)
    from scipy.ndimage import gaussian_filter
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = gaussian_filter(np.where(zc > 0, zs / zc, 0), sigma=5)
    check("3e 100 x 100 map recomputed from x, y and neuron 215", *same(mp.iloc[:, 1:].to_numpy(float), Z, 1e-12))
    check("3e grid and contour levels", same(num(mp, "y"), yg)[0] and num(lv, "contour level (a.u.)")[0] <= Z.min()
          and num(lv, "contour level (a.u.)")[-1] >= Z.max())
    # f
    mat = np.load(P.CACHE_DIR / "fig3f_matrix.npz", allow_pickle=True)
    drawn, mi, sf, ns = blocks(wb, "Fig3f")
    f = np.array(mat["mi_mean"], float)
    f[mat["sig_frac"] < 0.5] = np.nan
    f[np.eye(len(f), dtype=bool)] = np.nan
    check("3f drawn matrix (gray: diagonal, sig_frac < 0.5)", *same(drawn.iloc[:, 1:].to_numpy(float), f))
    check("3f MI, fraction significant, n sessions = fig3f_matrix.npz",
          same(mi.iloc[:, 1:].to_numpy(float), mat["mi_mean"])[0] and same(sf.iloc[:, 1:].to_numpy(float), mat["sig_frac"])[0]
          and same(ns.iloc[:, 1:].to_numpy(float), mat["n_sessions"])[0])
    return dict(per=per, sig=sf)


# ---------------------------------------------------------------------------------------------- Fig. 4
def fig4():
    wb = openpyxl.load_workbook(OUT / "Fig4_source_data.xlsx", read_only=True)
    tab = pd.read_csv(_F.V.get("saved_synthetic") / "full_thr001" / "summary_table.csv")
    is_int = tab.method == "intense"
    tab = tab[(tab["main"] & ~is_int) | (is_int & (tab.variant == "stage2"))]
    names = {"c": ["corr.", "MI", "corr. + shuffles", "INTENSE"], "d": ["average", "MI", "average + shuffles", "INTENSE"]}
    snrs, skips = [2, 4, 8, 16, 32, 64], np.round(np.linspace(0, 0.8, 9), 1)
    grids = {}
    for ftype, key, kpr in (("c", "Fig4b", "Fig4d"), ("d", "Fig4c", "Fig4e")):
        g, long = blocks(wb, key)
        q = tab[tab.ftype == ftype]
        ok = True
        for j, name in enumerate(names[ftype]):
            r = q[q.method_idx == j].pivot(index="snr", columns="skip_prob", values="f1").loc[snrs]
            got = g[g.method == name].iloc[:, 2:].to_numpy(float)
            ok &= same(got, r.to_numpy(float), 1e-12)[0]
            grids[(ftype, name)] = got
        check(f"{key[3:]} F1 grids of the four methods = summary_table.csv", ok)
        qq = q.sort_values(["method_idx", "snr", "skip_prob"])
        check(f"{key[3:]} benchmark rows = summary_table.csv ({len(qq)} rows)", len(long) == len(qq) and all(
            same(num(long, c2), qq[c1])[0] for c1, c2 in (("snr", "snr"), ("skip_prob", "p_skip"),
                                                         ("precision", "precision"), ("recall", "recall"),
                                                         ("f1", "f1"), ("tp", "tp"), ("fp", "fp"))))
        (pts,) = blocks(wb, kpr)
        ok = True
        for _, p in pts.iterrows():
            j = names[ftype].index(p["method"])
            r = q[(q.method_idx == j) & (q.snr == p["SNR"]) & np.isclose(q.skip_prob, p["p_skip"])]
            assert len(r) == 1
            r = r.iloc[0]
            ok &= same([p["precision (drawn)"], p["recall (drawn)"]],
                       [np.nan_to_num(r.precision, nan=0.0), r.recall])[0]
        check(f"{kpr[3:]} {len(pts)} precision-recall points = summary_table.csv (no detections drawn as 0)", ok)
        grids[f"pts_{ftype}"] = pts
        grids[f"long_{ftype}"] = long
    # a: the synthetic example generated again with the parameters and seed of the panel
    P.use_driada(tools=False)
    from scipy.ndimage import gaussian_filter1d
    from driada.experiment.synthetic import (apply_poisson_to_binary_series, delete_one_islands,
                                             generate_fbm_time_series, generate_pseudo_calcium_signal,
                                             select_signal_roi)
    par, sel, rnd, ser = blocks(wb, "Fig4a")
    L, H, SEED, FS = 9000, 0.7, 13, 20
    fbm = generate_fbm_time_series(L, H, SEED)
    _, lo, hi = select_signal_roi(fbm, seed=SEED)
    binary = np.zeros(L)
    binary[(fbm >= lo) & (fbm <= hi)] = 1
    res, i = binary.copy(), 0             # fill_binary_gaps of the panel, gaps up to 0.5 s
    while i < L:
        if res[i] == 0:
            g0 = i
            while i < L and res[i] == 0:
                i += 1
            if i - g0 <= 0.5 * FS and g0 > 0 and i < L:
                res[g0:i] = 1
        else:
            i += 1
    binary = res
    mod = delete_one_islands(binary, 0.6, seed=SEED)
    pois = apply_poisson_to_binary_series(mod, 0.1, 1.0, seed=SEED)
    calc = generate_pseudo_calcium_signal(duration=600, events=pois, sampling_rate=FS, event_rate=0.2,
                                          amplitude_range=(0.5, 2), decay_time=2, noise_std=0.1, seed=SEED)
    want = np.column_stack([np.arange(L) / FS, fbm + 2, binary, mod, gaussian_filter1d(pois.astype(float), sigma=60),
                            calc])
    check("4a series generated again with seed 13 (fBM, binary before/after, rate, pseudo-calcium)",
          *same(ser.to_numpy(float), want))
    v = dict(zip(par["quantity"], par["value"]))
    check("4a range of interest", same([v["range of interest, lower border (drawn, + offset 2)"],
                                         v["range of interest, upper border (drawn, + offset 2)"]], [lo + 2, hi + 2])[0])
    t = np.arange(L) / FS
    for df, b, nm in ((sel, binary, "row 1"), (rnd, mod, "rows 3, 4")):
        a, e = runs(b)
        check(f"4a periods of {nm} ({len(a)})", *same(df.to_numpy(float), np.column_stack([t[a], t[e - 1] + 1 / FS])))
    return grids


# ---------------------------------------------------------------------------------------------- ED Fig. 1
def ed1():
    src = OUT / "ED_calibration_source_data.xlsx"
    a, b = openpyxl.load_workbook(src, read_only=True), openpyxl.load_workbook(OUT / "ED_Fig1_source_data.xlsx",
                                                                               read_only=True)
    diff = []
    for ws in a.worksheets:
        ra, rb = list(ws.iter_rows(values_only=True)), list(b[ws.title].iter_rows(values_only=True))
        for i, (x, y) in enumerate(zip(ra, rb)):
            if strip(x) != strip(y):
                diff.append((ws.title, i + 1))
        if len([r for r in ra if strip(r)]) != len([r for r in rb if strip(r)]):
            diff.append((ws.title, "rows"))
    check("ED workbook = workbook of the figure except the title cell", diff == [("Contents", 1)], str(diff[:3]))
    check("ED title names Extended Data Fig. 1", "Extended Data Fig. 1" in str(b["Contents"]["A1"].value))
    cal = P.CALIBRATION_DIR
    tab, fp = pd.read_csv(cal / "calibration_table.csv"), pd.read_csv(cal / "false_pairs_delay.csv")
    (full,) = blocks(b, "FigEDtable")
    full = full.drop(columns=[c for c in full.columns if c is None])
    check("ED table sheet = calibration_table.csv", len(full) == len(tab) and all(
        same(num(full, c), tab[c].astype(float))[0] for c in tab.columns if c not in ("null", "ftype")))
    hist, pairs = blocks(b, "FigEDc")
    check("ED c false pairs = false_pairs_delay.csv", len(pairs) == len(fp) and all(
        same(num(pairs, c), fp[c])[0] for c in ("neuron", "me", "me_excess", "pval", "delay_frames"))
          and same(num(pairs, "delay_s"), num(pairs, "delay_frames") / num(pairs, "fps"))[0])
    bins = np.arange(-2.125, 2.126, 0.25)
    check("ED c histogram of the delays", np.array_equal(num(hist, "false pairs"),
                                                         np.histogram(num(pairs, "delay_s"), bins=bins)[0]))
    for sheet, suf in (("FigEDa", ""), ("FigEDb", "_mi04")):
        (df,) = blocks(b, sheet)
        got = sorted(zip(num(df, "pairs tested"), num(df, "false pairs"), np.round(num(df, "rate per 1000 pairs"), 9)))
        want = sorted(zip(tab["n_pairs"].astype(float), tab[f"k{suf}"].astype(float), np.round(1000 * tab[f"rate{suf}"], 9)))
        check(f"ED {sheet[-1]} rates = calibration_table.csv (k{suf}, rate{suf})", got == want)


# ---------------------------------------------------------------------------------------------- text
def text(f1, f2, f3, f4):
    tex = re.sub(r"\s+", " ", TEX.read_text(encoding="utf-8"))

    def has(s):
        return s in tex

    check("text: Fig. 1e thresholds 'p = 0.001 and 0.04 bits' = lines drawn in 1e",
          has(r"($p = 0.001$ and 0.04 bits") and np.isclose(f1["thr_p"], 0.001) and np.isclose(f1["thr_mi"], 0.04),
          f"{f1['thr_p']:.6g}, {f1['thr_mi']:.6g}")
    v, c = f2["v"], f2["cells"]
    ri = c[c.pc_int].sort_values("p_int", kind="mergesort").head(100).pc_cl.mean()
    rc = c[c.pc_cl].sort_values("z_cl", ascending=False, kind="mergesort").head(100).pc_int.mean()
    got = (round(100 * float(v["INTENSE place cells confirmed by the classic method (fraction)"])), round(100 * ri),
           round(100 * float(v["classic place cells confirmed by INTENSE (fraction)"])), round(100 * rc))
    check("text: place cells confirmed 'from 60% ... to 94% of the 100 most significant ones, and from 40% to 84%'",
          has("rising from 60\\% of the INTENSE place cells to 94\\% of the 100 most significant ones, and from 40\\% to 84\\%")
          and got == (60, 94, 40, 84), f"book: {got}")
    per = f3["per"]
    check("text: Fig. 3c 'n = 64 sessions of 16 mice'", has("$n = 64$ sessions of 16 mice")
          and len(per) == 64 and per.mouse.nunique() == 16)
    n = per["recorded neurons"].to_numpy(float)
    multi = int(round(float(np.sum(n * (per["% selective to 2 variables"] + per["% selective to 3 or more variables"]) / 100))))
    check("text: multi-selective neurons after disentanglement '868'", has("from 1{,}480 to 868") and multi == 868,
          f"book: {multi}")
    pos = np.sum(n * (per["% pairs with near-zero delay"] + per["% pairs with positive delay"]))
    allp = pos + np.sum(n * per["% pairs with negative delay"])
    check("text: '90.0% had near-zero or positive optimal delays'", has("90.0\\% had near-zero or positive optimal delays")
          and round(100 * pos / allp, 1) == 90.0, f"book: {100 * pos / allp:.2f}%")
    neg_least = all((g["% pairs with negative delay"] * g["recorded neurons"]).sum()
                    < min((g[cl] * g["recorded neurons"]).sum() for cl in ("% pairs with near-zero delay",
                                                                           "% pairs with positive delay"))
                    for _, g in per.groupby("mouse"))
    check("text: negative delays 'the least frequent in every mouse'", has("least frequent in every mouse") and neg_least)
    sig = f3["sig"].set_index("variable")

    def span(vs):
        vals = [float(sig.loc[a, b]) for i, a in enumerate(vs) for b in vs[i + 1:]]
        return round(100 * min(vals)), round(100 * max(vals))
    loco, zone = span(["speed", "rest", "walk", "run"]), span(["corners", "walls", "center"])
    check("text: Fig. 3f related pairs '92--100%' (speed, rest, walk, run) and '89--94%' (corners, walls, center)",
          has("92--100\\%") and has("89--94\\%") and loco == (92, 100) and zone == (89, 94), f"book: {loco}, {zone}")
    gi, gc = f4[("c", "INTENSE")], f4[("c", "corr. + shuffles")]
    k = int(np.sum(gi > gc))
    best = int(np.sum(gi >= np.maximum.reduce([f4[("c", m)] for m in ("corr.", "MI", "corr. + shuffles")])))
    check("text: INTENSE 'highest F1 score in 47 of the 54 conditions' (continuous, vs correlation with shuffles)",
          has("highest $F_1$ score in 47 of the 54 conditions") and k == 47,
          f"book: INTENSE > corr. + shuffles in {k} of 54; highest of all four in {best}")
    pc, pd_ = f4["pts_c"], f4["pts_d"]
    ri = pc[(pc.method == "INTENSE") & pc["plot"].str.startswith("right")]
    p64 = float(ri[ri.p_skip == 0]["precision (drawn)"].iloc[0])
    pdd = pd_[(pd_.method == "INTENSE") & pd_["plot"].str.startswith("right")]
    p64d = float(pdd[pdd.p_skip == 0]["precision (drawn)"].iloc[0])
    check("text: INTENSE precision '0.99 and 0.92' at SNR 64, p_skip 0 (continuous, discrete)",
          has("to 0.99 and 0.92 among the accepted pairs") and (round(p64, 2), round(p64d, 2)) == (0.99, 0.92),
          f"book: {p64:.4f}, {p64d:.4f}")
    lo_, hi_ = round(float(ri["precision (drawn)"].min()), 2), round(float(ri["precision (drawn)"].max()), 2)
    check("text: continuous precision at SNR 64 'stayed at 0.98--0.99 at every skip probability'",
          has("stayed at 0.98--0.99 at every skip probability") and (lo_, hi_) == (0.98, 0.99), f"book: {lo_}-{hi_}")
    f1d = pdd[pdd.p_skip <= 0.3]["F1"]
    check("text: discrete 'F1 score of 0.95 or above) up to a skip probability of 0.3' (SNR 64)",
          has("of 0.95 or above) up to a skip probability of 0.3") and float(f1d.min()) >= 0.95,
          f"book: min F1 {float(f1d.min()):.4f} for p_skip <= 0.3")
    means = []
    for ftype, ms in (("c", ("corr.", "MI")), ("d", ("average", "MI"))):
        lg = f4[f"long_{ftype}"]
        for mm in ms:
            means.append(float(lg[lg["method in the figure"] == mm]["precision"].mean()))
    rng = (round(100 * min(means)), round(100 * max(means)))
    check("text: analytical tests 'on average across conditions, only 6--13% of their detections were true'",
          has("only 6--13\\% of their") and rng == (6, 13), f"book: mean precision {[round(m, 3) for m in means]}")


def main():
    global TEX
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", type=Path, default=None, help="the manuscript, for the checks against its text")
    TEX = ap.parse_args().tex
    f1 = fig1()
    f2 = fig2()
    f3 = fig3()
    f4 = fig4()
    ed1()
    if TEX is not None:
        text(f1, f2, f3, f4)
    print(f"\n{'[OK] all checks passed' if not FAILED else f'[FAILED] {len(FAILED)} checks: ' + '; '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
