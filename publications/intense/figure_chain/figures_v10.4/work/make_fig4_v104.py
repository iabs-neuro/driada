"""Data and checks of Fig. 4; the figure of the paper is drawn by figures_v11.1/work/fig4/make_fig4_v111.py,
which imports this module with its copy of the panel scripts (fig4_v7src/).

The shaded periods of panel a span exactly their samples (EXACT_SHADING of fig4a_generation_pipeline.py;
plot_shadowed_groups of DRIADA adds one unit of the axis to each period). Data: [v11] saved_synthetic/full_thr001,
methods at the criterion they are reported with, INTENSE at Stage 2, PR points as slices (p_skip = 0 on the
left, SNR = 64 on the right), SNR = 2 left out of panels d, e.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # figure_chain/
import _paths_fig as _F  # noqa: E402  (puts the paper folder on sys.path)
import intense_paths  # noqa: E402

driada = intense_paths.use_driada()
sys.path.insert(0, str(HERE))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fig4_v7src import fig4_delegated as comp  # noqa: E402
from fig4_v7src import fig4a_generation_pipeline as f4a  # noqa: E402
from fig4_v7src import fig4de_pr_curves as pr  # noqa: E402

A = _F.D
# earlier figures and their tables, compared with only when they are under [v11] figure_artifacts
V7_PNG = _F.data_dir(_F.CHAIN / "science" / "intense_paper" / "extracted" / "figs") / "fig4_delegated.png"
V7_DATA = _F.data_dir(_F.CHAIN / "science" / "intense_figures")
V103_PNG = A / "figures_v10.3" / "fig4.png"
TABLE = _F.V.get("saved_synthetic") / "full_thr001" / "summary_table.csv"
V102_SRC = A / "figures_v10" / "work" / "fig4_out"
SNRS = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
SKIPS = np.linspace(0.0, 0.8, 9)
OUT = _F.out_dir(HERE.parent)
WORK_OUT = _F.out_dir(HERE) / "fig4_out"


def load_table(variant="stage2"):
    """Rows of the v10.2/v10.3 figure: the reported criterion, INTENSE at the given variant."""
    tab = pd.read_csv(TABLE)
    is_int = tab.method == "intense"
    keep = (tab["main"] & ~is_int) | (is_int & (tab.variant == variant))
    return tab[keep].copy()


def arrays(tab, ftype):
    a = np.full((2, 4, 2, len(SKIPS), len(SNRS), 1), np.nan)
    q = tab[tab.ftype == ftype]
    if len(q) != 4 * len(SKIPS) * len(SNRS):
        raise SystemExit(f"{ftype}: {len(q)} rows, expected {4 * len(SKIPS) * len(SNRS)}")
    for _, r in q.iterrows():
        si = int(np.argmin(np.abs(SKIPS - r.skip_prob)))
        assert np.isclose(SKIPS[si], r.skip_prob)
        ni = SNRS.index(float(r.snr))
        a[0, int(r.method_idx), :, si, ni, 0] = r.precision
        a[1, int(r.method_idx), :, si, ni, 0] = r.recall
    assert not np.isnan(a[1]).any()
    return a


def compare_png(ref, new):
    a, b = np.asarray(Image.open(ref)), np.asarray(Image.open(new))
    res = dict(reference=str(ref), output=str(new), same_shape=a.shape == b.shape)
    if a.shape == b.shape:
        d = np.any(a != b, axis=2)
        ys, xs = np.nonzero(d)
        res.update(identical=not d.any(), n_px_differ=int(d.sum()),
                   differing_px_bbox_rows_cols=[int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1]
                   if d.any() else None)
    return res


def runs(binary):
    b = (np.asarray(binary) == 1).astype(np.int8)
    e = np.flatnonzero(np.diff(np.r_[0, b, 0]) != 0)
    return list(zip(e[0::2], e[1::2]))


def shading_check(fig):
    """Shaded periods of panel a as drawn, against the runs of the series they come from."""
    from driada.experiment.synthetic import (apply_poisson_to_binary_series, delete_one_islands,
                                             generate_fbm_time_series, select_signal_roi)
    fbm = generate_fbm_time_series(f4a.LENGTH, f4a.HURST, f4a.SEED)
    _, lo, hi = select_signal_roi(fbm, seed=f4a.SEED)
    binary = np.zeros(f4a.LENGTH)
    binary[(fbm >= lo) & (fbm <= hi)] = 1
    binary = f4a.fill_binary_gaps(binary, f4a.MAX_GAP * f4a.SAMPLING_RATE)
    mod = delete_one_islands(binary, f4a.SKIP_PROB, seed=f4a.SEED)
    apply_poisson_to_binary_series(mod, f4a.RATE_0, f4a.RATE_1, seed=f4a.SEED)  # same call order as the panel
    fs = f4a.SAMPLING_RATE
    out = []
    axes = fig.get_axes()[:4]  # panel a is drawn first: four stacked axes
    for name, ax, series in (("1. continuous time series", axes[0], binary),
                             ("3. heterogeneous Poisson process", axes[2], mod),
                             ("4. pseudo calcium signal", axes[3], mod)):
        assert ax.get_title() == name, (ax.get_title(), name)
        spans = sorted((float(p.get_x()), float(p.get_x() + p.get_width())) for p in ax.patches)
        want = [(a / fs, e / fs) for a, e in runs(series)]
        out.append(dict(axes=name, n_periods_drawn=len(spans), n_runs=len(want),
                        max_abs_start_error_s=float(max(abs(s[0] - w[0]) for s, w in zip(spans, want))),
                        max_abs_end_error_s=float(max(abs(s[1] - w[1]) for s, w in zip(spans, want))),
                        shortest_period_s=float(min(w[1] - w[0] for w in want))))
    return out


def drawn_values(fig):
    """F1 images of the heatmaps and PR scatter offsets, as drawn."""
    heat, pts = [], []
    for ax in fig.get_axes():
        for im in ax.get_images():
            heat.append((ax.get_title(), np.asarray(im.get_array())))
        for coll in ax.collections:
            off = coll.get_offsets()
            if len(off) and ax.get_xlabel() == "precision":
                pts.append(np.asarray(off))
    return heat, pts


def value_checks(fig, tab):
    """Values as drawn against the table and, when they are there, the source tables of the figure of v10.2."""
    heat, pts = drawn_values(fig)
    have_old = _F.reference(V102_SRC / "fig4_source_BC.csv") is not None
    bc = pd.read_csv(V102_SRC / "fig4_source_BC.csv") if have_old else None
    de = pd.read_csv(V102_SRC / "fig4_source_DE.csv") if have_old else None
    names = {"c": ["corr.", "MI", "corr. + shuffles", "INTENSE"],
             "d": ["average", "MI", "average + shuffles", "INTENSE"]}
    checks = {"heatmaps": [], "pr": []}
    for k, (title, img) in enumerate(heat):
        ftype = "c" if k < 4 else "d"
        j = k % 4
        assert title == names[ftype][j], (title, ftype, j)
        q = tab[(tab.ftype == ftype) & (tab.method_idx == j)]
        ref = np.full((len(SNRS), len(SKIPS)), np.nan)
        for _, r in q.iterrows():
            ref[SNRS.index(float(r.snr)), int(np.argmin(np.abs(SKIPS - r.skip_prob)))] = r.f1
        one = dict(panel="b" if ftype == "c" else "c", method=title,
                   max_abs_diff_vs_table=float(np.nanmax(np.abs(img - ref))))
        if have_old:
            old = bc[(bc.ftype == ftype) & (bc.method == q.method.iloc[0])]
            ref_old = np.full_like(ref, np.nan)
            for _, r in old.iterrows():
                ref_old[SNRS.index(float(r.snr)), int(np.argmin(np.abs(SKIPS - r.skip_prob)))] = r.f1
            one["max_abs_diff_vs_v10_2_source"] = float(np.nanmax(np.abs(img - ref_old)))
        checks["heatmaps"].append(one)
    if not have_old:
        checks["max_abs_diff_all"] = max(h["max_abs_diff_vs_table"] for h in checks["heatmaps"])
        return checks
    k = 0
    for ftype in ("c", "d"):
        for along, fixed in (("snr", "p_skip=0"), ("skip", "SNR=max")):
            for j, name in enumerate(names[ftype]):
                p = pts[k]
                k += 1
                q = de[(de.ftype == ftype) & (de.method == name) & (de.along == along)]
                if along == "snr":
                    q = q[q.value > 2]  # SNR = 2 is not drawn (v7)
                diff = np.abs(p - q[["precision", "recall"]].to_numpy()).max()
                checks["pr"].append(dict(panel=("d" if ftype == "c" else "e") + ("-left" if along == "snr" else "-right"),
                                         method=name, n_points=len(p), fixed=fixed,
                                         max_abs_diff_vs_v10_2_source=float(diff)))
    checks["max_abs_diff_all"] = max(max(h["max_abs_diff_vs_table"] for h in checks["heatmaps"]),
                                     max(p["max_abs_diff_vs_v10_2_source"] for p in checks["pr"]))
    return checks


def main(mode):
    WORK_OUT.mkdir(exist_ok=True)
    (_F.out_dir(HERE) / "verify").mkdir(exist_ok=True)
    if mode == "control_v7":
        c = np.load(V7_DATA / "continuous INTENSE test mult v2.npz")["arr_0"]
        d = np.load(V7_DATA / "discrete INTENSE test mult v2.npz")["arr_0"]
        pr.PR_SELECT = "mean"
        fig = comp.create_fig4_delegated(output_dir=WORK_OUT, save_formats=("png",), c_data=c, d_data=d,
                                         output_name="fig4_control_v7", lowercase=False)
        plt.close(fig)
        if not V7_PNG.exists():
            # the v7 figure is an optional input: without it the control is drawn but not compared
            print(f"[SKIPPED] comparison with the v7 figure: {V7_PNG} not found")
            res = dict(reference=str(V7_PNG), reference_missing=True, output=str(WORK_OUT / "fig4_control_v7.png"))
        else:
            res = compare_png(V7_PNG, WORK_OUT / "fig4_control_v7.png")
        (_F.out_dir(HERE) / "verify" / "fig4_control_v7.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return

    tab = load_table()
    c, d = arrays(tab, "c"), arrays(tab, "d")
    pr.PR_SELECT = "slice"
    if mode == "control_v103":
        fig = comp.create_fig4_delegated(output_dir=WORK_OUT, save_formats=("png",), c_data=c, d_data=d,
                                         output_name="fig4_control_v103", lowercase=True)
        res = compare_png(V103_PNG, WORK_OUT / "fig4_control_v103.png")
        res["shading_panel_a"] = shading_check(fig)
        plt.close(fig)
        (_F.out_dir(HERE) / "verify" / "fig4_control_v103.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return

    if mode == "texts":
        # label clashes of the v10.3 settings and of the final settings, drawn without saving
        sys.path.insert(1, str(HERE))
        import overlaps
        reports = {}
        for shading in (False, True):
            f4a.EXACT_SHADING = shading
            fig = comp.create_fig4_delegated(output_dir=WORK_OUT, save_formats=(), c_data=c, d_data=d,
                                             output_name="unused", lowercase=True)
            reports["final" if shading else "control_v103"] = overlaps.text_report(fig)
            plt.close(fig)
        res = dict(text_clashes_vs_control=overlaps.diff_reports(reports["control_v103"], reports["final"]), **reports)
        (_F.out_dir(HERE) / "verify" / "fig4_texts.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(res, indent=1, ensure_ascii=False))
        return

    assert mode == "final", mode
    for f in ("fig4.png", "fig4.pdf"):
        if (OUT / f).exists():
            sys.exit(f"{OUT / f} exists; not overwriting")
    f4a.EXACT_SHADING = True
    fig = comp.create_fig4_delegated(output_dir=OUT, save_formats=("png", "pdf"), c_data=c, d_data=d,
                                     output_name="fig4", lowercase=True)
    res = compare_png(V103_PNG, OUT / "fig4.png")
    res["panel_a_region_rows_cols"] = [0, int(round(comp.TOP_ROW_FRAC * 2834)), 0,
                                       int(round(comp.LEFT_COL_CM / comp.CANVAS_WIDTH_CM * 2125))]
    res["shading_panel_a"] = shading_check(fig)
    res["values"] = value_checks(fig, tab)
    plt.close(fig)
    (_F.out_dir(HERE) / "verify" / "fig4_final.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
