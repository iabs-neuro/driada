# -*- coding: utf-8 -*-
"""Arbiter numbers of the SI (Section "Comparison with the dataset authors' cell-type labels")
from an arbiter_labels.csv:
  - share of chance labels at the authors' 95th / 99th percentile;
  - recall of the authors' cells and nesting (authors & GLM, of them also INTENSE) at the
    95th and 99th percentile;
  - recall as a function of label confidence (recall_vs_margin.csv and its working plot);
  - confidently labelled HD cells (margin > 4) by session order, with recording length;
  - HD cells INTENSE flags without an author label: mean vector length against the cutoff.
Recording length is read from the header of each session's calcium array ([mec] sessions_dir);
seconds use the frame rate the analysis used for the session (--fps-from: a cohort results
file with an `fps` column) or a constant (--fps-const).

usage: python -B mec/arbiter_stats.py --labels <arbiter_labels.csv> --out <dir>
                                      (--fps-from <results.csv> | --fps-const 7.52)
Outputs in <dir>: arbiter_stats.json, recall_vs_margin.csv, recall_vs_margin.png.
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402

SESS = P.get("mec", "sessions_dir")


def n_frames(path):
    """Number of frames of a session, from the calcium array header (the array is not read)."""
    with zipfile.ZipFile(path) as z, z.open("calcium.npy") as f:
        major, _ = np.lib.format.read_magic(f)
        read = (np.lib.format.read_array_header_1_0 if major == 1
                else np.lib.format.read_array_header_2_0)
        shape, _, _ = read(f)
    return int(shape[1])


def contamination(df):
    out = {}
    for t, lab95, lab99, sk, c95, c99 in [
            ("grid", "is_grid95", "is_grid99", "gridscore", "grid_95", "grid_99"),
            ("border", "is_border95", "is_border99", "borderscore", "border_95", "border_99"),
            ("HD", "is_hd95", "is_hd99", "mvl", "mvl_95", "mvl_99")]:
        U = df[np.isfinite(df[sk]) & np.isfinite(df[c95])].copy()
        n95, n99 = int(U[lab95].sum()), int(U[lab99].sum())
        exp_fp95, exp_fp99 = 0.05 * len(U), 0.01 * len(U)
        out[t] = dict(universe=len(U), labeled95=n95, labeled99=n99,
                      exp_chance95=round(exp_fp95), exp_chance99=round(exp_fp99),
                      chance_share95=round(exp_fp95 / n95, 2),
                      chance_share99=round(exp_fp99 / n99, 2))
    return out


def recall_nesting(df):
    """Recall of the authors' cells and nesting at both percentiles (99th: the nesting of the SI)."""
    out = {}
    for pct in ("95", "99"):
        for t, sk, ck, ik, gk in [("grid", "gridscore", "grid", "int_place", "glm_place"),
                                  ("border", "borderscore", "border", "int_place", "glm_place"),
                                  ("HD", "mvl", "mvl", "int_hd", "glm_hd")]:
            lab = {"grid": "is_grid", "border": "is_border", "HD": "is_hd"}[t] + pct
            U = df[np.isfinite(df[sk]) & np.isfinite(df[f"{ck}_{pct}"])]
            A = U[U[lab] == 1]
            out[f"{t}@{pct}"] = dict(
                universe=len(U), authors=len(A),
                intense_found=int((A[ik] == 1).sum()), glm_found=int((A[gk] == 1).sum()),
                recall_int=float((A[ik] == 1).mean()), recall_glm=float((A[gk] == 1).mean()),
                authors_and_glm=int((A[gk] == 1).sum()),
                of_them_also_int=int(((A[gk] == 1) & (A[ik] == 1)).sum()))
    for t, lab, ik, gk in [("grid", "is_grid95", "int_place", "glm_place"),
                           ("HD", "is_hd95", "int_hd", "glm_hd")]:
        out[f"{t}_precision_vs_authors95"] = dict(
            intense=float(df[df[ik] == 1][lab].mean()), glm=float(df[df[gk] == 1][lab].mean()))
    return out


def margin_curves(df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    types = [("grid", "gridscore", "grid_95", "grid_99", "int_place", "glm_place", "int_raw_place"),
             ("border", "borderscore", "border_95", "border_99", "int_place", "glm_place",
              "int_raw_place"),
             ("HD", "mvl", "mvl_95", "mvl_99", "int_hd", "glm_hd", "int_raw_hd")]
    bins = [(0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, np.inf)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=120)
    rows = []
    for ax, (t, sk, c95, c99, ik, gk, rk) in zip(axes, types):
        U = df[np.isfinite(df[sk]) & np.isfinite(df[c95]) & np.isfinite(df[c99])].copy()
        A = U[U[sk] > U[c95]].copy()                      # authors' labeled set
        span = (A[c99] - A[c95]).replace(0, np.nan)
        A["m"] = (A[sk] - A[c95]) / span                  # margin in null-tail units
        xs, ri, rg, rr, ns = [], [], [], [], []
        for lo, hi in bins:
            S = A[(A.m >= lo) & (A.m < hi)]
            if len(S) < 20:
                continue
            xs.append(f"{lo}-{'inf' if hi == np.inf else hi}")
            ri.append(S[ik].mean()); rg.append(S[gk].mean()); rr.append(S[rk].mean())
            ns.append(len(S))
            rows.append(dict(type=t, margin=xs[-1], n=len(S), recall_int=ri[-1],
                             recall_raw=rr[-1], recall_glm=rg[-1]))
        x = np.arange(len(xs))
        ax.plot(x, ri, "o-", color="#a50f15", label="INTENSE by_type")
        ax.plot(x, rr, "s--", color="#fb6a4a", label="INTENSE raw")
        ax.plot(x, rg, "^-", color="#2171b5", label="GLM")
        for xi, n in zip(x, ns):
            ax.text(xi, 1.02, f"n={n}", ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(xs); ax.set_ylim(0, 1.08)
        ax.set_xlabel("margin above authors' 95th pct cutoff\n"
                      "(in null-tail units, (score-c95)/(c99-c95))")
        ax.set_ylabel("recall of authors' cells"); ax.set_title(t); ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Recall vs confidence of the author label: does the miss mass sit at the borderline?")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "recall_vs_margin.png")
    plt.close(fig)
    rm = pd.DataFrame(rows)
    rm.to_csv(out_dir / "recall_vs_margin.csv", index=False)
    return rm


def hd_by_order(df, fps_of):
    U = df[np.isfinite(df.mvl) & np.isfinite(df.mvl_95) & np.isfinite(df.mvl_99)].copy()
    U["m"] = (U.mvl - U.mvl_95) / (U.mvl_99 - U.mvl_95).replace(0, np.nan)
    U["order"] = U.session.str.split("_").str[1]
    frames = {f.stem: n_frames(f) for f in SESS.glob("*.npz")}
    U["nframes"] = U.session.map(frames)
    U["seconds"] = U.nframes / U.session.map(fps_of)
    out = {}
    # confidently labelled HD cells (margin > 4): misses and the slice by session order
    S = U[U.m > 4]
    miss = S[S.int_hd == 0]
    out["high_margin"] = dict(
        n=len(S), intense_found=int((S.int_hd == 1).sum()), glm_found=int((S.glm_hd == 1).sum()),
        intense_misses=len(miss), miss_sessions=int(miss.session.nunique()),
        top_miss_sessions={k: int(v) for k, v in miss.session.value_counts().head(3).items()},
        misses_by_order={k: int(v) for k, v in miss.order.value_counts().sort_index().items()})
    g = S.groupby("order").agg(n=("int_hd", "size"), int_found=("int_hd", "sum"),
                               glm_found=("glm_hd", "sum"), recall_int=("int_hd", "mean"),
                               recall_glm=("glm_hd", "mean"), med_frames=("nframes", "median"),
                               med_seconds=("seconds", "median"))
    out["high_margin_by_order"] = {k: {c: (float(v) if c not in ("n", "int_found", "glm_found")
                                          else int(v)) for c, v in r.items()}
                                   for k, r in g.iterrows()}
    # all 95th-percentile HD labels by session order
    A = U[U.is_hd95 == 1]
    g = A.groupby("order").agg(n=("int_hd", "size"), INT=("int_hd", "mean"), GLM=("glm_hd", "mean"))
    out["hd95_by_order"] = {k: {c: (int(v) if c == "n" else float(v)) for c, v in r.items()}
                            for k, r in g.iterrows()}
    # HD cells INTENSE flags without an author label, against the population
    IO = U[(U.int_hd == 1) & (U.is_hd95 == 0)]
    r = IO.mvl / IO.mvl_95
    ALL = U.mvl / U.mvl_95
    out["intense_only_hd"] = dict(
        n=len(IO), ratio_median=float(r.median()), share_over_0p8=float((r > 0.8).mean()),
        share_over_0p5=float((r > 0.5).mean()),
        population_ratio_median=float(ALL.median()),
        population_share_over_0p8=float((ALL > 0.8).mean()),
        population_share_over_0p5=float((ALL > 0.5).mean()),
        miss_ratio_median_high_margin=float((miss.mvl / miss.mvl_95).median()),
        found_ratio_median_high_margin=float((S[S.int_hd == 1].mvl / S[S.int_hd == 1].mvl_95).median()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    fp = ap.add_mutually_exclusive_group(required=True)
    fp.add_argument("--fps-from", type=Path)
    fp.add_argument("--fps-const", type=float)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.labels)
    if a.fps_from is not None:
        cr = pd.read_csv(a.fps_from)
        fps_of = dict(zip(cr.session.astype(str), cr.fps.astype(float)))
    else:
        fps_of = {s: a.fps_const for s in df.session.unique()}
    res = {"labels": str(a.labels),
           "fps_source": str(a.fps_from) if a.fps_from is not None else a.fps_const,
           "contamination": contamination(df), "recall_nesting": recall_nesting(df)}
    rm = margin_curves(df, a.out)
    res.update(hd_by_order(df, fps_of))
    with open(a.out / "arbiter_stats.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res["contamination"]))
    for k, v in res["recall_nesting"].items():
        print(k, v)
    print(rm.to_string(index=False))
    for k in ("high_margin", "high_margin_by_order", "hd95_by_order", "intense_only_hd"):
        print(k, json.dumps(res[k]))


if __name__ == "__main__":
    main()
