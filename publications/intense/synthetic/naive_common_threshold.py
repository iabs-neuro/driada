"""Put the methods without shuffles on one common threshold for Figure 4.

Reads the per-pair p-values written by naive_pvalues.py, applies the same
threshold to every method without shuffles (two-sided MI test on continuous
features), and writes a summary_table.csv in which these rows replace the
original ones as the plotted ("main") variant. The shuffle methods are copied
unchanged. make_fig4.py can then be pointed at the new directory.

Usage: python naive_common_threshold.py --thr 0.001 --out results/full_thr001
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NAIVE = {"c": [("corr", 0, "p_corr"), ("mi_naive", 1, "p_mi_naive")],
         "d": [("average", 0, "p_average"), ("mi_naive", 1, "p_mi_naive")]}


def _f1(p, r):
    if not np.isfinite(r) or r == 0:
        return 0.0
    if not np.isfinite(p) or p + r == 0:
        return np.nan
    return 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.001)
    ap.add_argument("--src", required=True, help="<synthetic_dir>/full of run_benchmark.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for ftype, methods in NAIVE.items():
        for f in sorted((src / "naive_p" / ftype).glob("*.npz")):
            snr, skip, r = f.stem.split("_")
            z = np.load(f)
            gt = z["gt"]
            for method, idx, key in methods:
                det = z[key] < a.thr
                tp, fp, fn = int((det & gt).sum()), int((det & ~gt).sum()), int((~det & gt).sum())
                rows.append(dict(ftype=ftype, method=method, method_idx=idx,
                                 snr=int(snr[3:]), skip_prob=float(skip[4:]), realization=int(r[1:]),
                                 precision=tp / (tp + fp) if tp + fp else np.nan,
                                 recall=tp / (tp + fn) if tp + fn else np.nan,
                                 n_detected=tp + fp, tp=tp, fp=fp))
    long = pd.DataFrame(rows)
    keys = ["ftype", "method", "method_idx", "snr", "skip_prob"]
    new = long.groupby(keys, as_index=False).agg(
        precision=("precision", "mean"), recall=("recall", "mean"),
        n_detected=("n_detected", "mean"), tp=("tp", "mean"), fp=("fp", "mean"),
        n_realizations=("realization", "nunique"),
        n_no_detection=("n_detected", lambda x: int((x == 0).sum())),
    )
    new["variant"] = f"naive_p{a.thr:g}"
    new["main"] = True
    new["f1"] = [_f1(p, r) for p, r in zip(new.precision, new.recall)]

    tab = pd.read_csv(src / "summary_table.csv")
    tab.loc[tab.variant == "naive", "main"] = False
    tab = pd.concat([tab, new[tab.columns]], ignore_index=True)
    tab.to_csv(out / "summary_table.csv", index=False)
    print(f"[OK] {len(long)} config x method rows, threshold p < {a.thr:g} -> {out}")


if __name__ == "__main__":
    main()
