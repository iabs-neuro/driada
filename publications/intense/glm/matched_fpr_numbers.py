"""Numbers of the comparison at matched false-positive rates (Section S1.3) from the outputs of glm/matched_fpr.py.

* 5 false positives per 100 nonselective cells: F1 on linear mixed cells of each method at the
  threshold whose false-positive rate is closest to 0.05 (matched.csv, row target_fpr = 0.05);
* no false positive allowed: the largest F1 on linear mixed cells over the thresholds at which a
  method has no false positive on the nonselective cells, and the largest such threshold
  (curves.csv);
* the GLM with the circular-shift null: the smallest p-value it returns, 1 / (shifts + 1), and how
  many nonselective cells reach it (pvals.csv).

usage: python glm/matched_fpr_numbers.py  ->  <[glm] matched_fpr_dir>/matched_fpr_numbers.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
import intense_paths as P  # noqa: E402

DIR = P.get("glm", "matched_fpr_dir")
METHODS = ["INTENSE_nodelay", "INTENSE_delay", "M_cv_wilcox", "M_ho_shift"]


def at_target(matched, target):
    """F1 on linear mixed cells and the false-positive rate reached by each method at one target rate.

    Parameters
    ----------
    matched : pandas.DataFrame
        matched.csv of glm/matched_fpr.py.
    target : float
        Target false-positive rate; one of the values of ``matched.target_fpr``.

    Returns
    -------
    dict
        ``{method: {"f1_mixed_lin": float, "fpr": float}}``.
    """
    row = matched[(matched.target_fpr - target).abs() < 1e-12].iloc[0]
    return {m: {"f1_mixed_lin": float(row[m]), "fpr": float(row[f"{m}_actual_fpr"])} for m in METHODS}


def without_false_positives(curves):
    """Largest F1 on linear mixed cells over the thresholds without a false positive, per method.

    Parameters
    ----------
    curves : pandas.DataFrame
        curves.csv of glm/matched_fpr.py (one row per method and threshold ``alpha``).

    Returns
    -------
    dict
        ``{method: {"f1_mixed_lin", "at_threshold", "largest_threshold", "thresholds"}}``: the largest
        F1 (three decimals, as matched.csv), the smallest threshold that reaches it, the largest
        threshold without a false positive and the number of such thresholds; an empty dict for a
        method with none.
    """
    out = {}
    for m in METHODS:
        z = curves[(curves.method == m) & (curves.fpr_nonsel == 0)]
        if not len(z):
            out[m] = {}
            continue
        best = z.sort_values(["f1_mixed_lin", "alpha"], ascending=[False, True]).iloc[0]
        out[m] = {"f1_mixed_lin": round(float(best.f1_mixed_lin), 3), "at_threshold": float(best.alpha),
                  "largest_threshold": float(z.alpha.max()), "thresholds": int(len(z))}
    return out


def shift_null_floor(pvals):
    """Smallest p-value of the GLM with the circular-shift null and the nonselective cells that reach it.

    Parameters
    ----------
    pvals : pandas.DataFrame
        pvals.csv of glm/matched_fpr.py.

    Returns
    -------
    dict
        The smallest p-value of the arm M_ho_shift over all pairs, the number of nonselective cells
        (seed x neuron) and how many of them have a feature at that p-value.
    """
    h = pvals[pvals.method == "M_ho_shift"]
    floor = float(h.pval.min())
    cell_min = h[h.group == "nonselective"].groupby(["seed", "neuron"]).pval.min()
    return {"smallest_p": floor, "nonselective_cells": int(len(cell_min)),
            "nonselective_cells_at_smallest_p": int((cell_min == floor).sum())}


def main():
    matched = pd.read_csv(DIR / "matched.csv")
    curves = pd.read_csv(DIR / "curves.csv")
    pvals = pd.read_csv(DIR / "pvals.csv")
    res = {"seeds": int(pvals.seed.nunique()),
           "at_5_per_100_nonselective": at_target(matched, 0.05),
           "without_false_positives": without_false_positives(curves),
           "shift_null": shift_null_floor(pvals)}
    (DIR / "matched_fpr_numbers.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
