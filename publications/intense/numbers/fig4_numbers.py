"""Numbers of Results 2.4 read from the table Fig. 4 is drawn from (<[v11] saved_synthetic>/full_thr001/
summary_table.csv of synthetic/naive_common_threshold.py; figure_chain/figures_v10.4/work/
make_fig4_v104.py: non-INTENSE methods at their 'main' rows, INTENSE at variant stage2). ASCII output.

L1: precision of the analytical tests over the 54 conditions ("only 6-13% of their" detections).
L2: INTENSE on discrete variables at SNR 64 by skip probability (F1 0.957, 0.956, 0.953, 0.954 up to
    p_skip 0.3, 0.949 at 0.4: "F1 score of 0.95 or above up to a skip probability of 0.3"), and
    the precision on continuous variables at SNR 64 by skip probability ("0.98-0.99 at every skip
    probability").

usage: python numbers/fig4_numbers.py [--table CSV]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402,F401
import paths_v11 as V  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--table", default=str(V.get("saved_synthetic") / "full_thr001" / "summary_table.csv"))
tab = pd.read_csv(ap.parse_args().table)
is_int = tab.method == "intense"
fig = tab[(tab["main"] & ~is_int) | (is_int & (tab.variant == "stage2"))]
print("methods in Fig. 4:", fig.groupby(["ftype", "method", "variant"]).size().to_dict())

naive = fig[fig.variant.str.startswith("naive")]
print("\nL1: analytical tests (Fig. 4 rows), precision by method over the 54-condition grid")
for (ft, m), x in naive.groupby(["ftype", "method"]):
    print(f"  {ft} {m:12s} variant={x.variant.iloc[0]:13s} n={len(x)} mean={x.precision.mean():.3f} "
          f"min={x.precision.min():.3f} max={x.precision.max():.3f} median={x.precision.median():.3f}")
    top = x.loc[x.precision.idxmax()]
    print(f"      max at snr={top.snr} skip={top.skip_prob}; recall mean={x.recall.mean():.2f}")
print(f"  all analytical: per-condition min={naive.precision.min():.3f} max={naive.precision.max():.3f}; "
      f"method means {naive.groupby(['ftype','method']).precision.mean().min():.3f}-"
      f"{naive.groupby(['ftype','method']).precision.mean().max():.3f}")
# precision of the pooled detections (TP/(TP+FP) summed over the grid)
for (ft, m), x in naive.groupby(["ftype", "method"]):
    print(f"  pooled TP/(TP+FP) {ft} {m}: {x.tp.sum() / (x.tp.sum() + x.fp.sum()):.3f}")
# Fig. 4d,e left (p_skip = 0, SNR varies) and right (SNR = 64, p_skip varies): what the panels show
for ft in ("c", "d"):
    x = naive[naive.ftype == ft]
    l = x[x.skip_prob == 0]
    r = x[x.snr == 64]
    print(f"  panels {ft}: left (skip 0) precision {l.precision.min():.3f}-{l.precision.max():.3f}; "
          f"right (SNR 64) {r.precision.min():.3f}-{r.precision.max():.3f}")

print("\nL2: INTENSE stage2, discrete, SNR 64, by p_skip")
x = fig[is_int & (fig.ftype == "d") & (fig.snr == 64)].sort_values("skip_prob")
for _, r in x.iterrows():
    print(f"  skip={r.skip_prob:.1f} precision={r.precision:.4f} recall={r.recall:.4f} f1={r.f1:.4f}")
print("INTENSE stage2, discrete, all SNR >= 16 at skip 0.3 and 0.4:")
y = fig[is_int & (fig.ftype == "d") & (fig.skip_prob.isin([0.3, 0.4]))].sort_values(["skip_prob", "snr"])
for _, r in y.iterrows():
    print(f"  snr={r.snr} skip={r.skip_prob:.1f} precision={r.precision:.3f} recall={r.recall:.3f} f1={r.f1:.4f}")
print("INTENSE stage2, continuous, SNR 64: precision by skip")
z = fig[is_int & (fig.ftype == "c") & (fig.snr == 64)].sort_values("skip_prob")
print("  ", [(round(r.skip_prob, 1), round(r.precision, 3), round(r.recall, 3)) for _, r in z.iterrows()])
