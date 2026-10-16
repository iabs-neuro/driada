# -*- coding: utf-8 -*-
"""Paired comparisons over sessions and over mice for the Fig. 4 multisession run.

Usage: python paired_stats.py [sessions.csv]   (default: the table Figure 4 reads)
"""
import csv, os, sys
import numpy as np
from scipy.stats import wilcoxon, binomtest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import elife_paths

TABLE = sys.argv[1] if sys.argv[1:] else os.path.join(elife_paths.CACHE_DIR, "fig4_multisession_sessions.csv")
rows = [r for r in csv.DictReader(open(TABLE, encoding="utf-8")) if r["status"] == "ok"]
f = lambda r, k: float(r[k])
print(f"{len(rows)} sessions, {len({r['mouse'] for r in rows})} mice")

def cmp(a, b, label):
    x = np.array([f(r, a) - f(r, b) for r in rows])
    mice = {}
    for r in rows:
        mice.setdefault(r["mouse"], []).append(f(r, a) - f(r, b))
    m = np.array([np.mean(v) for v in mice.values()])
    print(f"{label:40s} sessions {int((x > 0).sum())}/{len(x)} Wilcoxon p={wilcoxon(x).pvalue:.1e} | "
          f"mice {int((m > 0).sum())}/{len(m)} sign p={binomtest(int((m > 0).sum()), len(m)).pvalue:.3f} "
          f"Wilcoxon p={wilcoxon(m).pvalue:.3f} | median diff {np.median(x):.3f}")

for m in ["r2cv", "r2", "dcorr", "dimratio"]:
    cmp(f"{m}_filt", f"{m}_all", f"{m}: filtered > all")
    cmp(f"{m}_filt", f"{m}_rand_mean", f"{m}: filtered > random same size")
    cmp(f"{m}_filt", f"{m}_shuf_filt", f"{m}: filtered > filtered shuffled")
    cmp(f"{m}_all", f"{m}_shuf_all", f"{m}: all > all shuffled")
cmp("base_pct_nonsel", "pct_nonsel_raw", "base rate > % non-sel (raw)")
cmp("base_pct_nonsel", "pct_nonsel_holm", "base rate > % non-sel (Holm)")
cmp("base_pct_nonsel", "pct_nonsel_raw_excl", "base rate > % non-sel (raw, excl. filtered)")
