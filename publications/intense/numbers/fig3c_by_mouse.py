"""Fig. 3c comparisons over sessions and over the 16 mice (session percentages averaged per mouse).

Reads <cache_dir>/fig3c_per_session.csv of fig3/fig3_data.py and writes the p-values of Methods,
"Statistical analysis of delay and selectivity distributions", to <numbers_dir>/fig3c_by_mouse.json.

usage: python numbers/fig3c_by_mouse.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402

RESULT = {}
for name, path in (("config", P.CACHE_DIR / "fig3c_per_session.csv"),):
    d = pd.read_csv(path)
    res = RESULT[name] = {"source": str(path), "n_sessions": len(d)}
    m = d.groupby("mouse")[["pct_1", "pct_2", "pct_3", "pct_neg", "pct_zero", "pct_pos"]].mean()
    print(name, "mice", len(m))
    res["n_mice"] = len(m)
    for a, b in [("pct_1", "pct_2"), ("pct_1", "pct_3"), ("pct_2", "pct_3"), ("pct_neg", "pct_zero"),
                 ("pct_neg", "pct_pos"), ("pct_zero", "pct_pos")]:
        print(f"  {a:8s} vs {b:8s}  sessions p={wilcoxon(d[a], d[b]).pvalue:.2e}  mice p={wilcoxon(m[a], m[b]).pvalue:.2e}"
              f"  mouse means {m[a].mean():.2f} vs {m[b].mean():.2f}")
        res[f"{a} vs {b}"] = {"p_sessions": float(wilcoxon(d[a], d[b]).pvalue),
                              "p_mice": float(wilcoxon(m[a], m[b]).pvalue),
                              "mice_same_direction": int(((m[a] > m[b]) == (m[a].mean() > m[b].mean())).sum())}
P.NUMBERS_DIR.mkdir(parents=True, exist_ok=True)
(P.NUMBERS_DIR / "fig3c_by_mouse.json").write_text(json.dumps(RESULT, indent=1), encoding="utf-8")
