"""Supplementary Fig. 3: F1 on linear mixed cells against the false-positive rate on nonselective cells.

Draws curves.csv of glm/matched_fpr.py: both INTENSE arms (zero lag, solid; delay search, dashed) and
the two GLM arms, each traced by sweeping its decision threshold.

usage: python glm/matched_fpr_figure.py  ->  <[glm] figures_dir>/glm_matched_fpr_v813.png
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
import intense_paths as P  # noqa: E402

FIGS = P.get("glm", "figures_dir")
FIGS.mkdir(parents=True, exist_ok=True)
cv = pd.read_csv(P.get("glm", "matched_fpr_dir") / "curves.csv")
style = {"M_cv_wilcox": ("C0", "-", "GLM, CV $R^2$ + Wilcoxon"),
         "M_ho_shift": ("C7", "-", "GLM, held-out $R^2$ + circular-shift null"),
         "INTENSE_nodelay": ("C3", "-", "INTENSE, zero lag"),
         "INTENSE_delay": ("C3", "--", "INTENSE, delay search")}
fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=200)
for m, (c, ls, lab) in style.items():
    s = cv[cv.method == m].sort_values("fpr_nonsel")
    ax.plot(s.fpr_nonsel, s.f1_mixed_lin, color=c, ls=ls, marker="o", ms=2.5,
            lw=2 if m.startswith("INTENSE") else 1.3, label=lab)
ax.set_xlabel("false-positive rate on nonselective cells")
ax.set_ylabel("$F_1$, linear mixed cells")
ax.set_xlim(-0.02, 0.55); ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(FIGS / "glm_matched_fpr_v813.png")
