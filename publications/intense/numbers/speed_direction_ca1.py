"""Direction of speed tuning of the CA1 speed cells selected by the GLM only, by INTENSE only and by both.

Hypothesis for the low overlap in Supplementary Table 4 (GLM 86, INTENSE 170, both 14): INTENSE does
not count anti-selective pairs (lower activity at higher values of the variable), the GLM selects
a feature whatever the sign of its effect. Per neuron: Spearman correlation of the fluorescence
with speed at zero lag and the signal ratio (mean fluorescence when speed is above its median over
the mean when below), on every fifth frame as in both analyses. Read-only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402
import paths_v11 as V  # noqa: E402

GLM = V.get("glm_ca1_per_neuron")
DB = P.DB_PRE
NPZ = V.get("nof_v1")
KEY = ["mouse", "session", "neuron_idx"]

g = pd.read_csv(GLM)
db = pd.read_csv(DB)
db = db[(db.pval < 0.001) & (db.me_excess > 0.04) & (db.feature == "speed")]
db = db[db.signal_ratio.isna() | (db.signal_ratio > 1)]
ints = set(map(tuple, db[KEY].astype({"neuron_idx": int}).values.tolist()))
glms = set(map(tuple, g[g.glm_speed == 1][KEY].values.tolist()))
groups = {"GLM only": glms - ints, "INTENSE only": ints - glms, "both": glms & ints}
print({k: len(v) for k, v in groups.items()})
rows = []
for (mouse, sess), _ in pd.DataFrame(sorted(glms | ints), columns=KEY).groupby(["mouse", "session"]):
    d = np.load(NPZ / f"NOF_{mouse}_{sess}_aligned.npz", allow_pickle=True)
    ca, sp = d["Calcium"][:, ::5], d["speed"][::5]
    med = np.median(sp)
    for name, members in groups.items():
        for (m, s, n) in members:
            if (m, s) != (mouse, sess):
                continue
            c = ca[int(n)]
            rho = spearmanr(c, sp).correlation
            ratio = c[sp > med].mean() / max(c[sp <= med].mean(), 1e-12)
            rows.append(dict(group=name, mouse=m, session=s, neuron_idx=int(n), rho=rho, ratio=ratio))
r = pd.DataFrame(rows)
out = P.NUMBERS_DIR / "speed_direction_ca1.csv"
out.parent.mkdir(parents=True, exist_ok=True)
r.to_csv(out, index=False)
for name, q in r.groupby("group"):
    print("%-13s n=%3d  negative rho %3d (%.0f%%)  ratio<=1 %3d (%.0f%%)  median rho %+.3f" % (
        name, len(q), (q.rho < 0).sum(), 100 * (q.rho < 0).mean(), (q.ratio <= 1).sum(),
        100 * (q.ratio <= 1).mean(), q.rho.median()))
print("[OK]", out)
