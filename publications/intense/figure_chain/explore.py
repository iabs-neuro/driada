"""Path length and arena coverage per session from x, y (cm; arena 44 x 44 cm).

Reads the synchronized sessions of [v11] nof_v1 and writes [v11] glm_explore_json. Two scripts take
path_m from it to keep only examples from sessions in which the mouse covered at least 10 m: the
picks of Supplementary Fig. 4 (figures_v10.6 and figures_v11.1 fig_examples_*.py) and the candidates
for the example of Fig. 3d,e (compute_pack_v11.1/scripts/f_candidates.py).

usage: python -X utf8 -B figure_chain/explore.py
"""
import json
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # figure_chain/
import _paths_fig as _F  # noqa: E402
D = _F.V.get("nof_v1")
OUT = _F.V.get("glm_explore_json")
rows = []
for p in sorted(D.glob("NOF_*_aligned.npz")):
    z = np.load(p, allow_pickle=True)
    x, y, sp = z["x"], z["y"], z["speed"]
    T = len(x)
    fps = 30 if T > 15000 else 20
    step = np.hypot(np.diff(x), np.diff(y))
    path_raw = step.sum() / 100.0
    path_speed = np.sum(np.clip(sp, 0, None)) / fps / 100.0
    # coverage: fraction of 44x44 cm covered by visited bins (origin-independent count of bins)
    cov = {}
    for b in (2.0, 4.0):
        ix = np.floor((x - x.min()) / b).astype(int); iy = np.floor((y - y.min()) / b).astype(int)
        nb = len(set(zip(ix.tolist(), iy.tolist())))
        cov[b] = nb * b * b / (44.0 * 44.0)
    bbox = (x.max() - x.min()) * (y.max() - y.min()) / (44.0 * 44.0)
    rows.append(dict(session=p.name[4:-12], frames=T, fps=fps, minutes=T / fps / 60,
                     path_m=round(path_raw, 2), path_speed_m=round(path_speed, 2),
                     cov2=round(cov[2.0], 3), cov4=round(cov[4.0], 3), bbox=round(bbox, 3),
                     xspan=round(x.max() - x.min(), 1), yspan=round(y.max() - y.min(), 1)))
paths = np.array([r["path_m"] for r in rows]); ps = np.array([r["path_speed_m"] for r in rows])
summ = dict(n=len(rows), median_path_m=float(np.median(paths)), median_path_speed_m=float(np.median(ps)),
            n_under_10m=int((paths < 10).sum()), under_10m=[r["session"] for r in rows if r["path_m"] < 10],
            n_under_10m_speed=int((ps < 10).sum()), under_10m_speed=[r["session"] for r in rows if r["path_speed_m"] < 10])
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(dict(summary=summ, rows=rows), indent=1), encoding="utf-8")
print(json.dumps(summ, indent=1))
for r in rows:
    if r["session"] in ("H39_4D", "H08_4D") or r["path_m"] < 12:
        print(r)
print("fps counts:", {f: sum(1 for r in rows if r["fps"] == f) for f in (20, 30)})
print("minutes range:", min(r["minutes"] for r in rows), max(r["minutes"] for r in rows))
print("max spans:", max(r["xspan"] for r in rows), max(r["yspan"] for r in rows))
