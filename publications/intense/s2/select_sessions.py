"""One session per CA1 mouse for the synthetic neurons of Supplementary Section S2, chosen by one rule.

Rule: among the sessions of a mouse in which at least two objects were each occupied for at
least 1.5% of the frames (MIN_OCC of the generator, synthetic/make_synth.py: an object counts
when its indicator is above 0.5 in at least that share of frames), the session with the longest
path. A mouse with no such session gets its longest-path session, and is flagged.

Path: sum of the steps of the tracked position (x, y in cm; body centre), in metres, as in
the path table [v11] glm_explore_json of figure_chain/explore.py; when that file is there, the paths are
checked against it.
Data: [data] nof_aligned, the files the generator reads.

Outputs (in [s2] out_dir):
  sessions_all64.csv   every session: path, frames, rate, object shares, eligible, chosen
  sessions_16.csv      the 16 chosen sessions, read by every later script of s2/
usage: python s2/select_sessions.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
import intense_paths as P  # noqa: E402
import paths_v11 as V  # noqa: E402

OUT = P.get("s2", "out_dir")
MIN_OCC = 0.015
OBJECTS = ["object1", "object2", "object3", "object4"]
# the path table is an optional cross-check; the choice of sessions does not read it
EXPLORE_JSON = V.get("glm_explore_json") if V.V11.get("glm_explore_json") else None


def main():
    rows = []
    for p in sorted(P.NOF_ALIGNED.glob("NOF_*_aligned.npz")):
        m = re.match(r"NOF_(H\d+)_(\dD)_aligned\.npz", p.name)
        with np.load(p, allow_pickle=True) as d:
            x, y = d["x"], d["y"]
            occ = {o: float((d[o] > 0.5).mean()) for o in OBJECTS}
            fps = float(d["_metadata"].item()["fps"])
        rows.append({"session": f"NOF_{m.group(1)}_{m.group(2)}", "mouse": m.group(1), "day": m.group(2),
                     "frames": len(x), "fps": fps, "minutes": round(len(x) / fps / 60, 2),
                     "path_m": float(np.hypot(np.diff(x), np.diff(y)).sum() / 100.0),
                     **{f"occ_{o}": occ[o] for o in OBJECTS},
                     "n_objects_ge_min": sum(v >= MIN_OCC for v in occ.values())})
    df = pd.DataFrame(rows)
    df["eligible"] = df.n_objects_ge_min >= 2
    assert len(df) == 64 and df.mouse.nunique() == 16 and (df.groupby("mouse").size() == 4).all()

    # one check of the path against the table the figure script reads (v1 files, same x, y)
    if EXPLORE_JSON is not None and EXPLORE_JSON.exists():
        ex = {r["session"]: r["path_m"] for r in json.loads(EXPLORE_JSON.read_text(encoding="utf-8"))["rows"]}
        diff = (df.path_m - df.session.str[4:].map(ex)).abs()
        print(f"[check] path vs explore.json: max abs difference {diff.max():.4f} m over {diff.notna().sum()} sessions")
    else:
        print("[check] path vs explore.json: skipped, figure_chain/explore.py has not written it")

    chosen, flags = [], []
    for mouse, g in df.groupby("mouse"):
        pool = g[g.eligible]
        if pool.empty:
            flags.append(mouse)
            pool = g
        chosen.append(pool.sort_values("path_m", ascending=False).session.iloc[0])
    df["chosen"] = df.session.isin(chosen)
    df["no_eligible_session"] = df.mouse.isin(flags)
    OUT.mkdir(parents=True, exist_ok=True)
    df.round(5).to_csv(OUT / "sessions_all64.csv", index=False)
    sel = df[df.chosen].sort_values("mouse").reset_index(drop=True)
    sel.round(5).to_csv(OUT / "sessions_16.csv", index=False)
    print("mice without an eligible session:", flags or "none")
    print("eligible sessions per mouse:", df.groupby("mouse").eligible.sum().to_dict())
    cols = ["session", "fps", "frames", "path_m"] + [f"occ_{o}" for o in OBJECTS] + ["n_objects_ge_min"]
    print(sel[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
