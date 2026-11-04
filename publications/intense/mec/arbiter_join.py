# -*- coding: utf-8 -*-
"""Arbiter: the authors' curated cell-type labels vs INTENSE vs GLM, for a given cell-set file.

Phase 2 of arbiter_labels.py with the paths as arguments. The authors' per-cell scores and
shuffle cutoffs (LABEL_COLS) are taken unchanged from the arbiter_labels.csv that
arbiter_labels.py decoded from the dump (<[mec] arbiter_dir>/arbiter_labels.csv); only the
detection columns come from the cell-set file, so the dump is read once for any number of
cell-set files. On the cell-set file arbiter_labels.py was run on, the outputs are those of
arbiter_labels.py.

usage: python -B mec/arbiter_join.py --sets <cell_sets.jsonl> --out <dir>
Outputs in <dir>: arbiter_labels.csv (per cell), arbiter_venn.csv (counts), arbiter_venn.png.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402

LBL_SRC = P.get("mec", "arbiter_dir") / "arbiter_labels.csv"
SESS = P.get("mec", "sessions_dir")

# Columns decoded from the dump by phase 1; everything after them is a detection column.
LABEL_COLS = ["session", "cell", "cell_id",
              "gridscore", "grid_95", "grid_99", "is_grid95", "is_grid99",
              "borderscore", "border_95", "border_99", "is_border95", "is_border99",
              "mvl", "mvl_95", "mvl_99", "is_hd95", "is_hd99"]


def fnum(x):
    try:
        return float(x)
    except ValueError:
        return np.nan


def load_labels():
    """(session, cell) -> the phase-1 columns as written, strings kept verbatim."""
    lab = {}
    with open(LBL_SRC, newline="") as fh:
        for r in csv.DictReader(fh):
            lab[(r["session"], int(r["cell"]))] = {c: r[c] for c in LABEL_COLS}
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    out_lbl, out_venn, out_fig = (a.out / "arbiter_labels.csv", a.out / "arbiter_venn.csv",
                                  a.out / "arbiter_venn.png")
    t00 = time.time()
    lab = load_labels()

    # cell index -> dump cell_id, per analysed session npz (as in arbiter_labels.py)
    id_map = {}
    for f in SESS.glob("*.npz"):
        with np.load(f) as d:
            id_map[f.stem] = d["cell_ids"]

    rows = []
    with open(a.sets) as fh:
        for line in fh:
            d = json.loads(line)
            stem = d["session"]
            cid = int(id_map[stem][d["cell"]])
            L = lab[(stem, d["cell"])]
            if int(L["cell_id"]) != cid:
                raise SystemExit(f"cell_id mismatch {stem} cell {d['cell']}: "
                                 f"{L['cell_id']} in labels, {cid} in npz")
            row = dict(L)
            row.update({
                "int_place": int("place" in d["int_bytype"]),
                "int_raw_place": int("place" in d["int_raw"]),
                "glm_place": int("place" in d["glm"]),
                "int_hd": int("head_direction" in d["int_bytype"]),
                "int_raw_hd": int("head_direction" in d["int_raw"]),
                "glm_hd": int("head_direction" in d["glm"]),
            })
            rows.append(row)
    if len(rows) != len(lab):
        raise SystemExit(f"{len(rows)} cells in the set file, {len(lab)} in the labels")
    with open(out_lbl, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"labels joined: {len(rows)} cells -> {out_lbl}", flush=True)

    # ---- three-circle counts per type (arbiter_labels.py) ----
    num = [{k: (fnum(v) if k in ("gridscore", "grid_95", "borderscore", "border_95",
                                 "mvl", "mvl_95") else v) for k, v in r.items()}
           for r in rows]
    types = [("grid", "is_grid95", "int_place", "glm_place"),
             ("border", "is_border95", "int_place", "glm_place"),
             ("head_direction", "is_hd95", "int_hd", "glm_hd")]
    # The authors' labels exist only where both the score and their shuffle cutoff
    # are defined (the 'standard'-curated sessions). Restrict each diagram to that
    # universe, otherwise the INTENSE/GLM circles are inflated by cells that have
    # no arbiter at all.
    universe_keys = {"grid": ("gridscore", "grid_95"), "border": ("borderscore", "border_95"),
                     "head_direction": ("mvl", "mvl_95")}
    venn = []
    for tname, labk, ikey, gkey in types:
        sk, ck = universe_keys[tname]
        U = {i for i, r in enumerate(num)
             if np.isfinite(r[sk]) and np.isfinite(r[ck])}
        A = {i for i in U if int(num[i][labk])}                # authors
        B = {i for i in U if num[i][ikey]}                     # INTENSE by_type
        C = {i for i in U if num[i][gkey]}                     # GLM
        reg = {
            "A_only": len(A - B - C), "B_only": len(B - A - C), "C_only": len(C - A - B),
            "AB": len((A & B) - C), "AC": len((A & C) - B), "BC": len((B & C) - A),
            "ABC": len(A & B & C),
        }
        cov_int = len(A & B) / len(A) if A else np.nan
        cov_glm = len(A & C) / len(A) if A else np.nan
        venn.append({"type": tname, "universe": len(U), "authors": len(A),
                     "intense": len(B), "glm": len(C),
                     "intense_recall_of_authors": cov_int, "glm_recall_of_authors": cov_glm,
                     **reg})
        print(f"  {tname}: authors={len(A)} INTENSE={len(B)} GLM={len(C)} "
              f"recall INT {cov_int:.4f} / GLM {cov_glm:.4f}", flush=True)
    with open(out_venn, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(venn[0].keys()))
        w.writeheader()
        w.writerows(venn)

    # ---- figure: three 3-circle diagrams (arbiter_labels.py) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
    titles = {"grid": "Grid cells (authors' 95th pct)",
              "border": "Border cells (authors' 95th pct)",
              "head_direction": "Head-direction cells (authors' 95th pct)"}
    feat = {"grid": "position", "border": "position", "head_direction": "head direction"}
    for ax, v in zip(axes, venn):
        centers = [(-0.35, 0.25), (0.35, 0.25), (0.0, -0.4)]
        cols = ["#6a51a3", "#cb181d", "#2171b5"]
        names = [f"Authors (n={v['authors']})", f"INTENSE {feat[v['type']]} (n={v['intense']})",
                 f"GLM {feat[v['type']]} (n={v['glm']})"]
        for (cx, cy), c, nm in zip(centers, cols, names):
            ax.add_patch(Circle((cx, cy), 0.62, alpha=0.25, color=c, ec=c, lw=2))
        ax.text(-0.75, 0.75, names[0], color=cols[0], fontsize=9, ha="center")
        ax.text(0.75, 0.75, names[1], color=cols[1], fontsize=9, ha="center")
        ax.text(0.0, -1.15, names[2], color=cols[2], fontsize=9, ha="center")
        pos = {"A_only": (-0.62, 0.42), "B_only": (0.62, 0.42), "C_only": (0.0, -0.72),
               "AB": (0.0, 0.5), "AC": (-0.38, -0.18), "BC": (0.38, -0.18), "ABC": (0.0, 0.05)}
        for k, (x, y) in pos.items():
            ax.text(x, y, str(v[k]), ha="center", va="center", fontsize=11)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.35, 1.1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{titles[v['type']]}\nrecall of authors' cells: INTENSE "
                     f"{v['intense_recall_of_authors']:.0%} vs GLM "
                     f"{v['glm_recall_of_authors']:.0%}", fontsize=10)
    fig.suptitle("Arbiter: authors' curated labels vs INTENSE (by_type) vs GLM, "
                 "analysed MEC cells", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_fig)
    print(f"ARBITER DONE in {time.time()-t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
