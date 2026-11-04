# -*- coding: utf-8 -*-
"""Arbiter analysis: the authors' curated cell-type labels vs INTENSE vs GLM.

Phase 1 (dump decode): extract per-cell functional scores and the authors' own shuffle
cutoffs from the Obenhaus DataJoint dump ([mec] dump, located by [mec] table_offsets):
  grid   — __grid_score.gridscore (signal_type=df_f) vs per-session
           __cutoffs_gridscore.gridscore_95/_99 (param_hash_id_cell='standard')
  border — __border_score.borderscore vs __cutoffs_border_score.borderscore_95/_99
  HD     — __angular_rate__stats.mvl (df_f) vs per-cell
           __shuffled__angular_rate_stats.mvl_95/mvl_99
Object-vector cells are skipped: their score is defined against object sessions,
which our open-field arms never analysed. Only text fields are read, so datajoint is not needed.

Phase 2 (join): map the per-cell detection sets (--sets: one JSON line per analysed cell with
the keys session, cell, glm, int_raw, int_bytype; cell index -> dump cell_id via each session
file's cell_ids in [mec] sessions_dir) to the labels, and count the three-way Venn per type —
authors' label vs INTENSE vs GLM — where the INTENSE/GLM feature is 'place' for grid and border
cells and 'head_direction' for HD cells. Primary threshold: the authors' 95th percentile; 99th
is also written.

Outputs in --out: arbiter_labels.csv (per cell), arbiter_venn.csv (counts), arbiter_venn.png.
arbiter_join.py takes the label columns of this arbiter_labels.csv for other cell-set files.

usage: python -B mec/arbiter_labels.py [--sets FILE] [--out DIR]
       defaults: <[mec] recompute_dir>/variants/new_on/cell_sets.jsonl, [mec] arbiter_dir
"""
import os, sys, time, json
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
import argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(HERE))
import intense_paths as P  # noqa: E402
import decode_cohort as DC          # stream_tuples, table_span, cols, DUMP
import dj_dump_reader as R

SESS = P.get("mec", "sessions_dir")
SETS = P.get("mec", "recompute_dir") / "variants" / "new_on" / "cell_sets.jsonl"
OUT = P.get("mec", "arbiter_dir")


def stream_table(table, want, where=None, label=None):
    """Yield dicts of `want` columns for each row of `table` (streaming, no blobs)."""
    names = DC.cols(table)
    idx = {nm: k for k, nm in enumerate(names)}
    s, e = DC.table_span(table)
    n = 0
    t0 = time.time()
    for tup in DC.stream_tuples(s, e):
        vals = R._split_fields(tup)
        if len(vals) != len(names):
            continue
        row = {}
        ok = True
        for c in want:
            raw = vals[idx[c]]
            row[c] = raw.strip().strip(b"'").decode(errors="replace")
        if where and not where(row):
            continue
        yield row
        n += 1
    print(f"  [{label or table}] {n} rows in {time.time()-t0:.0f}s", flush=True)


def fnum(x):
    try:
        return float(x)
    except ValueError:
        return np.nan


def collect():
    key3 = ["metasession_name", "session_order", "session_name"]

    # ---- per-cell scores (df_f rows only) ----
    grid, border, hd = {}, {}, {}
    for row in stream_table("__grid_score", key3 + ["cell_id", "signal_type", "gridscore"],
                            where=lambda r: r["signal_type"] == "df_f", label="grid_score"):
        k = (row["metasession_name"], row["session_order"], row["session_name"], int(row["cell_id"]))
        v = fnum(row["gridscore"])
        if k not in grid or (np.isfinite(v) and v > grid[k]):
            grid[k] = v
    for row in stream_table("__border_score", key3 + ["cell_id", "signal_type", "borderscore"],
                            where=lambda r: r["signal_type"] == "df_f", label="border_score"):
        k = (row["metasession_name"], row["session_order"], row["session_name"], int(row["cell_id"]))
        v = fnum(row["borderscore"])
        if k not in border or (np.isfinite(v) and v > border[k]):
            border[k] = v
    for row in stream_table("__angular_rate__stats", key3 + ["cell_id", "signal_type", "mvl"],
                            where=lambda r: r["signal_type"] == "df_f", label="hd_stats"):
        k = (row["metasession_name"], row["session_order"], row["session_name"], int(row["cell_id"]))
        hd[k] = fnum(row["mvl"])

    # ---- cutoffs: per-session for grid/border, per-cell for HD ----
    gcut, bcut, hcut = {}, {}, {}
    for row in stream_table("__cutoffs_gridscore",
                            key3 + ["param_hash_id_cell", "gridscore_95", "gridscore_99"],
                            where=lambda r: r["param_hash_id_cell"] == "standard",
                            label="cutoffs_grid"):
        gcut[(row["metasession_name"], row["session_order"], row["session_name"])] = (
            fnum(row["gridscore_95"]), fnum(row["gridscore_99"]))
    for row in stream_table("__cutoffs_border_score",
                            key3 + ["param_hash_id_cell", "borderscore_95", "borderscore_99"],
                            where=lambda r: r["param_hash_id_cell"] == "standard",
                            label="cutoffs_border"):
        bcut[(row["metasession_name"], row["session_order"], row["session_name"])] = (
            fnum(row["borderscore_95"]), fnum(row["borderscore_99"]))
    for row in stream_table("__shuffled__angular_rate_stats",
                            key3 + ["cell_id", "mvl_95", "mvl_99"], label="cutoffs_hd"):
        k = (row["metasession_name"], row["session_order"], row["session_name"], int(row["cell_id"]))
        hcut[k] = (fnum(row["mvl_95"]), fnum(row["mvl_99"]))

    return grid, border, hd, gcut, bcut, hcut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=Path, default=SETS)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    OUT_LBL, OUT_VENN, OUT_FIG = a.out / "arbiter_labels.csv", a.out / "arbiter_venn.csv", a.out / "arbiter_venn.png"
    t00 = time.time()
    grid, border, hd, gcut, bcut, hcut = collect()
    print(f"decode done in {time.time()-t00:.0f}s: grid={len(grid)} border={len(border)} "
          f"hd={len(hd)} gcut={len(gcut)} bcut={len(bcut)} hcut={len(hcut)}", flush=True)

    # cell index -> dump cell_id, per analysed session npz
    id_map = {}
    for f in SESS.glob("*.npz"):
        d = np.load(f)
        id_map[f.stem] = d["cell_ids"]

    import csv
    rows = []
    with open(a.sets) as fh:
        for line in fh:
            d = json.loads(line)
            stem = d["session"]
            meta, order, sess = stem.split("_", 2)
            cid = int(id_map[stem][d["cell"]])
            k4 = (meta, order, sess, cid)
            k3 = (meta, order, sess)
            g, g95, g99 = grid.get(k4, np.nan), *gcut.get(k3, (np.nan, np.nan))
            b, b95, b99 = border.get(k4, np.nan), *bcut.get(k3, (np.nan, np.nan))
            h, (h95, h99) = hd.get(k4, np.nan), hcut.get(k4, (np.nan, np.nan))
            rows.append({
                "session": stem, "cell": d["cell"], "cell_id": cid,
                "gridscore": g, "grid_95": g95, "grid_99": g99,
                "is_grid95": int(np.isfinite(g) and np.isfinite(g95) and g > g95),
                "is_grid99": int(np.isfinite(g) and np.isfinite(g99) and g > g99),
                "borderscore": b, "border_95": b95, "border_99": b99,
                "is_border95": int(np.isfinite(b) and np.isfinite(b95) and b > b95),
                "is_border99": int(np.isfinite(b) and np.isfinite(b99) and b > b99),
                "mvl": h, "mvl_95": h95, "mvl_99": h99,
                "is_hd95": int(np.isfinite(h) and np.isfinite(h95) and h > h95),
                "is_hd99": int(np.isfinite(h) and np.isfinite(h99) and h > h99),
                "int_place": int("place" in d["int_bytype"]),
                "int_raw_place": int("place" in d["int_raw"]),
                "glm_place": int("place" in d["glm"]),
                "int_hd": int("head_direction" in d["int_bytype"]),
                "int_raw_hd": int("head_direction" in d["int_raw"]),
                "glm_hd": int("head_direction" in d["glm"]),
            })
    with open(OUT_LBL, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"labels joined: {len(rows)} cells -> {OUT_LBL.name}", flush=True)

    # ---- three-circle counts per type ----
    import itertools
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
    for tname, lab, ikey, gkey in types:
        sk, ck = universe_keys[tname]
        U = {i for i, r in enumerate(rows)
             if np.isfinite(r[sk]) and np.isfinite(r[ck])}
        A = {i for i in U if rows[i][lab]}                     # authors
        B = {i for i in U if rows[i][ikey]}                    # INTENSE by_type
        C = {i for i in U if rows[i][gkey]}                    # GLM
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
              f"recall INT {cov_int:.2f} / GLM {cov_glm:.2f}", flush=True)
    with open(OUT_VENN, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(venn[0].keys()))
        w.writeheader(); w.writerows(venn)

    # ---- figure: three 3-circle diagrams ----
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
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.35, 1.1)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{titles[v['type']]}\nrecall of authors' cells: INTENSE "
                     f"{v['intense_recall_of_authors']:.0%} vs GLM "
                     f"{v['glm_recall_of_authors']:.0%}", fontsize=10)
    fig.suptitle("Arbiter: authors' curated labels vs INTENSE (by_type) vs GLM, "
                 "analysed MEC cells", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_FIG)
    print(f"ARBITER DONE in {time.time()-t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
