# -*- coding: utf-8 -*-
"""Finish the MEC decode after an interruption: decode_cohort_primary.py in what it keeps (channel
'primary' of the authors' signal dataset, curated cells, one row per cell, the same tracking stretch and
file layout), but the stream starts at --stream-from, the start of an INSERT statement found by
find_resume_offset.py, instead of the start of the table. Rows are in primary-key order, so every row of
the requested sessions lies after that statement start. Only --sessions are written; files of other
sessions are not touched. The per-session row census goes to decode_primary_log_resume.csv in the output
folder.

usage: python -B mec/decode_cohort_primary_resume.py --dump <section.sql> --offsets <offsets.txt>
                                                     --stream-from BYTE --sessions STEM [STEM ...]
                                                     [--out DIR]
       default --out: [mec] sessions_dir
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decode_cohort_primary as P  # noqa: E402

R, DC = P.R, P.DC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--offsets", type=Path, required=True)
    ap.add_argument("--stream-from", type=int, required=True)
    ap.add_argument("--out", type=Path, default=P.P.get("mec", "sessions_dir"))
    ap.add_argument("--sessions", nargs="+", required=True)
    a = ap.parse_args()
    DC.DUMP, DC.OFFS = a.dump, a.offsets
    s0, e0 = DC.table_span("_cell__traces")
    with open(a.dump, "rb") as fh:
        fh.seek(s0)
        if not fh.read(60).startswith(b"-- Table structure for table `_cell__traces`"):
            raise RuntimeError("the offsets file does not match this dump file")
        fh.seek(a.stream_from)
        if not fh.read(36).startswith(b"INSERT INTO `_cell__traces` VALUES "):
            raise RuntimeError("--stream-from is not the start of an INSERT statement of _cell__traces")
    if not s0 < a.stream_from < e0:
        raise RuntimeError("--stream-from lies outside the table")
    want = set(a.sessions)
    existing = sorted(s for s in want if (a.out / f"{s}.npz").exists())
    if existing:
        raise RuntimeError(f"refusing to overwrite existing files: {existing}")
    auth = P.authors_datasets()
    t0 = time.time()
    trk = DC.preload_tracking()
    cur = DC.preload_curated()
    names = DC.cols("_cell__traces")
    idx = {nm: k for k, nm in enumerate(names)}

    seen = defaultdict(lambda: defaultdict(int))
    failed = defaultdict(int)
    log_rows, done = [], set()
    cur_key, buf = None, {}

    def flush(key):
        if key is None or key not in trk:
            return
        stem = f"{key[0]}_{key[1]}_{key[2]}"
        cells = cur.get(key)
        keep = {c: v for c, v in buf.items() if (cells is None or c in cells)}
        log_rows.append(dict(session=stem, authors_dataset=auth.get(stem, ""),
                             rows=" ".join(f"{d}/{ch}:{n}" for (d, ch), n in sorted(seen[stem].items())),
                             kept_cells=len(keep), unpack_failed=failed[stem]))
        if not keep:
            return
        T = min(v.shape[0] for v in keep.values())
        ids = sorted(keep)
        cal = np.vstack([keep[c][:T] for c in ids])
        tk = trk[key]
        np.savez(a.out / f"{stem}.npz", calcium=cal, cell_ids=np.array(ids),
                 x=P.rs(tk["x_pos"], T), y=P.rs(tk["y_pos"], T), speed=P.rs(tk["speed"], T),
                 head=P.rs_circ(tk["head_angle"], T), fps=np.array([P.FPS]))
        done.add(stem)
        print(f"  {stem}: {len(ids)} cells x {T} frames; rows {log_rows[-1]['rows']} "
              f"({time.time() - t0:.0f} s)", flush=True)

    for tup in DC.stream_tuples(a.stream_from, e0):
        vals = R._split_fields(tup)
        if len(vals) != len(names):
            continue
        key = (DC._txt(vals[idx["metasession_name"]]).decode(errors="replace"),
               vals[idx["session_order"]].strip().decode(errors="replace"),
               DC._txt(vals[idx["session_name"]]).decode(errors="replace"))
        stem = f"{key[0]}_{key[1]}_{key[2]}"
        if key != cur_key:
            if cur_key is not None and f"{cur_key[0]}_{cur_key[1]}_{cur_key[2]}" in want:
                flush(cur_key)
            buf, cur_key = {}, key
            if want <= done:
                break
        if key not in trk or stem not in want:
            continue
        ds = DC._txt(vals[idx["dataset_name"]]).decode(errors="replace")
        ch = DC._txt(vals[idx["channel"]]).decode(errors="replace")
        seen[stem][(ds, ch)] += 1
        if ch != "primary":
            continue
        if stem in auth and ds != auth[stem]:
            continue
        cid = int(vals[idx["cell_id"]])
        cells = cur.get(key)
        if cells is not None and cid not in cells:
            continue
        if cid in buf:
            raise RuntimeError(f"{stem}: a second primary row for cell {cid} (dataset {ds})")
        try:
            buf[cid] = R.unpack_blob(R._decode_field(vals[idx["df_f"]], True))
        except Exception:
            failed[stem] += 1
    if cur_key is not None and f"{cur_key[0]}_{cur_key[1]}_{cur_key[2]}" in want:
        flush(cur_key)

    with open(a.out / "decode_primary_log_resume.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["session", "authors_dataset", "rows", "kept_cells", "unpack_failed"])
        w.writeheader()
        w.writerows(log_rows)
    missing = sorted(want - done)
    print(f"[{'OK' if not missing else 'FAILED'}] {len(done)} sessions written to {a.out}; "
          f"not found: {missing or 'none'} ({time.time() - t0:.0f} s)", flush=True)
    sys.exit(0 if not missing else 1)


if __name__ == "__main__":
    main()
