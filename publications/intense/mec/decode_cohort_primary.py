# -*- coding: utf-8 -*-
"""Session files of the MEC analysis: one streaming pass over _cell__traces of the Obenhaus et
al. (2022) dump, keeping for every cell only the row the authors' scores were computed from.

The primary key of _cell__traces is (metasession_name, session_order, session_name,
dataset_name, cell_id, channel), and the dump writes rows in key order, so a session can hold
several rows per cell: a 'secondary' channel, or a second signal dataset. Every df_f tuning
curve in the authors' __angular_rate is channel 'primary' of one signal dataset per session
([mec] signal_datasets, written by extract_dataset_map.py). This script keeps exactly those
rows, refuses a second row for the same cell, and records per session which (dataset, channel)
rows the table holds (decode_primary_log.csv in the output folder).

Per session it keeps the curated cells ('standard' in filtered_cells), trims all cells to the
shortest trace and stretches the tracking linearly onto the imaging frames (rs, rs_circ). The
stored fps is a placeholder: run_xyfix.py takes the rate of each session from [mec] fps_table.

Reads [mec] dump with [mec] table_offsets (or a file holding the _cell__traces section with an
offsets file for it) and the tracking and curated cells of [mec] dump_extract. Needs datajoint
0.12.7, the version that wrote the dump, to unpack the arrays.

usage: python -B mec/decode_cohort_primary.py [--dump FILE] [--offsets FILE] [--out DIR]
                                              [--sessions STEM ...]
       defaults: [mec] dump, [mec] table_offsets, [mec] sessions_dir;
       with --sessions the pass stops after the last requested session.
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(HERE))
import intense_paths as P  # noqa: E402
import dj_dump_reader as R  # noqa: E402
import decode_cohort as DC  # noqa: E402  (stream_tuples, table_span, cols, preload_* read DC.DUMP)

FPS = 7.52
SIGNAL_DATASETS = P.get("mec", "signal_datasets")


def authors_datasets():
    out = {}
    with open(SIGNAL_DATASETS, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["channel"] != "primary":
                continue
            if r["session"] in out and out[r["session"]] != r["signal_dataset"]:
                raise RuntimeError(f"two authors' signal datasets in {r['session']}")
            out[r["session"]] = r["signal_dataset"]
    return out


def rs(a, T):
    a = np.asarray(a, float); n = len(a)
    g = np.isfinite(a)
    if g.sum() >= 2:
        a = np.interp(np.arange(n), np.arange(n)[g], a[g])   # fill NaN
    return np.interp(np.linspace(0, n - 1, T), np.arange(n), a)


def rs_circ(a, T):
    a = np.asarray(a, float); n = len(a)
    g = np.isfinite(a)
    s = np.sin(a); c = np.cos(a)
    if g.sum() >= 2:
        s = np.interp(np.arange(n), np.arange(n)[g], s[g])
        c = np.interp(np.arange(n), np.arange(n)[g], c[g])
    gr = np.linspace(0, n - 1, T)
    return np.mod(np.arctan2(np.interp(gr, np.arange(n), s), np.interp(gr, np.arange(n), c)), 2 * np.pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, default=DC.DUMP)
    ap.add_argument("--offsets", type=Path, default=DC.OFFS, help="byte offsets of the tables in --dump")
    ap.add_argument("--out", type=Path, default=P.get("mec", "sessions_dir"))
    ap.add_argument("--sessions", nargs="*", default=None)
    a = ap.parse_args()
    if not a.dump.exists():
        print(f"[FAILED] the full dump is not at {a.dump}; _cell__traces is only in the full dump", flush=True)
        sys.exit(2)
    DC.DUMP, DC.OFFS = a.dump, a.offsets
    s0, e0 = DC.table_span("_cell__traces")
    with open(a.dump, "rb") as fh:
        fh.seek(s0)
        head = fh.read(60)
    if not head.startswith(b"-- Table structure for table `_cell__traces`"):
        raise RuntimeError("table_offsets.txt does not match this dump file")
    a.out.mkdir(parents=True, exist_ok=True)
    want = set(a.sessions) if a.sessions else None
    auth = authors_datasets()
    t0 = time.time()
    trk = DC.preload_tracking()
    cur = DC.preload_curated()
    names = DC.cols("_cell__traces")
    idx = {nm: k for k, nm in enumerate(names)}
    for c in ("dataset_name", "channel", "cell_id", "df_f"):
        assert c in idx, (c, names)

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
                 x=rs(tk["x_pos"], T), y=rs(tk["y_pos"], T), speed=rs(tk["speed"], T),
                 head=rs_circ(tk["head_angle"], T), fps=np.array([FPS]))
        done.add(stem)
        print(f"  {stem}: {len(ids)} cells x {T} frames; rows {log_rows[-1]['rows']} "
              f"({time.time() - t0:.0f} s)", flush=True)

    for tup in DC.stream_tuples(s0, e0):
        vals = R._split_fields(tup)
        if len(vals) != len(names):
            continue
        key = (DC._txt(vals[idx["metasession_name"]]).decode(errors="replace"),
               vals[idx["session_order"]].strip().decode(errors="replace"),
               DC._txt(vals[idx["session_name"]]).decode(errors="replace"))
        stem = f"{key[0]}_{key[1]}_{key[2]}"
        if key != cur_key:
            if cur_key is not None and (want is None or f"{cur_key[0]}_{cur_key[1]}_{cur_key[2]}" in want):
                flush(cur_key)
            buf, cur_key = {}, key
            if want is not None and want <= done:
                break
        if key not in trk or (want is not None and stem not in want):
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
        # a cell whose blob does not unpack is dropped, and counted per session
        try:
            buf[cid] = R.unpack_blob(R._decode_field(vals[idx["df_f"]], True))
        except Exception:
            failed[stem] += 1
    if cur_key is not None and (want is None or f"{cur_key[0]}_{cur_key[1]}_{cur_key[2]}" in want):
        flush(cur_key)

    with open(a.out / "decode_primary_log.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["session", "authors_dataset", "rows", "kept_cells", "unpack_failed"])
        w.writeheader()
        w.writerows(log_rows)
    missing = sorted(want - done) if want else []
    print(f"[OK] {len(done)} sessions written to {a.out}; not found: {missing or 'none'} "
          f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
