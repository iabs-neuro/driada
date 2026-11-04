# -*- coding: utf-8 -*-
"""Join the per-session frame rates from sync_streams.csv with the MEC analysis inputs.

Inputs (read only):
* <[mec] fps_dir>/sync_streams.csv (written by sync_fps.py);
* the session files of [mec] sessions_dir (written by decode_cohort_primary.py): number of
  frames of the calcium traces, number of cells, stored fps;
* <[mec] recompute_dir>/variants/new_on/results.csv - one row per session of the final MEC
  analysis (column in_si_final); <[mec] arbiter_dir>/arbiter_labels.csv, whose sessions are
  checked against them;
* `session` and `meta_session__scope` of [mec] dump_extract - setup and session names, to
  describe the sessions.

The imaging rate of a session is the mean rate over its frame times: (frames - 1) / (last - first),
from '2P frames' when the session has that stream, else from 'frames'. Deviation from the
constant 7.52 frames/s (FPS_USED) is rate / 7.52 - 1.

usage: python -B mec/join_fps.py > <[mec] fps_dir>/summary.txt
       -> <[mec] fps_dir>/fps_by_session.csv
"""
import csv
import glob
import os
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(HERE))
import intense_paths as P  # noqa: E402
import dj_dump_reader as R  # noqa: E402

FPS_DIR = P.get("mec", "fps_dir")
SQL = str(P.get("mec", "dump_extract"))
SESS = P.get("mec", "sessions_dir")
RESULTS = P.get("mec", "recompute_dir") / "variants" / "new_on" / "results.csv"
LABELS = P.get("mec", "arbiter_dir") / "arbiter_labels.csv"
FPS_USED = 7.52
THR = 0.01


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def main():
    streams = defaultdict(dict)
    with open(FPS_DIR / "sync_streams.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            streams[r["session"]][r["sync_name"]] = r

    sess_meta = {}
    for r in R.iter_rows(SQL, "session"):
        key = f"{r['metasession_name']}_{r['session_order']}_{r['session_name']}"
        sess_meta[key] = r
    scope = {r["metasession_name"]: r["scope_name"] for r in R.iter_rows(SQL, "meta_session__scope")}

    npz = {}
    for fn in sorted(glob.glob(os.path.join(SESS, "*.npz"))):
        # The shape is read from the .npy header, so the 1.5 GB of traces are not loaded.
        with zipfile.ZipFile(fn) as z, z.open("calcium.npy") as fh:
            version = np.lib.format.read_magic(fh)
            shape, _, _ = np.lib.format._read_array_header(fh, version)
        fps_stored = float(np.load(fn, allow_pickle=False)["fps"][0])
        npz[Path(fn).stem] = {"T": int(shape[-1]), "n_cells": int(shape[0]), "fps_stored": fps_stored}

    with open(RESULTS, encoding="utf-8") as f:
        si = [r["session"] for r in csv.DictReader(f)]
    si_set = set(si)
    with open(LABELS, encoding="utf-8") as f:
        arb = Counter(r["session"] for r in csv.DictReader(f))

    rows = []
    for s in sorted(set(streams) | set(npz)):
        st = streams.get(s, {})
        src = "2P frames" if "2P frames" in st else ("frames" if "frames" in st else "")
        m = sess_meta.get(s, {})
        rec = {"session": s, "in_sync": int(bool(st)), "frame_stream": src,
               "scope": scope.get(s.split("_")[0], ""), "experiment_type": m.get("experiment_type", ""),
               "animal_name": m.get("animal_name", ""), "timeseries_name": m.get("timeseries_name", ""),
               "in_npz": int(s in npz), "in_si_final": int(s in si_set), "arbiter_cells": arb.get(s, 0)}
        if src:
            r = st[src]
            fm, fmed = fnum(r["fps_mean"]), fnum(r["fps_median_interval"])
            rec.update(n_frames_sync=int(r["n_times"]), duration_s=fnum(r["duration_s"]),
                       fps_mean=fm, fps_median_interval=fmed,
                       interval_max_s=fnum(r["interval_max_s"]),
                       n_intervals_over_1p5_median=r["n_intervals_over_1p5_median"],
                       n_subsessions=r["n_subsessions"], subsession_lengths=r["subsession_lengths"],
                       subsession_fps_mean=r["subsession_fps_mean"],
                       subsession_gaps_s=r["subsession_gaps_s"],
                       dev_from_7p52_pct=100 * (fm / FPS_USED - 1))
            rec["over_1pct"] = int(abs(fm / FPS_USED - 1) > THR)
            if src == "frames" and "planes" in st:
                rec["planes_per_frame"] = int(st["planes"]["n_times"]) / int(r["n_times"])
            if "Tracking frames" in st:
                tr = st["Tracking frames"]
                rec.update(tracking_fps=fnum(tr["fps_mean"]),
                           tracking_start_minus_2p_start_s=fnum(tr["t_first_s"]) - fnum(r["t_first_s"]),
                           tracking_end_minus_2p_end_s=fnum(tr["t_last_s"]) - fnum(r["t_last_s"]))
        if s in npz:
            rec.update(T_npz=npz[s]["T"], n_cells_npz=npz[s]["n_cells"], fps_stored=npz[s]["fps_stored"])
            if src:
                rec["T_minus_sync_frames"] = npz[s]["T"] - rec["n_frames_sync"]
        rows.append(rec)

    fields = ["session", "in_sync", "in_npz", "in_si_final", "arbiter_cells", "scope",
              "experiment_type", "animal_name", "timeseries_name", "frame_stream", "n_frames_sync",
              "T_npz", "T_minus_sync_frames", "n_cells_npz", "duration_s", "fps_stored", "fps_mean",
              "fps_median_interval", "dev_from_7p52_pct", "over_1pct", "interval_max_s",
              "n_intervals_over_1p5_median", "n_subsessions", "subsession_lengths",
              "subsession_fps_mean", "subsession_gaps_s", "planes_per_frame", "tracking_fps",
              "tracking_start_minus_2p_start_s", "tracking_end_minus_2p_end_s"]
    FPS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FPS_DIR / "fps_by_session.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in rec.items()})

    def dist(sel, label):
        v = np.array([r["fps_mean"] for r in sel if "fps_mean" in r])
        dev = 100 * (v / FPS_USED - 1)
        print(f"\n[{label}] sessions {len(sel)}, with a frame-time stream {v.size}")
        if not v.size:
            return
        q = np.percentile(v, [0, 25, 50, 75, 100])
        print(f"  fps_mean min/q25/median/q75/max: " + " / ".join(f"{x:.4f}" for x in q))
        print(f"  deviation from 7.52, %: min {dev.min():+.3f}, max {dev.max():+.3f}")
        print(f"  |deviation| > 1 %: {int(np.sum(np.abs(dev) > 100 * THR))} sessions")
        by_stream = Counter(r["frame_stream"] for r in sel if "fps_mean" in r)
        print(f"  frame stream: {dict(by_stream)}")
        groups = Counter(round(r["fps_mean"], 3) for r in sel if "fps_mean" in r)
        print("  fps_mean rounded to 0.001, count: " +
              ", ".join(f"{k:.3f}: {n}" for k, n in sorted(groups.items())))
        for src in sorted(by_stream):
            vv = np.array([r["fps_mean"] for r in sel if r.get("frame_stream") == src])
            print(f"  {src!r}: n {vv.size}, fps_mean {vv.min():.4f}-{vv.max():.4f}, "
                  f"median {np.median(vv):.4f}")

    print(f"sessions in _sync: {sum(r['in_sync'] for r in rows)}; session files (npz): {len(npz)}; "
          f"final SI sessions ({RESULTS.name}): {len(si)}; distinct {len(si_set)}")
    print(f"final SI sessions without an npz: {sorted(si_set - set(npz))}")
    print(f"npz without a frame-time stream: {sorted(s for s in npz if not streams.get(s))}")
    print(f"npz not in final SI: {sorted(set(npz) - si_set)}")
    print(f"arbiter_labels.csv: {sum(arb.values())} cells in {len(arb)} sessions; "
          f"same set as final SI: {set(arb) == si_set}; cells per session equal to n_cells of the npz: "
          f"{all(arb[s] == npz[s]['n_cells'] for s in arb)}")
    st_fps = Counter(npz[s]["fps_stored"] for s in npz)
    print(f"fps stored in the npz files: {dict(st_fps)}")
    tm = Counter(r.get("T_minus_sync_frames") for r in rows if r["in_npz"] and r["frame_stream"])
    print(f"calcium frames (npz) minus frame times (sync), count: {dict(sorted(tm.items()))}")
    gaps = [r for r in rows if r.get("frame_stream") and int(r["n_intervals_over_1p5_median"] or 0) > 0]
    print(f"sessions with a frame interval > 1.5 x median: {len(gaps)} "
          f"(final SI: {sum(r['in_si_final'] for r in gaps)})")
    for r in gaps:
        print(f"    {r['session']} si={r['in_si_final']} max interval {r['interval_max_s']:.3f} s, "
              f"n {r['n_intervals_over_1p5_median']}, fps_mean {r['fps_mean']:.4f}, "
              f"fps_median {r['fps_median_interval']:.4f}")
    multi = [r for r in rows if r.get("subsession_fps_mean")]
    if multi:
        sub = np.array([float(x) for r in multi for x in r["subsession_fps_mean"].split()])
        gp = [float(x) for r in multi for x in r["subsession_gaps_s"].split()]
        print(f"combined sessions (>1 subsession): {len(multi)}; subsession fps {sub.min():.4f}-"
              f"{sub.max():.4f}; gaps between subsessions {min(gp):.4f}-{max(gp):.4f} s")
    dmed = np.array([abs(r["fps_mean"] / r["fps_median_interval"] - 1) for r in rows if "fps_mean" in r])
    print(f"|mean rate / median-interval rate - 1|: max {100 * dmed.max():.3f} %")

    dist([r for r in rows if r["in_sync"]], "all sessions in _sync")
    dist([r for r in rows if r["in_npz"]], "session files analysed")
    si_rows = [r for r in rows if r["in_si_final"]]
    dist(si_rows, f"final SI ({len(si_rows)})")

    print("\nfinal SI sessions with |deviation| > 1 %:")
    for r in sorted((r for r in si_rows if r.get("over_1pct")), key=lambda r: r["session"]):
        print(f"  {r['session']} {r['frame_stream']!r} fps {r['fps_mean']:.4f} "
              f"({r['dev_from_7p52_pct']:+.2f} %) T {r.get('T_npz')} frames {r.get('n_frames_sync')} "
              f"cells {r.get('n_cells_npz')} scope {r['scope']!r} type {r['experiment_type']!r} "
              f"planes/frame {r.get('planes_per_frame', '')} name {r['timeseries_name']!r}")
    print("\nother sessions with |deviation| > 1 % (not in final SI):")
    for r in sorted((r for r in rows if r.get("over_1pct") and not r["in_si_final"]),
                    key=lambda r: r["session"]):
        print(f"  {r['session']} in_npz={r['in_npz']} {r['frame_stream']!r} fps {r['fps_mean']:.4f} "
              f"({r['dev_from_7p52_pct']:+.2f} %) frames {r.get('n_frames_sync')} name {r['timeseries_name']!r}")

    tr = [r for r in si_rows if "tracking_fps" in r]
    if tr:
        a = np.array([r["tracking_start_minus_2p_start_s"] for r in tr])
        b = np.array([r["tracking_end_minus_2p_end_s"] for r in tr])
        tf = np.array([r["tracking_fps"] for r in tr])
        print(f"\nfinal SI sessions with 'Tracking frames': {len(tr)}; camera rate {tf.min():.3f}-{tf.max():.3f}")
        print(f"  first camera frame minus first imaging frame, s: {a.min():+.3f} .. {a.max():+.3f}")
        print(f"  last camera frame minus last imaging frame, s: {b.min():+.3f} .. {b.max():+.3f}")
    by_scope = Counter((r["scope"], r["experiment_type"], r["frame_stream"]) for r in si_rows)
    print("\nfinal SI sessions by scope / experiment_type / frame stream: " +
          "; ".join(f"{k}: {n}" for k, n in sorted(by_scope.items())))


if __name__ == "__main__":
    main()
