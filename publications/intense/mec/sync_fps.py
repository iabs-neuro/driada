# -*- coding: utf-8 -*-
"""Imaging frame rate of every MEC (Obenhaus et al., 2022) session, from the frame times in `_sync`.

Reads the `_sync` table of [mec] dump_extract with dj_dump_reader.py and decodes the
frame-time streams:

* '2P frames' (sample_rate 5000): `sync_data` holds the sample number of each imaging frame
  on the 5000 Hz acquisition clock;
* 'frames' / 'planes' (sample_rate 1, the sessions without a '2P frames' stream): `sync_data`
  holds the frame (or plane) times in seconds, so a rate of 1 sample per second is the unit;
* 'Tracking frames' (sample_rate 5000): camera frame times on the same clock as '2P frames'.

Per stream it writes the number of frames, first and last time, the mean rate over the
recording (frames - 1) / (last - first), the rate from the median interval, the longest
interval, and the same mean rate for each subsession (`lengths` splits a combined session).
Nothing in the extract is changed. Needs datajoint 0.12.7 to unpack the streams.

usage: python -B mec/sync_fps.py   -> <[mec] fps_dir>/sync_streams.csv (one row per session and stream)
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(HERE))
import intense_paths as P  # noqa: E402
import dj_dump_reader as R  # noqa: E402

SQL = str(P.get("mec", "dump_extract"))
OUT = P.get("mec", "fps_dir") / "sync_streams.csv"
STREAMS = ("2P frames", "frames", "planes", "Tracking frames")

FIELDS = ["session", "metasession_name", "session_order", "session_name", "dataset_name",
          "sync_name", "sample_rate", "num_indices", "n_times", "dtype", "t_first_s", "t_last_s",
          "duration_s", "fps_mean", "fps_median_interval", "interval_median_s", "interval_min_s",
          "interval_max_s", "n_intervals_over_1p5_median", "n_nonincreasing", "n_subsessions",
          "subsession_lengths", "subsession_fps_mean", "subsession_gaps_s",
          "lengths_sum_matches", "sample_boundaries"]


def describe(r):
    sr = float(r["sample_rate"])
    raw = np.asarray(R.unpack_blob(r["sync_data"]))
    t = raw.astype(np.float64) / sr                       # seconds on the acquisition clock
    lengths = R.unpack_blob(r["lengths"]) if r["lengths"] is not None else None
    bounds = R.unpack_blob(r["sample_boundaries"]) if r["sample_boundaries"] is not None else None
    lengths = np.asarray(lengths, float).ravel() if lengths is not None else np.zeros(0)
    # A stream with no usable data stores NaN lengths; it is kept and reported, not split.
    lengths = [int(x) for x in lengths] if np.all(np.isfinite(lengths)) else []
    bounds = [float(x) for x in np.asarray(bounds, float).ravel()] if bounds is not None else []
    rec = {"metasession_name": r["metasession_name"], "session_order": r["session_order"],
           "session_name": r["session_name"], "dataset_name": r["dataset_name"],
           "sync_name": r["sync_name"], "sample_rate": sr, "num_indices": int(r["num_indices"]),
           "n_times": int(t.size), "dtype": str(raw.dtype)}
    rec["session"] = f"{r['metasession_name']}_{r['session_order']}_{r['session_name']}"
    rec["sample_boundaries"] = " ".join(f"{b:.6g}" for b in bounds)
    rec["subsession_lengths"] = " ".join(str(x) for x in lengths)
    rec["lengths_sum_matches"] = int(sum(lengths) == t.size) if lengths else ""
    if t.size < 2 or not np.all(np.isfinite(t)):
        return rec
    d = np.diff(t)
    med = float(np.median(d))
    rec.update(t_first_s=t[0], t_last_s=t[-1], duration_s=t[-1] - t[0],
               fps_mean=(t.size - 1) / (t[-1] - t[0]), fps_median_interval=1.0 / med,
               interval_median_s=med, interval_min_s=float(d.min()), interval_max_s=float(d.max()),
               n_intervals_over_1p5_median=int(np.sum(d > 1.5 * med)),
               n_nonincreasing=int(np.sum(d <= 0)), n_subsessions=len(lengths) or 1)
    if len(lengths) > 1 and sum(lengths) == t.size:
        edges = np.cumsum([0] + lengths)
        fps_sub, gaps = [], []
        for k in range(len(lengths)):
            seg = t[edges[k]:edges[k + 1]]
            fps_sub.append((seg.size - 1) / (seg[-1] - seg[0]) if seg.size > 1 else np.nan)
            if k:
                gaps.append(seg[0] - t[edges[k] - 1])
        rec["subsession_fps_mean"] = " ".join(f"{f:.5f}" for f in fps_sub)
        rec["subsession_gaps_s"] = " ".join(f"{g:.4f}" for g in gaps)
    return rec


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in R.iter_rows(SQL, "_sync"):
        if r["sync_name"] in STREAMS:
            rows.append(describe(r))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for rec in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in rec.items()})
    by = {}
    for rec in rows:
        by.setdefault(rec["sync_name"], 0)
        by[rec["sync_name"]] += 1
    print(f"wrote {OUT.name}: {len(rows)} rows; per stream {by}")


if __name__ == "__main__":
    main()
