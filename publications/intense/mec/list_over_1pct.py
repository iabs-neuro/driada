# -*- coding: utf-8 -*-
"""Lists of the final SI sessions with their frame rates.

Reads <[mec] fps_dir>/fps_by_session.csv (written by join_fps.py) and writes into [mec] fps_dir:
* si_sessions_fps.csv - the sessions of the final MEC analysis (in_si_final), sorted by rate:
  session file name, the dataset's session label, microscope, frames, cells, rate, deviation
  from 7.52, whether it exceeds 1 %, and the time base the rate rests on: frame pulses on the
  5000 Hz acquisition clock ('2P frames', measured) or the exact 0.138 / 0.069 s grid of the
  'frames' stream (nominal). This is the table of [mec] fps_table;
* list_over_1pct.md - a markdown table of those exceeding 1 %, grouped by rate.

usage: python -B mec/list_over_1pct.py
"""
import csv
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
import intense_paths as P  # noqa: E402

FPS_DIR = P.get("mec", "fps_dir")
BASE = {"2P frames": "clock_5000Hz", "frames": "nominal_frames_grid"}
BASE_MD = {"2P frames": "5000 Hz clock", "frames": "nominal grid"}
with open(FPS_DIR / "fps_by_session.csv", encoding="utf-8") as f:
    si = [r for r in csv.DictReader(f) if r["in_si_final"] == "1"]
si.sort(key=lambda r: (round(float(r["fps_mean"]), 3), r["session"]))

with open(FPS_DIR / "si_sessions_fps.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["session", "label", "microscope", "frames", "cells", "fps_true", "dev_from_7p52_pct",
                "over_1pct", "time_base"])
    for r in si:
        w.writerow([r["session"], r["timeseries_name"], r["scope"], r["T_npz"], r["n_cells_npz"],
                    f"{float(r['fps_mean']):.4f}", f"{float(r['dev_from_7p52_pct']):+.3f}", r["over_1pct"],
                    BASE[r["frame_stream"]]])

rows = [r for r in si if r["over_1pct"] == "1"]
lines = ["| # | Session (file name in [mec] sessions_dir) | Session label in the dataset | Frames/s | Deviation | Time base | Frames | Cells |",
         "|---|---|---|---|---|---|---|---|"]
for k, r in enumerate(rows, 1):
    lines.append(f"| {k} | `{r['session']}` | {r['timeseries_name']} | {float(r['fps_mean']):.3f} | "
                 f"{float(r['dev_from_7p52_pct']):+.1f} % | {BASE_MD[r['frame_stream']]} | {r['T_npz']} | "
                 f"{r['n_cells_npz']} |")
(FPS_DIR / "list_over_1pct.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote si_sessions_fps.csv: {len(si)} sessions; list_over_1pct.md: {len(rows)} sessions")
