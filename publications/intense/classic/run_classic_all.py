# -*- coding: utf-8 -*-
"""Full run of the classic place-cell test over the 64 NOF sessions.

Two steps, on the clock of the frame rate of each session ('true', the one the paper uses) and, when
asked, on the nominal 20 frames/s ('old'):

1. detect_events.py  - wavelet events of DRIADA 1.2.2 per session and clock (the slow part;
                       sessions already cached are skipped, so a stopped run resumes);
2. classic_fps.py    - classic statistics, seed 0, and on the old clock also seed 1 (how far the
                       method agrees with itself), and the same with the event placed in the middle
                       of its run (tag _center); recomputed every time, seconds per session.

usage: python classic/run_classic_all.py [--n_jobs 4] [--clocks true] [--dry-run]
"""
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paths_v11 as V  # noqa: E402

FPS_TABLE = V.get("fps_table")


def sessions():
    with open(FPS_TABLE, newline="") as f:
        names = [r["session"] for r in csv.DictReader(f)]
    if len(names) != 64:
        raise RuntimeError(f"{FPS_TABLE}: {len(names)} sessions, expected 64")
    return names


def step(cmd, dry):
    print(">>", " ".join(cmd[:8]), "..." if len(cmd) > 8 else "", flush=True)
    if dry:
        return
    t0 = time.time()
    subprocess.run(cmd, cwd=HERE, check=True)
    print(f"[OK] {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--clocks", nargs="+", default=["true"], choices=["old", "true"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    S = sessions(); py = sys.executable
    step([py, "detect_events.py", "--n_jobs", str(a.n_jobs), "--clocks", *a.clocks, "--sessions", *S], a.dry_run)
    for extra in ([], ["--set", "event_time=run_center", "--tag", "_center"]):
        step([py, "classic_fps.py", "--clocks", *a.clocks, "--seed", "0", *extra, "--sessions", *S], a.dry_run)
        if "old" in a.clocks:
            step([py, "classic_fps.py", "--clocks", "old", "--seed", "1", *extra, "--sessions", *S], a.dry_run)
    print("[OK] full classic run done", flush=True)


if __name__ == "__main__":
    main()
