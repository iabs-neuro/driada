"""Frame rate of every NOF session -> fps_by_session.csv (session, fps, source).

The rate is the CaImAn parameter ``fr`` that BEARMIND wrote from the miniscope
timestamps. It was extracted before into the author-review checks
(fr_sorted.csv, 60 sessions); here it is read again from the same CaImAn
pickles, without unpickling them (opcode scan, as checks/pk4.py does), and the
two must agree. The four sessions of mouse H36 have no CaImAn file on this
machine: they get 29.8 frames/s, the rate of the sessions with the same number
of frames (about 17,860), with source ``frame count``.

Every rate is checked against the length of the synchronized session: frames /
fps must be about 10 minutes, which is how long every session lasted.

usage: python compute/make_fps_table.py   ->  <numbers_dir>/fps_by_session.csv
"""
import csv
import os
import pickletools
import re
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402
import paths_v11 as V  # noqa: E402

V1 = V.get("nof_v1")
FR_CSV = V.get("bearmind_fr")
PICKLES = V.get("bearmind_pickles")
NO_CAIMAN_FPS = 29.8
OUT = P.NUMBERS_DIR / "fps_by_session.csv"


def caiman_fr(path):
    """Value of the CaImAn parameter 'fr' in a pickle, read from its opcodes only."""
    with open(path, "rb") as f:
        k = -1
        for op, arg, _ in pickletools.genops(f):
            if k >= 0:
                if op.name in ("SHORT_BINBYTES", "BINBYTES") and len(arg) == 8:
                    return struct.unpack("<d", arg)[0]
                if op.name == "BINFLOAT":
                    return arg
                k += 1
                if k > 8:
                    return None
            if op.name in ("SHORT_BINUNICODE", "BINUNICODE") and arg == "fr":
                k = 0
    return None


def n_frames(path):
    with np.load(path, allow_pickle=False) as d:
        return d["Calcium"].shape[1]


def main():
    sessions = sorted(p.name[:-len("_aligned.npz")] for p in V1.glob("NOF_*_aligned.npz"))
    assert len(sessions) == 64, len(sessions)
    extracted = {}
    with open(FR_CSV, newline="") as f:
        for name, fr in csv.reader(f):
            extracted[name] = float(fr)
    assert len(extracted) == 60, len(extracted)

    pickles = {}
    for p in PICKLES.glob("NOF_*_estimates.pickle"):
        name = re.match(r"(NOF_H\d+_\dD)", p.name).group(1)
        assert name not in pickles, f"two CaImAn files for {name}"
        pickles[name] = p
    assert set(pickles) == set(extracted), set(pickles) ^ set(extracted)

    rows, problems = [], []
    for s in sessions:
        nf = n_frames(V1 / f"{s}_aligned.npz")
        if s in pickles:
            fr = caiman_fr(pickles[s])
            if fr is None or fr != extracted[s]:
                problems.append(f"{s}: fr in pickle {fr} != fr_sorted.csv {extracted[s]}")
            fps, source = extracted[s], "CaImAn fr"
        else:
            fps, source = NO_CAIMAN_FPS, "frame count"
            if not 17000 < nf < 18500:
                problems.append(f"{s}: no CaImAn file and {nf} frames, not the 30-fps length")
        dur = nf / fps
        if not 595 <= dur <= 605:
            problems.append(f"{s}: {nf} frames / {fps} = {dur:.1f} s, not about 10 min")
        rows.append((s, fps, source, nf, dur))

    for s, fps, source, nf, dur in rows:
        print(f"{s},{fps},{source},{nf} frames,{dur:.1f} s")
    if problems:
        print("[FAILED]", *problems, sep="\n  ")
        sys.exit(1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "fps", "source"])
        for s, fps, source, *_ in rows:
            w.writerow([s, repr(fps), source])
    os.replace(tmp, OUT)
    g20 = sum(fps < 25 for _, fps, *_ in rows)
    print(f"[OK] {len(rows)} sessions: {g20} at about 20 frames/s, {len(rows) - g20} at about 30; "
          f"{sum(r[2] == 'frame count' for r in rows)} by frame count; "
          f"durations {min(r[4] for r in rows):.1f}-{max(r[4] for r in rows):.1f} s -> {OUT}")


if __name__ == "__main__":
    main()
