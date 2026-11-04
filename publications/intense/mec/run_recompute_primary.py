# -*- coding: utf-8 -*-
"""The MEC analysis of the whole cohort: INTENSE and GLM, three parts each, all six processes at once.

Each process is single-threaded (run_xyfix.py: n_jobs=1, one BLAS thread); memory is the limit, and
run_xyfix.py itself waits before a session while less than 2.5 GB is free. A part whose sessions are
not all done after its first pass (a session that raised, for example on memory, is printed as FAILED
and not recorded) is run once more: run_xyfix.py skips the sessions it has finished. Logs and parts go
to --out, the summary to <out>/driver_summary.txt.

usage: python -X utf8 -B mec/run_recompute_primary.py [--npz-dir DIR] [--out DIR]
       defaults: [mec] sessions_dir, <[mec] recompute_dir>/parts
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import intense_paths as P  # noqa: E402

PY = sys.executable
N = 3
KINDS = ("intense", "glm")


def session_files():
    """Stems of the sessions that have a file, listed in [mec] fps_all (column in_npz)."""
    t = pd.read_csv(P.get("mec", "fps_all"))
    return set(t.loc[t.in_npz == 1, "session"].astype(str))


def log(msg, fh):
    line = f"{datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def expected(npz_dir, k):
    # the same split as run_xyfix.py: sorted(SESS.glob("*.npz"))[k::n]
    return {f.stem for f in sorted(npz_dir.glob("*.npz"))[k::N]}


def done(out, kind, k):
    p = out / f"{kind}_done_{k}of{N}.csv"
    return set(pd.read_csv(p)["session"].astype(str)) if p.exists() else set()


def launch(kind, k, npz_dir, out, logs):
    cmd = [PY, "-X", "utf8", "-B", str(HERE / "run_xyfix_primary.py"), "--npz-dir", str(npz_dir),
           "--out", str(out), "--shard", f"{k}/{N}", "--pass", kind]
    lf = open(logs / f"{kind}_{k}of{N}.log", "a", encoding="utf-8")
    lf.write(f"=== start {datetime.now():%Y-%m-%d %H:%M:%S} :: {' '.join(cmd)}\n")
    lf.flush()
    return subprocess.Popen(cmd, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT), lf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", type=Path, default=P.get("mec", "sessions_dir"))
    ap.add_argument("--out", type=Path, default=P.get("mec", "recompute_dir") / "parts")
    a = ap.parse_args()
    npz, out = a.npz_dir.resolve(), a.out.resolve()
    missing = sorted(session_files() - {f.stem for f in npz.glob("*.npz")})
    if missing:
        sys.exit(f"[FAILED] the decode has not finished: {len(missing)} sessions have no file, "
                 f"for example {missing[0]}")
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    fh = open(out / "driver_summary.txt", "a", encoding="utf-8")
    log(f"{len(list(npz.glob('*.npz')))} session files in {npz}; python {PY}", fh)
    jobs = [(kind, k) for kind in KINDS for k in range(N)]
    for attempt in (1, 2):
        todo = [(kind, k) for kind, k in jobs if expected(npz, k) - done(out, kind, k)]
        if not todo:
            break
        log(f"attempt {attempt}: " + ", ".join(f"{kind} {k}/{N}" for kind, k in todo), fh)
        procs = [(kind, k) + launch(kind, k, npz, out, logs) for kind, k in todo]
        for kind, k, p, lf in procs:
            rc = p.wait()
            lf.close()
            log(f"{kind} {k}/{N}: exit {rc}, done {len(done(out, kind, k))} of {len(expected(npz, k))}", fh)
    bad = {f"{kind} {k}/{N}": sorted(expected(npz, k) - done(out, kind, k)) for kind, k in jobs}
    bad = {j: m for j, m in bad.items() if m}
    if bad:
        log("[FAILED] sessions not done: " + "; ".join(f"{j}: {len(m)} ({', '.join(m[:5])})"
                                                      for j, m in bad.items()), fh)
        sys.exit(1)
    log("[OK] both passes finished in all shards", fh)


if __name__ == "__main__":
    main()
