"""Synthetic neurons of Supplementary Section S2 on the recorded behaviour of one session per mouse,
at the frame rate of each session.

synthetic/make_synth.py, synthetic/run_synth.py and synthetic/score.py are run unchanged, with
these settings made for the duration of the run:

* make: make_synth.py generates the calcium at FPS = 20 for every session
  (spike rates per frame, calcium decay, duration, 5-s activity islands of the
  null neurons). Here FPS is set, before each session, to _metadata['fps'] of
  the file of [data] nof_aligned whose behaviour is used, so the synthetic neurons
  live on the clock of the session. The behaviour carries its _metadata into the
  synthetic file, so INTENSE analyses it at the same rate.
* every folder (synth_npz, truth, results, tables) is <out>/ instead of the
  folder of the scripts.
* sessions: the 16 of sessions_16.csv (s2/select_sessions.py) instead of the 8 of make_synth.py;
* the session number in the seed [42, session number, neuron] is the position in SEED_ORDER:
  the 8 sessions of make_synth.py keep their numbers 0-7, the others follow in the order of
  the mice. make_synth.py takes 100 + len(name) for a session outside its list, which would give
  the same number, hence the same draws, to every other session;
* the search for a place field tries up to MAX_TRIES random fields instead of 500
  (widen_field_search below), with the same conditions on the field;
* N_WORKERS (environment, optional) caps the worker processes of INTENSE, which otherwise
  start one per core; the per-pair shuffles are drawn before the work is split, so the cap
  changes the time, not the result.

Steps: make, then run for each session, then score. Each step skips what is already done.

usage: python s2/s2_synth_fps_16.py make|run|score|all [session ...] [--out DIR]
       (default DIR: <[s2] out_dir>/s2)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if os.environ.get("N_WORKERS"):
    import multiprocessing
    multiprocessing.cpu_count = lambda: int(os.environ["N_WORKERS"])

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, str(PAPER / "synthetic"))
sys.path.insert(1, str(PAPER))
import intense_paths as P  # noqa: E402

S2_DIR = P.get("s2", "out_dir")
OLD8 = ["NOF_H01_2D", "NOF_H02_3D", "NOF_H07_1D", "NOF_H08_4D",
        "NOF_H14_1D", "NOF_H23_2D", "NOF_H31_4D", "NOF_H36_2D"]
SESSIONS16 = list(pd.read_csv(S2_DIR / "sessions_16.csv").session)
SEED_ORDER = OLD8 + [s for s in SESSIONS16 if s not in OLD8]


MAX_TRIES = 5000


def widen_field_search(M):
    """Let Session.place of make_synth.py try MAX_TRIES random fields instead of 500.

    The only range(500) of make_synth.py is the search loop of a place field. A session on
    which the search succeeds within 500 tries draws the same random numbers in the same
    order and gets the same neurons; a session on which it would fail (NOF_H26_1D: the mouse
    spent 45% of the frames in the centre, so a field near the walls kept more than 20% of
    itself in the centre for a place + centre neuron) gets a field satisfying the same
    conditions instead of a stop.
    """
    import builtins
    import inspect
    assert inspect.getsource(M).count("range(500)") == 1
    M.range = lambda *a: builtins.range(MAX_TRIES) if a == (500,) else builtins.range(*a)


def session_rate(P, name):
    with np.load(P.NOF_ALIGNED / f"{name}_aligned.npz", allow_pickle=True) as d:
        if "_metadata" not in d.files:
            raise SystemExit(f"{name}: no _metadata in {P.NOF_ALIGNED}; [data] nof_aligned must name the "
                             "files with the frame rate of each session (compute/make_v2_data.py)")
        return float(d["_metadata"].item()["fps"])


def make(out, names):
    import make_synth as M
    M.HERE = out
    for name in names:
        if (out / "truth" / f"{name}.json").exists():
            print(f"[SKIP] make {name}", flush=True)
            continue
        M.FPS = session_rate(M.P, name)
        M.make_session(name, M.SESSIONS.index(name) if name in M.SESSIONS else 100 + len(name))
        with np.load(out / "synth_npz" / f"{name}_aligned.npz", allow_pickle=True) as d:
            written = d["_metadata"].item() if "_metadata" in d.files else None
        if written is None or written.get("fps") != M.FPS:
            raise RuntimeError(f"{name}: synthetic file carries {written}, generated at {M.FPS}")
        print(f"  {name} generated at {M.FPS} frames/s; _metadata of the synthetic file: {written}", flush=True)


def run(out, names):
    import run_synth as R
    R.HERE, R.OUT = out, out / "results"
    assert R.CONFIG["pval_thr"] == R.PVAL_THR
    R.driada.set_parallel_backend("loky")
    for name in names:
        if (R.OUT / f"{name}.pkl").exists():
            print(f"[SKIP] run {name}", flush=True)
            continue
        t0 = time.time()
        R.ria.process_single_experiment(out / "synth_npz" / f"{name}_aligned.npz", R.CONFIG,
                                        output_dir=R.OUT, plot=False, use_filters=True,
                                        zone_rule=R.P.ZONE_RULE, zone_masks_dir=R.P.ZONE_MASKS)
        print(f"[TIME] {name}: {time.time() - t0:.0f} s", flush=True)


def score(out):
    import score as S
    S.HERE, S.OUT = out, out / "tables"
    S.main()


def main():
    args = sys.argv[1:]
    out = S2_DIR / "s2"
    if "--out" in args:
        i = args.index("--out")
        out = Path(args[i + 1])
        del args[i:i + 2]
    step, names = args[0], args[1:]
    import make_synth as M
    M.SESSIONS = SEED_ORDER
    widen_field_search(M)
    names = names or list(SESSIONS16)
    assert set(names) <= set(SESSIONS16), names
    out.mkdir(parents=True, exist_ok=True)
    if step in ("make", "all"):
        make(out, names)
    if step in ("run", "all"):
        run(out, names)
    if step in ("score", "all"):
        score(out)


if __name__ == "__main__":
    main()
