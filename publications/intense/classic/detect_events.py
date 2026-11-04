# -*- coding: utf-8 -*-
"""Calcium events of NOF sessions under a chosen clock, cached for the classic place-cell test.

The session is built exactly as in classic_multigrid_orig.py (the Python port of the
classic method): DRIADA 1.2.2 from the snapshot of the paper, data of the v1 aligned
files, wavelet event reconstruction. Only the frame rate handed to the library differs:

* ``old``  - 20 frames/s for every session (what DRIADA 1.2.2 assumed, because the
  aligned files carry no rate);
* ``true`` - the rate of the session from fps_by_session.csv of stage 0 (or, when that
  table is absent, from the CaImAn table of the frame-rate check, H36 = 29.8).

Saved per session and clock (pickle-free npz): positions, speed, the wavelet events of
every neuron (start and end frame as returned by the detector), the runs of the binary
event train ``neuron.sp`` and the calcium shuffle exclusion of the library (frames at
each end of the circular shift range).

usage: python detect_events.py --sessions NOF_H01_1D NOF_H03_4D --clocks old true
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# the detector draws a progress bar per session; in a background job it only fills the log
os.environ.setdefault("TQDM_DISABLE", "1")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import intense_paths as P  # noqa: E402
import paths_v11 as V  # noqa: E402

CODE_V122 = P.DRIADA_SRC.parent
NOF_V1 = V.get("nof_v1")
FPS_TABLE = V.get("fps_table")
FPS_FALLBACK = V.get("bearmind_fr")
OLD_FPS = 20.0
OUT = V.get("classic_dir") / "events"


def session_fps():
    """Frame rate of every session and where it came from."""
    if FPS_TABLE.exists():
        with open(FPS_TABLE, newline="") as f:
            return {r["session"]: (float(r["fps"]), r["source"]) for r in csv.DictReader(f)}, str(FPS_TABLE)
    table = {}
    with open(FPS_FALLBACK, newline="") as f:
        for name, fr in csv.reader(f):
            table[name] = (float(fr), "CaImAn fr")
    for day in range(1, 5):
        # no CaImAn files for H36; 17,858-17,860 frames in 10 min
        table[f"NOF_H36_{day}D"] = (29.8, "frame count")
    return table, str(FPS_FALLBACK)


def check_driada():
    """Put the 1.2.2 snapshot first on sys.path and make sure it is the one imported.

    Called in every worker process too: the development tree of DRIADA is installed in
    this Python and would otherwise be imported by a joblib worker."""
    for p in (P.TOOLS_DIR, P.DRIADA_SRC):
        if str(p) in sys.path:
            sys.path.remove(str(p))
        sys.path.insert(0, str(p))
    import driada
    where = Path(driada.__file__).resolve()
    if not where.is_relative_to(P.DRIADA_SRC) or driada.__version__ != "1.2.2":
        raise RuntimeError(f"driada {driada.__version__} from {where}, need 1.2.2 from {CODE_V122}")
    return driada.__version__


def build(path, fps):
    from selectivity_dynamics import get_experiment_config
    from driada.experiment.exp_build import load_exp_from_aligned_data
    from driada.utils.naming import parse_iabs_filename
    parsed = parse_iabs_filename(path.name)
    data = dict(np.load(path, allow_pickle=True))
    data.pop("_metadata", None)
    data.pop("_sync_info", None)
    ep = {k: parsed[k] for k in ("track", "animal_id", "session")}
    return load_exp_from_aligned_data(
        data_source="IABS", exp_params=ep, data=data,
        feature_types=get_experiment_config("NOF").get("feature_types"),
        static_features={"fps": float(fps)}, aggregate_features={("x", "y"): "place"},
        verbose=False, reconstruct_spikes="wavelet")


def runs_of(sp):
    """Start (inclusive) and end (exclusive) frames of the runs of sp > 0."""
    on = np.flatnonzero(np.diff(np.r_[0, (sp > 0).astype(np.int8), 0]) != 0)
    return on[0::2], on[1::2]


def process(session, clock, fps_table):
    out = OUT / f"events_{session}_{clock}.npz"
    if out.exists():
        return f"skip {out.name}"
    fps = OLD_FPS if clock == "old" else fps_table[session][0]
    t0 = time.time()
    check_driada()
    exp = build(NOF_V1 / f"{session}_aligned.npz", fps)
    if abs(float(exp.fps) - fps) > 1e-9:
        raise RuntimeError(f"{session}: exp.fps={exp.fps}, asked {fps}")
    meta = exp._reconstruction_metadata
    ev_cell, ev_st, ev_en, run_cell, run_st, run_en, excl = [], [], [], [], [], [], []
    for c in range(exp.n_cells):
        st = np.asarray(meta["start_events"][c], int)
        en = np.asarray(meta["end_events"][c], int)
        ev_cell.append(np.full(len(st), c)); ev_st.append(st); ev_en.append(en)
        rs, re_ = runs_of(np.asarray(exp.neurons[c].sp.data, float))
        run_cell.append(np.full(len(rs), c)); run_st.append(rs); run_en.append(re_)
        mask = np.asarray(exp.neurons[c].ca.shuffle_mask, bool)
        excl.append(int(np.argmax(mask)) if mask.any() else len(mask))
    cat = lambda a: np.concatenate(a) if a else np.zeros(0, int)
    np.savez_compressed(
        out, session=session, clock=clock, fps=float(exp.fps), n_cells=exp.n_cells,
        n_frames=exp.n_frames,
        x=np.asarray(exp.dynamic_features["x"].data, float),
        y=np.asarray(exp.dynamic_features["y"].data, float),
        speed=np.asarray(exp.dynamic_features["speed"].data, float),
        ev_cell=cat(ev_cell), ev_st=cat(ev_st), ev_en=cat(ev_en),
        run_cell=cat(run_cell), run_st=cat(run_st), run_en=cat(run_en),
        shuffle_excl=np.asarray(excl, int))
    return f"{session} {clock}: fps={exp.fps:.2f} cells={exp.n_cells} frames={exp.n_frames} " \
           f"events={len(cat(ev_st))} runs={len(cat(run_st))} ({time.time() - t0:.0f} s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--clocks", nargs="+", default=["old", "true"], choices=["old", "true"])
    ap.add_argument("--n_jobs", type=int, default=1)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"driada {check_driada()} from {CODE_V122}", flush=True)
    fps_table, src = session_fps()
    print(f"frame rates: {src}", flush=True)
    tasks = [(s, c) for s in a.sessions for c in a.clocks]
    if a.n_jobs == 1:
        for s, c in tasks:
            print(process(s, c, fps_table), flush=True)
    else:
        from joblib import Parallel, delayed
        for msg in Parallel(n_jobs=a.n_jobs, backend="loky", return_as="generator")(
                delayed(process)(s, c, fps_table) for s, c in tasks):
            print(msg, flush=True)
    print("[OK] events cached", flush=True)


if __name__ == "__main__":
    main()
