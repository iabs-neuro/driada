# -*- coding: utf-8 -*-
"""INTENSE and the forward-selection GLM on every session of the MEC cohort (DRIADA 1.2.2).

Each session file of [mec] sessions_dir (decode_cohort_primary.py) is cleaned as in load_session
and analysed at the frame rate of the session ([mec] fps_table, column fps_true), with seed 42.
The coordinates x and y are declared linear: left to the type detector, coordinates lying in
0-370 are taken for angles with a period of 360 in most sessions, and the type-based
representation would then test place as it is instead of its quadratic representation.

Two passes per part of the cohort:
  intense  INTENSE with type-based representations (two-stage, 100 / 10,000 shuffles,
           p < 0.001 without multiple-comparison correction, downsampling x2), with the delay
           search ('on', the setting of the Supplementary Information) and without it ('off'),
           every cell and feature. The statistics of every pair of the three features of
           Supplementary Table 5 are kept, so that the information threshold of the main text
           (excess MI over the mean of the second-stage shuffles above 0.04 bits) can be
           applied afterwards (the *_mi04 sets).
  glm      the forward-selection GLM (analyze_cohort.glm_select) on the rank-Gaussian transform
           of the fluorescence of every --glm-stride-th cell, with blocks place (7 x 7 bins),
           head direction (24 bins) and speed (10 quantile bins); the seed of a cell is its index.
n_jobs = 1: the shuffles are drawn in advance for the whole grid of pairs, so the result does not
depend on the number of worker processes.

usage: python -X utf8 -B mec/run_xyfix.py --shard k/N --pass intense|glm [--glm-stride S]
                                          [--stop-at ISO] [--sessions STEM ...] [--out DIR]
       default --out: <[mec] recompute_dir>/parts; finished sessions of the part are skipped, so a
       part is restarted with the same command
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys  # noqa: E402

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, str(PAPER))
import intense_paths as P  # noqa: E402

# The analysis is only meaningful on the version the paper names; use_driada checks it.
driada = P.use_driada()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402

sys.path.insert(0, str(PAPER / "glm"))  # glm_variants.py
sys.path.insert(0, str(HERE))
from analyze_cohort import bin_hd, bin_position, bin_speed, clean_series, glm_select  # noqa: E402
from glm_variants import rank_gaussian  # noqa: E402
from driada.experiment.exp_build import load_exp_from_aligned_data  # noqa: E402
from driada.intense.representations import get_representation_sources  # noqa: E402

ROOT = P.get("mec", "recompute_dir")
SESS = P.get("mec", "sessions_dir")
SEED = 42
FPS = pd.read_csv(P.get("mec", "fps_table")).set_index("session")["fps_true"]
CANON = ["place", "head_direction", "speed"]
NAME_MAP = {"head_angle": "head_direction", "head_angle_2d": "head_direction"}
# x and y are declared linear (see the module docstring).
FEATURE_TYPES = {"head_angle": "circular", "speed": "continuous", "x": "linear", "y": "linear"}
MI_EXCESS_THR = 0.04
ARMS = {"on": True, "off": False}
PAIR_COLS = ["session", "arm", "cell", "feature", "tested", "me", "me_null", "me_excess", "pval",
             "opt_delay", "signal_ratio", "stage1", "stage2"]
# Other computations may share the machine: a session is not started while less memory is free.
MIN_FREE_GB = 2.5


def wait_for_memory():
    waited = 0
    while psutil.virtual_memory().available / 2**30 < MIN_FREE_GB:
        if waited % 300 == 0:
            print(f"    waiting for memory: {psutil.virtual_memory().available / 2**30:.1f} GB free",
                  flush=True)
        time.sleep(10)
        waited += 10
    return waited


def load_session(npz):
    """Data cleaning of the cohort analysis; returns (arrays, None) or (None, reason)."""
    d = np.load(npz)
    cal_raw = d["calcium"].astype(float)
    if not np.isfinite(cal_raw).any():
        return None, "all-NaN calcium"
    cal = np.nan_to_num(cal_raw, nan=0.0, posinf=0.0, neginf=0.0)
    if cal.shape[0] < 5 or cal.shape[1] < 200:
        return None, "too small"
    x, y = clean_series(d["x"]), clean_series(d["y"])
    sp, hd = clean_series(d["speed"]), clean_series(d["head"])
    if min(np.ptp(x), np.ptp(y), np.ptp(sp), np.ptp(hd)) < 1e-6:
        return None, "constant behaviour"
    return (cal, x, y, sp, hd), None


def build_exp(cal, x, y, sp, hd, stem):
    """DRIADA experiment of a session: place aggregated from x and y, fps of the session, seed 42."""
    return load_exp_from_aligned_data(
        "MEC", {"track": "of", "animal_id": "topo", "session": stem},
        {"calcium": cal, "x": x, "y": y, "speed": sp, "head_angle": hd},
        feature_types=FEATURE_TYPES,
        aggregate_features={("x", "y"): "place"},
        static_features={"fps": float(FPS[stem]), "t_rise_sec": 0.25, "t_off_sec": 2.0},
        reconstruct_spikes=None, verbose=False, seed=SEED)


def canon(f, src):
    f = src.get(f, f)
    return NAME_MAP.get(f, f)


def run_arm(exp, delays):
    """Significant feature sets of every cell for one arm, with the statistics of the pairs."""
    stats, sig, info, _, _ = driada.compute_cell_feat_significance(
        exp, mode="two_stage", n_shuffles_stage1=100, n_shuffles_stage2=10000,
        pval_thr=0.001, multicomp_correction=None, ds=2, find_optimal_delays=delays,
        verbose=False, representation="by_type", n_jobs=1)
    src = get_representation_sources(exp)
    # the sets as the experiment reports them
    sets_lib = {int(ni): {canon(f, src) for f in fs} & set(CANON)
                for ni, fs in exp.get_significant_neurons().items()}
    rows = []
    for cell, fs in stats.items():
        for feat, st in fs.items():
            c = canon(feat, src)
            if c not in CANON:
                continue
            sg = sig[cell][feat]
            rows.append(dict(cell=int(cell), feature=c, tested=feat, me=st.get("me"),
                             me_null=st.get("me_null"), me_excess=st.get("me_excess"),
                             pval=st.get("pval"), opt_delay=st.get("opt_delay"),
                             signal_ratio=st.get("signal_ratio"),
                             stage1=bool(sg.get("stage1", False)), stage2=bool(sg.get("stage2", False))))
    pairs = pd.DataFrame(rows)
    sig_pairs = pairs[pairs.stage2]
    sets = {c: set(g.feature) for c, g in sig_pairs.groupby("cell")}
    if {k: v for k, v in sets_lib.items() if v} != {k: v for k, v in sets.items() if v}:
        raise RuntimeError("significant sets from the experiment and from the statistics differ")
    mi04 = sig_pairs[sig_pairs.me_excess.astype(float) > MI_EXCESS_THR]
    sets_mi04 = {c: set(g.feature) for c, g in mi04.groupby("cell")}
    tested = sorted(set(pairs.tested))
    return pairs, sets, sets_mi04, tested, int((info["optimal_delays"] != 0).sum())


def intense_session(npz):
    arrays, why = load_session(npz)
    if arrays is None:
        return None, why
    cal, x, y, sp, hd = arrays
    out = {"n_cells": cal.shape[0]}
    for arm, delays in ARMS.items():
        t0 = time.time()
        exp = build_exp(cal, x, y, sp, hd, npz.stem)
        if abs(float(exp.fps) - float(FPS[npz.stem])) > 1e-9:
            raise RuntimeError(f"exp.fps {exp.fps} != fps_true {FPS[npz.stem]}")
        comps = exp.dynamic_features["place"].ts_list
        if any(c.type_info is not None and c.type_info.is_circular for c in comps):
            raise RuntimeError("a component of place is circular although x and y are declared linear")
        pairs, sets, sets_mi04, tested, nz = run_arm(exp, delays)
        pairs.insert(0, "arm", arm)
        out[arm] = dict(pairs=pairs, sets=sets, sets_mi04=sets_mi04, tested=tested,
                        nonzero_delays=nz, seconds=time.time() - t0)
        del exp
    return out, None


def glm_session(npz, stride):
    arrays, why = load_session(npz)
    if arrays is None:
        return None, why
    cal, x, y, sp, hd = arrays
    blocks = {"place": bin_position(x, y), "head_direction": bin_hd(hd), "speed": bin_speed(sp)}
    return {ni: sorted(glm_select(blocks, rank_gaussian(cal[ni]), seed=ni))
            for ni in range(0, cal.shape[0], stride)}, None


def part_files(out, tag, kind):
    return out / f"{kind}_done_{tag}.csv", out / f"{kind}_sets_{tag}.jsonl", out / f"{kind}_pairs_{tag}.csv"


def restore(done_path, sets_path, pairs_path):
    """Keep only the rows of finished sessions (a session interrupted mid-way may have left some)."""
    done = set(pd.read_csv(done_path)["session"].astype(str)) if done_path.exists() else set()
    if pairs_path.exists():
        p = pd.read_csv(pairs_path)
        p[p.session.astype(str).isin(done)].to_csv(pairs_path, index=False)
    if sets_path.exists():
        keep = [ln for ln in sets_path.read_text().splitlines() if json.loads(ln)["session"] in done]
        sets_path.write_text("".join(ln + "\n" for ln in keep))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--pass", dest="kind", choices=["intense", "glm"], required=True)
    ap.add_argument("--glm-stride", type=int, default=1)
    ap.add_argument("--stop-at", default=None, help="local time (ISO) after which no session is started")
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "parts")
    a = ap.parse_args()
    k, n = (int(v) for v in a.shard.split("/"))
    a.out.mkdir(parents=True, exist_ok=True)
    tag = f"{k}of{n}"
    done_path, sets_path, pairs_path = part_files(a.out, tag, a.kind)
    files = sorted(SESS.glob("*.npz"))[k::n]
    if a.sessions:
        files = [f for f in files if f.stem in set(a.sessions)]
    done = restore(done_path, sets_path, pairs_path)
    todo = [f for f in files if f.stem not in done]
    stop_at = datetime.fromisoformat(a.stop_at) if a.stop_at else None
    print(f"driada {driada.__version__} from {driada.__file__}", flush=True)
    print(f"part {k}/{n}, pass {a.kind}: {len(files)} sessions, {len(done)} done, {len(todo)} to run; "
          f"feature_types {FEATURE_TYPES}; glm stride {a.glm_stride}; stop at {stop_at}", flush=True)
    t_all = time.time()
    for i, f in enumerate(todo):
        if stop_at is not None and datetime.now() >= stop_at:
            print(f"[STOPPED] {stop_at} reached; {len(todo) - i} sessions of the part not run", flush=True)
            break
        waited = wait_for_memory()
        t0 = time.time()
        try:
            if a.kind == "intense":
                res, why = intense_session(f)
            else:
                res, why = glm_session(f, a.glm_stride)
        except Exception:
            print(f"  [{i}] {f.stem} FAILED:\n{traceback.format_exc()}", flush=True)
            continue
        rec = {"session": f.stem, "excluded": why or "", "seconds": round(time.time() - t0, 1),
               "waited_for_memory": waited}
        if res is not None and a.kind == "intense":
            with open(sets_path, "a") as fs:
                for cell in range(res["n_cells"]):
                    row = {"session": f.stem, "cell": cell}
                    for arm in ARMS:
                        row[arm] = sorted(res[arm]["sets"].get(cell, set()))
                        row[f"{arm}_mi04"] = sorted(res[arm]["sets_mi04"].get(cell, set()))
                    fs.write(json.dumps(row) + "\n")
            pd.concat([res[arm]["pairs"].assign(session=f.stem) for arm in ARMS])[PAIR_COLS].to_csv(
                pairs_path, mode="a", header=not pairs_path.exists(), index=False)
            rec.update(n_cells=res["n_cells"], fps=float(FPS[f.stem]),
                       tested_on=" ".join(res["on"]["tested"]), tested_off=" ".join(res["off"]["tested"]),
                       **{f"{arm}_{c}": sum(c in s for s in res[arm]["sets"].values())
                          for arm in ARMS for c in CANON},
                       **{f"{arm}_mi04_{c}": sum(c in s for s in res[arm]["sets_mi04"].values())
                          for arm in ARMS for c in CANON},
                       **{f"seconds_{arm}": round(res[arm]["seconds"], 1) for arm in ARMS},
                       **{f"nonzero_delays_{arm}": res[arm]["nonzero_delays"] for arm in ARMS})
        elif res is not None:
            with open(sets_path, "a") as fs:
                for cell, g in res.items():
                    fs.write(json.dumps({"session": f.stem, "cell": cell, "glm": g}) + "\n")
            rec.update(cells_checked=len(res), stride=a.glm_stride)
        pd.DataFrame([rec]).to_csv(done_path, mode="a", header=not done_path.exists(), index=False)
        if res is None:
            print(f"  [{i}] {f.stem} EXCLUDED ({why})", flush=True)
            continue
        if a.kind == "intense":
            msg = " ".join(f"{arm}={rec[f'{arm}_place']}/{rec[f'{arm}_head_direction']}/{rec[f'{arm}_speed']} "
                           f"{arm}_mi04={rec[f'{arm}_mi04_place']}/{rec[f'{arm}_mi04_head_direction']}/"
                           f"{rec[f'{arm}_mi04_speed']} ({rec[f'seconds_{arm}']:.0f} s)" for arm in ARMS)
            print(f"  [{i}] {f.stem} n={res['n_cells']} at {rec['fps']:.3f} fps, tested {rec['tested_on']}: "
                  f"{msg}; total {time.time() - t_all:.0f} s", flush=True)
        else:
            print(f"  [{i}] {f.stem} GLM on {len(res)} cells ({rec['seconds']:.0f} s); "
                  f"total {time.time() - t_all:.0f} s", flush=True)
    print(f"[OK] part {k}/{n} pass {a.kind} finished", flush=True)


if __name__ == "__main__":
    main()
