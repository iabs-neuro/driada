"""The NOF sessions of Dandiset 002047 as the npz files the scripts of this folder read.

DANDI holds one NWB file per session; the analysis reads one npz file per session
(``NOF_<mouse>_<day>_aligned.npz`` in [v11] nof_v1) and one file of zone masks per
session (``<session>_zones_cm.npz`` in [data] zone_masks/NOF). This script writes
both from the NWB files, without changing any value:

* ``Calcium`` (neurons x frames) is ``processing/ophys/Fluorescence/Calcium`` transposed
  back; ``x``, ``y`` are the columns of ``processing/behavior/Position/SpatialSeries``;
  every other behavioral variable is ``processing/behavior/<name>``. The arrays keep
  their dtype, and the members are written in the order of the files the paper was
  computed from (``SESSION_KEYS``): DRIADA builds the features in that order, and the
  seeded tie-breaking noise and shuffles follow it.
* the zone masks: ``masks`` and ``zone_names`` from ``processing/behavior/ZoneMasks``,
  ``grid_step_cm`` from the resolution of the masks, ``corners_cm`` and
  ``object_centers_cm`` from ``processing/behavior/ArenaLandmarks``. DRIADA and
  synthetic/make_synth.py read ``zone_names``, ``masks`` and ``grid_step_cm``.

The frame rate of each session is not taken from NWB: compute/make_v2_data.py adds the
rate of fps_by_session.csv to every file. ``--check DIR`` compares the written files with
files of the same names in DIR (dtype, shape and values of every array).

usage: python compute/data_from_dandi.py <dandiset folder> [--check DIR_SESSIONS DIR_MASKS]
       (the dandiset folder holds sub-<mouse>/sub-<mouse>_ses-NOF-<mouse>-<day>_*.nwb,
       e.g. as written by ``dandi download DANDI:002047``)
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402
import paths_v11 as V  # noqa: E402

SESSION_KEYS = ["Calcium", "x", "y", "speed", "bodydirection", "headdirection", "rest", "walk",
                "locomotion", "freezing", "rear", "corners", "walls", "center", "object1",
                "object2", "object3", "object4", "objects"]
BEHAVIOR = "processing/behavior"


def read_session(path):
    """Session name, arrays of the session file and arrays of the zone-mask file of one NWB file.

    Parameters
    ----------
    path : Path
        NWB file of one session.

    Returns
    -------
    name : str
        Session name, e.g. ``'NOF_H01_1D'``.
    session : dict
        ``{key: array}`` in the order of ``SESSION_KEYS``.
    zones : dict
        ``masks``, ``zone_names``, ``grid_step_cm``, ``corners_cm``, ``object_centers_cm``,
        ``session``.
    """
    with h5py.File(path, "r") as h:
        name = h["general/session_id"][()].decode()
        beh = h[BEHAVIOR]
        xy = beh["Position/SpatialSeries/data"][()]
        session = {"Calcium": h["processing/ophys/Fluorescence/Calcium/data"][()].T.copy(),
                   "x": xy[:, 0].copy(), "y": xy[:, 1].copy()}
        for key in SESSION_KEYS[3:]:
            session[key] = beh[f"{key}/data"][()]
        extra = set(beh) - set(SESSION_KEYS) - {"Position", "ZoneMasks", "ArenaLandmarks"}
        if extra:
            raise SystemExit(f"{path}: behavioral variables {sorted(extra)} are not in the analysed files")
        zm = beh["ZoneMasks"]
        names = sorted(zm)
        steps = {1.0 / float(zm[n].attrs["resolution"]) for n in names}
        if len(steps) != 1:
            raise SystemExit(f"{path}: zone masks with different resolutions {steps}")
        lm = beh["ArenaLandmarks"]
        labels = [s.decode() for s in lm["landmark"][()]]
        pts = {lab: (x, y) for lab, x, y in zip(labels, lm["x_cm"][()], lm["y_cm"][()])}
        zones = {"masks": np.stack([zm[n][()] for n in names]),
                 "zone_names": np.array(names),
                 "grid_step_cm": np.float64(steps.pop()),
                 "corners_cm": np.array([pts[f"corner{k}"] for k in range(1, 5)], dtype=float),
                 "object_centers_cm": np.array([pts[f"object{k}"] for k in range(1, 5)], dtype=float),
                 "session": np.array(name)}
    return name, session, zones


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def check(name, session, zones, dir_sessions, dir_masks):
    """Differences between the arrays read from NWB and the files of the same session in two folders."""
    bad = []
    with np.load(Path(dir_sessions) / f"{name}_aligned.npz", allow_pickle=True) as ref:
        keys = [k for k in ref.files if k not in ("_sync_info", "_metadata")]
        if keys != SESSION_KEYS:
            bad.append(f"members {keys}")
        bad += [k for k in SESSION_KEYS if k in ref.files and not same(session[k], ref[k])]
    with np.load(Path(dir_masks) / f"{name}_zones_cm.npz") as ref:
        by_name = dict(zip([str(s) for s in ref["zone_names"]], ref["masks"]))
        if sorted(by_name) != list(zones["zone_names"]):
            bad.append("zone names")
        bad += [f"mask {n}" for n, m in zip(zones["zone_names"], zones["masks"]) if not same(m, by_name.get(str(n)))]
        bad += [k for k in ("grid_step_cm", "corners_cm", "object_centers_cm") if not same(zones[k], ref[k])]
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dandiset", type=Path)
    ap.add_argument("--check", nargs=2, metavar=("DIR_SESSIONS", "DIR_MASKS"))
    args = ap.parse_args()
    files = sorted(args.dandiset.glob("sub-*/*.nwb"))
    if not files:
        raise SystemExit(f"no NWB files under {args.dandiset}")
    out_sessions = V.get("nof_v1")
    out_masks = P.ZONE_MASKS / "NOF"
    if not args.check:
        out_sessions.mkdir(parents=True, exist_ok=True)
        out_masks.mkdir(parents=True, exist_ok=True)
    problems = []
    for f in files:
        name, session, zones = read_session(f)
        if args.check:
            bad = check(name, session, zones, *args.check)
            problems += [f"{name}: {b}" for b in bad]
            print(name, "[OK]" if not bad else f"[FAILED] {bad}", flush=True)
            continue
        for path, arrays in ((out_sessions / f"{name}_aligned.npz", session),
                             (out_masks / f"{name}_zones_cm.npz", zones)):
            if path.exists():
                raise SystemExit(f"{path} exists; this script does not overwrite")
            np.savez(path, **arrays)
        print(name, "->", out_sessions, flush=True)
    if problems:
        raise SystemExit(f"[FAILED] {len(problems)} differences")
    print(f"[OK] {len(files)} sessions")


if __name__ == "__main__":
    main()
