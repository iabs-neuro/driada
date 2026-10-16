"""Turn the NWB files of the FOF Dandiset back into the ``synchronized`` layout.

The figure and analysis scripts of the paper read one
``synchronized/<session>_aligned.npz`` per session: the array ``Calcium``
(neurons x frames) and one array per behavioural variable. This script rebuilds
those files from a downloaded copy of the Dandiset, whatever its folder
structure. Only ``h5py`` and ``numpy`` are needed.

Run:  python publications/elife/dandi/nwb_to_synchronized.py <folder with the NWB files> [--out <folder>]
      the default output folder is the ``synchronized`` folder of the data
      root (see ``elife_paths.py``)
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import elife_paths

# The order of the arrays in the original files. The loader builds the features
# in the order it meets them, so the order is part of what makes a rerun of the
# analysis reproduce the published numbers.
VARIABLE_ORDER = ["x", "y", "speed", "bodydirection", "headdirection", "rest", "walk",
                  "locomotion", "freezing", "rear", "walls", "centermiddle", "centertrue",
                  "center"]


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_session(path):
    """Read one NWB file of the dataset.

    Returns
    -------
    str
        Session name, e.g. ``FOF_F28_1D``.
    dict
        ``Calcium`` (neurons x frames) and the behavioural variables, in the
        order of the original synchronized files.
    """
    with h5py.File(path, "r") as f:
        session = _text(f["general/session_id"][()])
        arrays = {"Calcium": np.ascontiguousarray(f["processing/ophys/Fluorescence/Calcium/data"][:].T)}

        behavior = f["processing/behavior"]
        found = {}
        position = behavior["Position/SpatialSeries/data"][:]
        found["x"] = np.ascontiguousarray(position[:, 0])
        found["y"] = np.ascontiguousarray(position[:, 1])
        for name in behavior:
            if name != "Position":
                found[name] = behavior[name]["data"][:]

    for name in VARIABLE_ORDER + sorted(set(found) - set(VARIABLE_ORDER)):
        if name in found:
            arrays[name] = found[name]
    return session, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", help="folder with the NWB files (searched recursively)")
    parser.add_argument("--out", default=elife_paths.SYNC_DIR, help="folder for the *_aligned.npz files")
    parser.add_argument("--overwrite", action="store_true", help="replace npz files that already exist")
    parser.add_argument("sessions", nargs="*", help="convert only these sessions, e.g. FOF_F28_1D")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.source, "**", "*.nwb"), recursive=True))
    if not paths:
        print(f"[FAILED] no NWB files under {args.source}")
        return 1
    os.makedirs(args.out, exist_ok=True)

    done, wanted = set(), set(args.sessions)
    for path in paths:
        if wanted:
            # Reading the name alone avoids loading sessions that are not asked for.
            with h5py.File(path, "r") as f:
                if _text(f["general/session_id"][()]) not in wanted:
                    continue
        session, arrays = read_session(path)
        target = os.path.join(args.out, f"{session}_aligned.npz")
        if os.path.exists(target) and not args.overwrite:
            print(f"[FAILED] {target} exists; pass --overwrite to replace it")
            return 1
        np.savez(target, **arrays)
        done.add(session)
        print(f"[OK] {session}: {arrays['Calcium'].shape[0]} neurons x {arrays['Calcium'].shape[1]} frames, "
              f"{len(arrays) - 1} variables -> {target}", flush=True)

    if wanted - done:
        print(f"[FAILED] sessions not found among the NWB files: {' '.join(sorted(wanted - done))}")
        return 1
    print(f"[INFO] {len(done)} sessions written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
