"""Run one half of the calibration (two halves run as two background jobs).

The 64 NOF sessions of the shift null, 16 sessions of the cross null (one per
mouse, days rotated) and 8 synthetic experiments are split into two parts of
about equal number of neurons.

Usage: python run_part.py A|B
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import calib_null as cn  # noqa: E402  (checks the driada snapshot on import)


def n_cells(session):
    with np.load(cn.DATA / f"{session}_aligned.npz", allow_pickle=True) as d:
        return d["Calcium"].shape[0]


def plan():
    shift = [f"NOF_{m}_{d}D" for m in cn.MICE for d in range(1, 5)]
    cross = [f"NOF_{m}_{k % 4 + 1}D" for k, m in enumerate(cn.MICE)]
    tasks = [("shift", s, n_cells(s)) for s in shift] + [("cross", s, n_cells(s)) for s in cross]
    tasks += [("synth", k, 300) for k in range(8)]
    parts = {"A": [], "B": []}
    load = {"A": 0, "B": 0}
    for t in sorted(tasks, key=lambda t: -t[2]):
        p = min(load, key=load.get)
        parts[p].append(t)
        load[p] += t[2]
    return parts, load


def main():
    part = sys.argv[1]
    parts, load = plan()
    out = cn.P.CALIBRATION_DIR / "results"
    out.mkdir(exist_ok=True)
    print(f"part {part}: {len(parts[part])} tasks, {load[part]} neurons", flush=True)
    for null, unit, _ in parts[part]:
        if null == "synth":
            cn.run_synth(unit, out)
        else:
            cn.run_real(null, unit, out)
    print(f"[OK] part {part} done", flush=True)


if __name__ == "__main__":
    main()
