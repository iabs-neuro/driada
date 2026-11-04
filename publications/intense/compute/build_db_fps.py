"""Cross-session databases with the delay category at the frame rate of each session.

compute/build_db.py of the paper is run unchanged, except for one function of
tools/neuron_database of DRIADA 1.2.2: _add_delay_sign(data, fps) puts a pair
into the delay category -1, 0 or +1 by comparing its opt_delay (frames) with
0.75 s x fps, and loaders.py calls it with fps=20 for every session. For the
duration of the run it is replaced by the same rule with the rate of the pair's
session, taken from _metadata of the data files of the configuration
([data] nof_aligned). The databases keep their columns; only delay_sign can
differ.

With the 5-frame delay grid the 0.75 s boundary falls on a grid point only at
20 frames/s: at 19.76 frames/s a delay of 15 frames (0.76 s) is outside it, at
29.79 frames/s the boundary lies between 20 and 25 frames. The script prints
how many pairs change category against the rule at 20 frames/s, by rate group
and by delay, so that the effect of the boundary is seen, not assumed.

--round-boundary puts the boundary at round(0.75 s x fps) whole frames instead
(15 frames at 19.76, 22 at 29.79): a delay of 15 frames at about 20 frames/s
stays near zero, as before, while the 30-frame/s sessions get the 0.75 s
boundary within one frame. Which rule the paper uses is the authors' decision.

usage: python compute/build_db_fps.py --round-boundary
"""
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
import intense_paths as P  # noqa: E402

P.use_driada()
import neuron_database.loaders as L  # noqa: E402

BOUNDARY_S = 0.75  # the boundary of _add_delay_sign, in seconds
ROUND = "--round-boundary" in sys.argv[1:]


def session_rates():
    """{(mouse, day): fps} from _metadata of every synchronized file."""
    rates = {}
    for p in sorted(P.NOF_ALIGNED.glob("NOF_*_aligned.npz")):
        _, mouse, day = p.name[:-len("_aligned.npz")].split("_")
        with np.load(p, allow_pickle=True) as d:
            if "_metadata" not in d.files or "fps" not in d["_metadata"].item():
                raise SystemExit(f"{p} has no _metadata['fps']: this script needs the v2 data")
            rates[(mouse, day)] = float(d["_metadata"].item()["fps"])
    return rates


RATES = session_rates()


def add_delay_sign_session_rate(data, fps=None):
    """_add_delay_sign of loaders.py with the rate of each row's session (``fps`` is ignored)."""
    keys = list(zip(data["mouse"], data["session"]))
    missing = sorted({k for k in keys if k not in RATES})
    if missing:
        raise KeyError(f"no frame rate for {missing[:5]}")
    boundary = np.array([RATES[k] for k in keys]) * BOUNDARY_S
    if ROUND:
        boundary = np.round(boundary)
    delay = data["opt_delay"].to_numpy()
    sign = np.zeros(len(data), dtype=int)
    sign[delay < -boundary] = -1
    sign[delay > boundary] = 1
    data["delay_sign"] = sign


def report(path):
    """Pairs whose category differs from the rule at 20 frames/s."""
    d = pd.read_csv(path)
    fps = np.array([RATES[(m, s)] for m, s in zip(d["mouse"], d["session"])])
    at20 = np.where(d.opt_delay < -15, -1, np.where(d.opt_delay > 15, 1, 0))
    changed = d.delay_sign.to_numpy() != at20
    grp = np.where(fps < 25, "~20 fps", "~30 fps")
    print(f"  {path.name}: {len(d)} rows, {changed.sum()} change category against 20 frames/s")
    for g in ("~20 fps", "~30 fps"):
        sel = changed & (grp == g)
        by_delay = pd.Series(d.opt_delay[sel]).value_counts().sort_index()
        print(f"    {g}: {sel.sum()} of {(grp == g).sum()}; by delay (frames): "
              + (", ".join(f"{int(k)}: {v}" for k, v in by_delay.items()) or "-"))


def main():
    L._add_delay_sign = add_delay_sign_session_rate
    print(f"delay category at {BOUNDARY_S} s x the rate of each session"
          f"{', rounded to whole frames' if ROUND else ''} "
          f"({len(RATES)} sessions, {min(RATES.values())}-{max(RATES.values())} frames/s)", flush=True)
    sys.argv = [str(PAPER / "compute" / "build_db.py")]
    runpy.run_path(str(PAPER / "compute" / "build_db.py"), run_name="__main__")
    for name in ("NOF_pre_disent.csv", "NOF_post_disent.csv"):
        report(P.DB_DIR / name)


if __name__ == "__main__":
    main()
