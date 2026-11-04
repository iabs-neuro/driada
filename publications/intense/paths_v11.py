"""Locations added in version 11 of the manuscript: the [v11] section of the configuration.

They come from the configuration that intense_paths.py has read (config.toml, or the file
named by INTENSE_PAPER_CONFIG) and are resolved as in intense_paths.py. A script imports this module after
intense_paths and builds no path of its own.
"""
import csv
from pathlib import Path

import numpy as np

import intense_paths as P

V11 = P.CFG.get("v11", {})


def _path(key):
    value = V11.get(key)
    if not value:
        raise KeyError(f"[v11] {key} is not set in {P.CONFIG_FILE}")
    return P._path(value)


def get(key):
    """A [v11] location, resolved; KeyError names the configuration file if it is missing."""
    return _path(key)


# "aligned": conditional MI of disentanglement at the optimal delays of the pair (the paper);
# "asis": as DRIADA 1.2.2 computes it.
CMI_DELAYS = V11.get("cmi_delays", "asis")
if CMI_DELAYS not in ("aligned", "asis"):
    raise ValueError(f"[v11] cmi_delays = {CMI_DELAYS!r}; expected 'aligned' or 'asis'")


def session_name(path):
    """'NOF_H01_1D_aligned.npz' -> 'H01_1D'."""
    return Path(path).name[len("NOF_"):-len("_aligned.npz")]


def session_fps(folder=None):
    """{'H01_1D': fps} from _metadata['fps'] of every synchronized file of [data] nof_aligned."""
    rates = {}
    for p in sorted(Path(folder or P.NOF_ALIGNED).glob("NOF_*_aligned.npz")):
        with np.load(p, allow_pickle=True) as d:
            if "_metadata" not in d.files or "fps" not in d["_metadata"].item():
                raise SystemExit(f"{p} has no _metadata['fps']: [data] nof_aligned must name the files "
                                 f"with the frame rate of each session (compute/make_v2_data.py)")
            rates[session_name(p)] = float(d["_metadata"].item()["fps"])
    return rates


def fps_table(path=None):
    """{'H01_1D': fps} from the table of frame rates ([v11] fps_table, columns session and fps)."""
    with open(path or _path("fps_table"), newline="") as f:
        return {r["session"].replace("NOF_", ""): float(r["fps"]) for r in csv.DictReader(f)}
