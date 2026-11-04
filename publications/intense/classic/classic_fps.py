# -*- coding: utf-8 -*-
"""Classic place-cell test of the paper on a chosen clock (old 20 frames/s or the true rate).

Python port of the co-author's MATLAB routine, adapted from classic_multigrid_orig.py
(the earlier port: Skaggs information of event maps against circular shifts, events from
the DRIADA wavelet detector). The settings follow the description of the classic method
in Methods of the paper and the name of the MATLAB result file
(zscoreThr4_binsize4_center.mat); the reasons are in STAGE0B_REPORT.md:

* signal: calcium events of the DRIADA 1.2.2 wavelet detector, one event per run of the
  binary event train, placed at the first frame of the run (as in the earlier port);
* cells with fewer than 5 events get no map, z = 0 and are not place cells;
* arena split into 4 x 4 cm bins, counted from the smallest coordinate of the track;
* event counts and occupancy (frames) are smoothed separately by the same Gaussian of
  sigma = 1 bin, the rate map is their ratio, bins never visited are left out;
* Skaggs information in bits per event;
* null: 1000 circular shifts of the event train relative to the trajectory, drawn
  uniformly from the shifts the library allows for calcium (shifts shorter than five
  decay times of the indicator, 10 s, are excluded at either end, as in the earlier port);
* z = (SI - mean of the null) / standard deviation of the null; place cell: z > 4 (flag of
  the MATLAB file) and, as in the paper, z > 3.

The frame rate enters only through the cached events (the detector's windows are in
seconds) and through the shift exclusion. All frames are used (no downsampling): the
MATLAB routine has no downsampling step and event timing at full rate costs nothing here.

The shifts are drawn as uniform fractions of the allowed range from a generator seeded by
(seed, cell): the same cell gets the same fractions under both clocks, so differences
between the clocks are not shuffle noise.

usage: python classic_fps.py --sessions NOF_H01_1D NOF_H03_4D --clocks old true [--seed 0]
output: data/classic/classic_<session>_<clock>_s<seed>.npz
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path.insert(0, str(HERE.parent))
import paths_v11 as V  # noqa: E402

EVENTS = V.get("classic_dir") / "events"
OUT = V.get("classic_dir") / "classic"

SETTINGS = {
    "bin_cm": 4.0,
    "grid": "min",            # bins of bin_cm from the smallest coordinate of the track
    "sigma_bins": 1.0,        # Gaussian smoothing of counts and occupancy, in bins
    "border": "reflect",      # scipy.ndimage border rule of the smoothing
    "truncate": 4.0,          # kernel radius in sigmas (scipy default)
    "event_time": "run_start",
    "min_events": 5,
    "n_shuffles": 1000,
    "valid": "visited",       # bins entering the information: occupancy > 0
    "shifts": "mask",         # 'mask': library calcium shift range; 'all': any nonzero shift
    "std_ddof": 1,            # standard deviation of the null as in MATLAB std()
    # variants for the sensitivity check only (sensitivity.py); the defaults are the method
    "signal": "onset",        # 'onset': one count per event; 'frames': every frame of the event
    "speed_min": 0.0,         # cm/s; > 0 keeps only frames at or above this speed
}


def edges(v, size, mode):
    lo, hi = float(np.min(v)), float(np.max(v))
    if mode == "min":
        n = int(np.floor((hi - lo) / size)) + 1
        return lo + size * np.arange(n + 1)
    if mode == "center":
        n = int(np.ceil((hi - lo) / size))
        pad = (n * size - (hi - lo)) / 2
        return lo - pad + size * np.arange(n + 1)
    if mode == "zero":
        return size * np.arange(int(np.floor(lo / size)), int(np.floor(hi / size)) + 2)
    raise ValueError(mode)


def grid_index(x, y, st):
    xe, ye = edges(x, st["bin_cm"], st["grid"]), edges(y, st["bin_cm"], st["grid"])
    nx, ny = len(xe) - 1, len(ye) - 1
    bx = np.clip(np.searchsorted(xe, x, side="right") - 1, 0, nx - 1)
    by = np.clip(np.searchsorted(ye, y, side="right") - 1, 0, ny - 1)
    return bx * ny + by, (nx, ny)


def smoothing_matrix(shape, st):
    """Matrix G with G @ map.ravel() == gaussian_filter(map).ravel() for every map."""
    B = shape[0] * shape[1]
    if st["sigma_bins"] <= 0:
        return np.eye(B)
    G = np.empty((B, B))
    for k in range(B):
        e = np.zeros(B); e[k] = 1.0
        G[:, k] = gaussian_filter(e.reshape(shape), st["sigma_bins"], mode=st["border"],
                                  truncate=st["truncate"]).ravel()
    return G


def skaggs_columns(S, occ_s, valid):
    """Skaggs information (bits/event) of every column of the smoothed count maps S."""
    o = occ_s[valid]; p = o / o.sum()
    lam = S[valid] / o[:, None]
    lbar = (p[:, None] * lam).sum(0)
    r = lam / np.maximum(lbar[None, :], 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(r > 0, r * np.log2(np.where(r > 0, r, 1.0)), 0.0)
    return (p[:, None] * t).sum(0)


def event_frames(d, c, how, signal="onset"):
    if signal == "frames":
        s = d["run_st"][d["run_cell"] == c]; e = d["run_en"][d["run_cell"] == c]
        return np.concatenate([np.arange(a, b) for a, b in zip(s, e)]).astype(int) if len(s) else np.zeros(0, int)
    if how.startswith("run"):
        s = d["run_st"][d["run_cell"] == c]; e = d["run_en"][d["run_cell"] == c]
    else:
        s0 = d["ev_st"][d["ev_cell"] == c]; e0 = d["ev_en"][d["ev_cell"] == c]
        s, e = np.minimum(s0, e0), np.maximum(s0, e0)
    if how.endswith("start"):
        return s.astype(int)
    if how.endswith("center"):
        return ((s + e) // 2).astype(int)
    raise ValueError(how)


def classic_session(d, st, seed):
    """Per-cell classic statistics of one cached session."""
    T = int(d["n_frames"]); n = int(d["n_cells"])
    bin_id, shape = grid_index(d["x"], d["y"], st)
    B = shape[0] * shape[1]
    G = smoothing_matrix(shape, st)
    keep = np.ones(T, bool) if st["speed_min"] <= 0 else np.asarray(d["speed"]) >= st["speed_min"]
    occ = np.bincount(bin_id[keep], minlength=B).astype(float)
    occ_s = G @ occ
    valid = occ > 0 if st["valid"] == "visited" else occ_s > 1e-12
    res = {k: np.full(n, np.nan) for k in ("si", "mu", "sd", "z", "p")}
    res["n_events"] = np.zeros(n, int)
    N = st["n_shuffles"]
    for c in range(n):
        t = event_frames(d, c, st["event_time"], st["signal"])
        res["n_events"][c] = len(t) if st["signal"] == "onset" else int(np.sum(d["run_cell"] == c))
        if res["n_events"][c] < st["min_events"]:
            res["z"][c] = 0.0
            continue
        u = np.random.default_rng([seed, c]).random(N)
        if st["shifts"] == "mask":
            ex = int(d["shuffle_excl"][c])
            shifts = ex + np.floor(u * (T - 2 * ex)).astype(int)
        else:
            shifts = 1 + np.floor(u * (T - 1)).astype(int)
        E = (t[None, :] + np.r_[0, shifts][:, None]) % T            # (1+N, k) event frames
        rows = np.repeat(np.arange(N + 1), len(t))
        C = np.zeros((N + 1, B))
        np.add.at(C, (rows, bin_id[E.ravel()]), keep[E.ravel()].astype(float))
        vals = skaggs_columns(G @ C.T, occ_s, valid)
        null = vals[1:]
        mu, sd = null.mean(), null.std(ddof=st["std_ddof"])
        res["si"][c] = vals[0]; res["mu"][c] = mu; res["sd"][c] = sd
        res["z"][c] = (vals[0] - mu) / max(sd, 1e-12)
        res["p"][c] = (np.sum(null >= vals[0]) + 1) / (N + 1)
    return res, shape


def run(session, clock, seed, st, tag=""):
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"classic_{session}_{clock}_s{seed}{tag}.npz"
    t0 = time.time()
    d = np.load(EVENTS / f"events_{session}_{clock}.npz")
    res, shape = classic_session(d, st, seed)
    np.savez_compressed(out, session=session, clock=clock, fps=float(d["fps"]), seed=seed,
                        grid_shape=np.array(shape), settings=json.dumps(st), **res)
    z = res["z"]
    return (f"{session} {clock} seed {seed}{tag}: fps={float(d['fps']):.2f} grid {shape[0]}x{shape[1]} "
            f"cells={len(z)} >=5 events {int(np.sum(res['n_events'] >= st['min_events']))} "
            f"z>4 {int(np.sum(z > 4))} z>3 {int(np.sum(z > 3))} ({time.time() - t0:.0f} s) -> {out.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--clocks", nargs="+", default=["old", "true"], choices=["old", "true"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix of the output name for a variant")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override a setting, e.g. --set sigma_bins=1.5 border=constant")
    a = ap.parse_args()
    st = dict(SETTINGS)
    for kv in a.set:
        k, v = kv.split("=", 1)
        st[k] = type(SETTINGS[k])(v) if not isinstance(SETTINGS[k], str) else v
    for s in a.sessions:
        for c in a.clocks:
            print(run(s, c, a.seed, st, a.tag), flush=True)
    print("[OK] classic done", flush=True)


if __name__ == "__main__":
    main()
