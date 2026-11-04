"""Synthetic neurons at the information of CA1 neurons (folder s7/ of [s2] out_dir): pure place,
pure body direction and true mixed place + body direction (sum of two processes), on the recorded
behaviour of the 16 sessions of sessions_16.csv at the frame rate of each session.

Design:
  * the CA1 neurons that stay multiply selective after disentanglement are mostly body
    direction + place (544 of 868), so the mixed cells are place + body direction;
  * strengths: CA1 strongest-label excess MI is about 0.06-0.12 bits for place and 0.05-0.08 for
    body direction (data/s3_strength_overlap.csv); for the synthetic neurons of the S2 categories
    that is SNR about 3-6 for place and 2-5 for body direction, so:
      place_in, place_out       SNR 3, 4, 5, 6      15 per level  (60 + 60 per session)
      bodydirection             SNR 2, 3, 4, 5      10 per level  (40 per session)
      place+bodydirection (sum) SNR 4, 6, 8, 12     15 per level  (60 per session)
  * everything else as the generator of the S2 categories (synthetic/make_synth.py, from which
    Session, place(), direction() and calcium() are copied; place() here lacks the test that a
    field lies mostly outside a zone to avoid, which only the place + zone neurons of
    make_synth.py use, and no neuron here passes such a zone): circular
    field 6-11 cm, direction half-width 30-60 deg, Poisson events, calcium amplitudes 0.5-2,
    decay 2 s, noise 0.1, baseline 0.1 Hz; a sum cell is two processes with half the baseline each.
  * seed 1102 (the S2 categories use 42), so no neuron repeats a neuron of those categories; the
    session number in the seed [1102, session number, neuron] is the position in SEED_ORDER, as in
    s2/s2_synth_fps_16.py.
Output: <[s2] out_dir>/s7/synth_npz/<session>_aligned.npz, s7/truth/<session>.json.
usage: python s2/7_s7_make.py [session ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
import intense_paths as P  # noqa: E402

P.use_driada()
from driada.experiment.synthetic import (  # noqa: E402
    apply_poisson_to_binary_series,
    generate_pseudo_calcium_signal,
)

OUTDIR = P.get("s2", "out_dir") / "s7"
OLD8 = ["NOF_H01_2D", "NOF_H02_3D", "NOF_H07_1D", "NOF_H08_4D",
        "NOF_H14_1D", "NOF_H23_2D", "NOF_H31_4D", "NOF_H36_2D"]
SESSIONS = list(pd.read_csv(P.get("s2", "out_dir") / "sessions_16.csv").session)
SEED_ORDER = OLD8 + [s for s in SESSIONS if s not in OLD8]
SEED = 1102
BASELINE_RATE = 0.1
AMPLITUDE_RANGE = (0.5, 2.0)
CALCIUM_NOISE = 0.1
DECAY_TIME = 2.0
MIN_OCC = 0.015
PLAN = [("place_in", [3, 4, 5, 6], 15), ("place_out", [3, 4, 5, 6], 15),
        ("bodydirection", [2, 3, 4, 5], 10), ("place+bodydirection", [4, 6, 8, 12], 15)]
OBJECTS = ["object1", "object2", "object3", "object4"]
ZONES = ["corners", "walls", "center"]
MASK_NAMES = {"corner1": "ArenaCornerRealOut1", "corner2": "ArenaCornerRealOut2",
              "corner3": "ArenaCornerRealOut3", "corner4": "ArenaCornerRealOut4",
              "corners": "ArenaCornersAllRealOut", "walls": "ArenaWallsAllRealOut",
              "center": "Center", "object1": "Object1RealOut", "object2": "Object2RealOut",
              "object3": "Object3RealOut", "object4": "Object4RealOut",
              "objects": "ObjectAllRealOut"}
FPS = None  # set per session from _metadata['fps'] of its file in [data] nof_aligned


class Session:
    def __init__(self, name):
        self.name = name
        self.raw = dict(np.load(P.NOF_ALIGNED / f"{name}_aligned.npz", allow_pickle=True))
        self.n = self.raw["x"].size
        self.pos = np.vstack([self.raw["x"], self.raw["y"]])
        with np.load(P.ZONE_MASKS / "NOF" / f"{name}_zones_cm.npz") as z:
            masks = dict(zip([str(s) for s in z["zone_names"]], z["masks"].astype(bool)))
            step = float(z["grid_step_cm"])
        ix = np.floor(self.pos[0] / step + 0.5).astype(int)
        iy = np.floor(self.pos[1] / step + 0.5).astype(int)
        ny, nx = masks["ArenaReal"].shape
        ok = (ix >= 0) & (iy >= 0) & (ix < nx) & (iy < ny)
        self.in_mask = {}
        for key, mname in MASK_NAMES.items():
            m = np.zeros(self.n, bool)
            m[ok] = masks[mname][iy[ok], ix[ok]]
            self.in_mask[key] = m
        self.ind = {f: self.raw[f] > 0.5 for f in ZONES + OBJECTS + ["locomotion"]}
        self.objects = [o for o in OBJECTS if self.ind[o].mean() >= MIN_OCC]
        self.small_zones = [f"corner{k}" for k in range(1, 5)] + self.objects

    def zones_of(self, frame):
        return sorted(k for k, m in self.in_mask.items() if m[frame])

    def direction(self, rng, feat):
        pref = rng.uniform(-np.pi, np.pi)
        width = np.deg2rad(rng.uniform(30, 60))
        on = np.cos(self.raw[feat] - pref) > np.cos(width)
        return on, {f"{feat}_pref": round(pref, 3), f"{feat}_halfwidth_deg": round(np.rad2deg(width), 1)}

    def place(self, rng, where, avoid=None):
        small = np.zeros(self.n, bool)
        for k in ("corners", "objects"):
            small |= self.in_mask[k]
        for _ in range(500):
            r = rng.uniform(6, 11)
            if where == "in":
                zone = self.small_zones[rng.integers(len(self.small_zones))]
                frames = np.flatnonzero(self.in_mask[zone])
            else:
                frames = np.flatnonzero(~small)
            if avoid is not None:
                frames = frames[~self.ind[avoid][frames]]
            if frames.size == 0:
                continue
            c = self.pos[:, frames[rng.integers(frames.size)]]
            on = np.hypot(self.pos[0] - c[0], self.pos[1] - c[1]) < r
            if on.mean() < MIN_OCC:
                continue
            if where == "out" and small[on].mean() > 0.2:
                continue
            centre_frame = np.flatnonzero(on)[np.argmin(np.hypot(*(self.pos[:, on] - c[:, None])))]
            return on, {"field_center_cm": [round(float(c[0]), 2), round(float(c[1]), 2)],
                        "field_radius_cm": round(r, 2), "field_occupancy": round(float(on.mean()), 4),
                        "field_zones": self.zones_of(centre_frame),
                        "field_in_zone_share": {k: round(float(self.in_mask[k][on].mean()), 3)
                                                for k in ("corners", "walls", "center", "objects")}}
        raise RuntimeError(f"{self.name}: no place field for where={where}")


def calcium(rng, ons, snr, duration):
    base = BASELINE_RATE / FPS
    peak = BASELINE_RATE * snr / FPS
    share = base / len(ons)
    events = sum(apply_poisson_to_binary_series(on.astype(int), share, peak + share - base,
                                                seed=int(rng.integers(2**31)))
                 for on in ons)
    return generate_pseudo_calcium_signal(
        events=events, duration=duration, sampling_rate=FPS, amplitude_range=AMPLITUDE_RANGE,
        decay_time=DECAY_TIME, noise_std=CALCIUM_NOISE, seed=int(rng.integers(2**31)))


def make_session(name, s_index):
    global FPS
    S = Session(name)
    FPS = float(S.raw["_metadata"].item()["fps"])
    duration = S.n / FPS
    rows, traces, nid = [], [], 0
    for cat, levels, per in PLAN:
        for snr in levels:
            for _ in range(per):
                rng = np.random.default_rng([SEED, s_index, nid])
                if cat in ("place_in", "place_out"):
                    on, params = S.place(rng, cat.split("_")[1])
                    ons, truth = [on], ["place"]
                elif cat == "bodydirection":
                    on, params = S.direction(rng, "bodydirection")
                    ons, truth = [on], ["bodydirection"]
                else:
                    pon, params = S.place(rng, "out")
                    don, dp = S.direction(rng, "bodydirection")
                    params.update(dp)
                    ons, truth = [pon, don], ["place", "bodydirection"]
                traces.append(calcium(rng, ons, snr, duration))
                rows.append({"neuron": nid, "category": cat, "snr": snr, "truth": truth,
                             "on_share": [round(float(o.mean()), 4) for o in ons], **params})
                nid += 1
    out = dict(S.raw)
    out["Calcium"] = np.vstack(traces)
    (OUTDIR / "synth_npz").mkdir(parents=True, exist_ok=True)
    (OUTDIR / "truth").mkdir(parents=True, exist_ok=True)
    np.savez(OUTDIR / "synth_npz" / f"{name}_aligned.npz", **out)
    meta = {"session": name, "n_frames": S.n, "objects_used": S.objects,
            "settings": {"seed": SEED, "baseline_rate_hz": BASELINE_RATE, "plan": PLAN,
                         "amplitude_range": AMPLITUDE_RANGE, "calcium_noise": CALCIUM_NOISE,
                         "decay_time_s": DECAY_TIME, "fps": FPS},
            "neurons": rows}
    (OUTDIR / "truth" / f"{name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{name}: {len(rows)} neurons, {S.n} frames at {FPS} frames/s", flush=True)


if __name__ == "__main__":
    for i, name in enumerate(sys.argv[1:] or SESSIONS):
        if (OUTDIR / "truth" / f"{name}.json").exists():
            print(f"[SKIP] {name}", flush=True)
            continue
        make_session(name, SEED_ORDER.index(name))
