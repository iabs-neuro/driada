"""Synthetic neurons with known selectivity on top of the real behaviour of NOF sessions.

For every chosen session the behaviour (x, y, speed, locomotion/walk/rest/freezing,
zones, objects, head and body direction) is kept exactly as recorded; only the
calcium matrix is replaced by synthetic neurons. Each neuron fires as in the
Figure 4 benchmark (run_benchmark.py, generate_synthetic_data of DRIADA 1.2.2):
a binary "on" series of the variable it depends on drives a Poisson process with
the baseline rate 0.1 Hz off and baseline x SNR on, and the spikes are turned into
calcium by generate_pseudo_calcium_signal (amplitudes 0.5-2, decay 2 s, noise 0.1).
A neuron that depends on two variables is the sum of two such processes, each
with half of the baseline, so that it fires more when both are on.

Categories (truth in source feature names):
  speed, locomotion, place_in (field inside one corner or one object zone),
  place_out (field outside corners and objects), zone (corners, walls or center),
  object (one specific object), headdirection, bodydirection,
  mixed speed+locomotion, place+zone (field outside that zone),
  zone+object, headdirection+bodydirection, and null (random activity unrelated
  to behaviour).

Output: synth_npz/<session>_aligned.npz (same keys as the real file, synthetic
Calcium) and truth/<session>.json (one record per neuron).

usage: python make_synth.py [session ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402

P.use_driada()
from driada.experiment.synthetic import (  # noqa: E402
    apply_poisson_to_binary_series,
    generate_binary_time_series,
    generate_pseudo_calcium_signal,
)

SESSIONS = ["NOF_H01_2D", "NOF_H02_3D", "NOF_H07_1D", "NOF_H08_4D",
            "NOF_H14_1D", "NOF_H23_2D", "NOF_H31_4D", "NOF_H36_2D"]
SEED = 42
FPS = 20
BASELINE_RATE = 0.1
SNR_LEVELS = [4, 8, 16, 32]
PER_LEVEL = 10
AMPLITUDE_RANGE = (0.5, 2.0)
CALCIUM_NOISE = 0.1
DECAY_TIME = 2.0
MIN_OCC = 0.015  # smallest share of frames in an object zone or a place field

SINGLE = ["speed", "locomotion", "place_in", "place_out", "zone", "object",
          "headdirection", "bodydirection"]
MIXED = ["speed+locomotion", "place+zone", "zone+object", "headdirection+bodydirection"]
CATEGORIES = SINGLE + MIXED + ["null"]
ZONES = ["corners", "walls", "center"]
OBJECTS = ["object1", "object2", "object3", "object4"]
MASK_NAMES = {"corner1": "ArenaCornerRealOut1", "corner2": "ArenaCornerRealOut2",
              "corner3": "ArenaCornerRealOut3", "corner4": "ArenaCornerRealOut4",
              "corners": "ArenaCornersAllRealOut", "walls": "ArenaWallsAllRealOut",
              "center": "Center", "object1": "Object1RealOut", "object2": "Object2RealOut",
              "object3": "Object3RealOut", "object4": "Object4RealOut",
              "objects": "ObjectAllRealOut"}


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
        # Zone membership of every frame by the masks the zone rule uses
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

    # ------------------------------------------------------------- on-series
    def speed(self, rng):
        q = rng.uniform(0.5, 0.85)
        return self.raw["speed"] > np.quantile(self.raw["speed"], q), {"speed_quantile": round(q, 3)}

    def locomotion(self, rng):
        return self.ind["locomotion"], {}

    def direction(self, rng, feat):
        pref = rng.uniform(-np.pi, np.pi)
        width = np.deg2rad(rng.uniform(30, 60))
        on = np.cos(self.raw[feat] - pref) > np.cos(width)
        return on, {f"{feat}_pref": round(pref, 3), f"{feat}_halfwidth_deg": round(np.rad2deg(width), 1)}

    def zone(self, rng, zone=None):
        zone = zone or ZONES[rng.integers(len(ZONES))]
        return self.ind[zone], {"zone": zone}

    def obj(self, rng):
        o = self.objects[rng.integers(len(self.objects))]
        return self.ind[o], {"object": o}

    def place(self, rng, where, avoid=None):
        """Circular field of radius 6-11 cm centred on a visited position.

        where='in': centre inside one corner or one object zone;
        where='out': centre outside corners and objects, the field lying mostly outside them;
        avoid: a zone (indicator) the field must lie outside of (mixed place+zone).
        """
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
            if avoid is not None and self.ind[avoid][on].mean() > 0.2:
                continue
            centre_frame = np.flatnonzero(on)[np.argmin(np.hypot(*(self.pos[:, on] - c[:, None])))]
            return on, {"field_center_cm": [round(float(c[0]), 2), round(float(c[1]), 2)],
                        "field_radius_cm": round(r, 2), "field_occupancy": round(float(on.mean()), 4),
                        "field_zones": self.zones_of(centre_frame),
                        "field_in_zone_share": {k: round(float(self.in_mask[k][on].mean()), 3)
                                                for k in ("corners", "walls", "center", "objects")}}
        raise RuntimeError(f"{self.name}: no place field for where={where}, avoid={avoid}")

    def component(self, rng, what, **kw):
        if what == "speed":
            return self.speed(rng)
        if what == "locomotion":
            return self.locomotion(rng)
        if what in ("headdirection", "bodydirection"):
            return self.direction(rng, what)
        if what == "zone":
            return self.zone(rng, **kw)
        if what == "object":
            return self.obj(rng)
        raise ValueError(what)


def truth_name(what, params):
    return {"zone": params.get("zone"), "object": params.get("object")}.get(what, what)


def calcium(rng, ons, snr, duration):
    """Poisson spikes driven by the on-series, turned into calcium as in the benchmark."""
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
    S = Session(name)
    duration = S.n / FPS
    rows, traces = [], []
    nid = 0
    for cat in CATEGORIES:
        for snr in SNR_LEVELS:
            for rep in range(PER_LEVEL):
                rng = np.random.default_rng([SEED, s_index, nid])
                params, ons, truth = {}, [], []
                if cat == "null":
                    n_isl = max(2, int(round(duration / 60)))
                    ons = [generate_binary_time_series(S.n, n_isl, 5 * FPS,
                                                       seed=int(rng.integers(2**31))).astype(bool)]
                elif cat in ("place_in", "place_out"):
                    on, params = S.place(rng, cat.split("_")[1])
                    ons, truth = [on], ["place"]
                elif cat == "place+zone":
                    zone = ZONES[rng.integers(len(ZONES))]
                    zon, zp = S.zone(rng, zone)
                    pon, params = S.place(rng, "any", avoid=zone)
                    params.update(zp)
                    ons, truth = [pon, zon], ["place", zone]
                elif "+" in cat:
                    for what in cat.split("+"):
                        on, p = S.component(rng, what)
                        params.update(p)
                        ons.append(on)
                        truth.append(truth_name(what, p))
                else:
                    on, params = S.component(rng, cat)
                    ons, truth = [on], [truth_name(cat, params)]
                traces.append(calcium(rng, ons, snr, duration))
                rows.append({"neuron": nid, "category": cat, "snr": snr, "truth": truth,
                             "on_share": [round(float(o.mean()), 4) for o in ons], **params})
                nid += 1
    out = dict(S.raw)
    out["Calcium"] = np.vstack(traces)
    (HERE / "synth_npz").mkdir(exist_ok=True)
    (HERE / "truth").mkdir(exist_ok=True)
    np.savez(HERE / "synth_npz" / f"{name}_aligned.npz", **out)
    meta = {"session": name, "n_frames": S.n, "objects_used": S.objects,
            "settings": {"seed": SEED, "baseline_rate_hz": BASELINE_RATE, "snr_levels": SNR_LEVELS,
                         "per_level": PER_LEVEL, "amplitude_range": AMPLITUDE_RANGE,
                         "calcium_noise": CALCIUM_NOISE, "decay_time_s": DECAY_TIME, "fps": FPS},
            "neurons": rows}
    (HERE / "truth" / f"{name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{name}: {len(rows)} neurons, {S.n} frames, objects {S.objects}", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or SESSIONS
    for name in names:
        make_session(name, SESSIONS.index(name) if name in SESSIONS else 100 + len(name))
