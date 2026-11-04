"""False-positive calibration of INTENSE in the configuration of the NM paper.

Builds null experiments in which neural activity is independent of behaviour
and runs INTENSE on them exactly as the real NOF data are analysed
(tools/run_intense_analysis.py of v1.2.2: GCMI, by_type representation,
two-stage 100 / 10 000 shuffles, gamma_zi null, p < 0.001 per pair without
correction, ds = 5, delay search in +-2 s, anti-selective filter), with the
delay search switched on and off.

Null models
-----------
shift  real calcium of a session + all behavioural variables of the same
       session rotated together by a large circular shift (25-75 % of the
       recording). Keeps autocorrelation of both sides and the joint structure
       of the features; breaks the alignment.
cross  real calcium of mouse i + behaviour of mouse i+1 on the same day,
       both cut to the shorter recording. Independent by construction; keeps
       slow trends of both sides at their original place in the session.
synth  driada generator: neurons without selectivity (baseline Poisson spikes
       convolved with a calcium kernel) and synthetic place, speed, head
       direction and binary events.

Disentanglement only removes pairs, so it is not run: counts are those of
significant pairs before disentanglement (an upper bound for the paper's
final labels).

Usage
-----
    python calib_null.py --null shift --sessions NOF_H03_1D --out results
    python calib_null.py --null synth --n-synth 8 --out results
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402

P.use_driada()

import numpy as np
import pandas as pd

import driada

# The calibration is only meaningful for the code the paper is computed with
# (P.use_driada above checks the source tree and the version).

from selectivity_dynamics import DEFAULT_CONFIG, get_experiment_config  # noqa: E402
from selectivity_dynamics.analysis import _fix_normalized_circular_features  # noqa: E402
from selectivity_dynamics.loader import build_feature_list  # noqa: E402
from load_synchronized_experiments import get_npz_metadata  # noqa: E402
from driada.experiment.exp_build import load_exp_from_aligned_data  # noqa: E402
from driada.utils.naming import parse_iabs_filename  # noqa: E402

DATA = P.NOF_ALIGNED
MICE = ["H01", "H02", "H03", "H06", "H07", "H08", "H09", "H14", "H23", "H26",
        "H27", "H31", "H32", "H33", "H36", "H39"]
SEED = P.SEED
META_KEYS = ("_metadata", "_sync_info")

PAPER = dict(
    n_shuffles_stage1=DEFAULT_CONFIG["n_shuffles_stage1"],   # 100
    n_shuffles_stage2=DEFAULT_CONFIG["n_shuffles_stage2"],   # 10000
    pval_thr=DEFAULT_CONFIG["pval_thr"],                     # 0.001
    multicomp_correction=DEFAULT_CONFIG["multicomp_correction"],  # None
    ds=DEFAULT_CONFIG["ds"],                                 # 5
)


def feature_type(name):
    if name.startswith("place") or name.startswith("position"):
        return "place"
    if name.startswith("speed"):
        return "speed"
    if "direction" in name:
        return "direction"
    return "discrete"


def load_npz(path):
    data = dict(np.load(path, allow_pickle=True))
    for k in META_KEYS:
        data.pop(k, None)
    return data


def build_exp(data, name_for_params, fps):
    exp_config = get_experiment_config("NOF")
    parsed = parse_iabs_filename(Path(name_for_params).name)
    exp_params = {k: parsed[k] for k in ("track", "animal_id", "session")}
    return load_exp_from_aligned_data(
        data_source="IABS", exp_params=exp_params, data=data,
        feature_types=exp_config.get("feature_types"),
        # The paper loader passes fps only when the file has it; otherwise
        # the experiment takes the default of 20 frames/s.
        static_features={} if fps is None else {"fps": float(fps)},
        aggregate_features=exp_config["aggregate_features"],
        verbose=False, seed=SEED,
    ), exp_config["skip_for_intense"]


def null_data(null, session, rng_seed):
    """Calcium of `session` paired with behaviour independent of it."""
    path = DATA / f"{session}_aligned.npz"
    data = load_npz(path)
    fps = get_npz_metadata(path)["fps"]
    beh_keys = [k for k in data if k != "Calcium"]
    info = {}
    if null == "shift":
        n = data["Calcium"].shape[1]
        rng = np.random.default_rng([rng_seed, MICE.index(session.split("_")[1]),
                                     int(session.split("_")[2][0])])
        shift = int(rng.integers(n // 4, 3 * n // 4))
        for k in beh_keys:
            data[k] = np.roll(data[k], shift)
        info = {"shift_frames": shift, "beh_session": session}
    elif null == "cross":
        mouse, day = session.split("_")[1], session.split("_")[2]
        other = MICE[(MICE.index(mouse) + 1) % len(MICE)]
        other_session = f"NOF_{other}_{day}"
        opath = DATA / f"{other_session}_aligned.npz"
        odata = load_npz(opath)
        assert get_npz_metadata(opath)["fps"] == fps
        n = min(data["Calcium"].shape[1], odata["Calcium"].shape[1])
        data = {"Calcium": data["Calcium"][:, :n]}
        for k in beh_keys:
            data[k] = odata[k][:n]
        info = {"beh_session": other_session, "n_frames": n}
    else:
        raise ValueError(null)
    return data, fps, path, info


def run_intense(exp, feat_bunch, delays, cell_bunch=None):
    _fix_normalized_circular_features(exp)
    return driada.compute_cell_feat_significance(
        exp, cell_bunch=cell_bunch, feat_bunch=feat_bunch, metric="mi", mode="two_stage",
        find_optimal_delays=delays, with_disentanglement=False,
        enable_parallelization=False, verbose=False, engine="auto",
        remove_anti_selective=True, representation=P.REPRESENTATION, seed=SEED,
        use_precomputed_stats=False, **PAPER,
    )


def collect(stats, significance, info, label, extra):
    rows = []
    feats = list(next(iter(stats.values())).keys())
    delays = info["optimal_delays"]
    for i, cell in enumerate(stats):
        for j, feat in enumerate(feats):
            st = stats[cell][feat]
            sg = significance[cell][feat]
            rows.append(dict(
                unit=label, neuron=cell, feature=feat, ftype=feature_type(feat),
                me=st.get("me"), me_excess=st.get("me_excess"), pval=st.get("pval"),
                pre_rval=st.get("pre_rval"), rval=st.get("rval"),
                signal_ratio=st.get("signal_ratio"),
                stage1=bool(sg.get("stage1", False)), sig=bool(sg.get("stage2", False)),
                delay_frames=int(delays[i, j]), **extra,
            ))
    return rows


def run_real(null, session, out):
    for delays in (False, True):
        tag = "delay" if delays else "nodelay"
        fn = out / f"{null}_{tag}_{session}.csv"
        if fn.exists():
            print(f"[SKIP] {fn.name}", flush=True)
            continue
        t0 = time.time()
        data, fps, path, info = null_data(null, session, SEED)
        exp, skip = build_exp(data, path, fps)
        feat_bunch = build_feature_list(exp, skip)
        stats, sig, inf, _, _ = run_intense(exp, feat_bunch, delays)
        rows = collect(stats, sig, inf, session,
                       dict(null=null, delays=delays, **info))
        pd.DataFrame(rows).to_csv(str(fn) + ".tmp", index=False)
        os.replace(str(fn) + ".tmp", fn)
        df = pd.DataFrame(rows)
        print(f"[OK] {null} {tag} {session}: {exp.n_cells} cells, {len(df)} pairs, "
              f"{df.sig.sum()} significant, {time.time() - t0:.0f} s", flush=True)


def run_synth(k, out, n_cells=300, duration=900):
    from driada.experiment.synthetic import generate_tuned_selectivity_exp
    for delays in (False, True):
        tag = "delay" if delays else "nodelay"
        fn = out / f"synth_{tag}_{k:02d}.csv"
        if fn.exists():
            print(f"[SKIP] {fn.name}", flush=True)
            continue
        t0 = time.time()
        # One tuned neuron makes the generator produce the trajectory-derived
        # features; only the non-selective neurons are tested.
        population = [
            {"name": "driver", "count": 1, "features": ["position_2d", "head_direction", "speed"]},
            {"name": "nonselective", "count": n_cells, "features": []},
        ]
        exp = generate_tuned_selectivity_exp(
            population, duration=duration, fps=20, n_discrete_features=4,
            seed=SEED + k, verbose=False)
        cells = list(range(1, exp.n_cells))
        feat_bunch = [f for f in exp.dynamic_features if f not in ("x", "y")]
        stats, sig, inf, _, _ = run_intense(exp, feat_bunch, delays, cell_bunch=cells)
        rows = collect(stats, sig, inf, f"synth{k:02d}",
                       dict(null="synth", delays=delays))
        pd.DataFrame(rows).to_csv(str(fn) + ".tmp", index=False)
        os.replace(str(fn) + ".tmp", fn)
        df = pd.DataFrame(rows)
        print(f"[OK] synth {tag} {k}: {len(cells)} cells, {len(df)} pairs, "
              f"{df.sig.sum()} significant, {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", choices=["shift", "cross", "synth"], required=True)
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--synth-ids", nargs="*", type=int, default=None)
    ap.add_argument("--out", default=str(P.CALIBRATION_DIR / "results"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"driada {driada.__version__} from {Path(driada.__file__).parent}", flush=True)
    print(f"paper config: {PAPER}, by_type, gcmi, anti-selective on, seed {SEED}", flush=True)
    if a.null == "synth":
        for k in a.synth_ids:
            run_synth(k, out)
    else:
        for s in a.sessions:
            run_real(a.null, s, out)


if __name__ == "__main__":
    main()
