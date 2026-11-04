"""How much rate modulation each true feature of a Table 3 cell carries, set against its detection.

The detections are not recomputed: they are read from the parts of glm/synth_compute.py
(<[glm] synth_dir>/parts/t3_t3f1_<seed>.csv). Only the generator is called again, with the arguments
of synth_common.gen and the population of Supplementary Table 3 (synth_common.POP_STD plus five
head-direction + event linear-mixture cells), to read each cell's tuning parameters and the
behavioural features. Spike reconstruction is skipped: it runs after the rates are built and changes
neither the features nor the tuning. The tuning parameters come from the seeded generator, so they
are those of the saved run.

Per cell and true feature f the generator sets rate = baseline + (peak - baseline) * sum_f w_f r_f(t),
with r_f in [0, 1] and, for a two-feature weighted sum without explicit weights, w_f = 0.5 (w = 1 for a
pure cell); the script stops if a mixture cell of the population carries weights of its own. It reports
the modulation a feature contributes, (peak - baseline) * w_f * SD_t(r_f), in the rate units of the
generator, and whether INTENSE (p < 0.001, Table 3) and the strongest GLM on mixtures (alpha = 0.10)
detected that feature. Outputs: component_*.csv in [glm] mixtures_dir, read by glm/mixtures_summary.py.

usage: python glm/mixtures_components.py [--workers 4]
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys  # noqa: E402

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, str(PAPER))
import intense_paths as P  # noqa: E402

S1 = P.get("glm", "synth_dir")
OUT = P.get("glm", "mixtures_dir")

PEAK, BASE = 2.0, 0.1
GROUPS = {"hd_cells": ["head_direction"], "speed_cells": ["speed"], "event_cells": ["event_0"],
          "mixed_hd_speed_lin": ["head_direction", "speed"],
          "mixed_speed_event_lin": ["speed", "event_0"],
          "mixed_hd_event_lin": ["head_direction", "event_0"]}
SHORT = {"head_direction": "HD", "speed": "speed", "event_0": "event"}


def setup():
    """Load the DRIADA release of config.toml in a worker process and make synth_common importable.

    Returns
    -------
    module
        The ``driada`` package, after ``intense_paths.use_driada`` has checked its version.
    """
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    return P.use_driada(tools=False)


def one_seed(seed):
    import numpy as np
    driada = setup()
    from synth_common import POP_STD, group_map
    from driada.experiment.synthetic import generate_tuned_selectivity_exp
    from driada.experiment.synthetic.tuning import von_mises_tuning_curve, sigmoid_tuning_curve
    hd_event = {"name": "mixed_hd_event_lin", "count": 5, "features": ["head_direction", "event_0"],
                "combination": "weighted_sum"}
    pop = POP_STD + [hd_event]
    t0 = time.time()
    exp = generate_tuned_selectivity_exp(
        population=pop, duration=900.0, fps=20.0, seed=seed, n_discrete_features=1,
        baseline_rate=BASE, peak_rate=PEAK, decay_time=2.0, calcium_noise=0.05, verbose=False,
        reconstruct_spikes=None)
    groups = group_map(pop)
    hd = exp.dynamic_features["head_direction"].data
    sp = exp.dynamic_features["speed"].data
    ev = exp.dynamic_features["event_0"].data.astype(float)
    tp = exp.ground_truth["tuning_parameters"]
    rows = []
    for ni, g in groups.items():
        if g not in GROUPS:
            continue
        feats = GROUPS[g]
        comb = tp[ni].get("_combination")
        if len(feats) == 2:
            assert comb is not None and comb["mode"] == "weighted_sum" and comb["weights"] is None, comb
        w = 1.0 / len(feats)
        for f in feats:
            if f == "head_direction":
                r = von_mises_tuning_curve(hd, float(tp[ni][f]["pref_dir"]), float(tp[ni][f]["kappa"]))
                par = float(tp[ni][f]["pref_dir"])
            elif f == "speed":
                r = sigmoid_tuning_curve(sp, float(tp[ni][f]["threshold"]), float(tp[ni][f]["slope"]))
                par = float(tp[ni][f]["threshold"])
            else:
                r = ev
                par = float("nan")
            rows.append({"seed": seed, "neuron": ni, "group": g, "feature": SHORT[f], "weight": w,
                         "tuning_param": par, "mean_r": float(r.mean()), "sd_r": float(r.std()),
                         "mod_sd": (PEAK - BASE) * w * float(r.std())})
    return rows, time.time() - t0, driada.__version__, str(Path(driada.__file__).parent)


def det_set(s):
    return set() if (not isinstance(s, str) or s == "") else set(s.split("|"))


def main():
    import numpy as np
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    seeds = [42, 123, 7, 11, 19, 23, 31, 44, 57, 64, 73, 88, 91, 101, 113, 127, 139, 151, 163, 177]
    OUT.mkdir(parents=True, exist_ok=True)
    rows, info = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r, sec, ver, where in ex.map(one_seed, seeds):
            rows += r
            info.append({"seed": r[0]["seed"], "seconds": round(sec, 1), "driada": ver, "from": where})
    cells = pd.DataFrame(rows)
    full = {"HD": "head_direction", "speed": "speed", "event": "event_0"}
    det = pd.concat([pd.read_csv(S1 / "parts" / f"t3_t3f1_{s}.csv") for s in seeds], ignore_index=True)
    det = det[det.variant.isin(["INTENSE_p001", "V_alpha10"])]
    det = det.assign(det=det.detected.apply(det_set)).set_index(["seed", "neuron", "variant"]).det
    for v, col in (("INTENSE_p001", "found_INTENSE"), ("V_alpha10", "found_GLM")):
        cells[col] = [full[f] in det.loc[(s, n, v)] for s, n, f in zip(cells.seed, cells.neuron, cells.feature)]
    cells.to_csv(OUT / "component_strength_cells.csv", index=False)

    summ = (cells.groupby(["group", "feature"])
            .agg(n=("mod_sd", "size"), mod_sd_median=("mod_sd", "median"), mod_sd_q25=("mod_sd", lambda x: x.quantile(.25)),
                 mod_sd_q75=("mod_sd", lambda x: x.quantile(.75)), found_INTENSE=("found_INTENSE", "mean"),
                 found_GLM=("found_GLM", "mean"))
            .reset_index())
    summ.to_csv(OUT / "component_strength_summary.csv", index=False)

    # mixture cells where exactly one of the two features was found: is it the stronger component?
    stronger = []
    for v, col in (("INTENSE_p001", "found_INTENSE"), ("V_alpha10", "found_GLM")):
        for g in ["mixed_hd_speed_lin", "mixed_speed_event_lin", "mixed_hd_event_lin"]:
            sub = cells[cells.group == g]
            k = n = 0
            for (s, ni), cc in sub.groupby(["seed", "neuron"]):
                if cc[col].sum() == 1:
                    n += 1
                    k += int(cc.loc[cc[col], "mod_sd"].iloc[0] > cc.loc[~cc[col], "mod_sd"].iloc[0])
            stronger.append({"variant": v, "group": g, "one_found": n, "found_is_stronger": k})
    stronger = pd.DataFrame(stronger)
    stronger.to_csv(OUT / "component_one_found_stronger.csv", index=False)

    # detection as a function of the feature's own modulation, pure and mixed cells pooled per feature type
    edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    cells["mod_bin"] = pd.cut(cells.mod_sd, edges)
    cells["kind"] = np.where(cells.weight < 1, "mixture", "pure")
    curve = (cells.groupby(["feature", "kind", "mod_bin"], observed=True)
             .agg(n=("mod_sd", "size"), found_INTENSE=("found_INTENSE", "mean"), found_GLM=("found_GLM", "mean"))
             .reset_index())
    curve.to_csv(OUT / "component_detection_by_modulation.csv", index=False)
    (OUT / "component_strength_run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))
    print()
    print(stronger.to_string(index=False))
    print()
    print(curve.round(2).to_string(index=False))
    print()
    print(info[:2], "...", f"total {sum(i['seconds'] for i in info):.0f} s")


if __name__ == "__main__":
    main()
