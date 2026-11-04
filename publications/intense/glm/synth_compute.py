"""Synthetic benchmark of Supplementary Section S1: INTENSE and the GLM variants on the same recordings.

Each recording is generated once, and every method of its part is run on that recording: the
generator of DRIADA 1.2.2 adds tie-breaking noise without a seed to the synthetic calcium, so two
calls with one seed give slightly different recordings, and methods compared side by side must
share the recording.

Parts (one unit of work = one recording; the tag starts the names of the files of the part):
  t3f1  Supplementary Table 3 and Supplementary Fig. 1. 20 seeds; the 45 cells of the standard
        population (synth_common.POP_STD) plus 5 cells whose rate is the weighted sum of
        head-direction tuning and the event; 900 s at 20 Hz, SNR 20 (synth_common.gen). INTENSE at
        p < 0.001 per pair without correction (Table 3), INTENSE at p < 0.05 with Holm correction
        (Fig. 1), the eight GLM variants of glm_variants.py. On the first 10 seeds also the two
        GLM-only controls of Section S1.3 on the 45 standard cells: the GLM with pairwise
        interaction blocks and the GLM with four bases for speed (files t2_*).
  rep   seeds 42, 123 and 7 of t3f1 generated again, to measure the spread that the noise without
        a seed alone produces.
  snr   Supplementary Fig. 2. 10 seeds x 7 SNR levels, 35 cells, calcium noise 0.1: INTENSE at
        p < 0.05 with Holm correction and three GLM variants.
  t31   Three-feature mixtures. 15 seeds, 40 cells: INTENSE at p < 0.05 with Holm correction with
        the type-based representations, the same with untransformed features as a side arm, and
        three GLM variants.

INTENSE everywhere: two stages of 100 / 10,000 shuffles, ds = 5, delay search off, seed 42,
type-based representations (the default of DRIADA 1.2.2) unless stated. Each configuration is
called once with the removal of anti-selective pairs off, and both readings are kept: without the
removal (the reading of Table 3) and with it (the DRIADA default; the reading of Figs 1 and 2 and
of the three-feature mixtures). INTENSE runs its parallel path on 2 workers: its circular shifts
are drawn from the seed before the work is split, so the split does not decide them.

Workers take units in a fixed order by creating claims/<unit>.claim exclusively; a unit is done
when done/<unit>.ok exists, and a failed one leaves failed/<unit>.err (delete its claim to run it
again). Everything is written to [glm] synth_dir: parts/<kind>_<unit>.csv (per cell and method),
logs/worker_<K>.log. glm/synth_numbers.py, glm/synth_figures.py and glm/mixtures_*.py read parts/.

usage: python glm/synth_compute.py --worker K [--smoke]
       start one process per worker, K = 1, 2, ...; --smoke runs one short recording of each part
       into <synth_dir>/_smoke to check the code paths
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
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(HERE))
import intense_paths as P  # noqa: E402

driada = P.use_driada(tools=False)
OUTD = P.get("glm", "synth_dir")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from synth_common import POP_STD, SEEDS20, TRUE_STD, gen, group_map, prf1  # noqa: E402
from glm_variants import (VARIANTS, selected_to_features, build_blocks, rank_gaussian, cv_r2,  # noqa: E402
                          _forward, _wilcoxon_p, _speed_bins, _hd_bins, _binary_bins)
from driada.experiment.synthetic import generate_tuned_selectivity_exp  # noqa: E402
from driada.intense.representations import get_representation_sources  # noqa: E402

CANON = ("head_direction", "speed", "event_0")
GLM8 = ["V0_base", "V_ttest", "V_fdr", "V_alpha10", "V_folds20", "V_poisson", "V_max", "V_max_perfeat"]
GLM3 = ["V0_base", "V_alpha10", "V_max"]

HD_EVENT = {"name": "mixed_hd_event_lin", "count": 5, "features": ["head_direction", "event_0"],
            "combination": "weighted_sum"}
POP_T3 = POP_STD + [HD_EVENT]
TRUE_T3 = dict(TRUE_STD, mixed_hd_event_lin={"head_direction", "event_0"})

# Supplementary Fig. 2
SNR_VALUES = [2, 4, 8, 16, 32, 64, 128]
POP_SNR = [
    {"name": "hd_cells", "count": 5, "features": ["head_direction"]},
    {"name": "speed_cells", "count": 5, "features": ["speed"]},
    {"name": "event_cells", "count": 5, "features": ["event_0"]},
    {"name": "mixed_hd_speed_lin", "count": 5, "features": ["head_direction", "speed"], "combination": "weighted_sum"},
    {"name": "mixed_speed_event_lin", "count": 5, "features": ["speed", "event_0"], "combination": "weighted_sum"},
    {"name": "nonselective", "count": 10, "features": []},
]
TRUE_SNR = {"hd_cells": {"head_direction"}, "speed_cells": {"speed"}, "event_cells": {"event_0"},
            "mixed_hd_speed_lin": {"head_direction", "speed"},
            "mixed_speed_event_lin": {"speed", "event_0"}, "nonselective": set()}

# three-feature mixtures
POP3 = [
    {"name": "hd_cells", "count": 5, "features": ["head_direction"]},
    {"name": "speed_cells", "count": 5, "features": ["speed"]},
    {"name": "event_cells", "count": 5, "features": ["event_0"]},
    {"name": "mixed2_lin", "count": 5, "features": ["head_direction", "speed"], "combination": "weighted_sum"},
    {"name": "mixed3_lin", "count": 5, "features": ["head_direction", "speed", "event_0"], "combination": "weighted_sum"},
    {"name": "mixed3_and", "count": 5, "features": ["head_direction", "speed", "event_0"], "combination": "and"},
    {"name": "nonselective", "count": 10, "features": []},
]
TRUE3 = {"hd_cells": {"head_direction"}, "speed_cells": {"speed"}, "event_cells": {"event_0"},
         "mixed2_lin": {"head_direction", "speed"}, "mixed3_lin": {"head_direction", "speed", "event_0"},
         "mixed3_and": {"head_direction", "speed", "event_0"}, "nonselective": set()}

DURATION = 900.0
BASE_KW = dict(mode="two_stage", n_shuffles_stage1=100, n_shuffles_stage2=10000, ds=5, topk1=1, topk2=5,
               shift_window=2, seed=42)
CFG = {"p001": dict(pval_thr=0.001, multicomp_correction=None),
       "holm": dict(pval_thr=0.05, multicomp_correction="holm")}
REP_SEEDS = (42, 123, 7)
T2_SEEDS = SEEDS20[:10]

# GLM-only controls of Section S1.3: interaction blocks and bases for speed
INT_MAP = {"head_direction": {"head_direction"}, "speed": {"speed"}, "event_0": {"event_0"},
           "hd_x_speed": {"head_direction", "speed"}, "speed_x_event": {"speed", "event_0"}}
T24_GROUPS = ["speed_cells", "mixed_hd_speed_lin", "mixed_speed_event_lin"]
T24_BASES = ["quantile10", "linear10", "quantile20", "spline"]

LOG = None


def log(msg):
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    if LOG is not None:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def canon(feat, src):
    f = src.get(feat, feat) if isinstance(feat, str) else feat
    return "head_direction" if f == "head_direction_2d" else f


def run_intense(exp, cfg, representation="by_type"):
    """Detected features per cell without and with the removal of anti-selective pairs, and pair stats."""
    stats, sig, _, _, _ = driada.compute_cell_feat_significance(
        exp, find_optimal_delays=False, with_disentanglement=False,
        multifeature_map=driada.intense.DEFAULT_MULTIFEATURE_MAP,
        remove_anti_selective=False, use_precomputed_stats=False, save_computed_stats=False,
        verbose=False, n_jobs=2, representation=representation, **BASE_KW, **CFG[cfg])
    src = get_representation_sources(exp)
    det, det_f, pairs = {}, {}, []
    for cell in stats:
        for feat in stats[cell]:
            c = canon(feat, src)
            if c not in CANON:
                continue
            st, sg = stats[cell][feat], sig[cell][feat]
            s2 = bool(sg.get("stage2", False))
            sr = st.get("signal_ratio")
            anti = sr is not None and sr <= 1.0
            if s2:
                det.setdefault(int(cell), set()).add(c)
                if not anti:
                    det_f.setdefault(int(cell), set()).add(c)
            pairs.append({"neuron": int(cell), "feature": c, "tested": str(feat), "pval": st.get("pval"),
                          "me": st.get("me"), "me_excess": st.get("me_excess"), "signal_ratio": sr,
                          "stage1": bool(sg.get("stage1", False)), "stage2": s2})
    return det, det_f, pairs


def rows_for(part, seed, tag, variant, dets, groups, truth, extra=None):
    out = []
    for ni, g in groups.items():
        d = dets.get(ni, set())
        r = {"part": part, "seed": seed, "tag": tag, "variant": variant, "neuron": ni, "group": g,
             "f1": prf1(d, truth[g]), "detected": "|".join(sorted(d))}
        if extra:
            r.update(extra)
        out.append(r)
    return out


def glm_dets(exp, variant, seed):
    return {ni: selected_to_features(VARIANTS[variant](exp, ni, seed=seed)["selected"])
            for ni in range(exp.n_cells)}


def write_csv(rows, name):
    path = OUTD / "parts" / name
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def _inter(a, b):
    t = a.shape[0]
    return (a[:, :, None] * b[:, None, :]).reshape(t, -1).astype(np.float32)


def speed_basis(v, kind):
    if kind == "quantile10":
        return _speed_bins(v, 10, "quantile")
    if kind == "linear10":
        return _speed_bins(v, 10, "linear")
    if kind == "quantile20":
        return _speed_bins(v, 20, "quantile")
    from sklearn.preprocessing import SplineTransformer
    return SplineTransformer(n_knots=8, degree=3, include_bias=False).fit_transform(v.reshape(-1, 1)).astype(np.float32)


def t2_controls(exp, seed, groups):
    """GLM-only controls of Section S1.3 on the standard cells.

    The forward selection of the baseline GLM, fit to the rank-Gaussian transform of the
    fluorescence, once with pairwise interaction blocks (head direction x speed, speed x event)
    added to the candidates, and once with each of four bases for speed on the speed cells and
    the two linear mixtures with speed.
    """
    rows = []
    base = build_blocks(exp)
    for ni, g in groups.items():
        if g not in TRUE_STD:
            continue
        y = rank_gaussian(exp.calcium.data[ni].astype(np.float64))
        blocks = dict(base)
        blocks["hd_x_speed"] = _inter(base["head_direction"], base["speed"])
        blocks["speed_x_event"] = _inter(base["speed"], base["event_0"])

        def sc(sel, blocks=blocks, y=y):
            x = np.hstack([blocks[f] for f in sel]).astype(np.float32) if sel else np.zeros((len(y), 0), np.float32)
            return cv_r2(x, y, 10, 1.0, seed)
        sel = _forward(sc, list(blocks.keys()), _wilcoxon_p, alpha=0.05, mc="bonferroni")["selected"]
        det = set().union(*[INT_MAP[f] for f in sel]) if sel else set()
        rows.append({"seed": seed, "method": "GLM_interaction", "basis": "", "neuron": ni, "group": g,
                     "f1": prf1(det & set(CANON), TRUE_STD[g])})
    hd = exp.dynamic_features["head_direction"].data
    sp = exp.dynamic_features["speed"].data
    ev = exp.dynamic_features["event_0"].data.astype(float)
    for kind in T24_BASES:
        for ni, g in groups.items():
            if g not in T24_GROUPS:
                continue
            y = rank_gaussian(exp.calcium.data[ni].astype(np.float64))
            blocks = {"speed": speed_basis(sp, kind), "head_direction": _hd_bins(hd, 24),
                      "event_0": _binary_bins(ev)}

            def sc(sel, blocks=blocks, y=y):
                x = np.hstack([blocks[f] for f in sel]).astype(np.float32) if sel else np.zeros((len(y), 0), np.float32)
                return cv_r2(x, y, 10, 1.0, seed)
            sel = selected_to_features(_forward(sc, list(blocks.keys()), _wilcoxon_p, 0.05, "bonferroni")["selected"])
            rows.append({"seed": seed, "method": "GLM_speed_basis", "basis": kind, "neuron": ni, "group": g,
                         "f1": prf1(sel, TRUE_STD[g])})
    return rows


def unit_t3f1(seed, tag):
    name = f"{tag}_{seed}"
    t = {}
    t0 = time.time()
    exp = gen(POP_T3, seed, duration=DURATION)
    t["gen"] = time.time() - t0
    groups = group_map(POP_T3)
    t0 = time.time()
    d001, d001f, p001 = run_intense(exp, "p001")
    t["intense_p001"] = time.time() - t0
    t0 = time.time()
    dh, dhf, ph = run_intense(exp, "holm")
    t["intense_holm"] = time.time() - t0
    rows_t3 = (rows_for("t3", seed, tag, "INTENSE_p001", d001, groups, TRUE_T3)
               + rows_for("t3", seed, tag, "INTENSE_p001_filter", d001f, groups, TRUE_T3))
    rows_f1 = (rows_for("f1", seed, tag, "INTENSE_holm_filter", dhf, groups, TRUE_T3)
               + rows_for("f1", seed, tag, "INTENSE_holm", dh, groups, TRUE_T3))
    glm = {}
    t0 = time.time()
    for v in GLM3:
        glm[v] = glm_dets(exp, v, seed)
        rows_t3 += rows_for("t3", seed, tag, v, glm[v], groups, TRUE_T3)
    t["glm3"] = time.time() - t0
    write_csv(rows_t3, f"t3_{name}.csv")
    t0 = time.time()
    for v in GLM8:
        if v not in glm:
            glm[v] = glm_dets(exp, v, seed)
    t["glm5"] = time.time() - t0
    for v in GLM8:
        rows_f1 += rows_for("f1", seed, tag, v, glm[v], groups, TRUE_T3)
    write_csv(rows_f1, f"f1_{name}.csv")
    pairs = ([dict(p, seed=seed, tag=tag, cfg="p001", group=groups[p["neuron"]],
                   true=p["feature"] in TRUE_T3[groups[p["neuron"]]]) for p in p001]
             + [dict(p, seed=seed, tag=tag, cfg="holm", group=groups[p["neuron"]],
                     true=p["feature"] in TRUE_T3[groups[p["neuron"]]]) for p in ph])
    write_csv(pairs, f"pairs_{name}.csv")
    if tag == "t3f1" and seed in T2_SEEDS:
        t0 = time.time()
        write_csv(t2_controls(exp, seed, groups), f"t2_{name}.csv")
        t["t2"] = time.time() - t0
    new = {v: round(float(np.mean([prf1(glm[v].get(ni, set()), TRUE_T3[g]) for ni, g in groups.items()
                                   if g == "mixed_hd_event_lin"])), 2) for v in GLM3}
    log(f"  {name}: " + ", ".join(f"{k} {v:.0f} s" for k, v in t.items())
        + f"; HD+event F1 INTENSE_p001 "
        + f"{np.mean([prf1(d001.get(ni, set()), TRUE_T3[g]) for ni, g in groups.items() if g == 'mixed_hd_event_lin']):.2f}"
        + f", GLM {new}")
    return t


def unit_snr(snr, seed):
    name = f"snr{snr}_{seed}"
    t = {}
    t0 = time.time()
    exp = generate_tuned_selectivity_exp(
        population=POP_SNR, duration=DURATION, fps=20.0, seed=seed, n_discrete_features=1,
        baseline_rate=0.1, peak_rate=0.1 * snr, decay_time=2.0, calcium_noise=0.1, verbose=False,
        reconstruct_spikes="wavelet")
    t["gen"] = time.time() - t0
    groups = group_map(POP_SNR)
    t0 = time.time()
    dh, dhf, ph = run_intense(exp, "holm")
    t["intense_holm"] = time.time() - t0
    ex = {"snr": snr}
    rows = (rows_for("snr", seed, "snr", "INTENSE_holm_filter", dhf, groups, TRUE_SNR, ex)
            + rows_for("snr", seed, "snr", "INTENSE_holm", dh, groups, TRUE_SNR, ex))
    t0 = time.time()
    for v in GLM3:
        rows += rows_for("snr", seed, "snr", v, glm_dets(exp, v, seed), groups, TRUE_SNR, ex)
    t["glm3"] = time.time() - t0
    write_csv(rows, f"snr_{name}.csv")
    write_csv([dict(p, seed=seed, snr=snr, cfg="holm", group=groups[p["neuron"]],
                    true=p["feature"] in TRUE_SNR[groups[p["neuron"]]]) for p in ph], f"pairs_{name}.csv")
    log(f"  {name}: " + ", ".join(f"{k} {v:.0f} s" for k, v in t.items()))
    return t


def unit_t31(seed):
    name = f"t31_{seed}"
    t = {}
    t0 = time.time()
    exp = gen(POP3, seed, duration=DURATION)
    t["gen"] = time.time() - t0
    groups = group_map(POP3)
    t0 = time.time()
    dh, dhf, ph = run_intense(exp, "holm")
    t["intense_holm"] = time.time() - t0
    t0 = time.time()
    rh, rhf, _ = run_intense(exp, "holm", representation="raw")
    t["intense_holm_raw"] = time.time() - t0
    rows = (rows_for("t31", seed, "t31", "INTENSE_holm_filter", dhf, groups, TRUE3)
            + rows_for("t31", seed, "t31", "INTENSE_holm", dh, groups, TRUE3)
            + rows_for("t31", seed, "t31", "INTENSE_raw_holm_filter", rhf, groups, TRUE3)
            + rows_for("t31", seed, "t31", "INTENSE_raw_holm", rh, groups, TRUE3))
    t0 = time.time()
    for v in GLM3:
        rows += rows_for("t31", seed, "t31", v, glm_dets(exp, v, seed), groups, TRUE3)
    t["glm3"] = time.time() - t0
    write_csv(rows, f"t31_{name}.csv")
    write_csv([dict(p, seed=seed, cfg="holm", group=groups[p["neuron"]],
                    true=p["feature"] in TRUE3[groups[p["neuron"]]]) for p in ph], f"pairs_{name}.csv")
    log(f"  {name}: " + ", ".join(f"{k} {v:.0f} s" for k, v in t.items()))
    return t


def all_units():
    u = [(f"t3f1_{s}", unit_t3f1, (s, "t3f1")) for s in SEEDS20]
    u += [(f"rep_{s}", unit_t3f1, (s, "rep")) for s in REP_SEEDS]
    u += [(f"snr{snr}_{s}", unit_snr, (snr, s)) for snr in SNR_VALUES for s in SEEDS20[:10]]
    u += [(f"t31_{s}", unit_t31, (s,)) for s in SEEDS20[:15]]
    return u


def smoke_setup():
    """Short recordings, one cell per group and fewer shuffles: a quick check of every code path."""
    global OUTD, DURATION, POP_T3, POP_SNR, POP3, SNR_VALUES, REP_SEEDS, T2_SEEDS
    OUTD = OUTD / "_smoke"
    DURATION = 120.0
    shrink = lambda pop: [dict(g, count=2 if g["name"] == "nonselective" else 1) for g in pop]  # noqa: E731
    POP_T3, POP_SNR, POP3 = shrink(POP_T3), shrink(POP_SNR), shrink(POP3)
    SNR_VALUES = [16]
    REP_SEEDS = ()
    BASE_KW["n_shuffles_stage2"] = 1000


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        smoke_setup()
    for sub in ("parts", "claims", "done", "failed", "logs"):
        (OUTD / sub).mkdir(parents=True, exist_ok=True)
    LOG = OUTD / "logs" / f"worker_{a.worker}.log"
    log(f"worker {a.worker} start; driada {driada.__version__} from {Path(driada.__file__).parent}")
    units = all_units()
    if a.smoke:
        units = [units[0], next(u for u in units if u[0].startswith("snr")),
                 next(u for u in units if u[0].startswith("t31"))]
    n_done = 0
    for name, fn, args in units:
        if (OUTD / "done" / f"{name}.ok").exists():
            continue
        try:
            fd = os.open(OUTD / "claims" / f"{name}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(fd, f"worker {a.worker} pid {os.getpid()} {datetime.now().isoformat(timespec='seconds')}\n".encode())
        os.close(fd)
        log(f"worker {a.worker}: {name} start")
        t0 = time.time()
        try:
            t = fn(*args)
            (OUTD / "done" / f"{name}.ok").write_text(
                json.dumps({"worker": a.worker, "seconds": round(time.time() - t0, 1),
                            "parts": {k: round(v, 1) for k, v in t.items()}}), encoding="utf-8")
            n_done += 1
        except Exception:
            (OUTD / "failed" / f"{name}.err").write_text(traceback.format_exc(), encoding="utf-8")
            log(f"[FAILED] {name}:\n{traceback.format_exc()}")
    log(f"[OK] worker {a.worker} finished, {n_done} units")


if __name__ == "__main__":
    main()
