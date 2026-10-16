# -*- coding: utf-8 -*-
"""
Figure 4 analyses (spatial filtering + reverse analysis) repeated on every FOF
session, with the controls needed to interpret them.

Per session:
  * Isomap embeddings (ds=5, nn=40, dim=2, as in fig4/fig4_composite.py) of
      - all neurons,
      - the spatially filtered set (INTENSE stage-2 significant for place,
        walls or centertrue),
      - N_RANDOM random subsets of the same size drawn from all neurons,
      - N_SHUF circular-shift shuffles of the all-neuron and filtered sets
        (each neuron shifted independently, which destroys neuron-behaviour
        coupling but keeps single-cell statistics).
  * Metrics on the time points Isomap kept (disconnected k-NN graphs drop a few
    frames; positions are indexed by the kept frames, not truncated):
      eff. dim. ratio, in-sample position R^2, cross-validated position R^2
      (5 contiguous blocks), Spearman distance correlation.
  * Reverse analysis on the filtered embedding: neurons significant for an
    embedding component (two-stage INTENSE), reported with no correction and
    with Holm correction, with and without the neurons that built the
    embedding, next to the session base rate of non-selective neurons.

Resumable: finished sessions are read back from the CSV and skipped.
Run:  python multisession_fig4_controls.py [mouse ids...]
Env:  N_RANDOM, N_SHUF, N_SHUF2, MIN_FILT, OUT_DIR, SMOKE=1 (tiny settings)
"""
import csv
import glob
import json
import os
import subprocess
import sys
import time
import warnings
import zlib

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, t as tdist
from sklearn.linear_model import LinearRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import elife_paths

from driada.experiment import load_exp_from_aligned_data
from driada.dim_reduction.data import MVData
from driada.dim_reduction.dr_base import merge_params_with_defaults
from driada.intense import compute_embedding_selectivity
from driada.intense.correction import get_multicomp_correction_thr

SYNC_DIR = elife_paths.SYNC_DIR
INTENSE_DIR = elife_paths.INTENSE_DIR
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(elife_paths.OUTPUT_DIR, "multisession", "results"))

SMOKE = bool(os.environ.get("SMOKE"))
DS, NN, DIM = 5, 40, 2
N_RANDOM = int(os.environ.get("N_RANDOM", 2 if SMOKE else 20))
N_SHUF = int(os.environ.get("N_SHUF", 1 if SMOKE else 5))
N_SHUF1 = 100
N_SHUF2 = int(os.environ.get("N_SHUF2", 200 if SMOKE else 10000))
MIN_FILT = int(os.environ.get("MIN_FILT", 10))
MAX_LOST_FRAC = 0.05  # reverse analysis is skipped if Isomap dropped more frames
PVAL_THR = 0.05
SEED = 42
# k-NN graph symmetrisation for Isomap: 'intersection' (mutual neighbours, driada default)
# or 'union' (an edge if either point is among the other's neighbours, as in textbook Isomap)
SYMMETRIZATION = os.environ.get("SYMMETRIZATION", "intersection")

# by_type representations name the place feature place_quad, direction
# features *_harm2 and speed speed_quad
SPATIAL_FEATURES = ["place_quad", "place", "walls", "centertrue"]
FEATURE_GROUPS = {
    "place": ["place_quad", "place"], "HD": ["bodydirection_harm2", "headdirection_harm2"],
    "speed": ["speed_quad", "speed"], "locomotion": ["locomotion", "walk"],
    "rest": ["rest", "freezing"], "rear": ["rear"],
    "spatial": ["walls", "centertrue"],
}
FORCE_CONTINUOUS = ["x", "y", "speed", "bodydirection", "headdirection"]
STATIC_FEATURES = {"fps": 30, "t_rise_sec": 0.05, "t_off_sec": 1.5}


def is_sig(entry):
    return entry.get("stage2", entry.get("stage1", False))


def spatial_idx(n, sig):
    return [i for i in range(n) if str(i) in sig and
            any(f in sig[str(i)] and is_sig(sig[str(i)][f]) for f in SPATIAL_FEATURES)]


def primary_selectivity(i, sig, stats):
    c = str(i)
    if c not in sig:
        return "non-sel."
    best, best_mi = None, -np.inf
    for g, feats in FEATURE_GROUPS.items():
        for f in feats:
            if f in sig[c] and is_sig(sig[c][f]) and stats[c][f]["me"] > best_mi:
                best_mi, best = stats[c][f]["me"], g
    return best or "non-sel."


def embed(cal):
    """Isomap coordinates (dim x T_kept) and the indices of the kept frames."""
    prm = merge_params_with_defaults("isomap", {"e_params": {"dim": DIM},
                                                "g_params": {"nn": NN, "symmetrization": SYMMETRIZATION}})
    emb = MVData(cal, downsampling=1).get_embedding(e_params=prm["e_params"], g_params=prm["g_params"],
                                                    m_params=prm["m_params"])
    lost = getattr(emb.graph, "lost_nodes", set()) or set()
    kept = np.array(sorted(set(range(cal.shape[1])) - set(lost)))
    assert emb.coords.shape[1] == len(kept), (emb.coords.shape, len(kept))
    return emb.coords, kept


def metrics(coords, kept, x, y):
    pos = np.c_[x[kept], y[kept]]
    c = coords.T
    v = np.var(c, axis=0)
    dimratio = v.min() / v.max()
    r2_in = LinearRegression().fit(c, pos).score(c, pos)
    # held-out R^2 on 5 contiguous time blocks
    folds = np.array_split(np.arange(len(pos)), 5)
    ss_res = ss_tot = 0.0
    for te in folds:
        tr = np.setdiff1d(np.arange(len(pos)), te)
        pred = LinearRegression().fit(c[tr], pos[tr]).predict(c[te])
        ss_res += ((pos[te] - pred) ** 2).sum()
        ss_tot += ((pos[te] - pos[tr].mean(0)) ** 2).sum()
    r2_cv = 1 - ss_res / ss_tot
    rng = np.random.RandomState(SEED)
    sub = rng.choice(len(pos), min(2000, len(pos)), replace=False)
    dcorr = spearmanr(pdist(c[sub]), pdist(pos[sub]))[0]
    return dict(dimratio=dimratio, r2=r2_in, r2cv=r2_cv, dcorr=dcorr)


def roll_shuffle(cal, rng):
    T = cal.shape[1]
    lo = max(1, T // 10)
    return np.stack([np.roll(row, rng.integers(lo, T - lo)) for row in cal])


def reverse_analysis(exp, coords, kept, T, sig, stats, filt, n):
    """Percent of embedding-informative neurons whose primary selectivity is non-sel."""
    # INTENSE needs a value for every frame: fill frames Isomap dropped with the
    # nearest kept frame in time (only done when the dropped fraction is small).
    full = np.empty((DIM, T))
    nearest = kept[np.clip(np.searchsorted(kept, np.arange(T)), 0, len(kept) - 1)]
    idx = np.searchsorted(kept, nearest)
    full[:] = coords[:, idx]
    exp.store_embedding(full.T, method_name="isomap", data_type="calcium",
                        metadata={"ds": DS, "n_components": DIM, "n_neighbors": NN})
    res = compute_embedding_selectivity(
        exp, embedding_methods=["isomap"], mode="two_stage",
        n_shuffles_stage1=N_SHUF1, n_shuffles_stage2=N_SHUF2,
        multicomp_correction=None, pval_thr=PVAL_THR, find_optimal_delays=True,
        remove_anti_selective=False, verbose=False, seed=SEED)["isomap"]
    raw = sorted({int(i) for c in range(DIM) for i in res["component_selectivity"][c]})
    pvals = [s.get("pval") for nd in res["stats"].values() for s in nd.values()
             if isinstance(s, dict) and s.get("pval") is not None]
    holm_thr = get_multicomp_correction_thr(PVAL_THR, mode="holm", all_pvals=pvals) if pvals else 0
    holm = sorted({int(nid) for nid, nd in res["stats"].items() for s in nd.values()
                   if isinstance(s, dict) and s.get("pval") is not None and s["pval"] < holm_thr
                   and int(nid) in raw})
    fset = set(filt)
    base = 100.0 * sum(primary_selectivity(i, sig, stats) == "non-sel." for i in range(n)) / n

    def pct(ids):
        return (100.0 * sum(primary_selectivity(i, sig, stats) == "non-sel." for i in ids) / len(ids)
                if ids else np.nan)

    out = dict(base_pct_nonsel=base, holm_thr=holm_thr)
    for tag, ids in (("raw", raw), ("holm", holm)):
        ext = [i for i in ids if i not in fset]
        out.update({f"n_inf_{tag}": len(ids), f"pct_nonsel_{tag}": pct(ids),
                    f"n_inf_{tag}_excl": len(ext), f"pct_nonsel_{tag}_excl": pct(ext)})
    return out


def process(session):
    t0 = time.time()
    rng = np.random.default_rng(SEED + zlib.crc32(session.encode()) % 10000)
    data = dict(np.load(os.path.join(SYNC_DIR, f"{session}_aligned.npz")))
    track, animal, day = session.split("_")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp = load_exp_from_aligned_data(
            data_source="FOF", exp_params={"track": track, "animal_id": animal, "session": day},
            data=data, force_continuous=FORCE_CONTINUOUS, static_features=STATIC_FEATURES,
            verbose=False, seed=SEED)
    cal = exp.calcium.scdata[:, ::DS]
    x = exp.dynamic_features["x"].data[::DS]
    y = exp.dynamic_features["y"].data[::DS]
    n, T = cal.shape
    res = np.load(os.path.join(INTENSE_DIR, f"{session}_results.npz"), allow_pickle=True)
    sig = json.loads(str(res["_significance_json"][0]))
    stats = json.loads(str(res["_stats_json"][0]))
    filt = spatial_idx(n, sig)
    row = dict(session=session, mouse=animal, day=day, n_neurons=n, n_filtered=len(filt), T=T)
    if len(filt) < MIN_FILT:
        row["status"] = "too_few_filtered"
        return row

    c_all, k_all = embed(cal)
    c_f, k_f = embed(cal[filt])
    row["lost_frac_all"] = 1 - len(k_all) / T
    row["lost_frac_filt"] = 1 - len(k_f) / T
    for tag, (c, k) in (("all", (c_all, k_all)), ("filt", (c_f, k_f))):
        for m, v in metrics(c, k, x, y).items():
            row[f"{m}_{tag}"] = v

    # random subsets of the same size as the filtered set
    rnd = []
    for _ in range(N_RANDOM):
        ids = rng.choice(n, len(filt), replace=False)
        rnd.append(metrics(*embed(cal[ids]), x, y))
    # circular-shift shuffles
    sh = {"all": [], "filt": []}
    for _ in range(N_SHUF):
        sh["all"].append(metrics(*embed(roll_shuffle(cal, rng)), x, y))
        sh["filt"].append(metrics(*embed(roll_shuffle(cal[filt], rng)), x, y))
    for m in ("dimratio", "r2", "r2cv", "dcorr"):
        vals = np.array([r[m] for r in rnd])
        row[f"{m}_rand_mean"] = vals.mean()
        row[f"{m}_rand_sd"] = vals.std(ddof=1) if len(vals) > 1 else np.nan
        # share of random subsets at least as good as the filtered set
        row[f"{m}_rand_p"] = (1 + (vals >= row[f"{m}_filt"]).sum()) / (1 + len(vals))
        for tag in ("all", "filt"):
            row[f"{m}_shuf_{tag}"] = np.mean([r[m] for r in sh[tag]])

    if row["lost_frac_filt"] <= MAX_LOST_FRAC:
        try:
            row.update(reverse_analysis(exp, c_f, k_f, T, sig, stats, filt, n))
        except Exception as e:  # keep the quality metrics for this session
            row["reverse_error"] = f"{type(e).__name__}: {str(e)[:100]}"
    else:
        row["reverse_error"] = "too many frames dropped by Isomap"
    row["status"] = "ok"
    row["seconds"] = round(time.time() - t0)
    return row


def ci95(a):
    a = np.asarray([v for v in a if v not in (None, "") and not np.isnan(float(v))], float)
    if len(a) < 2:
        return np.nan, np.nan, np.nan
    m = a.mean()
    h = tdist.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return m, m - h, m + h


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    jsonl = os.path.join(OUT_DIR, "sessions.jsonl")
    done = set()
    if os.path.exists(jsonl):
        done = {json.loads(l)["session"] for l in open(jsonl, encoding="utf-8") if l.strip()}
    names = sorted(os.path.basename(f).replace("_aligned.npz", "")
                   for f in glob.glob(os.path.join(SYNC_DIR, "FOF_*_aligned.npz")))
    names = [s for s in names if os.path.exists(os.path.join(INTENSE_DIR, f"{s}_results.npz"))]
    if sys.argv[1:]:
        names = [s for s in names if any(m in s for m in sys.argv[1:])]
    todo = [s for s in names if s not in done]
    import driada
    print(f"[INFO] driada from {os.path.dirname(driada.__file__)}", flush=True)
    print(f"[INFO] symmetrization={SYMMETRIZATION}; {len(names)} sessions, {len(todo)} to do; N_RANDOM={N_RANDOM} N_SHUF={N_SHUF} "
          f"N_SHUF2={N_SHUF2} MIN_FILT={MIN_FILT}", flush=True)
    for s in todo:
        try:
            row = process(s)
        except Exception as e:
            row = dict(session=s, status=f"FAILED {type(e).__name__}: {str(e)[:150]}")
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                                for k, v in row.items()}) + "\n")
        print(f"[{row.get('status')}] {s} n_filt={row.get('n_filtered')} "
              f"r2cv all/filt/rand={row.get('r2cv_all', float('nan')):.3f}/"
              f"{row.get('r2cv_filt', float('nan')):.3f}/{row.get('r2cv_rand_mean', float('nan')):.3f} "
              f"t={row.get('seconds')}s", flush=True)

    aggregate([jsonl], OUT_DIR)


def aggregate(jsonl_paths, out_dir):
    """Write sessions.csv and summary.txt (mean and 95% CI over sessions and over mice)."""
    rows = [json.loads(l) for p in jsonl_paths for l in open(p, encoding="utf-8") if l.strip()]
    ok = [r for r in rows if r.get("status") == "ok"]
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(out_dir, "sessions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    lines = [f"sessions ok: {len(ok)} of {len(rows)}; mice: {len({r['mouse'] for r in ok})}"]
    num_keys = [k for k in keys if any(isinstance(r.get(k), (int, float)) for r in ok)
                and k != "seconds"]
    for k in num_keys:
        vals = [r[k] for r in ok if isinstance(r.get(k), (int, float))]
        per_mouse = {}
        for r in ok:
            if isinstance(r.get(k), (int, float)):
                per_mouse.setdefault(r["mouse"], []).append(r[k])
        ms = ci95(vals)
        mm = ci95([np.nanmean(v) for v in per_mouse.values()])
        lines.append(f"{k:28s} sessions {ms[0]:.3f} [{ms[1]:.3f}, {ms[2]:.3f}] n={len(vals)} | "
                     f"mice {mm[0]:.3f} [{mm[1]:.3f}, {mm[2]:.3f}] n={len(per_mouse)}")
    open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    # Windows puts windowless background processes into power-saving mode
    if os.name == "nt":
        subprocess.run([sys.executable, os.path.join(HERE, "set_high_qos.py"), str(os.getpid())])
    if os.environ.get("MERGE_DIRS"):
        dirs = os.environ["MERGE_DIRS"].split(";")
        os.makedirs(OUT_DIR, exist_ok=True)
        aggregate([os.path.join(d, "sessions.jsonl") for d in dirs
                   if os.path.exists(os.path.join(d, "sessions.jsonl"))], OUT_DIR)
    else:
        main()
