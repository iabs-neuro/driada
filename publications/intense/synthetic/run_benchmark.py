#!/usr/bin/env python3
"""Synthetic benchmark of Figure 4 recomputed on DRIADA 1.2.2 with seed 42.

Every Figure 4 quantity (F1 heatmaps B-C, precision-recall maps D-E and the
numbers quoted in the synthetic-validation Results) is derived from the
per-configuration results this script writes.

Data
----
``generate_synthetic_data`` of DRIADA 1.2.2 with the parameters of the Methods
(500 neurons, 20 features of one type, 2400 s at 20 Hz, Hurst 0.3, amplitudes
0.5-2, noise 0.1, decay 2 s, baseline 0.1 Hz, peak = baseline x SNR, at most 10
islands of mean 5 s), on the grid SNR {2..64} x p_skip {0..0.8} x 5
realizations. Features are generated here with explicit seeds and passed in as
``pregenerated_features``: the generator itself draws discrete features without
a seed, which would make the benchmark irreproducible. Realization r uses the
data seed 42 + 1000 r, so realizations share no feature, while all grid cells of
one realization share features, receptive fields and spike-generator seeds
(common random numbers: grid cells differ only in SNR and p_skip).

Methods
-------
Continuous: correlation (Pearson t-test, p < 1e-50), MI without shuffles (GCMI
converted to an effective correlation, one-sided t-test, p < 1e-50),
correlation + shuffles (INTENSE pipeline, metric ``fast_pearsonr``), INTENSE.
Discrete: average (Welch t-test, p < 0.01 / 10 000), MI without shuffles
(G-test, p < 1e-50), average + shuffles (INTENSE pipeline, metric ``av``),
INTENSE.

The shuffle-based methods run ``compute_cell_feat_significance`` exactly as the
real NOF data were analysed: two stages of 100 / 10 000 shuffles, p < 0.001 per
pair without correction, ds = 5, delay search within +-2 s, by_type
representation, seed 42. The excess-information criterion (MI above chance
> 0.04 bits) is applied afterwards, as in the paper, and only to MI: for the
correlation and activity-ratio metrics it has no counterpart in bits. The
anti-selectivity filter is not applied inside the pipeline; its signal ratio is
saved so that the filtered variant is reported alongside (for continuous
features it compares activity above and below the median, which removes neurons
tuned to the lower half of the feature by construction of this benchmark).

Outputs (``<out>/<mode>/``)
---------------------------
raw/<type>/snr<S>_skip<P>_r<R>.json   counts, timings, peak memory per config
raw/<type>/snr<S>_skip<P>_r<R>.npz    per-pair arrays of the shuffle methods
summary_long.csv                      one row per config x method x variant
summary_table.csv                     means over realizations, F1 from means
numbers_text.csv                      quantities quoted in the Results text
run_info.json                         settings, versions, wall time

Usage
-----
    python run_benchmark.py --quick              # 4 configs per type, ~minutes
    python run_benchmark.py --workers 3          # full grid, 540 configs
    python run_benchmark.py --aggregate-only     # rebuild CSVs from raw/
"""

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402
import paths_v11 as V  # noqa: E402

P.use_driada(tools=False)      # the DRIADA release of [driada] src, version checked

import numpy as np
import pandas as pd
import psutil
from scipy.stats import chi2 as chi2_dist
from scipy.stats import t as t_dist

import driada

# The benchmark is only meaningful for the code the paper is computed with.
assert driada.__version__ == "1.2.2", driada.__version__
_driada_path = Path(driada.__file__).resolve()

from driada.experiment.exp_base import Experiment  # noqa: E402
from driada.experiment.synthetic import (  # noqa: E402
    generate_binary_time_series,
    generate_fbm_time_series,
    generate_synthetic_data,
)
from driada.information.info_base import TimeSeries, get_mi  # noqa: E402

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# Settings (Methods: synthetic data generation, validation dataset)
# ----------------------------------------------------------------------------
SEED = 42
REALIZATION_STRIDE = 1000
SNR_VALUES = [2, 4, 8, 16, 32, 64]
SKIP_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
N_REALIZATIONS = 5
QUICK_SNR = [4, 64]
QUICK_SKIP = [0.0, 0.8]

N_FEATURES = 20
N_NEURONS = 500
DURATION = 2400
FPS = 20
HURST = 0.3
BASELINE_RATE = 0.1
AMPLITUDE_RANGE = (0.5, 2.0)
CALCIUM_NOISE = 0.1
DECAY_TIME = 2.0
AVG_ISLANDS = 10
AVG_DURATION = 5
T_RISE = 0.25

# INTENSE settings of the real-data analysis (Methods: implementation)
PAPER = dict(
    mode="two_stage",
    n_shuffles_stage1=100,
    n_shuffles_stage2=10000,
    pval_thr=0.001,
    multicomp_correction=None,
    ds=5,
    topk1=1,
    topk2=5,
    shift_window=2,
    representation="by_type",
    seed=SEED,
    engine="auto",
)
MI_EXCESS_THR = 0.04  # bits above the mean of the Stage 2 shuffles

# Comparison methods without shuffles
NAIVE_PVAL_THR = 1e-50
NAIVE_AV_ALPHA = 0.01  # Bonferroni over all neuron-feature pairs

METHODS = {
    "c": ["corr", "mi_naive", "corr_shuffles", "intense"],
    "d": ["average", "mi_naive", "average_shuffles", "intense"],
}
SHUFFLE_METRIC = {"corr_shuffles": "fast_pearsonr", "average_shuffles": "av", "intense": "mi"}
# Correlation and activity-ratio nulls are not bounded at zero; the normal fit
# is the one the original comparison used for them.
DISTR = {"fast_pearsonr": "norm", "av": "norm", "mi": "gamma_zi"}
# The variant each method is reported with in the figure.
MAIN_VARIANT = {
    "corr": "naive", "average": "naive", "mi_naive": "naive",
    "corr_shuffles": "stage2", "average_shuffles": "stage2", "intense": "stage2_mi04",
}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def make_features(ftype, data_seed):
    length = int(DURATION * FPS)
    feats = []
    for i in range(N_FEATURES):
        if ftype == "c":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # global-seed notice
                feats.append(generate_fbm_time_series(length, HURST, seed=data_seed + i))
        else:
            feats.append(generate_binary_time_series(
                length, AVG_ISLANDS, AVG_DURATION * FPS, seed=data_seed + i))
    return feats


def make_data(ftype, snr, skip, r):
    data_seed = SEED + REALIZATION_STRIDE * r
    feats, calcium, gt = generate_synthetic_data(
        nfeats=N_FEATURES, nneurons=N_NEURONS, ftype=ftype, duration=DURATION,
        seed=data_seed, sampling_rate=FPS, baseline_rate=BASELINE_RATE,
        peak_rate=BASELINE_RATE * snr, skip_prob=skip, hurst=HURST,
        calcium_amplitude_range=AMPLITUDE_RANGE, decay_time=DECAY_TIME,
        avg_islands=AVG_ISLANDS, avg_duration=AVG_DURATION,
        calcium_noise=CALCIUM_NOISE, verbose=False,
        pregenerated_features=make_features(ftype, data_seed),
    )
    return feats, calcium, gt.T.astype(bool), data_seed  # gt -> (neurons, features)


def feature_names(ftype):
    return [f"{'fbm' if ftype == 'c' else 'event'}_{i}" for i in range(N_FEATURES)]


def make_exp(ftype, feats, calcium):
    # Built as generate_tuned_selectivity_exp builds its synthetic experiments.
    dyn = {
        name: TimeSeries(data=f, discrete=(ftype == "d"), name=name)
        for name, f in zip(feature_names(ftype), feats)
    }
    return Experiment(
        signature=f"fig4_synthetic_{ftype}",
        calcium=calcium, spikes=None,
        static_features={"fps": FPS, "t_rise_sec": T_RISE, "t_off_sec": DECAY_TIME},
        dynamic_features=dyn,
        exp_identificators={"type": "fig4_benchmark"},
        verbose=False, optimize_kinetics=False, reconstruct_spikes=None,
    )


# ----------------------------------------------------------------------------
# Methods without shuffles (on the same ds-downsampled data)
# ----------------------------------------------------------------------------
def _zscore_rows(x):
    z = x - x.mean(axis=1, keepdims=True)
    sd = z.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return z / sd


def naive_corr(calcium, feats, ds):
    ca, ft = calcium[:, ::ds], np.asarray(feats)[:, ::ds]
    n = ca.shape[1]
    r = _zscore_rows(ca) @ _zscore_rows(ft).T / n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt((n - 2) / (1 - r ** 2))
    p = 2 * t_dist.sf(np.abs(t), df=n - 2)
    return np.nan_to_num(p, nan=1.0) < NAIVE_PVAL_THR


def naive_average(calcium, feats, ds):
    ca = calcium[:, ::ds]
    n_neurons = ca.shape[0]
    thr = NAIVE_AV_ALPHA / (n_neurons * len(feats))
    p = np.ones((n_neurons, len(feats)))
    for j, f in enumerate(feats):
        on = np.asarray(f)[::ds] == 1
        n1, n0 = on.sum(), (~on).sum()
        if n1 < 2 or n0 < 2:
            continue
        a, b = ca[:, on], ca[:, ~on]
        v1, v0 = a.var(axis=1, ddof=1) / n1, b.var(axis=1, ddof=1) / n0
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (a.mean(axis=1) - b.mean(axis=1)) / np.sqrt(v1 + v0)
            dof = (v1 + v0) ** 2 / (v1 ** 2 / (n1 - 1) + v0 ** 2 / (n0 - 1))
        p[:, j] = np.nan_to_num(2 * t_dist.sf(np.abs(t), df=dof), nan=1.0)
    return p < thr


def naive_mi(exp, ftype, calcium, feats, ds):
    names = feature_names(ftype)
    n_neurons = calcium.shape[0]
    if ftype == "c":
        ca, ft = calcium[:, ::ds], np.asarray(feats)[:, ::ds]
        n = ca.shape[1]
        signs = np.sign(_zscore_rows(ca) @ _zscore_rows(ft).T)
        mi = np.zeros((n_neurons, len(names)))
        for i in range(n_neurons):
            ca_ts = exp.neurons[i].ca
            for j, name in enumerate(names):
                mi[i, j] = get_mi(ca_ts, exp.dynamic_features[name], ds=ds)
        # GCMI is in bits: rho^2 = 1 - 2^(-2 I).
        rho = signs * np.sqrt(np.clip(1 - 2.0 ** (-2 * mi), 0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
        p = np.nan_to_num(t_dist.sf(t, df=n - 2), nan=1.0)
        p[np.abs(rho) >= 1.0] = 0.0
        return p < NAIVE_PVAL_THR
    # Discrete: G-test on quantile-binned calcium, Sturges bins.
    ca = calcium[:, ::ds]
    n = ca.shape[1]
    n_bins = int(np.ceil(np.log2(n)) + 1)
    q = np.linspace(0, 1, n_bins + 1)[1:-1]
    binned = np.stack([np.digitize(row, np.quantile(row, q)) for row in ca])
    p = np.ones((n_neurons, len(feats)))
    for j, f in enumerate(feats):
        fv = np.asarray(f)[::ds]
        vals = np.unique(fv)
        if len(vals) < 2:
            continue
        for i in range(n_neurons):
            ct = np.stack([np.bincount(binned[i][fv == v], minlength=n_bins) for v in vals])
            ct = ct[:, ct.sum(axis=0) > 0]
            ct = ct[ct.sum(axis=1) > 0]
            if min(ct.shape) < 2:
                continue
            exp_ct = ct.sum(axis=1, keepdims=True) * ct.sum(axis=0, keepdims=True) / ct.sum()
            with np.errstate(divide="ignore", invalid="ignore"):
                g = 2 * np.sum(np.where(ct > 0, ct * np.log(ct / exp_ct), 0.0))
            p[i, j] = chi2_dist.sf(g, (ct.shape[0] - 1) * (ct.shape[1] - 1))
    return p < NAIVE_PVAL_THR


# ----------------------------------------------------------------------------
# Shuffle-based methods (INTENSE pipeline)
# ----------------------------------------------------------------------------
def run_pipeline(exp, ftype, metric, delays):
    stats, sig, info, _, _ = driada.compute_cell_feat_significance(
        exp, metric=metric, metric_distr_type=DISTR[metric],
        find_optimal_delays=delays, with_disentanglement=False,
        remove_anti_selective=False, use_precomputed_stats=False,
        save_computed_stats=False, enable_parallelization=False, verbose=False,
        **PAPER,
    )
    names = feature_names(ftype)
    n = len(stats)
    out = {k: np.full((n, len(names)), np.nan, dtype=np.float32)
           for k in ("pval", "me", "me_excess", "signal_ratio")}
    out["stage1"] = np.zeros((n, len(names)), bool)
    out["stage2"] = np.zeros((n, len(names)), bool)
    out["delay"] = np.zeros((n, len(names)), np.int16)
    tested = list(next(iter(stats.values())).keys())
    for jj, feat in enumerate(tested):
        # by_type renames linear features (fbm_3 -> fbm_3_quad); map by index.
        j = int(str(feat).split("_")[1])
        assert names[j] in str(feat), (feat, names[j])
        for i, cell in enumerate(stats):
            st, sg = stats[cell][feat], sig[cell][feat]
            for k in ("pval", "me", "me_excess", "signal_ratio"):
                v = st.get(k)
                out[k][i, j] = np.nan if v is None else v
            out["stage1"][i, j] = bool(sg.get("stage1", False))
            out["stage2"][i, j] = bool(sg.get("stage2", False))
            out["delay"][i, j] = int(info["optimal_delays"][i, jj])
    out["tested_features"] = np.array([str(f) for f in tested])
    return out


def variants(method, arr):
    """Detection matrices of one method under each reported criterion."""
    if not isinstance(arr, dict):
        return {"naive": arr}
    anti_ok = np.nan_to_num(arr["signal_ratio"], nan=0.0) > 1.0
    v = {"stage1": arr["stage1"], "stage2": arr["stage2"],
         "stage2_antisel": arr["stage2"] & anti_ok}
    if method == "intense":
        thr = np.nan_to_num(arr["me_excess"], nan=-np.inf) > MI_EXCESS_THR
        v["stage2_mi04"] = arr["stage2"] & thr
        v["stage2_mi04_antisel"] = arr["stage2"] & thr & anti_ok
    return v


def counts(det, gt):
    tp = int(np.sum(det & gt))
    fp = int(np.sum(det & ~gt))
    fn = int(np.sum(~det & gt))
    return dict(tp=tp, fp=fp, fn=fn, n_detected=tp + fp, n_true=int(gt.sum()))


# ----------------------------------------------------------------------------
# One configuration
# ----------------------------------------------------------------------------
def config_stem(snr, skip, r):
    return f"snr{snr}_skip{skip:.1f}_r{r}"


def run_config(ftype, snr, skip, r, raw_dir, delays):
    stem = config_stem(snr, skip, r)
    fjson = raw_dir / f"{stem}.json"
    if fjson.exists():
        return json.loads(fjson.read_text())
    t_all = time.time()
    t0 = time.time()
    feats, calcium, gt, data_seed = make_data(ftype, snr, skip, r)
    exp = make_exp(ftype, feats, calcium)
    timing = {"generate": time.time() - t0}
    ds = PAPER["ds"]
    res, arrays = {}, {}
    for method in METHODS[ftype]:
        t0 = time.time()
        if method == "corr":
            det = naive_corr(calcium, feats, ds)
        elif method == "average":
            det = naive_average(calcium, feats, ds)
        elif method == "mi_naive":
            det = naive_mi(exp, ftype, calcium, feats, ds)
        else:
            det = run_pipeline(exp, ftype, SHUFFLE_METRIC[method], delays)
            arrays[method] = det
        timing[method] = time.time() - t0
        res[method] = {name: counts(d, gt) for name, d in variants(method, det).items()}
    np.savez_compressed(
        raw_dir / f"{stem}.npz", gt=gt,
        **{f"{m}__{k}": v for m, a in arrays.items() for k, v in a.items()},
    )
    mem = psutil.Process().memory_info()
    out = dict(
        ftype=ftype, snr=snr, skip_prob=skip, realization=r, data_seed=data_seed,
        intense_seed=SEED, find_optimal_delays=delays, methods=res,
        timing_sec={k: round(v, 2) for k, v in timing.items()},
        wall_sec=round(time.time() - t_all, 2),
        peak_rss_gb=round(getattr(mem, "peak_wset", mem.rss) / 2 ** 30, 3),
    )
    tmp = fjson.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, fjson)
    return out


def _worker(args):
    ftype, snr, skip, r, raw_dir, delays = args
    return run_config(ftype, snr, skip, r, Path(raw_dir), delays)


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
def _pr(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan
    return precision, recall


def _f1(p, r):
    if not np.isfinite(r) or r == 0:
        return 0.0
    if not np.isfinite(p) or p + r == 0:
        return np.nan
    return 2 * p * r / (p + r)


def aggregate(mode_dir):
    rows = []
    for fjson in sorted((mode_dir / "raw").glob("*/*.json")):
        d = json.loads(fjson.read_text())
        ftype = d["ftype"]
        for m_idx, method in enumerate(METHODS[ftype]):
            for variant, c in d["methods"][method].items():
                p, r = _pr(c["tp"], c["fp"], c["fn"])
                rows.append(dict(
                    ftype=ftype, method=method, method_idx=m_idx, variant=variant,
                    main=(variant == MAIN_VARIANT[method]), snr=d["snr"],
                    skip_prob=d["skip_prob"], realization=d["realization"],
                    precision=p, recall=r, **c, wall_sec=d["wall_sec"],
                ))
    long = pd.DataFrame(rows)
    long.to_csv(mode_dir / "summary_long.csv", index=False)
    keys = ["ftype", "method", "method_idx", "variant", "main", "snr", "skip_prob"]
    tab = long.groupby(keys, as_index=False).agg(
        precision=("precision", "mean"), recall=("recall", "mean"),
        n_detected=("n_detected", "mean"), tp=("tp", "mean"), fp=("fp", "mean"),
        n_realizations=("realization", "nunique"),
        n_no_detection=("n_detected", lambda x: int((x == 0).sum())),
    )
    # Methods: F1 from the precision and recall averaged over realizations.
    tab["f1"] = [_f1(p, r) for p, r in zip(tab.precision, tab.recall)]
    tab.to_csv(mode_dir / "summary_table.csv", index=False)
    text_numbers(tab, mode_dir)
    return long, tab


def text_numbers(tab, mode_dir):
    """Quantities quoted in the Results on the synthetic benchmark."""
    rows = []

    def cell(ftype, method, variant, snr, skip, col):
        q = tab[(tab.ftype == ftype) & (tab.method == method) & (tab.variant == variant)
                & (tab.snr == snr) & np.isclose(tab.skip_prob, skip)]
        return float(q[col].iloc[0]) if len(q) else np.nan

    def add(key, value, ftype, method, variant, where):
        rows.append(dict(key=key, value=value, ftype=ftype, method=method,
                         variant=variant, where=where))

    for ftype in ("c", "d"):
        for method in METHODS[ftype]:
            v = MAIN_VARIANT[method]
            q = tab[(tab.ftype == ftype) & (tab.method == method) & (tab.variant == v)]
            add("precision_mean_grid", q.precision.mean(), ftype, method, v, "mean over grid")
            add("recall_mean_grid", q.recall.mean(), ftype, method, v, "mean over grid")
            add("recall_max_grid", q.recall.max(), ftype, method, v, "max over grid")
            add("f1_mean_grid", q.f1.mean(), ftype, method, v, "mean over grid")
            for snr in sorted(tab.snr.unique()):
                for col in ("precision", "recall", "f1"):
                    add(f"{col}_snr{snr}_skip0", cell(ftype, method, v, snr, 0.0, col),
                        ftype, method, v, f"SNR={snr}, p_skip=0")
            for skip in sorted(tab.skip_prob.unique()):
                for col in ("precision", "recall", "f1"):
                    add(f"{col}_snr64_skip{skip:.1f}", cell(ftype, method, v, 64, skip, col),
                        ftype, method, v, f"SNR=64, p_skip={skip:.1f}")
        # Two-stage testing: Stage 1 candidates and precision Stage 1 -> final.
        q = tab[(tab.ftype == ftype) & (tab.method == "intense") & (tab.variant == "stage1")]
        frac = q.n_detected / (N_NEURONS * N_FEATURES)
        add("stage1_candidate_fraction_min", frac.min(), ftype, "intense", "stage1", "min over grid")
        add("stage1_candidate_fraction_max", frac.max(), ftype, "intense", "stage1", "max over grid")
        for variant in ("stage1", "stage2", "stage2_mi04", "stage2_mi04_antisel"):
            for col in ("precision", "recall"):
                add(f"{col}_snr64_skip0", cell(ftype, "intense", variant, 64, 0.0, col),
                    ftype, "intense", variant, "SNR=64, p_skip=0")
    pd.DataFrame(rows).to_csv(mode_dir / "numbers_text.csv", index=False)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help=f"SNR {QUICK_SNR} x p_skip {QUICK_SKIP} x 1 realization, both types")
    ap.add_argument("--workers", type=int, default=1, help="configs run in parallel")
    ap.add_argument("--types", nargs="+", choices=["c", "d"], default=["c", "d"])
    ap.add_argument("--no-delays", action="store_true",
                    help="fix the delay at zero instead of searching +-2 s as for the real data")
    ap.add_argument("--out", default=str(V.get("synthetic_dir")))
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()

    mode = "quick" if a.quick else "full"
    if a.no_delays:
        mode += "_nodelay"
    mode_dir = Path(a.out) / mode
    delays = not a.no_delays
    snrs, skips, n_real = (QUICK_SNR, QUICK_SKIP, 1) if a.quick else (SNR_VALUES, SKIP_VALUES, N_REALIZATIONS)

    if not a.aggregate_only:
        tasks = []
        for ftype in a.types:
            raw_dir = mode_dir / "raw" / ftype
            raw_dir.mkdir(parents=True, exist_ok=True)
            for r in range(n_real):
                for snr in snrs:
                    for skip in skips:
                        tasks.append((ftype, snr, skip, r, str(raw_dir), delays))
        todo = [t for t in tasks
                if not (Path(t[4]) / f"{config_stem(t[1], t[2], t[3])}.json").exists()]
        print(f"driada {driada.__version__} from {_driada_path}", flush=True)
        print(f"mode={mode}: {len(tasks)} configs, {len(todo)} to run, "
              f"workers={a.workers}, delays={delays}", flush=True)
        info = dict(mode=mode, driada_version=driada.__version__, driada_path=str(_driada_path),
                    seed=SEED, realization_stride=REALIZATION_STRIDE, paper=PAPER,
                    mi_excess_thr=MI_EXCESS_THR, find_optimal_delays=delays,
                    snr_values=snrs, skip_values=skips, n_realizations=n_real,
                    n_neurons=N_NEURONS, n_features=N_FEATURES, duration=DURATION, fps=FPS,
                    workers=a.workers, started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        t_start = time.time()
        done = 0
        if a.workers <= 1:
            for t in todo:
                d = _worker(t)
                done += 1
                print(f"[{done}/{len(todo)}] {t[0]} SNR={t[1]} skip={t[2]} r={t[3]}: "
                      f"{d['wall_sec']:.0f} s, peak {d['peak_rss_gb']:.2f} GB", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                futs = {ex.submit(_worker, t): t for t in todo}
                for fut in as_completed(futs):
                    t = futs[fut]
                    d = fut.result()
                    done += 1
                    print(f"[{done}/{len(todo)}] {t[0]} SNR={t[1]} skip={t[2]} r={t[3]}: "
                          f"{d['wall_sec']:.0f} s, peak {d['peak_rss_gb']:.2f} GB", flush=True)
        info["wall_sec"] = round(time.time() - t_start, 1)
        info["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        (mode_dir / "run_info.json").write_text(json.dumps(info, indent=1))

    _, tab = aggregate(mode_dir)
    main_tab = tab[tab.main & np.isclose(tab.skip_prob, 0.0)]
    print(main_tab[["ftype", "method", "snr", "precision", "recall", "f1"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)
    print(f"[OK] results in {mode_dir}", flush=True)


if __name__ == "__main__":
    main()
