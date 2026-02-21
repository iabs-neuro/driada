#!/usr/bin/env python3
"""
INTENSE Validation Benchmark: SNR x skip_prob Sweep

Compares INTENSE selectivity detection against baseline methods across
varying signal quality (SNR) and response reliability (skip_prob).

Two symmetric tests:
  1. Discrete: 20 event features, 500 neurons
  2. Continuous: 20 FBM features, 500 neurons

Four methods per test type:
  Continuous: dummy_corr, dummy_mi, fast_pearsonr, mi
  Discrete:   dummy_av,   dummy_mi, av,             mi

Reports both stage-1 (screening) and stage-2 (validated) metrics.
Output: per-run JSON, aggregated NPZ (backward-compatible with old notebook),
and heatmap PNGs.

Usage:
    python validate_intense_benchmark.py                    # full sweep (sequential)
    python validate_intense_benchmark.py --workers 4        # 4 configs in parallel
    python validate_intense_benchmark.py --quick            # smoke test (1 config)
    python validate_intense_benchmark.py --quick --workers 2
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# Suppress numpy implicit multi-threading (MKL/OpenBLAS).
# Small FFT arrays (14,400 elements) are dominated by thread spawn/join
# overhead (~50-200us per call x 60,000 calls = seconds of pure waste).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as t_dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from driada.experiment.synthetic import generate_synthetic_data
from driada.experiment.event_detection import CA_SHIFT_N_TOFF, MIN_FEAT_SHIFT_SEC
from driada.intense.intense_base import compute_me_stats
from driada.intense.fft import _build_fft_cache
from driada.information.info_base import TimeSeries, get_mi


# =============================================================================
# CONFIGURATION
# =============================================================================

BASELINE_RATE = 0.1
SNR_VALUES = [2, 4, 8, 16, 32, 64, 128]
SKIP_PROB_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
N_REALIZATIONS = 5

N_FEATURES = 20
N_NEURONS = 500
DURATION = 3600
FPS = 20
HURST = 0.3
CALCIUM_AMPLITUDE_RANGE = (0.5, 2.0)
CALCIUM_NOISE = 0.1
DECAY_TIME = 2.0

# INTENSE pipeline parameters (matching old notebook)
N_SHUFFLES_STAGE1 = 100
N_SHUFFLES_STAGE2 = 10000
PVAL_THR = 0.05
DS = 5
MULTICOMP_CORRECTION = "holm"
TOPK1 = 1
TOPK2 = 5

# Methods per feature type
CONTINUOUS_METHODS = ['dummy_corr', 'dummy_mi', 'fast_pearsonr', 'mi']
DISCRETE_METHODS = ['dummy_av', 'dummy_mi', 'av', 'mi']

# Dummy method thresholds (matching old notebook)
DUMMY_PVAL_THRESHOLD = 1e-50
DUMMY_AV_ALPHA = 0.01  # Bonferroni-corrected: alpha / (n_neurons * n_features)


# =============================================================================
# PRECISION / RECALL COMPUTATION
# =============================================================================

def compute_pr_rec(ans, gt):
    """Compute precision and recall from significance matrix and ground truth.

    Parameters
    ----------
    ans : ndarray of shape (n_neurons, n_features) or (n_features, n_neurons)
        Binary significance matrix (detected pairs).
    gt : ndarray of shape (n_features, n_neurons)
        Ground truth matrix.

    Returns
    -------
    precision, recall : float
    """
    if ans.shape != gt.shape:
        ans = ans.T

    tp = np.sum(ans * gt)
    total_detected = np.sum(ans)
    total_true = np.sum(gt)

    precision = float(tp / total_detected) if total_detected > 0 else 0.0
    recall = float(tp / total_true) if total_true > 0 else 0.0
    return precision, recall


# =============================================================================
# DUMMY BASELINE METHODS (vectorized)
# =============================================================================

def run_dummy_corr(calcium, feats, gt, ds=DS):
    """Pearson correlation baseline (continuous features) — vectorized.

    Computes all n_neurons x n_features correlations via matrix multiply,
    then converts to p-values analytically. Threshold: p < 1e-50.
    """
    cal_ds = calcium[:, ::ds]
    feat_ds = feats[:, ::ds]
    n = cal_ds.shape[1]

    # Standardize
    cal_z = cal_ds - cal_ds.mean(axis=1, keepdims=True)
    cal_std = cal_ds.std(axis=1, keepdims=True)
    cal_std[cal_std == 0] = 1.0
    cal_z /= cal_std

    feat_z = feat_ds - feat_ds.mean(axis=1, keepdims=True)
    feat_std = feat_ds.std(axis=1, keepdims=True)
    feat_std[feat_std == 0] = 1.0
    feat_z /= feat_std

    # Correlation matrix: (n_neurons, n_features)
    corr = cal_z @ feat_z.T / n

    # Convert to two-sided p-values
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = corr * np.sqrt((n - 2) / (1 - corr ** 2))
    pmat = 2 * (1 - t_dist.cdf(np.abs(t_stat), df=n - 2))
    pmat = np.nan_to_num(pmat, nan=1.0)

    ans = (pmat < DUMMY_PVAL_THRESHOLD).astype(int)
    return compute_pr_rec(ans, gt)


def run_dummy_av(calcium, feats, gt, ds=DS):
    """T-test baseline (discrete features) — vectorized per feature.

    Splits calcium into ON/OFF groups based on binary feature,
    runs vectorized Welch's t-test across all neurons simultaneously.
    Threshold: p < 0.01 / (n_neurons * n_features).
    """
    n_neurons, n_features = len(calcium), len(feats)
    threshold = DUMMY_AV_ALPHA / (n_neurons * n_features)
    cal_ds = calcium[:, ::ds]
    pmat = np.ones((n_neurons, n_features))

    for j in range(n_features):
        f = feats[j, ::ds]
        mask_on = f == 1
        mask_off = f == 0
        n_on = mask_on.sum()
        n_off = mask_off.sum()
        if n_on < 2 or n_off < 2:
            continue

        on_vals = cal_ds[:, mask_on]   # (n_neurons, n_on)
        off_vals = cal_ds[:, mask_off]  # (n_neurons, n_off)

        mean_on = on_vals.mean(axis=1)
        mean_off = off_vals.mean(axis=1)
        var_on = on_vals.var(axis=1, ddof=1)
        var_off = off_vals.var(axis=1, ddof=1)

        se = np.sqrt(var_on / n_on + var_off / n_off)
        with np.errstate(divide='ignore', invalid='ignore'):
            t_stat = (mean_on - mean_off) / (se + 1e-30)

            # Welch-Satterthwaite degrees of freedom
            num = (var_on / n_on + var_off / n_off) ** 2
            denom = ((var_on / n_on) ** 2 / (n_on - 1)
                     + (var_off / n_off) ** 2 / (n_off - 1))
            df = num / (denom + 1e-30)

        p = 2 * (1 - t_dist.cdf(np.abs(t_stat), df=df))
        pmat[:, j] = np.nan_to_num(p, nan=1.0)

    ans = (pmat < threshold).astype(int)
    return compute_pr_rec(ans, gt)


def _mi_to_pearson_pval_vec(mi_vals, corr_signs, n):
    """Convert GCMI values to one-sided p-values via Pearson rho t-test (vectorized)."""
    rho_sq = 1 - np.exp(-2 * mi_vals)
    rho = np.sqrt(np.clip(rho_sq, 0, None)) * corr_signs

    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    pvals = 1 - t_dist.cdf(t_stat, df=n - 2)
    pvals = np.nan_to_num(pvals, nan=1.0)
    # Perfect correlation
    pvals[np.abs(rho) >= 1.0] = 0.0
    return pvals


def run_dummy_mi(calcium, feats, gt, ftype, ds=DS):
    """Analytical MI-based baseline.

    Continuous: GCMI → Pearson rho → t-test p-value (vectorized correlation sign).
    Discrete: chi-squared contingency test (G-test).
    Threshold: p < 1e-50.
    """
    n_neurons, n_features = len(calcium), len(feats)
    cal_ds = calcium[:, ::ds]
    feat_ds = feats[:, ::ds]
    n = cal_ds.shape[1]

    if ftype == 'c':
        # Vectorized: compute correlation signs via matrix multiply
        cal_z = cal_ds - cal_ds.mean(axis=1, keepdims=True)
        feat_z = feat_ds - feat_ds.mean(axis=1, keepdims=True)
        cal_std = cal_ds.std(axis=1, keepdims=True)
        cal_std[cal_std == 0] = 1.0
        feat_std = feat_ds.std(axis=1, keepdims=True)
        feat_std[feat_std == 0] = 1.0
        corr_signs = np.sign(cal_z @ feat_z.T / n)  # (n_neurons, n_features)

        # MI via FFT cache: build once, read mi_all[0] for zero-shift MI
        ts_ca = [TimeSeries(ca, ts_type='linear', name=f"_dn{i}") for i, ca in enumerate(calcium)]
        ts_f = [TimeSeries(f, ts_type='linear', name=f"_df{j}") for j, f in enumerate(feats)]
        fft_cache, _ = _build_fft_cache(ts_ca, ts_f, 'mi', 'gcmi', ds, 'auto', False, n_jobs=-1)
        mi_mat = np.zeros((n_neurons, n_features))
        for (k1, k2), entry in fft_cache.items():
            i = int(k1[3:])   # "_dn0" -> 0
            j = int(k2[3:])   # "_df0" -> 0
            mi_mat[i, j] = entry.mi_all[0]

        pmat = _mi_to_pearson_pval_vec(mi_mat, corr_signs, n)

    else:
        # Discrete: G-test via contingency table (vectorized)
        from scipy.stats import chi2 as chi2_dist
        pmat = np.ones((n_neurons, n_features))
        n_bins = int(np.ceil(np.log2(n)) + 1)

        # Precompute binned signals for all neurons
        binned_all = np.empty((n_neurons, n), dtype=int)
        bin_edges = np.linspace(0, 1, n_bins + 1)[1:-1]
        for i in range(n_neurons):
            edges = np.quantile(cal_ds[i], bin_edges)
            binned_all[i] = np.digitize(cal_ds[i], edges)

        for j in range(n_features):
            f = feat_ds[j]
            f_vals = np.unique(f)
            n_fv = len(f_vals)
            if n_fv < 2:
                continue
            masks = [f == fv for fv in f_vals]

            for i in range(n_neurons):
                sig_binned = binned_all[i]
                ct = np.zeros((n_fv, n_bins), dtype=int)
                for fi in range(n_fv):
                    np.add.at(ct[fi], sig_binned[masks[fi]], 1)
                ct = ct[:, ct.sum(axis=0) > 0]
                ct = ct[ct.sum(axis=1) > 0, :]
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    continue
                n_total = ct.sum()
                row_sums = ct.sum(axis=1, keepdims=True)
                col_sums_k = ct.sum(axis=0, keepdims=True)
                expected = row_sums * col_sums_k / n_total
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where(ct > 0, ct / expected, 1.0)
                    g_stat = 2 * np.sum(
                        ct * np.where(ct > 0, np.log(ratio), 0.0))
                dof = (ct.shape[0] - 1) * (ct.shape[1] - 1)
                pmat[i, j] = chi2_dist.sf(g_stat, dof)

    ans = (pmat < DUMMY_PVAL_THRESHOLD).astype(int)
    return compute_pr_rec(ans, gt)


# =============================================================================
# INTENSE METHODS
# =============================================================================

def run_intense_method(method, ts_calcium, ts_feats, gt, n_jobs=-1):
    """Run INTENSE with given metric. Return stage1 and stage2 (precision, recall).

    Parameters
    ----------
    method : str
        One of 'mi', 'fast_pearsonr', 'av'.
    ts_calcium : list of TimeSeries
        Pre-created calcium TimeSeries objects.
    ts_feats : list of TimeSeries
        Pre-created feature TimeSeries objects.
    n_jobs : int
        Number of parallel jobs for INTENSE. -1 = all cores.
    """
    stats, significance, info = compute_me_stats(
        ts_calcium,
        ts_feats,
        metric=method,
        mode='two_stage',
        n_shuffles_stage1=N_SHUFFLES_STAGE1,
        n_shuffles_stage2=N_SHUFFLES_STAGE2,
        ds=DS,
        topk1=TOPK1,
        topk2=TOPK2,
        multicomp_correction=MULTICOMP_CORRECTION,
        pval_thr=PVAL_THR,
        find_optimal_delays=False,
        shift_window=2,
        noise_ampl=1e-4,
        metric_distr_type='gamma_zi' if method in ('mi', 'fast_pearsonr') else 'norm',

        verbose=False,
        enable_parallelization=True,
        n_jobs=n_jobs,
    )

    # Extract stage1 significance matrix
    sig1_tables = info['stage_1_significance']
    ans1 = sig1_tables['stage1']  # (n_neurons, n_features)
    pr1, rec1 = compute_pr_rec(ans1, gt)

    # Extract stage2 significance matrix
    if 'stage_2_significance' in info:
        sig2_tables = info['stage_2_significance']
        ans2 = sig2_tables['stage2']  # (n_neurons, n_features)
        pr2, rec2 = compute_pr_rec(ans2, gt)
    else:
        pr2, rec2 = 0.0, 0.0

    return (
        {'precision': pr1, 'recall': rec1},
        {'precision': pr2, 'recall': rec2},
    )


# =============================================================================
# CORE ANALYSIS
# =============================================================================

def run_single_config(test_type, snr, skip_prob, seed, n_jobs=-1):
    """Run one configuration: generate data, run all methods, validate.

    Parameters
    ----------
    n_jobs : int
        Number of parallel jobs for INTENSE methods. -1 = all cores.
    """
    ftype = 'c' if test_type == 'continuous' else 'd'
    peak_rate = BASELINE_RATE * snr
    methods = CONTINUOUS_METHODS if ftype == 'c' else DISCRETE_METHODS

    # Generate synthetic data
    feats, calcium, gt = generate_synthetic_data(
        nfeats=N_FEATURES,
        nneurons=N_NEURONS,
        ftype=ftype,
        duration=DURATION,
        sampling_rate=FPS,
        baseline_rate=BASELINE_RATE,
        peak_rate=peak_rate,
        skip_prob=skip_prob,
        hurst=HURST,
        calcium_amplitude_range=CALCIUM_AMPLITUDE_RANGE,
        decay_time=DECAY_TIME,
        calcium_noise=CALCIUM_NOISE,
        seed=seed,
        verbose=False,
    )

    # Create TimeSeries objects once, shared across all INTENSE methods
    is_discrete = (ftype == 'd')
    intense_methods = [m for m in methods if m not in ('dummy_corr', 'dummy_av', 'dummy_mi')]
    if intense_methods:
        ts_calcium = [TimeSeries(ca, ts_type='linear') for ca in calcium]
        ts_feats = [TimeSeries(f, ts_type='binary' if is_discrete else 'linear') for f in feats]
        # Set shuffle masks matching Neuron.__init__() behavior:
        # exclude first/last min_shift frames to prevent near-zero shifts
        # (within calcium autocorrelation window) from contaminating the null.
        t_off_frames = DECAY_TIME * FPS
        min_shift = int(t_off_frames * CA_SHIFT_N_TOFF)
        for ts in ts_calcium:
            ts.shuffle_mask[:min_shift] = False
            ts.shuffle_mask[-min_shift:] = False
        # Set shuffle masks on features to exclude near-zero shifts
        min_feat_shift = int(MIN_FEAT_SHIFT_SEC * FPS)
        for ts in ts_feats:
            ts.shuffle_mask[:min_feat_shift] = False
            ts.shuffle_mask[-min_feat_shift:] = False

    method_results = {}
    for method in methods:
        if method == 'dummy_corr':
            pr, rec = run_dummy_corr(calcium, feats, gt)
            method_results[method] = {
                'stage1': {'precision': round(pr, 6), 'recall': round(rec, 6)},
            }
        elif method == 'dummy_av':
            pr, rec = run_dummy_av(calcium, feats, gt)
            method_results[method] = {
                'stage1': {'precision': round(pr, 6), 'recall': round(rec, 6)},
            }
        elif method == 'dummy_mi':
            pr, rec = run_dummy_mi(calcium, feats, gt, ftype)
            method_results[method] = {
                'stage1': {'precision': round(pr, 6), 'recall': round(rec, 6)},
            }
        else:
            s1, s2 = run_intense_method(method, ts_calcium, ts_feats, gt,
                                        n_jobs=n_jobs)
            method_results[method] = {
                'stage1': {k: round(v, 6) for k, v in s1.items()},
                'stage2': {k: round(v, 6) for k, v in s2.items()},
            }

    return {
        'test_type': test_type,
        'snr': snr,
        'skip_prob': skip_prob,
        'seed': seed,
        'peak_rate': round(peak_rate, 4),
        'methods': method_results,
    }


def _worker(args):
    """Picklable top-level worker for ProcessPoolExecutor."""
    test_type, snr, skip_prob, seed, n_jobs, results_dir = args
    result = run_single_config(test_type, snr, skip_prob, seed, n_jobs=n_jobs)

    # Save per-run JSON
    fname = f"snr{snr}_skip{skip_prob}_seed{seed}.json"
    fpath = os.path.join(results_dir, fname)
    with open(fpath, "w") as f:
        json.dump(result, f, indent=2)

    return result


def run_sweep(test_type, output_dir, n_workers=1,
              snr_values=None, skip_prob_values=None, n_realizations=None,
              resume=False):
    """Run full SNR x skip_prob sweep for one feature type.

    Parameters
    ----------
    n_workers : int
        Number of parallel worker processes. 1 = sequential (INTENSE uses all cores).
        >1 = parallel configs, each INTENSE limited to cpu_count // n_workers cores.
    resume : bool
        If True, skip configs that already have a JSON result file and load
        cached results instead.
    """
    snr_values = snr_values or SNR_VALUES
    skip_prob_values = skip_prob_values or SKIP_PROB_VALUES
    n_realizations = n_realizations or N_REALIZATIONS

    results_dir = os.path.join(output_dir, f"results_{test_type}")
    os.makedirs(results_dir, exist_ok=True)

    # Threading parallelism within INTENSE gives <1.4x speedup on Windows
    # (GIL limits FFT cache build loop). Use --workers for process-level
    # parallelism instead: each worker runs INTENSE serially.
    total_cores = cpu_count()
    n_jobs = 1

    # Build task list
    all_task_args = []
    for snr in snr_values:
        for skip_prob in skip_prob_values:
            for seed_idx in range(n_realizations):
                all_task_args.append((test_type, snr, skip_prob, seed_idx,
                              n_jobs, results_dir))

    # If resuming, load cached results and filter out completed tasks
    tasks = []
    all_results = []
    if resume:
        n_cached = 0
        for task_args in all_task_args:
            _, snr, skip_prob, seed_idx, _, _ = task_args
            fname = f"snr{snr}_skip{skip_prob}_seed{seed_idx}.json"
            fpath = os.path.join(results_dir, fname)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    cached = json.load(f)
                all_results.append(cached)
                n_cached += 1
            else:
                tasks.append(task_args)
        print(f"  Resume: {n_cached} cached, {len(tasks)} remaining")
    else:
        tasks = all_task_args

    total = len(all_task_args)
    n_todo = len(tasks)

    if n_todo == 0:
        print("  All configs already completed.")
        return all_results

    if n_workers <= 1:
        # Sequential — verbose per-config output
        for idx, task_args in enumerate(tasks):
            _, snr, skip_prob, seed_idx, _, _ = task_args
            global_idx = total - n_todo + idx + 1
            label = (f"[{global_idx}/{total}] {test_type} "
                     f"SNR={snr} skip={skip_prob} seed={seed_idx}")
            print(f"  {label} ...", end=" ", flush=True)

            t0 = time.time()
            result = _worker(task_args)
            elapsed = time.time() - t0

            mi_key = 'mi'
            if mi_key in result['methods'] and 'stage2' in result['methods'][mi_key]:
                s2 = result['methods'][mi_key]['stage2']
                print(f"MI_s2: P={s2['precision']:.2f} R={s2['recall']:.2f} "
                      f"({elapsed:.1f}s)")
            else:
                print(f"({elapsed:.1f}s)")

            all_results.append(result)
    else:
        # Parallel — use ProcessPoolExecutor
        print(f"  Launching {n_workers} workers "
              f"({n_jobs} cores/worker, {n_todo} configs)...")

        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker, t): t for t in tasks}
            for future in as_completed(futures):
                done += 1
                task_args = futures[future]
                _, snr, skip_prob, seed_idx, _, _ = task_args
                result = future.result()
                all_results.append(result)

                mi_key = 'mi'
                if mi_key in result['methods'] and 'stage2' in result['methods'][mi_key]:
                    s2 = result['methods'][mi_key]['stage2']
                    print(f"  [{done}/{n_todo}] SNR={snr} skip={skip_prob} "
                          f"seed={seed_idx} — "
                          f"MI_s2: P={s2['precision']:.2f} R={s2['recall']:.2f}")
                else:
                    print(f"  [{done}/{n_todo}] SNR={snr} skip={skip_prob} "
                          f"seed={seed_idx} — done")

    return all_results


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_to_npz(all_results, methods, snr_values, skip_prob_values,
                     n_realizations):
    """Aggregate results into NPZ array matching old notebook format.

    Returns
    -------
    arr : ndarray of shape (2, n_methods, 2, n_skip, n_snr, n_realizations)
        Dimensions: (metrics, methods, stages, skip_probs, snrs, seeds)
        metrics: [precision, recall]
        stages: [stage1, stage2]
    """
    n_metrics = 2
    n_methods = len(methods)
    n_stages = 2
    n_skip = len(skip_prob_values)
    n_snr = len(snr_values)
    n_rand = n_realizations

    arr = np.full((n_metrics, n_methods, n_stages, n_skip, n_snr, n_rand), np.nan)

    for result in all_results:
        snr = result['snr']
        skip_prob = result['skip_prob']
        seed = result['seed']

        i_skip = skip_prob_values.index(skip_prob)
        i_snr = snr_values.index(snr)
        i_seed = seed

        for i_method, method in enumerate(methods):
            if method not in result['methods']:
                continue
            mdata = result['methods'][method]

            if 'stage1' in mdata:
                arr[0, i_method, 0, i_skip, i_snr, i_seed] = mdata['stage1']['precision']
                arr[1, i_method, 0, i_skip, i_snr, i_seed] = mdata['stage1']['recall']

            if 'stage2' in mdata:
                arr[0, i_method, 1, i_skip, i_snr, i_seed] = mdata['stage2']['precision']
                arr[1, i_method, 1, i_skip, i_snr, i_seed] = mdata['stage2']['recall']

    return arr


def save_summary_npz(arr, test_type, methods, snr_values, skip_prob_values,
                     output_dir):
    """Save aggregated array as NPZ."""
    fpath = os.path.join(output_dir, f"{test_type}_results.npz")
    np.savez(
        fpath,
        arr=arr,
        methods=np.array(methods),
        snr_values=np.array(snr_values, dtype=float),
        skip_prob_values=np.array(skip_prob_values, dtype=float),
        metric_names=np.array(['precision', 'recall']),
        stage_names=np.array(['stage1', 'stage2']),
    )
    print(f"  Saved: {fpath}")


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_single_heatmap(ax, data, snr_values, skip_prob_values, title,
                        vmin=0, vmax=1):
    """Plot a single 2D heatmap on the given axis."""
    im = ax.imshow(
        data, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax,
        origin="lower",
    )
    ax.set_xticks(range(len(skip_prob_values)))
    ax.set_xticklabels([f"{sp:.1f}" for sp in skip_prob_values], fontsize=6)
    ax.set_yticks(range(len(snr_values)))
    ax.set_yticklabels([str(s) for s in snr_values])
    ax.set_xlabel("skip_prob")
    ax.set_ylabel("SNR")
    ax.set_title(title, fontsize=9)

    for i in range(len(snr_values)):
        for j in range(len(skip_prob_values)):
            val = data[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=5, color=color)

    return im


def plot_method_heatmaps(arr, methods, snr_values, skip_prob_values,
                         test_name, output_dir):
    """Create heatmap grid: rows=methods, cols=[precision, recall] for stage2.

    For methods with only stage1, uses stage1 data.
    """
    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods, 2, figsize=(12, 3 * n_methods))
    if n_methods == 1:
        axes = axes[np.newaxis, :]

    for i_method, method in enumerate(methods):
        stage_idx = 1 if not method.startswith('dummy_') else 0
        stage_label = 'stage2' if stage_idx == 1 else 'stage1'

        for i_metric, metric_name in enumerate(['Precision', 'Recall']):
            data = np.nanmean(arr[i_metric, i_method, stage_idx, :, :, :], axis=-1)
            plot_single_heatmap(
                axes[i_method, i_metric],
                data.T,
                snr_values,
                skip_prob_values,
                f"{method} ({stage_label}) — {metric_name}",
            )

    fig.suptitle(f"INTENSE Benchmark: {test_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fpath = os.path.join(output_dir, f"heatmap_{test_name.lower()}_all_methods.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="INTENSE validation benchmark")
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick smoke test: 1 SNR, 1 skip_prob, 1 realization per test",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1 = sequential). "
             "Each worker gets cpu_count/workers cores for INTENSE.",
    )
    parser.add_argument(
        "--only", choices=["discrete", "continuous"],
        help="Run only one test type (skip the other).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip configs that already have a JSON result file.",
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(__file__))
    n_workers = max(1, args.workers)

    if args.quick:
        snr_values = [16]
        skip_prob_values = [0.0]
        n_realizations = 1
        print("=== QUICK MODE: 1 config per test ===\n")
    else:
        snr_values = SNR_VALUES
        skip_prob_values = SKIP_PROB_VALUES
        n_realizations = N_REALIZATIONS
        total_runs = len(snr_values) * len(skip_prob_values) * n_realizations * 2
        print(f"=== FULL SWEEP: {total_runs} total runs ===\n")

    total_cores = cpu_count()
    cores_per_worker = max(1, total_cores // n_workers) if n_workers > 1 else total_cores
    print(f"Config: {N_FEATURES} features, {N_NEURONS} neurons, "
          f"{DURATION}s, baseline={BASELINE_RATE}, noise={CALCIUM_NOISE}")
    print(f"INTENSE: shuffles={N_SHUFFLES_STAGE1}/{N_SHUFFLES_STAGE2}, "
          f"correction={MULTICOMP_CORRECTION}, ds={DS}")
    print(f"Parallelism: {n_workers} worker(s), "
          f"{cores_per_worker} cores/worker ({total_cores} total)\n")

    run_discrete = args.only in (None, "discrete")
    run_continuous = args.only in (None, "continuous")

    results_discrete = []
    results_continuous = []
    t_discrete = 0.0
    t_continuous = 0.0

    if run_discrete:
        # --- Discrete test ---
        print("=" * 60)
        print(f"TEST 1: DISCRETE ({N_FEATURES} event features, {N_NEURONS} neurons)")
        print(f"Methods: {DISCRETE_METHODS}")
        print("=" * 60)
        t0 = time.time()
        results_discrete = run_sweep(
            "discrete", output_dir, n_workers=n_workers,
            snr_values=snr_values, skip_prob_values=skip_prob_values,
            n_realizations=n_realizations, resume=args.resume,
        )
        t_discrete = time.time() - t0
        print(f"\nDiscrete test completed in {t_discrete:.1f}s\n")

    if run_continuous:
        # --- Continuous test ---
        print("=" * 60)
        print(f"TEST 2: CONTINUOUS ({N_FEATURES} FBM features, {N_NEURONS} neurons)")
        print(f"Methods: {CONTINUOUS_METHODS}")
        print("=" * 60)
        t0 = time.time()
        results_continuous = run_sweep(
            "continuous", output_dir, n_workers=n_workers,
            snr_values=snr_values, skip_prob_values=skip_prob_values,
            n_realizations=n_realizations, resume=args.resume,
        )
        t_continuous = time.time() - t0
        print(f"\nContinuous test completed in {t_continuous:.1f}s\n")

    # --- Aggregate ---
    print("=" * 60)
    print("AGGREGATION & VISUALIZATION")
    print("=" * 60)

    if run_discrete and results_discrete:
        arr_discrete = aggregate_to_npz(
            results_discrete, DISCRETE_METHODS,
            snr_values, skip_prob_values, n_realizations,
        )
        save_summary_npz(arr_discrete, "discrete", DISCRETE_METHODS,
                         snr_values, skip_prob_values, output_dir)
        if len(snr_values) > 1 and len(skip_prob_values) > 1:
            plot_method_heatmaps(arr_discrete, DISCRETE_METHODS,
                                 snr_values, skip_prob_values, "Discrete", output_dir)

    if run_continuous and results_continuous:
        arr_continuous = aggregate_to_npz(
            results_continuous, CONTINUOUS_METHODS,
            snr_values, skip_prob_values, n_realizations,
        )
        save_summary_npz(arr_continuous, "continuous", CONTINUOUS_METHODS,
                         snr_values, skip_prob_values, output_dir)
        if len(snr_values) > 1 and len(skip_prob_values) > 1:
            plot_method_heatmaps(arr_continuous, CONTINUOUS_METHODS,
                                 snr_values, skip_prob_values, "Continuous", output_dir)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Total time: {t_discrete + t_continuous:.1f}s")
    print(f"Output directory: {output_dir}")

    if not args.quick:
        test_list = []
        if run_discrete and results_discrete:
            test_list.append(("Discrete", arr_discrete, DISCRETE_METHODS))
        if run_continuous and results_continuous:
            test_list.append(("Continuous", arr_continuous, CONTINUOUS_METHODS))
        for test_name, arr, methods in test_list:
            print(f"\n{test_name} — mean precision / recall across all configs:")
            for i_m, method in enumerate(methods):
                stage_idx = 1 if not method.startswith('dummy_') else 0
                stage_label = 'S2' if stage_idx == 1 else 'S1'
                p = np.nanmean(arr[0, i_m, stage_idx, :, :, :])
                r = np.nanmean(arr[1, i_m, stage_idx, :, :, :])
                print(f"  {method:16s} ({stage_label}): P={p:.3f}  R={r:.3f}")
    else:
        test_list = []
        if run_discrete:
            test_list.append(("Discrete", results_discrete, DISCRETE_METHODS))
        if run_continuous:
            test_list.append(("Continuous", results_continuous, CONTINUOUS_METHODS))
        for test_name, results, methods in test_list:
            print(f"\n{test_name}:")
            for result in results:
                for method in methods:
                    if method not in result['methods']:
                        continue
                    mdata = result['methods'][method]
                    for stage in ['stage1', 'stage2']:
                        if stage in mdata:
                            p = mdata[stage]['precision']
                            r = mdata[stage]['recall']
                            print(f"  {method:16s} {stage}: P={p:.4f}  R={r:.4f}")


if __name__ == "__main__":
    main()
