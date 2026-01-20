"""Summary statistics extraction and printing for INTENSE analysis.

Contains:
- extract_* functions: Extract metrics from INTENSE results
- compute_summary_metrics: Aggregate all metrics into one dict
- print_* functions: Display summaries to console
- save_batch_summary_csv: Save batch results to CSV
- load_batch_summary_csv: Load existing batch results from CSV
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_batch_summary_csv(csv_path):
    """Load existing batch summary from CSV.

    Parameters
    ----------
    csv_path : Path or str
        Path to batch_summary.csv

    Returns
    -------
    list[dict]
        List of summary dicts (excluding aggregate row)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path)

    # Filter out aggregate row (has exp_name == 'AGGREGATE' or NaN exp_name)
    if 'exp_name' in df.columns:
        df = df[df['exp_name'].notna() & (df['exp_name'] != 'AGGREGATE')]

    # Convert to list of dicts
    return df.to_dict('records')


def extract_metadata(exp):
    """Extract basic experiment metadata."""
    duration_sec = exp.n_frames / exp.fps if exp.fps > 0 else np.nan
    return {
        'n_cells': exp.n_cells,
        'n_frames': exp.n_frames,
        'fps': exp.fps,
        'duration_sec': duration_sec,
    }


def extract_selectivity_stats(exp, stats, significance):
    """Extract selectivity statistics from INTENSE results."""
    significant_neurons = exp.get_significant_neurons()
    n_significant_cells = len(significant_neurons)
    pct_selective = 100.0 * n_significant_cells / exp.n_cells if exp.n_cells > 0 else 0.0

    total_sig_pairs = sum(len(features) for features in significant_neurons.values())
    avg_features_per_neuron = total_sig_pairs / n_significant_cells if n_significant_cells > 0 else 0.0

    # Count single vs multi-feature neurons
    single_feature_neurons = {n: f for n, f in significant_neurons.items() if len(f) == 1}
    multi_feature_neurons = {n: f for n, f in significant_neurons.items() if len(f) >= 2}
    n_single_feature = len(single_feature_neurons)
    n_multi_feature = len(multi_feature_neurons)
    pct_multi_feature = 100.0 * n_multi_feature / n_significant_cells if n_significant_cells > 0 else 0.0

    return {
        'n_significant_cells': n_significant_cells,
        'pct_selective': pct_selective,
        'total_sig_pairs': total_sig_pairs,
        'avg_features_per_neuron': avg_features_per_neuron,
        'n_single_feature': n_single_feature,
        'n_multi_feature': n_multi_feature,
        'pct_multi_feature': pct_multi_feature,
    }


def extract_mi_stats(stats, significance):
    """Extract MI statistics from significant pairs only."""
    mi_values = []
    rel_mi_beh_values = []
    rel_mi_ca_values = []

    for cell_id, feat_dict in stats.items():
        for feat_name, pair_stats in feat_dict.items():
            # Check if this pair is significant
            is_sig = False
            if cell_id in significance and feat_name in significance[cell_id]:
                sig_info = significance[cell_id][feat_name]
                if isinstance(sig_info, dict):
                    is_sig = sig_info.get('stage2', sig_info.get('stage1', False))
                else:
                    is_sig = bool(sig_info)

            if is_sig:
                if 'me' in pair_stats:
                    mi_values.append(pair_stats['me'])
                if 'rel_me_beh' in pair_stats:
                    rel_mi_beh_values.append(pair_stats['rel_me_beh'])
                if 'rel_me_ca' in pair_stats:
                    rel_mi_ca_values.append(pair_stats['rel_me_ca'])

    return {
        'mi_mean': np.mean(mi_values) if mi_values else np.nan,
        'mi_median': np.median(mi_values) if mi_values else np.nan,
        'mi_std': np.std(mi_values) if mi_values else np.nan,
        'mi_max': np.max(mi_values) if mi_values else np.nan,
        'rel_mi_beh_mean': np.mean(rel_mi_beh_values) if rel_mi_beh_values else np.nan,
        'rel_mi_ca_mean': np.mean(rel_mi_ca_values) if rel_mi_ca_values else np.nan,
    }


def extract_timing_stats(info, t_load, t_intense):
    """Extract timing statistics from info dict."""
    timings = info.get('timings', {}) if info else {}

    return {
        't_load': t_load,
        't_delay_opt': timings.get('stage1_delay_optimization', np.nan),
        't_stage1_scan': timings.get('stage1_pair_scanning', np.nan),
        't_stage2_scan': timings.get('stage2_pair_scanning', np.nan),
        't_disentangle': timings.get('disentanglement', np.nan),
        't_total': t_intense,
    }


def _parse_fft_counts(fft_counts):
    """Parse FFT type counts dict into structured stats."""
    # Keys match FFT_* constants in intense_base.py
    cc = fft_counts.get('cc', 0)           # FFT_CONTINUOUS
    gd = fft_counts.get('gd', 0)           # FFT_DISCRETE (Gaussian-discrete)
    dd = fft_counts.get('dd', 0)           # FFT_DISCRETE_DISCRETE
    mts = fft_counts.get('mts', 0)         # FFT_MULTIVARIATE
    mts_discrete = fft_counts.get('mts_discrete', 0)  # FFT_MTS_DISCRETE
    mts_mts = fft_counts.get('mts_mts', 0) # FFT_MTS_MTS
    loop = fft_counts.get('loop', 0)

    total = cc + gd + dd + mts + mts_discrete + mts_mts + loop
    fft_total = cc + gd + dd + mts + mts_discrete + mts_mts
    coverage = 100.0 * fft_total / total if total > 0 else 0.0

    return {
        'cc': cc, 'gd': gd, 'dd': dd, 'mts': mts,
        'mts_discrete': mts_discrete, 'mts_mts': mts_mts,
        'loop': loop, 'total': total, 'coverage': coverage,
    }


def extract_fft_stats(info):
    """Extract FFT usage statistics from info dict."""
    timings = info.get('timings', {}) if info else {}

    # Main INTENSE FFT stats
    fft_counts = timings.get('fft_type_counts', {})
    main = _parse_fft_counts(fft_counts)

    # Disentanglement FFT stats (if available)
    disent_counts = timings.get('fft_type_counts_disentanglement', {})
    disent = _parse_fft_counts(disent_counts)

    return {
        'fft_cc_pairs': main['cc'],
        'fft_gd_pairs': main['gd'],
        'fft_dd_pairs': main['dd'],
        'fft_mts_pairs': main['mts'],
        'fft_mts_discrete_pairs': main['mts_discrete'],
        'fft_mts_mts_pairs': main['mts_mts'],
        'loop_pairs': main['loop'],
        'total_pairs': main['total'],
        'fft_coverage_pct': main['coverage'],
        # Disentanglement FFT stats
        'disent_fft_cc_pairs': disent['cc'],
        'disent_fft_gd_pairs': disent['gd'],
        'disent_fft_mts_pairs': disent['mts'],
        'disent_loop_pairs': disent['loop'],
        'disent_total_pairs': disent['total'],
        'disent_fft_coverage_pct': disent['coverage'],
    }


def extract_disentanglement_stats(disent_results):
    """Extract disentanglement statistics."""
    if not disent_results or 'feat_feat_significance' not in disent_results:
        return {
            'n_correlated_feat_pairs': 0,
            'redundancy_rate': np.nan,
            'true_mixed_selectivity_rate': np.nan,
            'independence_rate': np.nan,
        }

    # Count correlated feature pairs from feat_feat_significance matrix
    feat_sig = disent_results['feat_feat_significance']
    n_features = feat_sig.shape[0]
    n_correlated_feat_pairs = 0
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if feat_sig[i, j] == 1:
                n_correlated_feat_pairs += 1

    # Extract rates from summary
    summary = disent_results.get('summary', {})
    overall_stats = summary.get('overall_stats', {})

    return {
        'n_correlated_feat_pairs': n_correlated_feat_pairs,
        'redundancy_rate': overall_stats.get('redundancy_rate', np.nan),
        'true_mixed_selectivity_rate': overall_stats.get('true_mixed_selectivity_rate', np.nan),
        'independence_rate': overall_stats.get('independence_rate', np.nan),
    }


def compute_summary_metrics(exp, stats, significance, info, disent_results, t_load, t_intense):
    """Compute comprehensive summary metrics for an experiment.

    Parameters
    ----------
    exp : Experiment
        Experiment object
    stats : dict
        Stats dict from INTENSE
    significance : dict
        Significance dict from INTENSE
    info : dict
        Info dict from INTENSE
    disent_results : dict
        Disentanglement results
    t_load : float
        Loading time in seconds
    t_intense : float
        INTENSE analysis time in seconds

    Returns
    -------
    dict
        Comprehensive summary with all metrics
    """
    summary = {}

    # Extract all metric categories
    summary.update(extract_metadata(exp))
    summary.update(extract_selectivity_stats(exp, stats, significance))
    summary.update(extract_mi_stats(stats, significance))
    summary.update(extract_timing_stats(info, t_load, t_intense))
    summary.update(extract_fft_stats(info))
    summary.update(extract_disentanglement_stats(disent_results))

    return summary


def format_metric(value, format_spec=".1f", missing_str="N/A"):
    """Format a metric value, showing N/A for missing data."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return missing_str
    if isinstance(value, float):
        return f"{value:{format_spec}}"
    return str(value)


def print_per_file_summary(summary, exp_name):
    """Print comprehensive per-file summary."""
    print(f"\n{'='*60}")
    print(f"SUMMARY: {exp_name}")
    print('='*60)

    # Experiment Info
    print(f"\n  Experiment Info:")
    print(f"    Neurons: {summary['n_cells']}")
    print(f"    Frames: {summary['n_frames']}")
    print(f"    FPS: {format_metric(summary['fps'], '.1f')}")
    print(f"    Duration: {format_metric(summary['duration_sec'], '.1f')}s")

    # Selectivity
    print(f"\n  Selectivity:")
    print(f"    Significant neurons: {summary['n_significant_cells']}/{summary['n_cells']} ({format_metric(summary['pct_selective'], '.1f')}%)")
    print(f"    Total significant pairs: {summary['total_sig_pairs']}")
    print(f"    Avg features/neuron: {format_metric(summary['avg_features_per_neuron'], '.2f')}")
    print(f"    Single-feature neurons: {summary['n_single_feature']}")
    print(f"    Multi-feature neurons: {summary['n_multi_feature']} ({format_metric(summary['pct_multi_feature'], '.1f')}%)")

    # MI Statistics
    print(f"\n  MI Statistics (significant pairs only):")
    print(f"    Mean MI: {format_metric(summary['mi_mean'], '.4f')} bits")
    print(f"    Median MI: {format_metric(summary['mi_median'], '.4f')} bits")
    print(f"    Std MI: {format_metric(summary['mi_std'], '.4f')} bits")
    print(f"    Max MI: {format_metric(summary['mi_max'], '.4f')} bits")
    print(f"    Rel MI (behavior): {format_metric(summary['rel_mi_beh_mean'], '.4f')}")
    print(f"    Rel MI (calcium): {format_metric(summary['rel_mi_ca_mean'], '.4f')}")

    # Profiling / Timing
    print(f"\n  Profiling:")
    print(f"    Load time: {format_metric(summary['t_load'], '.1f')}s")
    print(f"    Delay optimization: {format_metric(summary['t_delay_opt'], '.1f')}s")
    print(f"    Stage 1 scanning: {format_metric(summary['t_stage1_scan'], '.1f')}s")
    print(f"    Stage 2 scanning: {format_metric(summary['t_stage2_scan'], '.1f')}s")
    print(f"    Disentanglement: {format_metric(summary['t_disentangle'], '.1f')}s")
    print(f"    Total INTENSE time: {format_metric(summary['t_total'], '.1f')}s")

    # FFT Optimization
    print(f"\n  FFT Optimization:")
    print(f"    FFT continuous-continuous: {summary['fft_cc_pairs']} pairs")
    print(f"    FFT GCMI discrete: {summary['fft_gd_pairs']} pairs")
    print(f"    FFT discrete-discrete: {summary['fft_dd_pairs']} pairs")
    print(f"    FFT MTS: {summary['fft_mts_pairs']} pairs")
    print(f"    FFT MTS-discrete: {summary['fft_mts_discrete_pairs']} pairs")
    print(f"    FFT MTS-MTS: {summary['fft_mts_mts_pairs']} pairs")
    print(f"    Loop (no FFT): {summary['loop_pairs']} pairs")
    print(f"    Total pairs: {summary['total_pairs']}")
    print(f"    FFT coverage: {format_metric(summary['fft_coverage_pct'], '.1f')}%")

    # Disentanglement FFT stats (if available)
    if summary.get('disent_total_pairs', 0) > 0:
        print(f"\n  Disentanglement FFT:")
        print(f"    FFT continuous-continuous: {summary['disent_fft_cc_pairs']} pairs")
        print(f"    FFT GCMI discrete: {summary['disent_fft_gd_pairs']} pairs")
        print(f"    FFT MTS: {summary['disent_fft_mts_pairs']} pairs")
        print(f"    Loop (no FFT): {summary['disent_loop_pairs']} pairs")
        print(f"    Total pairs: {summary['disent_total_pairs']}")
        print(f"    FFT coverage: {format_metric(summary['disent_fft_coverage_pct'], '.1f')}%")

    # Disentanglement
    print(f"\n  Disentanglement:")
    print(f"    Correlated feature pairs: {summary['n_correlated_feat_pairs']}")
    print(f"    Redundancy rate: {format_metric(summary['redundancy_rate'], '.1f')}%")
    print(f"    True mixed selectivity: {format_metric(summary['true_mixed_selectivity_rate'], '.1f')}%")
    print(f"    Independence rate: {format_metric(summary['independence_rate'], '.1f')}%")


def compute_aggregate_row(summary_rows):
    """Compute aggregate statistics across experiments."""
    if not summary_rows:
        return {}

    df = pd.DataFrame(summary_rows)

    agg = {'exp_name': 'AGGREGATE'}

    # Sum counts
    for col in ['n_cells', 'n_frames', 'n_significant_cells', 'total_sig_pairs',
                'n_single_feature', 'n_multi_feature', 'n_correlated_feat_pairs',
                'fft_cc_pairs', 'fft_gd_pairs', 'fft_dd_pairs', 'fft_mts_pairs',
                'fft_mts_discrete_pairs', 'fft_mts_mts_pairs', 'loop_pairs', 'total_pairs']:
        if col in df.columns:
            agg[col] = df[col].sum()

    # Weighted means for percentages
    if 'pct_selective' in df.columns and 'n_cells' in df.columns:
        total_cells = df['n_cells'].sum()
        total_sig = df['n_significant_cells'].sum()
        agg['pct_selective'] = 100.0 * total_sig / total_cells if total_cells > 0 else np.nan

    if 'pct_multi_feature' in df.columns and 'n_significant_cells' in df.columns:
        total_sig = df['n_significant_cells'].sum()
        total_multi = df['n_multi_feature'].sum()
        agg['pct_multi_feature'] = 100.0 * total_multi / total_sig if total_sig > 0 else np.nan

    if 'fft_coverage_pct' in df.columns and 'total_pairs' in df.columns:
        total_pairs = df['total_pairs'].sum()
        fft_pairs = (df['fft_cc_pairs'].sum() + df['fft_gd_pairs'].sum() +
                     df['fft_dd_pairs'].sum() + df['fft_mts_pairs'].sum() +
                     df['fft_mts_discrete_pairs'].sum() + df['fft_mts_mts_pairs'].sum())
        agg['fft_coverage_pct'] = 100.0 * fft_pairs / total_pairs if total_pairs > 0 else np.nan

    # Mean for other metrics
    for col in ['fps', 'duration_sec', 'avg_features_per_neuron', 'mi_mean', 'mi_median',
                'mi_std', 'mi_max', 'rel_mi_beh_mean', 'rel_mi_ca_mean',
                't_load', 't_delay_opt', 't_stage1_scan', 't_stage2_scan', 't_disentangle', 't_total',
                'redundancy_rate', 'true_mixed_selectivity_rate', 'independence_rate']:
        if col in df.columns:
            agg[col] = df[col].mean()

    return agg


def save_batch_summary_csv(summary_rows, output_path):
    """Save batch summary to CSV with aggregate row."""
    df = pd.DataFrame(summary_rows)

    # Compute aggregate row
    agg_row = compute_aggregate_row(summary_rows)

    # Append aggregate row
    df = pd.concat([df, pd.DataFrame([agg_row])], ignore_index=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\n  Batch summary saved to: {output_path}")


def print_batch_summary(summaries, t_batch_total, output_dir):
    """Print enhanced batch summary with aggregate statistics."""
    print(f"\n{'='*60}")
    print("BATCH SUMMARY")
    print('='*60)
    print(f"  Experiments processed: {len(summaries)}")
    print(f"  Total batch time: {t_batch_total:.1f}s")

    # Per-experiment list
    print(f"\n  Per-experiment results:")
    for s in summaries:
        print(f"    {s['exp_name']}: {s['n_significant_cells']}/{s['n_cells']} significant "
              f"({s['pct_selective']:.1f}%), {s['t_total']:.1f}s")

    # Aggregate statistics
    agg = compute_aggregate_row(summaries)
    print(f"\n  Aggregate Statistics:")
    print(f"    Total neurons: {agg.get('n_cells', 0)}")
    print(f"    Total significant neurons: {agg.get('n_significant_cells', 0)}")
    print(f"    Overall selectivity: {format_metric(agg.get('pct_selective'), '.1f')}%")
    print(f"    Total significant pairs: {agg.get('total_sig_pairs', 0)}")
    print(f"    Avg features/neuron: {format_metric(agg.get('avg_features_per_neuron'), '.2f')}")
    print(f"    FFT coverage: {format_metric(agg.get('fft_coverage_pct'), '.1f')}%")
    print(f"    Avg redundancy rate: {format_metric(agg.get('redundancy_rate'), '.1f')}%")
    print(f"    Avg true mixed selectivity: {format_metric(agg.get('true_mixed_selectivity_rate'), '.1f')}%")

    if output_dir:
        print(f"\n  Results saved to: {output_dir}")
