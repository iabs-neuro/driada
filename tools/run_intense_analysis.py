#!/usr/bin/env python
"""
Run full INTENSE analysis on .npz experiment data.

Loads experiment data, creates aggregated features, and runs
INTENSE analysis with disentanglement to identify neural selectivity.

Usage
-----
    # Process a single file
    python tools/run_intense_analysis.py "DRIADA data/LNOF_J01_4D_aligned.npz"

    # Process all .npz files in a directory
    python tools/run_intense_analysis.py --dir "DRIADA data" --output-dir INTENSE

    # Process multiple specific files
    python tools/run_intense_analysis.py "file1.npz" "file2.npz" "file3.npz"

    # With custom parameters
    python tools/run_intense_analysis.py --dir "DRIADA data" \
        --n-shuffles-stage2 5000 \
        --pval 0.0001 \
        --ds 10 \
        --output-dir results

    # Save single file results to specific output
    python tools/run_intense_analysis.py "DRIADA data/LNOF_J01_4D_aligned.npz" \
        --output results.json
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import driada
from driada.experiment.exp_build import load_exp_from_aligned_data
from driada.information.info_base import MultiTimeSeries
from driada.intense.io import save_results as save_intense_results

# Import utility for parsing IABS filenames
from load_synchronized_experiments import parse_iabs_filename, get_npz_metadata

# Import filter utilities
from disentanglement_filters import (
    build_priority_filter,
    compose_filters,
    build_mi_ratio_filter,
    build_exclusion_filter,
)


# ==============================================================================
# Features to skip from INTENSE analysis (will be aggregated instead)
skip_for_intense = ['x', 'y', 'Reconstructions']

# Aggregation mapping: {(tuple_of_features): 'new_aggregated_name'}
# These features are combined into MultiTimeSeries during experiment construction
# The component features (e.g., 'x', 'y') are consumed and replaced by the combined name ('xy')
aggregate_features = {
    ('x', 'y'): 'xy',
}

# Default INTENSE configuration
DEFAULT_CONFIG = {
    'n_shuffles_stage1': 100,
    'n_shuffles_stage2': 10000,
    'pval_thr': 0.001,            # Strict threshold
    'multicomp_correction': None,  # No correction
    'ds': 5,                       # Downsampling for speed
}


# ==============================================================================
# Disentanglement Pre-Filters
# ==============================================================================
# These filters run BEFORE parallel disentanglement processing. They allow
# experiment-specific rules to pre-decide feature pair outcomes without
# running the full disentanglement algorithm.
#
# Filter Protocol:
#   def my_filter(neuron_selectivities, pair_decisions, renames, **kwargs):
#       - neuron_selectivities: {nid: [feat1, feat2, ...]} - can modify
#       - pair_decisions: {nid: {(f1, f2): 0/0.5/1}} - 0=f1 wins, 1=f2 wins, 0.5=keep both
#       - renames: {nid: {new_name: (old1, old2)}} - for merged features
#
# See tools/disentanglement_filters.py for utility functions.
# ==============================================================================

# General priority rules: first feature wins over second when both present
GENERAL_PRIORITY_RULES = [
    ('headdirection', 'bodydirection'),  # head direction > body direction
    ('freezing', 'rest'),                # freezing > rest
    ('locomotion', 'speed'),             # locomotion > speed
    ('rest', 'speed'),                   # rest > speed
    ('freezing', 'speed'),               # freezing > speed
    ('walk', 'speed'),                   # walk > speed
]

# NOF experiment: specific objects > general 'objects' > 'center'
def nof_filter(neuron_selectivities, pair_decisions, renames, **kwargs):
    """NOF experiment filter: specific objects beat general categories."""
    specific_objs = {'object1', 'object2', 'object3', 'object4'}

    for nid, sels in neuron_selectivities.items():
        has_specific = bool(specific_objs.intersection(sels))

        # Specific object > 'objects'
        if has_specific and 'objects' in sels:
            for obj in specific_objs:
                if obj in sels:
                    pair_decisions[nid][(obj, 'objects')] = 0

        # Any object feature > 'center'
        if (has_specific or 'objects' in sels) and 'center' in sels:
            if 'objects' in sels:
                pair_decisions[nid][('objects', 'center')] = 0
            for obj in specific_objs:
                if obj in sels:
                    pair_decisions[nid][(obj, 'center')] = 0


# 3DM experiment: 3d-place vs place based on MI ratio
def tdm_filter(neuron_selectivities, pair_decisions, renames,
               cell_feat_stats=None, mi_ratio_threshold=1.5, **kwargs):
    """3DM experiment filter: 3d-place vs place based on MI ratio."""
    if cell_feat_stats is None:
        return

    for nid, sels in neuron_selectivities.items():
        if 'place' in sels and '3d-place' in sels:
            mi_2d = cell_feat_stats.get(nid, {}).get('place', {}).get('me', 0)
            mi_3d = cell_feat_stats.get(nid, {}).get('3d-place', {}).get('me', 0)

            if mi_3d >= mi_ratio_threshold * mi_2d:
                pair_decisions[nid][('3d-place', 'place')] = 0
            else:
                pair_decisions[nid][('place', '3d-place')] = 0

        # 3d-place > z (z is a component)
        if 'z' in sels and '3d-place' in sels:
            pair_decisions[nid][('3d-place', 'z')] = 0

        # start_box > 3d-place (discrete trumps continuous place)
        if '3d-place' in sels and 'start_box' in sels:
            pair_decisions[nid][('start_box', '3d-place')] = 0

        # speed > speed_z
        if 'speed' in sels and 'speed_z' in sels:
            pair_decisions[nid][('speed', 'speed_z')] = 0


def get_filter_for_experiment(exp_type):
    """Get the composed filter for a specific experiment type.

    Parameters
    ----------
    exp_type : str
        Experiment type identifier: 'NOF', 'LNOF', '3DM', 'BOF', or None

    Returns
    -------
    callable or None
        Composed filter function, or None for no filtering
    """
    # Always start with general priority rules
    general_filter = build_priority_filter(GENERAL_PRIORITY_RULES)
    filters = [general_filter]

    if exp_type in ('NOF', 'LNOF'):
        filters.append(nof_filter)
    elif exp_type == '3DM':
        filters.append(tdm_filter)
    # BOF and other experiments: just use general rules

    return compose_filters(*filters) if filters else None


def build_feature_list(exp, skip_features):
    """Build feature list excluding specified features."""
    feat_bunch = [
        feat_name for feat_name in exp.dynamic_features.keys()
        if feat_name not in skip_features
    ]
    return feat_bunch


def get_skip_delays(exp):
    """Get list of features to skip delay optimization (MultiTimeSeries)."""
    return [
        feat_name for feat_name, feat_data in exp.dynamic_features.items()
        if isinstance(feat_data, MultiTimeSeries)
    ]


# ==============================================================================
# Summary Helper Functions
# ==============================================================================

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


def extract_fft_stats(info):
    """Extract FFT usage statistics from info dict."""
    timings = info.get('timings', {}) if info else {}
    fft_counts = timings.get('fft_type_counts', {})

    # Parse FFT type counts
    fft_cc_pairs = fft_counts.get('fft_continuous_continuous', 0)
    fft_gd_pairs = fft_counts.get('fft_gcmi_discrete', 0)
    fft_dd_pairs = fft_counts.get('fft_discrete_discrete', 0)
    fft_mts_pairs = fft_counts.get('fft_mts', 0)
    fft_mts_discrete_pairs = fft_counts.get('fft_mts_discrete', 0)
    fft_mts_mts_pairs = fft_counts.get('fft_mts_mts', 0)
    loop_pairs = fft_counts.get('loop', 0)

    total_pairs = sum([fft_cc_pairs, fft_gd_pairs, fft_dd_pairs, fft_mts_pairs,
                       fft_mts_discrete_pairs, fft_mts_mts_pairs, loop_pairs])

    fft_total = fft_cc_pairs + fft_gd_pairs + fft_dd_pairs + fft_mts_pairs + fft_mts_discrete_pairs + fft_mts_mts_pairs
    fft_coverage_pct = 100.0 * fft_total / total_pairs if total_pairs > 0 else 0.0

    return {
        'fft_cc_pairs': fft_cc_pairs,
        'fft_gd_pairs': fft_gd_pairs,
        'fft_dd_pairs': fft_dd_pairs,
        'fft_mts_pairs': fft_mts_pairs,
        'fft_mts_discrete_pairs': fft_mts_discrete_pairs,
        'fft_mts_mts_pairs': fft_mts_mts_pairs,
        'loop_pairs': loop_pairs,
        'total_pairs': total_pairs,
        'fft_coverage_pct': fft_coverage_pct,
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


def run_intense_analysis(exp, config, skip_features, pre_filter_func=None, filter_kwargs=None):
    """Run INTENSE analysis with disentanglement.

    Parameters
    ----------
    exp : Experiment
        Experiment object to analyze
    config : dict
        Configuration parameters for INTENSE
    skip_features : list
        Feature names to exclude from analysis
    pre_filter_func : callable, optional
        Pre-filter function for disentanglement. Runs BEFORE parallel processing.
        See tools/disentanglement_filters.py for utilities.
    filter_kwargs : dict, optional
        Keyword arguments to pass to pre_filter_func.

    Returns
    -------
    tuple
        (stats, significance, info, results, disent_results, timings)
    """
    feat_bunch = build_feature_list(exp, skip_features)
    skip_delays = get_skip_delays(exp)

    print(f"\nFeatures to analyze: {feat_bunch}")
    print(f"Features skipped: {skip_features}")
    print(f"Skip delays for: {skip_delays}")
    if pre_filter_func:
        print(f"Pre-filter: {pre_filter_func.__name__ if hasattr(pre_filter_func, '__name__') else 'composed'}")

    stats, significance, info, results, disent_results = driada.compute_cell_feat_significance(
        exp,
        feat_bunch=feat_bunch,
        mode='two_stage',
        n_shuffles_stage1=config['n_shuffles_stage1'],
        n_shuffles_stage2=config['n_shuffles_stage2'],
        allow_mixed_dimensions=True,
        find_optimal_delays=True,
        skip_delays=skip_delays,
        ds=config['ds'],
        pval_thr=config['pval_thr'],
        multicomp_correction=config['multicomp_correction'],
        with_disentanglement=True,
        enable_parallelization=True,
        verbose=True,
        engine=config['engine'],
        profile=True,
        pre_filter_func=pre_filter_func,
        filter_kwargs=filter_kwargs,
    )

    # Extract timing info from results if available
    timings = {}
    if hasattr(results, 'timings'):
        timings = results.timings
    elif info and 'timings' in info:
        timings = info['timings']

    return stats, significance, info, results, disent_results, timings


def print_results(exp, stats, significance, info, results, disent_results):
    """Print analysis results summary."""
    # Print significant neurons
    significant_neurons = exp.get_significant_neurons()
    print(f"\n{'='*60}")
    print("SIGNIFICANT NEURON-FEATURE PAIRS")
    print('='*60)
    total_pairs = sum(len(features) for features in significant_neurons.values())
    print(f"  Significant neurons: {len(significant_neurons)}/{exp.n_cells}")
    print(f"  Total significant pairs: {total_pairs}")

    # Separate single-feature and multi-feature (nontrivial) neurons
    single_feature_neurons = {n: f for n, f in significant_neurons.items() if len(f) == 1}
    multi_feature_neurons = {n: f for n, f in significant_neurons.items() if len(f) >= 2}

    print(f"\n  Single-feature neurons: {len(single_feature_neurons)}")
    print(f"  Multi-feature neurons (nontrivial): {len(multi_feature_neurons)}")

    # Print multi-feature neurons with disentanglement info
    if multi_feature_neurons:
        print(f"\n{'='*60}")
        print("NONTRIVIAL PAIRS (MIXED SELECTIVITY)")
        print('='*60)

        for neuron_id, features in sorted(multi_feature_neurons.items()):
            print(f"\n  Neuron {neuron_id}: {features}")

            # Show MI values for each feature
            for feat in features:
                pair_stats = exp.get_neuron_feature_pair_stats(neuron_id, feat)
                mi = pair_stats.get('me', 0)
                pval = pair_stats.get('pval', 1)
                print(f"    - {feat}: MI={mi:.4f}, p={pval:.2e}")

    # Print disentanglement summary with feature pair details
    if disent_results and 'summary' in disent_results:
        print(f"\n{'='*60}")
        print("DISENTANGLEMENT SUMMARY")
        print('='*60)
        summary = disent_results['summary']

        # Print correlated feature pairs (from feat_feat_significance matrix)
        if 'feat_feat_significance' in disent_results and 'feature_names' in disent_results:
            feat_sig = disent_results['feat_feat_significance']
            feat_names = disent_results['feature_names']
            print(f"\n  Correlated Feature Pairs (behavioral):")
            correlated_pairs = []
            for i in range(len(feat_names)):
                for j in range(i + 1, len(feat_names)):
                    if feat_sig[i, j] == 1:
                        correlated_pairs.append((feat_names[i], feat_names[j]))
            print(f"    Total: {len(correlated_pairs)} correlated pairs")
            for f1, f2 in correlated_pairs:
                print(f"    - {f1} <-> {f2}")

        # Overall stats
        if 'overall_stats' in summary:
            ds = summary['overall_stats']
            print(f"\n  Overall Statistics:")
            print(f"    Total pairs analyzed: {ds.get('total_neuron_pairs', 0)}")
            print(f"    Redundancy rate: {ds.get('redundancy_rate', 0):.1f}%")
            print(f"    True mixed selectivity: {ds.get('true_mixed_selectivity_rate', 0):.1f}%")

        # Feature pair details
        if 'feature_pairs' in summary and summary['feature_pairs']:
            print(f"\n  Feature Pair Analysis:")
            for pair_key, pair_info in summary['feature_pairs'].items():
                n_total = pair_info.get('total_neurons', 0)
                if n_total > 0:
                    parts = pair_key.split('_vs_')
                    feat1, feat2 = parts[0], parts[1]
                    feat1_pct = pair_info.get(f'{feat1}_primary', 0)
                    feat2_pct = pair_info.get(f'{feat2}_primary', 0)
                    undist_pct = pair_info.get('undistinguishable_pct', 0)
                    redundant_pct = pair_info.get('redundant_pct', 0)
                    print(f"    {pair_key}: (n={n_total})")
                    print(f"      {feat1} primary: {feat1_pct:.0f}%, {feat2} primary: {feat2_pct:.0f}%")
                    print(f"      Undistinguishable: {undist_pct:.0f}%, Redundant: {redundant_pct:.0f}%")

    # Print optimal delays info
    if info and 'optimal_delays' in info:
        print(f"\n{'='*60}")
        print("OPTIMAL DELAYS")
        print('='*60)
        print(f"  Delay matrix shape: {info['optimal_delays'].shape}")


def save_results(output_path, exp, stats, significance, info, results, disent_results):
    """Save results to JSON file (legacy format)."""
    significant_neurons = exp.get_significant_neurons()

    output_data = {
        'experiment': exp.signature,
        'n_cells': exp.n_cells,
        'n_frames': exp.n_frames,
        'significant_neurons': {
            str(k): list(v) for k, v in significant_neurons.items()
        },
        'total_significant_pairs': sum(len(f) for f in significant_neurons.values()),
    }

    if disent_results and 'summary' in disent_results:
        summary = disent_results['summary']
        if 'overall_stats' in summary:
            output_data['disentanglement'] = summary['overall_stats']

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# ==============================================================================
# Disentangled Stats/Significance Functions
# ==============================================================================

def build_disentangled_stats(stats, significance, disent_results, exp):
    """Build stats/significance dicts with disentanglement applied.

    Uses `final_sels` from disentanglement results to determine which features
    to keep per neuron. Redundant features are removed, merged features get
    combined stats from their component features.

    Parameters
    ----------
    stats : dict
        Original stats dict: stats[cell_id][feat_name] -> {me, pval, ...}
    significance : dict
        Original significance dict: significance[cell_id][feat_name] -> {...}
    disent_results : dict
        Disentanglement results containing 'per_neuron_disent'
    exp : Experiment
        Experiment object for getting significant neurons

    Returns
    -------
    tuple
        (disent_stats, disent_significance) with redundant features removed
        and merged features combined
    """
    per_neuron_disent = disent_results.get('per_neuron_disent', {})
    significant_neurons = exp.get_significant_neurons()

    disent_stats = {}
    disent_significance = {}

    for nid in stats.keys():
        disent_stats[nid] = {}
        disent_significance[nid] = {}

        # Get final selectivities for this neuron
        if nid in per_neuron_disent:
            final_sels = per_neuron_disent[nid]['final_sels']
            renames = per_neuron_disent[nid].get('renames', {})
        else:
            # Neuron not in disent results (single-feature or not processed)
            # Use original significant features
            final_sels = list(significant_neurons.get(nid, []))
            renames = {}

        for feat in final_sels:
            if feat in renames:
                # Merged feature - combine component stats
                old1, old2 = renames[feat]
                combined_stats = _combine_feature_stats(
                    stats[nid].get(old1, {}),
                    stats[nid].get(old2, {})
                )
                combined_sig = _combine_feature_significance(
                    significance[nid].get(old1, {}),
                    significance[nid].get(old2, {})
                )
                disent_stats[nid][feat] = combined_stats
                disent_significance[nid][feat] = combined_sig
            else:
                # Regular feature - copy from original if exists
                if feat in stats[nid]:
                    disent_stats[nid][feat] = stats[nid][feat].copy()
                if feat in significance[nid]:
                    disent_significance[nid][feat] = significance[nid][feat]

    return disent_stats, disent_significance


def _combine_feature_stats(stats1, stats2):
    """Combine stats from two features into one merged entry.

    Uses max for MI values, min for p-values.
    """
    combined = {}

    # MI: use max
    if 'me' in stats1 or 'me' in stats2:
        combined['me'] = max(stats1.get('me', 0), stats2.get('me', 0))

    # p-value: use min (most significant)
    if 'pval' in stats1 or 'pval' in stats2:
        combined['pval'] = min(stats1.get('pval', 1), stats2.get('pval', 1))

    # Relative MI: use max
    for key in ['rel_me_beh', 'rel_me_ca']:
        if key in stats1 or key in stats2:
            combined[key] = max(stats1.get(key, 0), stats2.get(key, 0))

    # Delay: use from feature with higher MI
    if stats1.get('me', 0) >= stats2.get('me', 0):
        if 'delay' in stats1:
            combined['delay'] = stats1['delay']
    else:
        if 'delay' in stats2:
            combined['delay'] = stats2['delay']

    # Copy other keys from the dominant feature
    dominant = stats1 if stats1.get('me', 0) >= stats2.get('me', 0) else stats2
    for key in dominant:
        if key not in combined:
            combined[key] = dominant[key]

    # Mark as merged
    combined['merged_from'] = [
        stats1.get('feature_name', 'feat1'),
        stats2.get('feature_name', 'feat2')
    ]

    return combined


def _combine_feature_significance(sig1, sig2):
    """Combine significance from two features.

    Both features being significant means the merged feature is significant.
    """
    # If either is a simple bool, convert to dict
    if isinstance(sig1, bool):
        sig1 = {'significant': sig1}
    if isinstance(sig2, bool):
        sig2 = {'significant': sig2}

    combined = {}

    # Stage significance: both must be significant
    for stage in ['stage1', 'stage2']:
        if stage in sig1 or stage in sig2:
            combined[stage] = sig1.get(stage, False) and sig2.get(stage, False)

    # Overall significant if stage2 (or stage1 if no stage2)
    if 'stage2' in combined:
        combined['significant'] = combined['stage2']
    elif 'stage1' in combined:
        combined['significant'] = combined['stage1']

    return combined


def save_disentangled_tables(disent_stats, disent_significance, feat_names, exp_name, output_dir):
    """Save disentangled stats and significance to CSV files.

    Parameters
    ----------
    disent_stats : dict
        Disentangled stats dict
    disent_significance : dict
        Disentangled significance dict
    feat_names : list
        List of feature names (original + any merged names)
    exp_name : str
        Experiment name for file naming
    output_dir : Path
        Output directory (will create tables_disentangled/ subdirectory)
    """
    output_dir = Path(output_dir)
    tables_dir = output_dir / 'tables_disentangled'
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Collect all feature names (including merged ones)
    all_feats = set(feat_names)
    for nid_stats in disent_stats.values():
        all_feats.update(nid_stats.keys())
    all_feat_names = sorted(all_feats)

    # Save stats CSV
    save_stats_csv(disent_stats, all_feat_names, tables_dir / f'{exp_name} INTENSE stats.csv')

    # Save significance CSV
    save_significance_csv(disent_significance, all_feat_names, tables_dir / f'{exp_name} INTENSE significance.csv')

    print(f"  Disentangled tables saved to: {tables_dir}")


def get_exp_name(npz_path):
    """Extract experiment name from NPZ filename.

    Examples: 'NOF_H01_1D syn data.npz' -> 'NOF_H01_1D'
              'LNOF_J01_1D_aligned.npz' -> 'LNOF_J01_1D'
    """
    name = Path(npz_path).stem
    # Remove common suffixes
    for suffix in [' syn data', '_syn_data', '_aligned', ' aligned']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def save_stats_csv(stats_dict, feat_names, output_path):
    """Save stats dict to CSV matching reference format.

    Parameters
    ----------
    stats_dict : dict
        Nested dict: stats_dict[cell_id][feat_name] -> {me, pval, stage1, stage2, ...}
    feat_names : list
        List of feature names (column headers)
    output_path : Path
        Output CSV file path
    """
    # Build DataFrame from nested dict
    rows = []
    cell_ids = sorted(stats_dict.keys(), key=int)

    for cell_id in cell_ids:
        row = {}
        for feat in feat_names:
            if feat in stats_dict[cell_id]:
                row[feat] = repr(stats_dict[cell_id][feat])
            else:
                row[feat] = repr({})
        rows.append(row)

    df = pd.DataFrame(rows, index=cell_ids)
    df.index.name = None
    df.to_csv(output_path)
    print(f"  Saved: {output_path}")


def save_significance_csv(sig_dict, feat_names, output_path):
    """Save significance dict to CSV matching reference format.

    Parameters
    ----------
    sig_dict : dict
        Nested dict: sig_dict[cell_id][feat_name] -> significance value/dict
    feat_names : list
        List of feature names (column headers)
    output_path : Path
        Output CSV file path
    """
    # Build DataFrame from nested dict
    rows = []
    cell_ids = sorted(sig_dict.keys(), key=int)

    for cell_id in cell_ids:
        row = {}
        for feat in feat_names:
            if feat in sig_dict[cell_id]:
                val = sig_dict[cell_id][feat]
                # Convert to dict format if it's a simple bool
                if isinstance(val, bool):
                    row[feat] = repr({'significant': val})
                else:
                    row[feat] = repr(val)
            else:
                row[feat] = repr({})
        rows.append(row)

    df = pd.DataFrame(rows, index=cell_ids)
    df.index.name = None
    df.to_csv(output_path)
    print(f"  Saved: {output_path}")


def save_disentanglement(disent_results, exp_name, output_dir):
    """Save all disentanglement outputs.

    Parameters
    ----------
    disent_results : dict
        Disentanglement results from INTENSE analysis
    exp_name : str
        Experiment name for file naming
    output_dir : Path
        Output directory for disentanglement files
    """
    output_dir = Path(output_dir)
    feat_names = disent_results.get('feature_names', [])

    # Save feature-feature significance matrix
    if 'feat_feat_significance' in disent_results:
        feat_sig = disent_results['feat_feat_significance']
        df = pd.DataFrame(feat_sig, index=feat_names, columns=feat_names)
        path = output_dir / f'{exp_name}_feat_feat_sig.csv'
        df.to_csv(path)
        print(f"  Saved: {path}")

    # Save disentanglement matrix
    if 'disent_matrix' in disent_results:
        disent_matrix = disent_results['disent_matrix']
        df = pd.DataFrame(disent_matrix, index=feat_names, columns=feat_names)
        path = output_dir / f'{exp_name}_disent_matrix.csv'
        df.to_csv(path)
        print(f"  Saved: {path}")

    # Save count matrix
    if 'count_matrix' in disent_results:
        count_matrix = disent_results['count_matrix']
        df = pd.DataFrame(count_matrix, index=feat_names, columns=feat_names)
        path = output_dir / f'{exp_name}_count_matrix.csv'
        df.to_csv(path)
        print(f"  Saved: {path}")

    # Save summary as JSON
    if 'summary' in disent_results:
        path = output_dir / f'{exp_name}_summary.json'
        with open(path, 'w') as f:
            json.dump(disent_results['summary'], f, indent=2)
        print(f"  Saved: {path}")


def save_all_results(exp_name, exp, stats, significance, info, results, disent_results, output_dir):
    """Save all INTENSE results to structured output directory.

    Parameters
    ----------
    exp_name : str
        Experiment name for file naming
    exp : Experiment
        Experiment object
    stats : dict
        Stats dict from INTENSE
    significance : dict
        Significance dict from INTENSE
    info : dict
        Info dict from INTENSE
    results : IntenseResults
        IntenseResults object
    disent_results : dict
        Disentanglement results
    output_dir : Path
        Base output directory
    """
    output_dir = Path(output_dir)

    # Create directory structure
    tables_dir = output_dir / 'tables'
    results_dir = output_dir / 'results'
    disent_dir = output_dir / 'disentanglement'

    tables_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    disent_dir.mkdir(exist_ok=True)

    # Get feature names from stats dict (first cell's keys)
    first_cell = next(iter(stats.values()))
    feat_names = list(first_cell.keys())

    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print('='*60)

    # Save stats CSV
    save_stats_csv(stats, feat_names, tables_dir / f'{exp_name} INTENSE stats.csv')

    # Save significance CSV
    save_significance_csv(significance, feat_names, tables_dir / f'{exp_name} INTENSE significance.csv')

    # Save IntenseResults to NPZ (fast serialization)
    npz_path = results_dir / f'{exp_name}_results.npz'
    t_save_start = time.time()
    save_intense_results(results, npz_path)
    t_save = time.time() - t_save_start
    print(f"  Saved: {npz_path} ({t_save:.2f}s)")

    # Save disentanglement outputs
    if disent_results:
        save_disentanglement(disent_results, exp_name, disent_dir)

        # Save disentangled stats/significance tables
        disent_stats, disent_significance = build_disentangled_stats(
            stats, significance, disent_results, exp
        )
        save_disentangled_tables(
            disent_stats, disent_significance, feat_names, exp_name, output_dir
        )

    print(f"\nAll results saved to: {output_dir}")


def load_experiment_from_npz(npz_path, agg_features=None, verbose=True):
    """Load experiment from NPZ file.

    Parameters
    ----------
    npz_path : Path
        Path to the .npz file
    agg_features : dict, optional
        Feature aggregation mapping {(key1, key2): 'combined_name'}
        These features are combined into MultiTimeSeries during construction.
    verbose : bool
        Whether to print loading info

    Returns
    -------
    Experiment
        Loaded experiment object
    """
    # Parse filename to get exp_params
    parsed = parse_iabs_filename(npz_path.name)
    if parsed is None:
        raise ValueError(f"Could not parse filename: {npz_path.name}")

    # Load data
    data = dict(np.load(npz_path, allow_pickle=True))

    # Get metadata for fps
    metadata = get_npz_metadata(npz_path)

    # Build static features
    static_features = {}
    if metadata['fps'] is not None:
        static_features['fps'] = float(metadata['fps'])

    # Build exp_params
    exp_params = parsed.exp_params.copy()

    # Remove metadata keys from data dict
    data.pop('_metadata', None)
    data.pop('_sync_info', None)

    # Create experiment with aggregate_features parameter
    # This creates MultiTimeSeries during construction, ensuring proper data hashes
    exp = load_exp_from_aligned_data(
        data_source='IABS',
        exp_params=exp_params,
        data=data,
        static_features=static_features,
        aggregate_features=agg_features,
        verbose=verbose,
    )

    return exp


def process_single_experiment(npz_path, config, output_dir=None, plot=False, use_filters=True):
    """Process a single experiment file.

    Parameters
    ----------
    npz_path : Path
        Path to the NPZ file
    config : dict
        INTENSE configuration
    output_dir : Path, optional
        Output directory for saving results
    plot : bool
        Whether to plot disentanglement heatmap
    use_filters : bool
        Whether to use experiment-specific disentanglement filters (default: True)

    Returns
    -------
    dict
        Summary of results
    """
    npz_path = Path(npz_path)
    exp_name = get_exp_name(npz_path)

    print(f"\n{'='*60}")
    print(f"PROCESSING: {exp_name}")
    print('='*60)

    # Load experiment
    print(f"\nLoading experiment...")
    t_start = time.time()
    exp = load_experiment_from_npz(npz_path, agg_features=aggregate_features)
    t_load = time.time() - t_start

    print(f"  Loaded: {exp.signature}")
    print(f"  Neurons: {exp.n_cells}, Frames: {exp.n_frames}, FPS: {exp.fps}")

    # Get experiment type and filter
    pre_filter_func = None
    filter_kwargs = None
    if use_filters:
        # Extract experiment type from name (e.g., 'NOF_H01_1D' -> 'NOF')
        exp_type = exp_name.split('_')[0] if '_' in exp_name else None
        pre_filter_func = get_filter_for_experiment(exp_type)
        if pre_filter_func:
            print(f"  Using filter for experiment type: {exp_type}")
            # filter_kwargs can include thresholds, pre-extracted data, etc.
            filter_kwargs = {'mi_ratio_threshold': 1.5}

    # Run INTENSE analysis
    print(f"\nRunning INTENSE analysis...")
    t_start = time.time()
    stats, significance, info, results, disent_results, timings = run_intense_analysis(
        exp, config, skip_for_intense,
        pre_filter_func=pre_filter_func,
        filter_kwargs=filter_kwargs,
    )
    t_intense = time.time() - t_start

    # Print results summary
    print_results(exp, stats, significance, info, results, disent_results)

    # Compute and display per-file summary
    summary_dict = compute_summary_metrics(
        exp, stats, significance, info, disent_results, t_load, t_intense
    )
    summary_dict['exp_name'] = exp_name
    print_per_file_summary(summary_dict, exp_name)

    # Save results if output directory specified
    if output_dir:
        save_all_results(exp_name, exp, stats, significance, info, results, disent_results, output_dir)

    # Plot disentanglement heatmap if requested
    if plot and disent_results:
        plot_disentanglement(disent_results, str(output_dir / 'disentanglement' / exp_name) if output_dir else None)

    # Print timing summary
    t_total = t_load + t_intense
    print(f"\n  Timing: Load {t_load:.1f}s, INTENSE {t_intense:.1f}s, Total {t_total:.1f}s")

    # Return full comprehensive summary
    return summary_dict


def main():
    parser = argparse.ArgumentParser(
        description='Run INTENSE analysis on NPZ data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python tools/run_intense_analysis.py "DRIADA data/NOF_H01_1D syn data.npz" --output-dir INTENSE

  # Process all NPZ files in a directory
  python tools/run_intense_analysis.py --dir "DRIADA data" --output-dir INTENSE

  # Multiple files (batch mode)
  python tools/run_intense_analysis.py "DRIADA data/NOF_H01_1D syn data.npz" "DRIADA data/NOF_H01_2D syn data.npz" --output-dir INTENSE

  # Glob pattern
  python tools/run_intense_analysis.py "DRIADA data/NOF_*.npz" --output-dir INTENSE

  # Legacy single-file mode with JSON output
  python tools/run_intense_analysis.py "DRIADA data/LNOF_J01_1D_aligned.npz" --output results.json
        """
    )
    parser.add_argument('npz_paths', nargs='*', type=str,
                        help='Path(s) to .npz file(s), supports glob patterns')
    parser.add_argument('--dir', type=str,
                        help='Process all .npz files in this directory (batch mode)')
    parser.add_argument('--n-shuffles-stage1', type=int, default=DEFAULT_CONFIG['n_shuffles_stage1'],
                        help=f"Stage 1 shuffles (default: {DEFAULT_CONFIG['n_shuffles_stage1']})")
    parser.add_argument('--n-shuffles-stage2', type=int, default=DEFAULT_CONFIG['n_shuffles_stage2'],
                        help=f"Stage 2 shuffles (default: {DEFAULT_CONFIG['n_shuffles_stage2']})")
    parser.add_argument('--pval', type=float, default=DEFAULT_CONFIG['pval_thr'],
                        help=f"P-value threshold (default: {DEFAULT_CONFIG['pval_thr']})")
    parser.add_argument('--ds', type=int, default=DEFAULT_CONFIG['ds'],
                        help=f"Downsampling factor (default: {DEFAULT_CONFIG['ds']})")
    parser.add_argument('--output-dir', type=str, help='Output directory for batch results')
    parser.add_argument('--output', type=str, help='Output file for results (JSON, legacy mode)')
    parser.add_argument('--plot', action='store_true', help='Plot disentanglement heatmap')
    parser.add_argument('--engine', type=str, default='auto', choices=['auto', 'fft', 'loop'],
                        help='Computation engine: auto (default), fft, or loop')
    parser.add_argument('--no-filters', action='store_true',
                        help='Disable experiment-specific disentanglement filters')
    args = parser.parse_args()

    config = {
        'n_shuffles_stage1': args.n_shuffles_stage1,
        'n_shuffles_stage2': args.n_shuffles_stage2,
        'pval_thr': args.pval,
        'multicomp_correction': None,  # No correction
        'ds': args.ds,
        'engine': args.engine,
    }

    # Expand paths from --dir option or from positional arguments
    all_paths = []

    if args.dir:
        # Process all .npz files in directory
        dir_path = Path(args.dir)
        if not dir_path.exists():
            print(f"Error: Directory not found: {args.dir}")
            sys.exit(1)
        if not dir_path.is_dir():
            print(f"Error: Not a directory: {args.dir}")
            sys.exit(1)
        all_paths = list(dir_path.glob('*.npz'))
        if not all_paths:
            print(f"Error: No .npz files found in {args.dir}")
            sys.exit(1)
    else:
        # Expand glob patterns from positional arguments
        if not args.npz_paths:
            print("Error: Either provide npz_paths or use --dir option")
            parser.print_help()
            sys.exit(1)
        for pattern in args.npz_paths:
            expanded = glob.glob(pattern)
            if expanded:
                all_paths.extend(expanded)
            else:
                # Not a glob, use as-is
                all_paths.append(pattern)

    # Filter to only .npz files and convert to strings
    npz_paths = [str(p) for p in all_paths if str(p).endswith('.npz')]

    if not npz_paths:
        print("Error: No .npz files found")
        sys.exit(1)

    # Print configuration
    print("=" * 60)
    print("INTENSE BATCH ANALYSIS")
    print("=" * 60)
    print(f"Input files: {len(npz_paths)}")
    for p in npz_paths:
        print(f"  - {p}")
    print(f"\nConfiguration:")
    print(f"  Stage 1 shuffles: {config['n_shuffles_stage1']}")
    print(f"  Stage 2 shuffles: {config['n_shuffles_stage2']}")
    print(f"  P-value threshold: {config['pval_thr']}")
    print(f"  Multiple comparison correction: {config['multicomp_correction']}")
    print(f"  Downsampling factor: {config['ds']}")
    print(f"  Engine: {config['engine']}")
    print(f"  Skip features: {skip_for_intense}")
    print(f"  Aggregate features: {aggregate_features}")
    print(f"  Disentanglement filters: {'disabled' if args.no_filters else 'enabled (experiment-specific)'}")
    if args.output_dir:
        print(f"  Output directory: {args.output_dir}")

    # Validate files exist
    for npz_path in npz_paths:
        if not Path(npz_path).exists():
            print(f"Error: File not found: {npz_path}")
            sys.exit(1)

    # Process each file
    output_dir = Path(args.output_dir) if args.output_dir else None
    summaries = []
    t_batch_start = time.time()

    use_filters = not args.no_filters
    for i, npz_path in enumerate(npz_paths):
        print(f"\n[{i+1}/{len(npz_paths)}] Processing {Path(npz_path).name}")
        summary = process_single_experiment(npz_path, config, output_dir, args.plot, use_filters)
        summaries.append(summary)

        # Legacy JSON output for single file mode
        if args.output and len(npz_paths) == 1:
            # Re-run to get full results for JSON save (not ideal but maintains compatibility)
            exp = load_experiment_from_npz(Path(npz_path), agg_features=aggregate_features, verbose=False)
            exp_name = get_exp_name(Path(npz_path))
            exp_type = exp_name.split('_')[0] if '_' in exp_name else None
            pre_filter = get_filter_for_experiment(exp_type) if use_filters else None
            stats, significance, info, results, disent_results, _ = run_intense_analysis(
                exp, config, skip_for_intense,
                pre_filter_func=pre_filter,
                filter_kwargs={'mi_ratio_threshold': 1.5} if pre_filter else None,
            )
            save_results(args.output, exp, stats, significance, info, results, disent_results)

    t_batch_total = time.time() - t_batch_start

    # Save batch CSV summary
    if output_dir:
        save_batch_summary_csv(summaries, output_dir / 'batch_summary.csv')

    # Print enhanced batch summary
    print_batch_summary(summaries, t_batch_total, output_dir)

    print(f"\n{'='*60}")
    print("BATCH ANALYSIS COMPLETE")
    print('='*60)


def plot_disentanglement(disent_results, output_base=None):
    """Plot disentanglement heatmap."""
    import matplotlib.pyplot as plt

    disent_matrix = disent_results.get('disent_matrix')
    count_matrix = disent_results.get('count_matrix')
    feat_names = disent_results.get('feature_names')

    if disent_matrix is None or count_matrix is None:
        print("No disentanglement matrices to plot")
        return

    print(f"\n{'='*60}")
    print("PLOTTING DISENTANGLEMENT HEATMAP")
    print('='*60)

    fig, ax = driada.intense.plot_disentanglement_heatmap(
        disent_matrix,
        count_matrix,
        feat_names,
        title="Feature Disentanglement Analysis",
        figsize=(12, 10),
    )

    if output_base:
        # Save next to output file or with default name
        plot_path = output_base.replace('.json', '_disentanglement.png') if output_base.endswith('.json') else f"{output_base}_disentanglement.png"
    else:
        plot_path = "disentanglement_heatmap.png"

    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.show()


if __name__ == '__main__':
    main()
