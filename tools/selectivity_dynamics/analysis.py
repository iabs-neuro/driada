"""Core INTENSE analysis functions.

Contains:
- run_intense_analysis: Run INTENSE with disentanglement
- print_results: Display significant neurons and disentanglement summary
- build_disentangled_stats: Apply disentanglement to raw results
- _combine_feature_stats: Merge stats from two features
- _combine_feature_significance: Combine significance from merged features
"""

import sys
import warnings
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import numpy as np
import driada

from .loader import build_feature_list


def _fix_normalized_circular_features(exp):
    """Detect and rescale circular features erroneously stored in [0, 1].

    Some data pipelines normalize circular angles to [0, 1] instead of
    keeping them in radians. On this range cos/sin are monotonically
    related, making the _2d encoding degenerate. This function detects
    such features and rescales them to [0, 2*pi].

    Parameters
    ----------
    exp : Experiment
        Experiment object (modified in-place).

    Returns
    -------
    list of str
        Names of features that were rescaled.
    """
    from driada.information import TimeSeries, MultiTimeSeries
    from driada.information.circular_transform import circular_to_cos_sin

    rescaled = []

    for name, ts in list(exp.dynamic_features.items()):
        if name.endswith("_2d"):
            continue
        if not isinstance(ts, TimeSeries):
            continue
        if ts.discrete:
            continue
        if not (hasattr(ts, "type_info") and ts.type_info and ts.type_info.is_circular):
            continue

        data_min = np.nanmin(ts.data)
        data_max = np.nanmax(ts.data)
        data_range = data_max - data_min

        # Detect [0, 1] normalized circular data:
        # range ≈ 1, min ≈ 0, max ≈ 1 (with some tolerance)
        if data_range < 1.5 and data_min >= -0.1 and data_max <= 1.1:
            # Rescale to [0, 2*pi]
            ts.data = ts.data * (2 * np.pi)
            ts.type_info.circular_period = 2 * np.pi

            # Rebuild copula normal cache
            from driada.information.gcmi import copnorm
            ts.copula_normal_data = copnorm(ts.data).ravel()

            # Rebuild _2d version if it exists
            name_2d = f"{name}_2d"
            if name_2d in exp.dynamic_features:
                new_2d = circular_to_cos_sin(ts.data, period=2 * np.pi, name=name_2d)
                exp.dynamic_features[name_2d] = new_2d

            rescaled.append(name)

    if rescaled:
        warnings.warn(
            f"Circular features rescaled from [0,1] to [0,2π]: {rescaled}. "
            f"The source data likely has a normalization error.",
            UserWarning,
        )

    return rescaled


def run_intense_analysis(exp, config, skip_features, pre_filter_func=None, post_filter_func=None, filter_kwargs=None):
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
        See tools/selectivity_dynamics/filters.py for utilities.
    post_filter_func : callable, optional
        Post-filter function for disentanglement. Runs AFTER parallel processing
        to modify results (e.g., tie-breaking). See filters.py for utilities.
    filter_kwargs : dict, optional
        Keyword arguments to pass to pre_filter_func and post_filter_func.

    Returns
    -------
    tuple
        (stats, significance, info, results, disent_results, timings)
    """
    # Fix circular features erroneously normalized to [0, 1]
    _fix_normalized_circular_features(exp)

    metric = config.get('metric', 'mi')
    feat_bunch = build_feature_list(exp, skip_features)

    # Non-MI metrics cannot handle multi-dimensional features
    use_circular_2d = True
    if metric != 'mi':
        use_circular_2d = False
        from driada.information import MultiTimeSeries
        multidim = [
            name for name, ts in exp.dynamic_features.items()
            if isinstance(ts, MultiTimeSeries) and name in feat_bunch
        ]
        if multidim:
            feat_bunch = [f for f in feat_bunch if f not in multidim]
            print(f"\nSkipping multi-dimensional features (incompatible with {metric}): {multidim}")

    print(f"\nFeatures to analyze: {feat_bunch}")
    print(f"Features skipped: {skip_features}")
    if pre_filter_func:
        print(f"Pre-filter: {pre_filter_func.__name__ if hasattr(pre_filter_func, '__name__') else 'composed'}")
    if post_filter_func:
        print(f"Post-filter: {post_filter_func.__name__ if hasattr(post_filter_func, '__name__') else 'composed'}")

    with_disentanglement = config.get('with_disentanglement', True)
    result = driada.compute_cell_feat_significance(
        exp,
        feat_bunch=feat_bunch,
        metric=metric,
        mode='two_stage',
        n_shuffles_stage1=config['n_shuffles_stage1'],
        n_shuffles_stage2=config['n_shuffles_stage2'],
        find_optimal_delays=True,
        ds=config['ds'],
        pval_thr=config['pval_thr'],
        multicomp_correction=config['multicomp_correction'],
        with_disentanglement=with_disentanglement,
        enable_parallelization=True,
        verbose=True,
        engine=config['engine'],
        profile=True,
        pre_filter_func=pre_filter_func,
        post_filter_func=post_filter_func,
        filter_kwargs=filter_kwargs,
        use_circular_2d=use_circular_2d,
    )
    if with_disentanglement:
        stats, significance, info, results, disent_results = result
    else:
        stats, significance, info, results = result
        disent_results = None

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


def build_disentangled_stats(stats, significance, disent_results, exp):
    """Build stats/significance dicts with disentanglement applied.

    Uses `final_sels` and `pairs` from disentanglement results to determine
    which features to keep per neuron. Redundant features are removed based
    on pair decisions, merged features get combined stats.

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
