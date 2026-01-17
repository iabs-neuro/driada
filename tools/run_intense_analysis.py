#!/usr/bin/env python
"""
Run full INTENSE analysis on .npz experiment data.

Loads experiment data, creates aggregated features, and runs
INTENSE analysis with disentanglement to identify neural selectivity.

Usage
-----
    python tools/run_intense_analysis.py "DRIADA data/LNOF_J01_4D_aligned.npz"

    # With custom parameters
    python tools/run_intense_analysis.py "DRIADA data/LNOF_J05_1D_aligned.npz" \
        --n-shuffles-stage2 5000 \
        --pval 0.0001 \
        --ds 10

    # Save results to file
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


def run_intense_analysis(exp, config, skip_features):
    """Run INTENSE analysis with disentanglement.

    Parameters
    ----------
    exp : Experiment
        Experiment object to analyze
    config : dict
        Configuration parameters for INTENSE
    skip_features : list
        Feature names to exclude from analysis

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


def process_single_experiment(npz_path, config, output_dir=None, plot=False):
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

    # Run INTENSE analysis
    print(f"\nRunning INTENSE analysis...")
    t_start = time.time()
    stats, significance, info, results, disent_results, timings = run_intense_analysis(
        exp, config, skip_for_intense
    )
    t_intense = time.time() - t_start

    # Print results summary
    print_results(exp, stats, significance, info, results, disent_results)

    # Save results if output directory specified
    if output_dir:
        save_all_results(exp_name, exp, stats, significance, info, results, disent_results, output_dir)

    # Plot disentanglement heatmap if requested
    if plot and disent_results:
        plot_disentanglement(disent_results, str(output_dir / 'disentanglement' / exp_name) if output_dir else None)

    # Print timing summary
    t_total = t_load + t_intense
    print(f"\n  Timing: Load {t_load:.1f}s, INTENSE {t_intense:.1f}s, Total {t_total:.1f}s")

    return {
        'exp_name': exp_name,
        'n_cells': exp.n_cells,
        'n_frames': exp.n_frames,
        'n_significant': len(exp.get_significant_neurons()),
        't_load': t_load,
        't_intense': t_intense,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run INTENSE analysis on NPZ data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python tools/run_intense_analysis.py "DRIADA data/NOF_H01_1D syn data.npz" --output-dir INTENSE

  # Multiple files (batch mode)
  python tools/run_intense_analysis.py "DRIADA data/NOF_H01_1D syn data.npz" "DRIADA data/NOF_H01_2D syn data.npz" --output-dir INTENSE

  # Glob pattern
  python tools/run_intense_analysis.py "DRIADA data/NOF_*.npz" --output-dir INTENSE

  # Legacy single-file mode with JSON output
  python tools/run_intense_analysis.py "DRIADA data/LNOF_J01_1D_aligned.npz" --output results.json
        """
    )
    parser.add_argument('npz_paths', nargs='+', type=str, help='Path(s) to .npz file(s)')
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
    args = parser.parse_args()

    config = {
        'n_shuffles_stage1': args.n_shuffles_stage1,
        'n_shuffles_stage2': args.n_shuffles_stage2,
        'pval_thr': args.pval,
        'multicomp_correction': None,  # No correction
        'ds': args.ds,
    }

    # Expand glob patterns
    all_paths = []
    for pattern in args.npz_paths:
        expanded = glob.glob(pattern)
        if expanded:
            all_paths.extend(expanded)
        else:
            # Not a glob, use as-is
            all_paths.append(pattern)

    # Filter to only .npz files
    npz_paths = [p for p in all_paths if p.endswith('.npz')]

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
    print(f"  Skip features: {skip_for_intense}")
    print(f"  Aggregate features: {aggregate_features}")
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

    for i, npz_path in enumerate(npz_paths):
        print(f"\n[{i+1}/{len(npz_paths)}] Processing {Path(npz_path).name}")
        summary = process_single_experiment(npz_path, config, output_dir, args.plot)
        summaries.append(summary)

        # Legacy JSON output for single file mode
        if args.output and len(npz_paths) == 1:
            # Re-run to get full results for JSON save (not ideal but maintains compatibility)
            exp = load_experiment_from_npz(Path(npz_path), agg_features=aggregate_features, verbose=False)
            stats, significance, info, results, disent_results, _ = run_intense_analysis(
                exp, config, skip_for_intense
            )
            save_results(args.output, exp, stats, significance, info, results, disent_results)

    t_batch_total = time.time() - t_batch_start

    # Print batch summary
    print(f"\n{'='*60}")
    print("BATCH SUMMARY")
    print('='*60)
    print(f"  Experiments processed: {len(summaries)}")
    print(f"  Total time: {t_batch_total:.1f}s")
    print(f"\n  Per-experiment summary:")
    for s in summaries:
        print(f"    {s['exp_name']}: {s['n_significant']}/{s['n_cells']} significant ({s['t_intense']:.1f}s)")

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
