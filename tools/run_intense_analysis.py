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

    # Resume batch processing, skipping already-computed files
    python tools/run_intense_analysis.py --dir "DRIADA data" --output-dir INTENSE --skip-computed

    # Save single file results to specific output
    python tools/run_intense_analysis.py "DRIADA data/LNOF_J01_4D_aligned.npz" \
        --output results.json
"""

import argparse
import gc
import glob
import os
import sys
import time
from pathlib import Path

# Limit BLAS threads to prevent conflicts with joblib parallelism
# Must be set before importing numpy/scipy
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Import from selectivity_dynamics package
from selectivity_dynamics import (
    DEFAULT_CONFIG,
    get_experiment_config,
    get_filter_for_experiment,
    extract_filter_data,
    load_experiment_from_npz,
    run_intense_analysis,
    print_results,
    compute_summary_metrics,
    print_per_file_summary,
    print_batch_summary,
    save_batch_summary_csv,
    load_batch_summary_csv,
    save_all_results,
    save_results,
    get_exp_name,
    plot_disentanglement,
)


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

    # Extract experiment type from name (e.g., 'NOF_H01_1D' -> 'NOF')
    exp_type = exp_name.split('_')[0] if '_' in exp_name else None

    # Get experiment config
    exp_config = get_experiment_config(exp_type)
    print(f"  Experiment type: {exp_type}")

    # Load experiment with config-based aggregation
    print(f"\nLoading experiment...")
    t_start = time.time()
    exp = load_experiment_from_npz(npz_path, agg_features=exp_config['aggregate_features'])
    t_load = time.time() - t_start

    print(f"  Loaded: {exp.signature}")
    print(f"  Neurons: {exp.n_cells}, Frames: {exp.n_frames}, FPS: {exp.fps}")

    # Get filter and build filter_kwargs from config
    pre_filter_func = None
    filter_kwargs = None
    if use_filters:
        pre_filter_func = get_filter_for_experiment(exp_type)
        print(f"  Using filter for experiment type: {exp_type}")

        # Build filter_kwargs from config
        filter_kwargs = {
            'mi_ratio_threshold': 1.5,
            'place_feat_name': exp_config['place_feat_name'],
            'discrete_place_features': exp_config['discrete_place_features'],
            'feature_renaming': exp_config['feature_renaming'],
            'correspondence_threshold': 0.4,
        }

        # Extract calcium/feature data for spatial_filter if needed
        if exp_config['discrete_place_features']:
            spatial_data = extract_filter_data(exp, discrete_place_features=exp_config['discrete_place_features'])
            filter_kwargs.update(spatial_data)
            print(f"  Extracted spatial filter data for {len(spatial_data['calcium_data'])} neurons")

    # Run INTENSE analysis
    print(f"\nRunning INTENSE analysis...")
    t_start = time.time()
    stats, significance, info, results, disent_results, timings = run_intense_analysis(
        exp, config, exp_config['skip_for_intense'],
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

    # Clear TimeSeries caches to prevent memory accumulation
    for neuron in exp.neurons:
        if hasattr(neuron.ca, 'clear_caches'):
            neuron.ca.clear_caches()
    for feature in exp.dynamic_features.values():
        if hasattr(feature, 'clear_caches'):
            feature.clear_caches()
    del exp

    # Return full comprehensive summary
    return summary_dict


def is_already_processed(exp_name, output_dir):
    """Check if experiment already has results in output directory.

    Parameters
    ----------
    exp_name : str
        Experiment name (e.g., 'NOF_H01_1D')
    output_dir : Path
        Output directory path

    Returns
    -------
    bool
        True if results NPZ file exists
    """
    results_path = Path(output_dir) / 'results' / f'{exp_name}_results.npz'
    return results_path.exists()


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
    parser.add_argument('--skip-computed', action='store_true',
                        help='Skip files that already have results in output directory')
    parser.add_argument('--parallel-backend', type=str, default='loky',
                        choices=['loky', 'threading', 'multiprocessing'],
                        help='Joblib parallel backend: loky (default, true parallelism), '
                             'threading (stable, good for NumPy), multiprocessing (legacy)')
    args = parser.parse_args()

    # Set parallel backend before any heavy computation
    import driada
    driada.set_parallel_backend(args.parallel_backend)

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
    print(f"  Skip/aggregate features: experiment-specific (from EXPERIMENT_CONFIGS)")
    print(f"  Disentanglement filters: {'disabled' if args.no_filters else 'enabled (experiment-specific)'}")
    print(f"  Skip computed: {args.skip_computed}")
    print(f"  Parallel backend: {args.parallel_backend}")
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
    skipped_count = 0
    processed_count = 0

    # Load existing batch summary if --skip-computed and output_dir exists
    existing_summaries = {}
    if args.skip_computed and output_dir:
        batch_csv = output_dir / 'batch_summary.csv'
        if batch_csv.exists():
            existing_list = load_batch_summary_csv(batch_csv)
            existing_summaries = {s['exp_name']: s for s in existing_list if 'exp_name' in s}
            print(f"  Loaded {len(existing_summaries)} existing summaries from {batch_csv}")

    for i, npz_path in enumerate(npz_paths):
        npz_name = Path(npz_path).name
        exp_name = get_exp_name(Path(npz_path))

        # Skip if already processed (when --skip-computed is set)
        if args.skip_computed and output_dir and is_already_processed(exp_name, output_dir):
            print(f"[{i+1}/{len(npz_paths)}] SKIPPED (exists): {npz_name}")
            skipped_count += 1
            # Add existing summary if available
            if exp_name in existing_summaries:
                summaries.append(existing_summaries[exp_name])
            continue

        print(f"\n[{i+1}/{len(npz_paths)}] Processing {npz_name}")
        summary = process_single_experiment(npz_path, config, output_dir, args.plot, use_filters)
        summaries.append(summary)
        processed_count += 1

        # Save batch summary after each file to preserve progress
        if output_dir:
            save_batch_summary_csv(summaries, output_dir / 'batch_summary.csv')

        # Force garbage collection and worker pool cleanup
        gc.collect()
        time.sleep(2)  # Allow workers to shut down (idle_worker_timeout=60s)

        # Legacy JSON output for single file mode
        if args.output and len(npz_paths) == 1:
            # Re-run to get full results for JSON save (not ideal but maintains compatibility)
            exp_name = get_exp_name(Path(npz_path))
            exp_type = exp_name.split('_')[0] if '_' in exp_name else None
            exp_config = get_experiment_config(exp_type)
            exp = load_experiment_from_npz(Path(npz_path), agg_features=exp_config['aggregate_features'], verbose=False)
            pre_filter = get_filter_for_experiment(exp_type) if use_filters else None
            filter_kwargs = None
            if pre_filter:
                filter_kwargs = {
                    'mi_ratio_threshold': 1.5,
                    'place_feat_name': exp_config['place_feat_name'],
                    'discrete_place_features': exp_config['discrete_place_features'],
                    'feature_renaming': exp_config['feature_renaming'],
                }
            stats, significance, info, results, disent_results, _ = run_intense_analysis(
                exp, config, exp_config['skip_for_intense'],
                pre_filter_func=pre_filter,
                filter_kwargs=filter_kwargs,
            )
            save_results(args.output, exp, stats, significance, info, results, disent_results)

    t_batch_total = time.time() - t_batch_start

    # Save batch CSV summary
    if output_dir:
        save_batch_summary_csv(summaries, output_dir / 'batch_summary.csv')

    # Print enhanced batch summary
    print_batch_summary(summaries, t_batch_total, output_dir)

    # Print skip summary if --skip-computed was used
    if args.skip_computed:
        loaded_count = len(existing_summaries)
        print(f"\nSkip summary: {processed_count} processed, {skipped_count} skipped ({loaded_count} loaded from cache)")

    print(f"\n{'='*60}")
    print("BATCH ANALYSIS COMPLETE")
    print('='*60)


if __name__ == '__main__':
    main()
