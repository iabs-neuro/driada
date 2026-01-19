"""Export functions for saving INTENSE analysis results.

Contains:
- save_results: Legacy JSON output
- save_stats_csv / save_significance_csv: Raw table outputs
- save_disentangled_tables: Disentangled CSV outputs
- save_disentanglement: Disentanglement matrices and summary
- save_all_results: Combined save operation
"""

import json
import time
from pathlib import Path

import pandas as pd

from driada.intense.io import save_results as save_intense_results


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


def filter_stats_by_stage2(stats, significance):
    """Filter stats to only include features that pass stage2 criterion.

    This ensures consistency between CSV tables and disentanglement,
    which both use the stage2 criterion (pval AND rval checks).

    Parameters
    ----------
    stats : dict
        Nested dict: stats[cell_id][feat_name] -> {me, pval, ...}
    significance : dict
        Nested dict: significance[cell_id][feat_name] -> {stage2: bool, ...}

    Returns
    -------
    filtered_stats : dict
        Stats dict with only stage2-significant features
    filtered_significance : dict
        Significance dict with only stage2-significant features
    """
    filtered_stats = {}
    filtered_significance = {}

    for cell_id in stats.keys():
        filtered_stats[cell_id] = {}
        filtered_significance[cell_id] = {}

        for feat_name in stats[cell_id].keys():
            # Check if this feature passes stage2 criterion
            sig_entry = significance.get(cell_id, {}).get(feat_name, {})
            if sig_entry.get('stage2', False):
                filtered_stats[cell_id][feat_name] = stats[cell_id][feat_name]
                filtered_significance[cell_id][feat_name] = significance[cell_id][feat_name]

    return filtered_stats, filtered_significance


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
    # Import here to avoid circular imports
    from .analysis import build_disentangled_stats

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

    # Filter stats to only include stage2-significant features
    # This ensures CSV tables use the same criterion as disentanglement
    filtered_stats, filtered_significance = filter_stats_by_stage2(stats, significance)

    # Save stats CSV (only stage2-significant features)
    save_stats_csv(filtered_stats, feat_names, tables_dir / f'{exp_name} INTENSE stats.csv')

    # Save significance CSV (only stage2-significant features)
    save_significance_csv(filtered_significance, feat_names, tables_dir / f'{exp_name} INTENSE significance.csv')

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
        # Use filtered stats to ensure consistency with raw tables
        disent_stats, disent_significance = build_disentangled_stats(
            filtered_stats, filtered_significance, disent_results, exp
        )
        save_disentangled_tables(
            disent_stats, disent_significance, feat_names, exp_name, output_dir
        )

    print(f"\nAll results saved to: {output_dir}")
