#!/usr/bin/env python3
"""Compare raw vs disentangled INTENSE tables for consistency verification.

Usage:
    python tools/compare_raw_disent.py INTENSE_fixed
    python tools/compare_raw_disent.py INTENSE_fixed --exp NOF_H01_1D
"""

import argparse
from pathlib import Path
import pandas as pd


def load_stats_table(path: Path) -> pd.DataFrame:
    """Load stats CSV and return DataFrame."""
    df = pd.read_csv(path, index_col=0)
    return df


def get_neurons_features(df: pd.DataFrame) -> dict:
    """Extract {neuron_id: [features]} from stats table.

    The CSV is a matrix: rows=neurons, columns=features.
    Cell value is {} if not significant, or a dict with stats.
    """
    result = {}
    for nid in df.index:
        result[int(nid)] = []
        for feat in df.columns:
            cell_val = str(df.loc[nid, feat])
            # Non-empty dict means significant
            if cell_val != '{}':
                result[int(nid)].append(feat)
    return result


def compare_tables(raw_path: Path, disent_path: Path, exp_name: str):
    """Compare raw and disentangled tables."""
    print(f"\n{'='*60}")
    print(f"Comparing: {exp_name}")
    print(f"{'='*60}")

    # Load tables
    raw_df = load_stats_table(raw_path)
    disent_df = load_stats_table(disent_path)

    print(f"\n--- Table Sizes ---")
    print(f"RAW:   {len(raw_df)} rows")
    print(f"DISENT: {len(disent_df)} rows")

    # Extract neuron -> features mapping
    raw_nf = get_neurons_features(raw_df)
    disent_nf = get_neurons_features(disent_df)

    print(f"\n--- Neuron Counts ---")
    print(f"RAW:   {len(raw_nf)} neurons")
    print(f"DISENT: {len(disent_nf)} neurons")

    # Find differences
    raw_neurons = set(raw_nf.keys())
    disent_neurons = set(disent_nf.keys())

    only_in_raw = raw_neurons - disent_neurons
    only_in_disent = disent_neurons - raw_neurons
    in_both = raw_neurons & disent_neurons

    print(f"\n--- Consistency Check ---")
    print(f"Neurons in both:       {len(in_both)}")
    print(f"Only in RAW:           {len(only_in_raw)}")
    print(f"Only in DISENT:        {len(only_in_disent)}")

    if only_in_raw:
        print(f"\n--- Neurons in RAW but not DISENT (PROBLEM!) ---")
        for nid in sorted(only_in_raw):
            print(f"  Neuron {nid}: {raw_nf[nid]}")

    if only_in_disent:
        print(f"\n--- Neurons in DISENT but not RAW (merged features?) ---")
        for nid in sorted(only_in_disent):
            print(f"  Neuron {nid}: {disent_nf[nid]}")

    # Compare features for neurons in both
    changed = []
    for nid in sorted(in_both):
        raw_feats = set(raw_nf[nid])
        disent_feats = set(disent_nf[nid])
        if raw_feats != disent_feats:
            changed.append({
                'neuron': nid,
                'raw': sorted(raw_feats),
                'disent': sorted(disent_feats),
                'removed': sorted(raw_feats - disent_feats),
                'added': sorted(disent_feats - raw_feats)
            })

    print(f"\n--- Feature Changes ---")
    print(f"Neurons with different features: {len(changed)}")

    if changed:
        print(f"\n--- Detailed Changes (first 20) ---")
        for c in changed[:20]:
            print(f"\nNeuron {c['neuron']}:")
            print(f"  RAW:     {c['raw']}")
            print(f"  DISENT:  {c['disent']}")
            if c['removed']:
                print(f"  Removed: {c['removed']}")
            if c['added']:
                print(f"  Added:   {c['added']}")

    # Summary stats
    total_raw_features = sum(len(f) for f in raw_nf.values())
    total_disent_features = sum(len(f) for f in disent_nf.values())

    print(f"\n--- Summary ---")
    print(f"Total feature entries: RAW={total_raw_features}, DISENT={total_disent_features}")
    print(f"Reduction: {total_raw_features - total_disent_features} ({100*(total_raw_features-total_disent_features)/total_raw_features:.1f}%)")

    return {
        'exp_name': exp_name,
        'raw_rows': len(raw_df),
        'disent_rows': len(disent_df),
        'raw_neurons': len(raw_nf),
        'disent_neurons': len(disent_nf),
        'only_in_raw': len(only_in_raw),
        'only_in_disent': len(only_in_disent),
        'changed': len(changed),
        'problem_neurons': list(only_in_raw)
    }


def main():
    parser = argparse.ArgumentParser(description='Compare raw vs disentangled INTENSE tables')
    parser.add_argument('output_dir', type=Path, help='INTENSE output directory')
    parser.add_argument('--exp', type=str, help='Specific experiment name (optional)')
    args = parser.parse_args()

    output_dir = args.output_dir
    tables_dir = output_dir / 'tables'
    disent_dir = output_dir / 'tables_disentangled'

    if not tables_dir.exists():
        print(f"Error: tables directory not found: {tables_dir}")
        return 1

    if not disent_dir.exists():
        print(f"Error: tables_disentangled directory not found: {disent_dir}")
        return 1

    # Find all stats files
    raw_files = list(tables_dir.glob('* INTENSE stats.csv'))

    if args.exp:
        raw_files = [f for f in raw_files if args.exp in f.name]

    if not raw_files:
        print(f"No stats files found in {tables_dir}")
        return 1

    results = []
    for raw_path in sorted(raw_files):
        exp_name = raw_path.name.replace(' INTENSE stats.csv', '')
        disent_path = disent_dir / raw_path.name

        if not disent_path.exists():
            print(f"Warning: No disentangled file for {exp_name}")
            continue

        result = compare_tables(raw_path, disent_path, exp_name)
        results.append(result)

    # Overall summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")
        total_problems = sum(r['only_in_raw'] for r in results)
        print(f"Total experiments: {len(results)}")
        print(f"Total problem neurons (in RAW but not DISENT): {total_problems}")
        if total_problems > 0:
            print("\nProblem neurons by experiment:")
            for r in results:
                if r['problem_neurons']:
                    print(f"  {r['exp_name']}: {r['problem_neurons']}")

    return 0


if __name__ == '__main__':
    exit(main())
