#!/usr/bin/env python
"""Run cross-session analysis from an INTENSE results folder.

Usage
-----
    # Auto-detect experiment type from folder name
    python tools/run_cross_analysis.py "DRIADA data/FOF" --intense INTENSE_FOF_v1

    # Explicit experiment type
    python tools/run_cross_analysis.py "DRIADA data/NOF" --intense INTENSE_NOF_v3 --exp NOF

    # Custom output tag
    python tools/run_cross_analysis.py "DRIADA data/FOF" --intense INTENSE_FOF_v1 --tag v2
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from neuron_database import (
    load_experiment, ExperimentConfig, EXPERIMENT_CONFIGS,
    export_all,
)


def detect_experiment_type(data_dir):
    """Detect experiment type from the folder name."""
    name = Path(data_dir).name.upper()
    for exp_id in EXPERIMENT_CONFIGS:
        if name == exp_id or name.startswith(exp_id + '_'):
            return exp_id
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Run cross-session analysis from INTENSE results')
    parser.add_argument('data_dir', type=str,
                        help='Experiment data directory (e.g., "DRIADA data/FOF")')
    parser.add_argument('--intense', required=True,
                        help='INTENSE results subdirectory name (e.g., INTENSE_FOF_v1)')
    parser.add_argument('--exp', default=None,
                        help='Experiment type (e.g., FOF, NOF). Auto-detected if omitted.')
    parser.add_argument('--tag', default='v1',
                        help='Output folder version tag (default: v1)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)

    # Detect or validate experiment type
    exp_id = args.exp or detect_experiment_type(data_dir)
    if exp_id is None:
        print(f"Error: Cannot detect experiment type from '{data_dir.name}'. Use --exp.")
        sys.exit(1)
    if exp_id not in EXPERIMENT_CONFIGS:
        print(f"Error: Unknown experiment type '{exp_id}'. "
              f"Available: {sorted(EXPERIMENT_CONFIGS.keys())}")
        sys.exit(1)

    # Verify INTENSE subdirectory exists
    intense_dir = data_dir / args.intense
    tables_dir = intense_dir / 'tables_disentangled'
    if not tables_dir.exists():
        print(f"Error: Tables not found at {tables_dir}")
        sys.exit(1)

    out_dir = data_dir / f"cross-analysis 3.0 {args.tag}" / exp_id

    # Build config with overridden tables_subdir
    base = EXPERIMENT_CONFIGS[exp_id]
    config = ExperimentConfig(
        experiment_id=base.experiment_id,
        sessions=base.sessions,
        matching_subdir=base.matching_subdir,
        tables_subdir=f'{args.intense}/tables_disentangled',
        nontrivial_matching=base.nontrivial_matching,
        delay_strategy=base.delay_strategy,
        sessions_to_match=base.sessions_to_match,
        mice_metadata=list(base.mice_metadata),
        killed_sessions=list(base.killed_sessions),
        excluded_mice=list(base.excluded_mice),
        discrete_place_features=list(base.discrete_place_features),
        aggregate_features=dict(base.aggregate_features),
    )

    print(f"Experiment: {exp_id}")
    print(f"Data dir:   {data_dir}")
    print(f"INTENSE:    {args.intense}")
    print(f"Output:     {out_dir}")
    print()

    db = load_experiment(exp_id, str(data_dir), config=config)
    db.summary()

    print(f"\nExporting to: {out_dir}")
    export_all(db, out_dir)

    # Save config for reproducibility
    config_path = out_dir / 'config.json'
    config_dict = asdict(config)
    config_dict['_command'] = ' '.join(sys.argv)
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved to: {config_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
