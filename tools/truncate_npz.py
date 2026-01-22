#!/usr/bin/env python
"""
Truncate .npz experiment files along the time dimension.

Usage:
    # Single file, keep middle 60%
    python tools/truncate_npz.py file.npz --range 0.2,0.8

    # Batch mode
    python tools/truncate_npz.py --dir "DRIADA data" --range 0.1,0.9 --output-dir truncated/

    # Multiple files
    python tools/truncate_npz.py file1.npz file2.npz --range 0.2,0.8
"""

import argparse
import glob
from pathlib import Path
import numpy as np

METADATA_KEYS = {'_metadata', '_sync_info'}


def truncate_npz(input_path, output_path, start_rel, end_rel):
    """Truncate NPZ file along time dimension."""
    data = dict(np.load(input_path, allow_pickle=True))

    # Get reference n_timepoints from calcium
    calcium = data.get('calcium', data.get('Calcium'))
    if calcium is None:
        raise ValueError(f"No calcium data found in {input_path}")
    n_timepoints = calcium.shape[1]

    # Calculate indices
    start_idx = int(n_timepoints * start_rel)
    end_idx = int(n_timepoints * end_rel)

    # Truncate arrays
    truncated = {}
    for key, arr in data.items():
        if key in METADATA_KEYS:
            truncated[key] = arr  # Preserve metadata as-is
        elif arr.ndim == 1 and arr.shape[0] == n_timepoints:
            truncated[key] = arr[start_idx:end_idx]
        elif arr.ndim == 2 and arr.shape[1] == n_timepoints:
            truncated[key] = arr[:, start_idx:end_idx]
        else:
            truncated[key] = arr  # Non-time arrays preserved

    np.savez(output_path, **truncated)
    return end_idx - start_idx


def main():
    parser = argparse.ArgumentParser(description='Truncate .npz files along time dimension')
    parser.add_argument('npz_paths', nargs='*', help='Path(s) to .npz file(s)')
    parser.add_argument('--dir', type=str, help='Process all .npz files in directory')
    parser.add_argument('--range', type=str, required=True,
                        help='Relative range as "start,end" (e.g., "0.2,0.8")')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--suffix', type=str, default='_truncated',
                        help='Suffix for output files (default: _truncated)')
    args = parser.parse_args()

    # Parse range
    start_rel, end_rel = map(float, args.range.split(','))

    # Collect input files
    npz_paths = []
    if args.dir:
        npz_paths = list(Path(args.dir).glob('*.npz'))
    npz_paths.extend(Path(p) for pattern in args.npz_paths
                     for p in glob.glob(pattern) if p.endswith('.npz'))

    if not npz_paths:
        print("No .npz files found")
        return

    # Process files
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for npz_path in npz_paths:
        out_name = f"{npz_path.stem}{args.suffix}.npz"
        out_path = (output_dir / out_name) if output_dir else npz_path.parent / out_name

        n_frames = truncate_npz(npz_path, out_path, start_rel, end_rel)
        print(f"{npz_path.name} -> {out_path.name} ({n_frames} frames)")


if __name__ == '__main__':
    main()
