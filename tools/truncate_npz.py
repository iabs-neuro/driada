#!/usr/bin/env python
"""
Truncate .npz experiment files along the time dimension.

Usage:
    # Single file, keep middle 60% (relative range)
    python tools/truncate_npz.py file.npz --range 0.2,0.8

    # Absolute frame indices
    python tools/truncate_npz.py file.npz --sf 100 --ef 500

    # Time in seconds (requires fps in metadata)
    python tools/truncate_npz.py file.npz --st 5.0 --et 25.0

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


def get_fps_from_metadata(data):
    """Extract fps from _metadata if available."""
    if '_metadata' in data:
        metadata = data['_metadata']
        if hasattr(metadata, 'item'):
            metadata = metadata.item()
        if isinstance(metadata, dict) and 'fps' in metadata:
            return metadata['fps']
    return None


def truncate_npz(input_path, output_path, start_idx=None, end_idx=None,
                 start_rel=None, end_rel=None, start_sec=None, end_sec=None):
    """Truncate NPZ file along time dimension."""
    data = dict(np.load(input_path, allow_pickle=True))

    # Get reference n_timepoints from calcium
    calcium = data.get('calcium', data.get('Calcium'))
    if calcium is None:
        raise ValueError(f"No calcium data found in {input_path}")
    n_timepoints = calcium.shape[1]

    # Calculate indices based on provided mode
    if start_idx is not None or end_idx is not None:
        # Absolute frame indices mode
        start_idx = start_idx if start_idx is not None else 0
        end_idx = end_idx if end_idx is not None else n_timepoints
    elif start_sec is not None or end_sec is not None:
        # Time in seconds mode
        fps = get_fps_from_metadata(data)
        if fps is None:
            raise ValueError(f"No fps found in metadata for {input_path}, cannot use time-based truncation")
        start_idx = int(start_sec * fps) if start_sec is not None else 0
        end_idx = int(end_sec * fps) if end_sec is not None else n_timepoints
    elif start_rel is not None or end_rel is not None:
        # Relative range mode
        start_rel = start_rel if start_rel is not None else 0.0
        end_rel = end_rel if end_rel is not None else 1.0
        start_idx = int(n_timepoints * start_rel)
        end_idx = int(n_timepoints * end_rel)
    else:
        raise ValueError("No truncation parameters provided")

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
    parser.add_argument('--range', type=str,
                        help='Relative range as "start,end" (e.g., "0.2,0.8")')
    parser.add_argument('--sf', type=int, help='Start frame (absolute index)')
    parser.add_argument('--ef', type=int, help='End frame (absolute index)')
    parser.add_argument('--st', type=float, help='Start time in seconds')
    parser.add_argument('--et', type=float, help='End time in seconds')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--suffix', type=str, default='_truncated',
                        help='Suffix for output files (default: _truncated)')
    args = parser.parse_args()

    # Determine truncation mode and validate arguments
    has_range = args.range is not None
    has_frame = args.sf is not None or args.ef is not None
    has_time = args.st is not None or args.et is not None

    modes_used = sum([has_range, has_frame, has_time])
    if modes_used == 0:
        parser.error("Must specify one of: --range, --sf/--ef, or --st/--et")
    if modes_used > 1:
        parser.error("Cannot mix --range, --sf/--ef, and --st/--et modes")

    # Prepare truncation parameters
    trunc_kwargs = {}
    if has_range:
        start_rel, end_rel = map(float, args.range.split(','))
        trunc_kwargs = {'start_rel': start_rel, 'end_rel': end_rel}
    elif has_frame:
        trunc_kwargs = {'start_idx': args.sf, 'end_idx': args.ef}
    elif has_time:
        trunc_kwargs = {'start_sec': args.st, 'end_sec': args.et}

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

        n_frames = truncate_npz(npz_path, out_path, **trunc_kwargs)
        print(f"{npz_path.name} -> {out_path.name} ({n_frames} frames)")


if __name__ == '__main__':
    main()
