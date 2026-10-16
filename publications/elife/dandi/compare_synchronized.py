"""Compare rebuilt ``*_aligned.npz`` files with the original synchronized files.

For every file of the rebuilt folder the original of the same name must hold
the same arrays in the same order, with the same type, shape and bytes. Whole
files are also compared byte by byte; they may differ there while every array
is identical, because an npz file is a zip archive that records the time each
member was written.

Run:  python publications/elife/dandi/compare_synchronized.py <rebuilt folder> [--original <folder>]
"""
import argparse
import filecmp
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import elife_paths


def compare(rebuilt_path, original_path):
    """Return the list of differences between two npz files (empty if none)."""
    a, b = np.load(rebuilt_path, allow_pickle=False), np.load(original_path, allow_pickle=False)
    if a.files != b.files:
        return [f"arrays or their order differ: {a.files} vs {b.files}"]
    problems = []
    for key in b.files:
        x, y = a[key], b[key]
        if x.dtype != y.dtype or x.shape != y.shape:
            problems.append(f"{key}: {x.dtype}{x.shape} vs {y.dtype}{y.shape}")
        elif x.tobytes() != y.tobytes():
            problems.append(f"{key}: values differ, largest difference {np.max(np.abs(x - y))}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("rebuilt", help="folder with the rebuilt *_aligned.npz files")
    parser.add_argument("--original", default=elife_paths.SYNC_DIR, help="folder with the original files")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.rebuilt, "*_aligned.npz")))
    if not paths:
        print(f"[FAILED] no *_aligned.npz files in {args.rebuilt}")
        return 1
    failed = 0
    for path in paths:
        name = os.path.basename(path)
        original = os.path.join(args.original, name)
        if not os.path.exists(original):
            problems = ["no original file of this name"]
        else:
            problems = compare(path, original)
        if problems:
            failed += 1
            print(f"[FAILED] {name}: {'; '.join(problems)}")
        else:
            same_file = filecmp.cmp(path, original, shallow=False)
            n = len(np.load(path).files)
            print(f"[OK] {name}: {n} arrays identical in order, type, shape and bytes; "
                  f"whole file byte-identical: {'yes' if same_file else 'no'}")
    print(f"files compared: {len(paths)}, with differences: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
