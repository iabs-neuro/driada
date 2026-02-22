#!/usr/bin/env python
"""Verify all notebook code cells execute against the current DRIADA API.

Loads each .ipynb, extracts code cells, and runs them sequentially in a
shared namespace — exactly as a user would in Jupyter.  Reports the first
failing cell per notebook with full traceback.

Usage:
    python tools/verify_notebooks.py              # all notebooks
    python tools/verify_notebooks.py 01 03        # specific notebooks
    python tools/verify_notebooks.py --quick      # skip slow cells (>30s)
"""

import argparse
import json
import os
import sys
import traceback
import time

# Headless matplotlib before any driada import
import matplotlib
matplotlib.use("Agg")


NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")


def extract_code_cells(notebook_path):
    """Return list of (cell_index, source) for code cells."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if source.strip():
                cells.append((i, source))
    return cells


def strip_install_lines(source):
    """Remove pip install / magic lines, keep the rest (imports etc)."""
    lines = source.split("\n")
    kept = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("!") or stripped.startswith("%"):
            continue
        kept.append(line)
    return "\n".join(kept)


def run_notebook(notebook_path, verbose=False):
    """Execute all code cells in order. Returns (passed, n_cells, errors)."""
    name = os.path.basename(notebook_path)
    cells = extract_code_cells(notebook_path)

    namespace = {"__name__": "__main__"}
    errors = []
    executed = 0

    for cell_idx, source in cells:
        source = strip_install_lines(source)
        if not source.strip():
            if verbose:
                print(f"  [cell {cell_idx}] SKIP (install-only cell)")
            continue

        try:
            exec(compile(source, f"{name}:cell_{cell_idx}", "exec"), namespace)
            executed += 1
            if verbose:
                print(f"  [cell {cell_idx}] OK")
        except Exception:
            tb = traceback.format_exc()
            # Extract the last line for summary
            error_line = tb.strip().split("\n")[-1]
            errors.append({
                "cell": cell_idx,
                "error": error_line,
                "traceback": tb,
                "source_preview": source[:200],
            })
            if verbose:
                print(f"  [cell {cell_idx}] FAIL: {error_line}")
            # Stop on first error — later cells depend on earlier ones
            break

    return executed, len(cells), errors


def main():
    parser = argparse.ArgumentParser(description="Verify notebook code cells execute")
    parser.add_argument("notebooks", nargs="*",
                        help="Notebook numbers to verify (e.g. 01 03). Default: all.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Find notebooks
    all_notebooks = sorted(
        f for f in os.listdir(NOTEBOOKS_DIR) if f.endswith(".ipynb")
    )

    if args.notebooks:
        selected = []
        for num in args.notebooks:
            matches = [f for f in all_notebooks if f"_{num}_" in f or f.startswith(f"{num}_")]
            selected.extend(matches)
        all_notebooks = sorted(set(selected))

    if not all_notebooks:
        print("No notebooks found.")
        return 1

    print(f"Verifying {len(all_notebooks)} notebook(s)...\n")

    results = {}
    total_errors = 0
    t0 = time.time()

    for nb_file in all_notebooks:
        nb_path = os.path.join(NOTEBOOKS_DIR, nb_file)
        print(f"{nb_file}")

        executed, total_cells, errors = run_notebook(nb_path, verbose=args.verbose)

        if errors:
            total_errors += len(errors)
            results[nb_file] = "FAIL"
            for err in errors:
                print(f"  FAIL at cell {err['cell']}: {err['error']}")
                if args.verbose:
                    print()
                    print(err["traceback"])
        else:
            results[nb_file] = "PASS"
            print(f"  PASS ({executed}/{total_cells} code cells)")

        print()

    elapsed = time.time() - t0

    # Summary
    print("=" * 60)
    print(f"{'NOTEBOOK':<50} {'STATUS'}")
    print("-" * 60)
    for nb_file, status in results.items():
        marker = "OK" if status == "PASS" else "FAIL"
        print(f"{nb_file:<50} {marker}")
    print("-" * 60)
    passed = sum(1 for s in results.values() if s == "PASS")
    print(f"{passed}/{len(results)} passed ({elapsed:.1f}s)")

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
