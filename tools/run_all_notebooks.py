#!/usr/bin/env python
"""Regenerate and syntax-check all DRIADA notebooks."""

import os
import sys
import time
import subprocess
import glob
import json

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")
NOTEBOOKS_DIR = os.path.join(REPO_ROOT, "notebooks")


def find_generators():
    """Find all create_notebook_*.py generator scripts."""
    pattern = os.path.join(TOOLS_DIR, "create_notebook_*.py")
    return sorted(glob.glob(pattern))


def find_notebooks():
    """Find all .ipynb notebook files."""
    pattern = os.path.join(NOTEBOOKS_DIR, "*.ipynb")
    return sorted(glob.glob(pattern))


def run_generators():
    """Run all generators and report results."""
    generators = find_generators()
    print(f"Found {len(generators)} generators\n")

    results = []
    for gen in generators:
        name = os.path.basename(gen)
        print(f"  {name} ... ", end="", flush=True)
        start = time.time()
        result = subprocess.run(
            [sys.executable, gen],
            capture_output=True, text=True, timeout=30,
            cwd=REPO_ROOT,
        )
        elapsed = time.time() - start
        ok = result.returncode == 0
        print(f"{'OK' if ok else 'FAIL'} ({elapsed:.1f}s)")
        if not ok:
            print(f"    {result.stderr.strip()[-200:]}")
        results.append((name, ok))
    return results


def check_notebook_syntax(nb_path):
    """Compile all code cells in a notebook, return list of errors."""
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    errors = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # Skip Jupyter magic lines
        lines = [l for l in source.split("\n") if not l.strip().startswith(("!", "%"))]
        clean = "\n".join(lines)
        if not clean.strip():
            continue
        try:
            compile(clean, f"cell_{i}", "exec")
        except SyntaxError as e:
            errors.append(f"cell {i}, line {e.lineno}: {e.msg}")
    return errors


def check_notebook_content(nb_path):
    """Check notebook for common issues, return list of warnings."""
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    warnings = []
    all_source = ""
    for cell in nb["cells"]:
        all_source += "".join(cell["source"])

    if "blob/dev" in all_source:
        warnings.append("contains blob/dev Colab links (should be blob/main)")
    if "git+https" in all_source:
        warnings.append("contains git+https install (should be pip install driada)")
    if "TODO" in all_source:
        warnings.append("contains TODO comments")

    return warnings


def check_ascii(nb_path):
    """Check for non-ASCII characters, return list of hits."""
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    hits = []
    for ci, cell in enumerate(nb["cells"]):
        for li, line in enumerate(cell["source"]):
            for ch in line:
                if ord(ch) > 127:
                    hits.append(f"cell {ci}, line {li}: U+{ord(ch):04X}")
                    break
    return hits


def main():
    print("=" * 60)
    print("Phase 1: Regenerate notebooks from generators")
    print("=" * 60)
    gen_results = run_generators()

    print(f"\n{'=' * 60}")
    print("Phase 2: Syntax check all notebooks")
    print("=" * 60)

    notebooks = find_notebooks()
    print(f"\nFound {len(notebooks)} notebooks\n")

    all_pass = True
    for nb_path in notebooks:
        name = os.path.basename(nb_path)
        errors = check_notebook_syntax(nb_path)
        warnings = check_notebook_content(nb_path)
        ascii_hits = check_ascii(nb_path)

        status = "OK"
        if errors:
            status = "FAIL"
            all_pass = False

        detail = ""
        if warnings:
            detail += f" [{len(warnings)} warnings]"
        if ascii_hits:
            detail += f" [{len(ascii_hits)} non-ASCII]"

        print(f"  {name}: {status}{detail}")
        for e in errors:
            print(f"    ERROR: {e}")
        for w in warnings:
            print(f"    WARN: {w}")

    # Summary
    gen_passed = sum(1 for _, ok in gen_results if ok)
    gen_failed = sum(1 for _, ok in gen_results if not ok)

    print(f"\n{'=' * 60}")
    print(f"Generators: {gen_passed} passed, {gen_failed} failed")
    print(f"Notebooks:  {'all pass' if all_pass else 'FAILURES FOUND'}")
    print("=" * 60)

    if not all_pass or gen_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
