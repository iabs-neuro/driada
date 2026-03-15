#!/usr/bin/env python
"""Regenerate, syntax-check, and execute all DRIADA notebooks."""

import os
import sys
import time
import subprocess
import glob
import json
import argparse

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")
NOTEBOOKS_DIR = os.path.join(REPO_ROOT, "notebooks")
EXECUTE_TIMEOUT = 900  # 15 minutes per notebook


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


def execute_notebook(nb_path):
    """Execute a notebook top-to-bottom, return (success, elapsed, error_msg)."""
    out_path = nb_path.replace(".ipynb", "_executed.ipynb")
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    start = time.time()
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=600",
                "--output", os.path.basename(out_path),
                nb_path,
            ],
            capture_output=True, text=True, timeout=EXECUTE_TIMEOUT,
            cwd=REPO_ROOT, env=env,
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            # Clean up executed notebook
            if os.path.exists(out_path):
                os.remove(out_path)
            return True, elapsed, ""
        else:
            error = result.stderr[-500:] if result.stderr else "unknown error"
            if os.path.exists(out_path):
                os.remove(out_path)
            return False, elapsed, error
    except subprocess.TimeoutExpired:
        if os.path.exists(out_path):
            os.remove(out_path)
        return False, EXECUTE_TIMEOUT, "TIMEOUT"


def main():
    parser = argparse.ArgumentParser(description="Validate DRIADA notebooks")
    parser.add_argument("--execute", action="store_true",
                        help="Execute notebooks top-to-bottom (slow)")
    parser.add_argument("--only", type=str, default=None,
                        help="Only process notebooks matching this pattern (e.g. '06')")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 1: Regenerate notebooks from generators")
    print("=" * 60)
    gen_results = run_generators()

    print(f"\n{'=' * 60}")
    print("Phase 2: Syntax and content check")
    print("=" * 60)

    notebooks = find_notebooks()
    if args.only:
        notebooks = [n for n in notebooks if args.only in os.path.basename(n)]
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

    if args.execute:
        print(f"\n{'=' * 60}")
        print("Phase 3: Execute notebooks")
        print("=" * 60)
        print()

        exec_results = []
        for nb_path in notebooks:
            name = os.path.basename(nb_path)
            print(f"  {name} ... ", end="", flush=True)
            ok, elapsed, error = execute_notebook(nb_path)
            status = "OK" if ok else "FAIL"
            print(f"{status} ({elapsed:.0f}s)")
            if not ok:
                all_pass = False
                for line in error.strip().split("\n")[-5:]:
                    print(f"    {line}")
            exec_results.append((name, ok, elapsed))

        total_exec = sum(t for _, _, t in exec_results)
        exec_passed = sum(1 for _, ok, _ in exec_results if ok)
        exec_failed = sum(1 for _, ok, _ in exec_results if not ok)
        print(f"\n  Execution: {exec_passed} passed, {exec_failed} failed ({total_exec:.0f}s total)")

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
