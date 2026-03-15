#!/usr/bin/env python
"""Run all DRIADA example scripts and report results."""

import os
import sys
import time
import subprocess
import glob

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")
TIMEOUT = 600  # 10 minutes per example


def find_examples():
    """Find all example .py files, excluding archived and __init__."""
    scripts = []
    for path in sorted(glob.glob(os.path.join(EXAMPLES_DIR, "**", "*.py"), recursive=True)):
        rel = os.path.relpath(path, REPO_ROOT)
        if "_archived" in rel or "__pycache__" in rel or "__init__" in rel:
            continue
        scripts.append(path)
    return scripts


def run_example(path):
    """Run a single example, return (success, elapsed, output)."""
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=TIMEOUT,
            cwd=REPO_ROOT, env=env,
        )
        elapsed = time.time() - start
        output = result.stdout[-500:] if result.stdout else ""
        if result.returncode != 0:
            output += "\n--- STDERR ---\n" + (result.stderr[-500:] if result.stderr else "")
        return result.returncode == 0, elapsed, output
    except subprocess.TimeoutExpired:
        return False, TIMEOUT, "TIMEOUT"


def main():
    scripts = find_examples()
    print(f"Found {len(scripts)} example scripts\n")

    results = []
    for path in scripts:
        rel = os.path.relpath(path, REPO_ROOT)
        print(f"  {rel} ... ", end="", flush=True)
        ok, elapsed, output = run_example(path)
        status = "OK" if ok else "FAIL"
        print(f"{status} ({elapsed:.1f}s)")
        if not ok:
            for line in output.strip().split("\n")[-5:]:
                print(f"    {line}")
        results.append((rel, ok, elapsed))

    # Summary
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total_time = sum(t for _, _, t in results)

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {len(results)} total ({total_time:.0f}s)")

    if failed:
        print("\nFailed examples:")
        for rel, ok, _ in results:
            if not ok:
                print(f"  {rel}")
        sys.exit(1)


if __name__ == "__main__":
    main()
