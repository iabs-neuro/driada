#!/usr/bin/env python3
"""Pre-commit hook to check documentation consistency for changed files."""

import subprocess
import sys
from pathlib import Path
import ast
from verify_doc_consistency import DocConsistencyChecker


def get_changed_python_files():
    """Get list of Python files that have been staged for commit."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True,
            text=True,
            check=True
        )
        
        files = result.stdout.strip().split('\n')
        python_files = [f for f in files if f.endswith('.py') and f.startswith('src/')]
        return python_files
    except subprocess.CalledProcessError:
        return []


def main():
    """Check documentation consistency for staged files."""
    changed_files = get_changed_python_files()
    
    if not changed_files:
        sys.exit(0)
    
    print("Checking documentation consistency for changed files...")
    
    checker = DocConsistencyChecker()
    all_issues = []
    
    for filepath in changed_files:
        if Path(filepath).exists():
            issues = checker.check_file(filepath)
            all_issues.extend(issues)
    
    if all_issues:
        print("\n❌ Documentation consistency issues found:\n")
        for issue in all_issues:
            print(f"  - {issue}")
        print("\nPlease fix the documentation issues before committing.")
        sys.exit(1)
    else:
        print("✅ Documentation consistency check passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()