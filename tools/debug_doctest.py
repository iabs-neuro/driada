#!/usr/bin/env python3
"""Debug doctest failures by running with verbose output."""

import sys
import doctest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

if len(sys.argv) < 2:
    print("Usage: python debug_doctest.py <module_path>")
    sys.exit(1)

file_path = sys.argv[1]

# Import the module
import importlib.util
spec = importlib.util.spec_from_file_location("test_module", file_path)
if spec and spec.loader:
    module = importlib.util.module_from_spec(spec)
    
    # Execute module
    spec.loader.exec_module(module)
    
    # Run doctests verbosely
    print(f"Running doctests for {file_path}:\n")
    failures, tests = doctest.testmod(
        module, 
        verbose=True,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    
    print(f"\n\nSummary: {tests} tests, {failures} failures")
else:
    print(f"Could not load module from {file_path}")