#!/usr/bin/env python
import subprocess
import sys
import os

# Run in a fresh Python process
code = """
import numpy as np
from driada.rsa.core_jit import fast_correlation_distance

patterns = np.array([
    [1, 2, 3, 4],
    [1, 2, 3, 4],  # Identical to first
    [2, 4, 6, 8],  # Perfectly correlated but different scale
    [4, 3, 2, 1],  # Anti-correlated
])

rdm = fast_correlation_distance(patterns)
print(f"Distance [0,1] (identical): {rdm[0, 1]}")
print(f"Distance [0,2] (correlated): {rdm[0, 2]}")
print(f"Distance [0,3] (anti-corr): {rdm[0, 3]}")
"""

# Run with JIT disabled to ensure we get the latest code
env = {"DRIADA_DISABLE_NUMBA": "1"}
result = subprocess.run([sys.executable, "-c", code], env={**os.environ, **env}, capture_output=True, text=True)
print("STDOUT:")
print(result.stdout)
if result.stderr:
    print("STDERR:")
    print(result.stderr)