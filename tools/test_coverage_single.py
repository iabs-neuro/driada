#!/usr/bin/env python
"""Quick test of coverage parsing on utils module."""

import subprocess
import re

cmd = [
    "conda", "run", "-n", "driada",
    "python", "-m", "pytest",
    "tests/unit/utils/test_output.py",
    "tests/unit/utils/test_naming.py",
    "tests/unit/utils/test_matrix.py",
    "--cov=driada.utils",
    "--cov-report=term",
    "-q"
]

result = subprocess.run(cmd, capture_output=True, text=True)
output = result.stdout + result.stderr

print("=== RAW OUTPUT ===")
print(output[-2000:])  # Last 2000 chars

print("\n=== PARSING COVERAGE ===")
coverage_pattern = r'(?:src/)?driada/(\S+?)\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+(?:\.\d+)?)%'
for match in re.finditer(coverage_pattern, output):
    module_path = match.group(1)
    percentage = float(match.group(2))
    print(f"Found: {module_path} -> {percentage}%")

total_match = re.search(r'TOTAL\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+(?:\.\d+)?)%', output)
if total_match:
    print(f"TOTAL: {total_match.group(1)}%")
else:
    print("No TOTAL found")