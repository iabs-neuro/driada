#!/usr/bin/env python3
"""
Quick analysis of test suite without running tests.
"""

import subprocess
import json
from pathlib import Path

def analyze_test_files():
    """Analyze test files without running them."""
    print("DRIADA Test Suite Quick Analysis")
    print("=" * 80)
    
    # Get all test files
    test_files = list(Path('tests').rglob('test_*.py'))
    test_files = [f for f in test_files if '__pycache__' not in str(f)]
    
    print(f"Found {len(test_files)} test files\n")
    
    # Run pytest with --collect-only to get test counts
    cmd = ['pytest', '--collect-only', '-q']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse output
    for line in result.stdout.split('\n'):
        if 'errors' in line:
            print(f"Collection errors: {line}")
        elif 'selected' in line:
            print(f"Total tests collected: {line}")
    
    # Check specific problematic files
    print("\nChecking reported problematic test files:")
    print("-" * 80)
    
    problem_files = [
        'tests/unit/intense/test_intense_pipelines.py',
        'tests/unit/experiment/synthetic/test_mixed_population.py',
        'tests/unit/experiment/test_calcium_dynamics.py',
        'tests/unit/experiment/synthetic/test_2d_spatial_manifold.py',
        'tests/unit/experiment/synthetic/test_3d_spatial_manifold.py',
        'tests/unit/experiment/synthetic/test_circular_manifold.py'
    ]
    
    for file in problem_files:
        if Path(file).exists():
            # Get test count
            cmd = ['pytest', '--collect-only', file, '-q']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            test_count = 0
            for line in result.stdout.split('\n'):
                if 'selected' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'selected':
                            test_count = int(parts[i-1])
                            break
            
            print(f"{Path(file).name:<40} {test_count} tests")
    
    # Check if we can get coverage without running tests
    print("\nCurrent test coverage (from previous runs):")
    print("-" * 80)
    
    # Try to read existing coverage data
    cov_file = Path('.coverage')
    if cov_file.exists():
        cmd = ['coverage', 'report', '--skip-covered']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Find TOTAL line
        for line in result.stdout.split('\n'):
            if 'TOTAL' in line:
                print(line)
    else:
        print("No coverage data found. Run tests to generate coverage.")
    
    # Look for test markers
    print("\nTest markers analysis:")
    print("-" * 80)
    
    # Search for slow markers
    slow_marked = []
    for file in test_files:
        content = file.read_text()
        if '@pytest.mark.slow' in content:
            slow_marked.append(file.name)
    
    print(f"Files with @pytest.mark.slow: {len(slow_marked)}")
    if slow_marked:
        for f in slow_marked[:5]:  # Show first 5
            print(f"  - {f}")
    
    # Check for fixture usage
    print("\nFixture usage in key files:")
    print("-" * 80)
    
    fixture_files = [
        'tests/unit/intense/conftest.py',
        'tests/integration/conftest.py',
        'tests/conftest.py'
    ]
    
    for file in fixture_files:
        if Path(file).exists():
            content = Path(file).read_text()
            fixtures = [line.strip() for line in content.split('\n') if line.strip().startswith('@pytest.fixture')]
            print(f"{file}: {len(fixtures)} fixtures defined")

if __name__ == '__main__':
    analyze_test_files()