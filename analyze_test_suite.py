#!/usr/bin/env python3
"""
Analyze the current state of the test suite after optimizations.
"""

import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any

def run_pytest_collect(test_path: str) -> int:
    """Get number of tests in a file without running them."""
    cmd = ['pytest', '--collect-only', test_path, '-q']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse output for test count
    for line in result.stdout.split('\n'):
        if 'selected' in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'selected':
                    return int(parts[i-1])
    return 0

def check_test_status(test_path: str) -> Dict[str, Any]:
    """Check if a test file has issues by running a quick test."""
    print(f"Checking: {test_path}")
    
    # Run just the first test with short timeout
    cmd = ['pytest', test_path, '-x', '--tb=short', '--durations=1']
    
    try:
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        duration = time.time() - start
        
        output = result.stdout + result.stderr
        
        # Check results
        if 'PASSED' in output:
            status = 'working'
        elif 'FAILED' in output:
            status = 'failing'
        elif 'SKIPPED' in output:
            status = 'skipped'
        else:
            status = 'unknown'
            
        # Extract any errors
        errors = []
        if 'FAILED' in output or 'ERROR' in output:
            for line in output.split('\n'):
                if 'assert' in line or 'Error' in line or 'FAILED' in line:
                    errors.append(line.strip())
                    
        return {
            'status': status,
            'duration': round(duration, 1),
            'errors': errors[:3]  # First 3 errors
        }
        
    except subprocess.TimeoutExpired:
        return {
            'status': 'slow',
            'duration': 30,
            'errors': ['Test takes >30s even with -x flag']
        }
    except Exception as e:
        return {
            'status': 'error',
            'duration': 0,
            'errors': [str(e)]
        }

def main():
    """Analyze current test suite state."""
    print("DRIADA Test Suite Analysis - Post-Optimization Status")
    print("=" * 80)
    
    # Key test files to check based on previous reports
    critical_tests = {
        'intense_pipelines': 'tests/unit/intense/test_intense_pipelines.py',
        'mixed_population': 'tests/unit/experiment/synthetic/test_mixed_population.py', 
        'calcium_dynamics': 'tests/unit/experiment/test_calcium_dynamics.py',
        '2d_spatial_manifold': 'tests/unit/experiment/synthetic/test_2d_spatial_manifold.py',
        '3d_spatial_manifold': 'tests/unit/experiment/synthetic/test_3d_spatial_manifold.py',
        'circular_manifold': 'tests/unit/experiment/synthetic/test_circular_manifold.py',
        'intense': 'tests/unit/intense/test_intense.py',
        'integration': 'tests/integration/test_integration.py',
        'selectivity_mapper': 'tests/integration/test_selectivity_mapper.py'
    }
    
    results = {}
    
    for name, path in critical_tests.items():
        if Path(path).exists():
            test_count = run_pytest_collect(path)
            status = check_test_status(path)
            results[name] = {
                'path': path,
                'test_count': test_count,
                **status
            }
        else:
            results[name] = {
                'path': path,
                'test_count': 0,
                'status': 'missing',
                'duration': 0,
                'errors': ['File not found']
            }
    
    # Print summary
    print("\nTest Suite Status Summary:")
    print(f"{'Test Module':<25} {'Status':<12} {'Tests':<8} {'Time(s)':<10} {'Issues'}")
    print("-" * 80)
    
    for name, info in results.items():
        status = info['status']
        
        # Color coding for terminal
        if status == 'working':
            status_str = status
        elif status == 'slow':
            status_str = f"{status.upper()}"
        elif status == 'failing':
            status_str = f"FAIL"
        else:
            status_str = status
            
        print(f"{name:<25} {status_str:<12} {info['test_count']:<8} {info['duration']:<10.1f} {info['errors'][0] if info['errors'] else ''}")
    
    # Check overall coverage
    print("\n" + "="*80)
    print("Checking Overall Test Coverage...")
    
    cov_cmd = ['pytest', 'tests/unit/test_api_imports.py', '--cov=driada', '--cov-report=term', '--tb=no']
    cov_result = subprocess.run(cov_cmd, capture_output=True, text=True)
    
    # Extract coverage
    for line in cov_result.stdout.split('\n'):
        if 'TOTAL' in line:
            print(f"Current Coverage: {line.strip()}")
            
    # Save detailed results
    with open('test_status_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*80)
    print("Analysis complete. Results saved to test_status_analysis.json")

if __name__ == '__main__':
    main()