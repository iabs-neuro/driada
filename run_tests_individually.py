#!/usr/bin/env python3
"""
Run all tests individually to collect timing, coverage, and failure information.
"""

import subprocess
import time
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Any
import re

def run_single_test(test_path: str, timeout: int = 300) -> Dict[str, Any]:
    """Run a single test file and collect metrics."""
    print(f"\n{'=' * 80}")
    print(f"Running: {test_path}")
    print(f"{'=' * 80}")
    
    start_time = time.time()
    result = {
        'test_file': test_path,
        'status': 'unknown',
        'duration': 0,
        'coverage': 0,
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'errors': [],
        'timeout': False,
        'simplification_needed': False
    }
    
    # Run test with coverage
    cmd = [
        'pytest', 
        test_path,
        '-xvs',
        '--tb=short',
        '--cov=driada',
        '--cov-report=term-missing:skip-covered',
        '--durations=10'
    ]
    
    try:
        # Run with timeout
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        duration = time.time() - start_time
        result['duration'] = round(duration, 2)
        
        # Parse output
        output = proc.stdout + proc.stderr
        
        # Extract test counts
        test_pattern = r'(\d+) passed|(\d+) failed|(\d+) skipped|(\d+) error'
        matches = re.findall(test_pattern, output)
        
        for match in matches:
            if match[0]:  # passed
                result['passed'] = int(match[0])
            elif match[1]:  # failed
                result['failed'] = int(match[1])
            elif match[2]:  # skipped
                result['skipped'] = int(match[2])
            elif match[3]:  # error
                result['failed'] += int(match[3])
        
        # Extract coverage
        cov_pattern = r'TOTAL\s+\d+\s+\d+\s+(\d+)%'
        cov_match = re.search(cov_pattern, output)
        if cov_match:
            result['coverage'] = int(cov_match.group(1))
        
        # Check for specific errors
        if 'FAILED' in output or 'ERROR' in output:
            result['status'] = 'failed'
            # Extract error messages
            error_lines = []
            for line in output.split('\n'):
                if 'FAILED' in line or 'ERROR' in line or 'AssertionError' in line:
                    error_lines.append(line.strip())
            result['errors'] = error_lines[:5]  # Keep first 5 error lines
        elif result['passed'] > 0 or result['skipped'] > 0:
            result['status'] = 'passed'
        else:
            result['status'] = 'no_tests'
        
        # Check if simplification needed (>20s)
        if duration > 20:
            result['simplification_needed'] = True
            
    except subprocess.TimeoutExpired:
        result['timeout'] = True
        result['status'] = 'timeout'
        result['duration'] = timeout
        result['simplification_needed'] = True
        print(f"TIMEOUT: Test took longer than {timeout}s")
    except Exception as e:
        result['status'] = 'error'
        result['errors'] = [str(e)]
    
    return result

def get_all_test_files() -> List[str]:
    """Get all test files in the tests directory."""
    test_files = []
    for pattern in ['tests/**/test_*.py', 'tests/**/*_test.py']:
        test_files.extend(glob.glob(pattern, recursive=True))
    
    # Filter out __pycache__ and other non-test files
    test_files = [f for f in test_files if '__pycache__' not in f and '__init__' not in f]
    
    return sorted(test_files)

def main():
    """Run all tests individually and generate report."""
    print("DRIADA Test Analysis - Individual Test Execution")
    print("=" * 80)
    
    # Skip environment check - we're already in the correct environment
    
    # Get all test files
    test_files = get_all_test_files()
    print(f"Found {len(test_files)} test files")
    
    # Run each test
    results = []
    total_coverage_lines = 0
    total_lines = 0
    
    for i, test_file in enumerate(test_files):
        print(f"\nProgress: {i+1}/{len(test_files)}")
        result = run_single_test(test_file)
        results.append(result)
        
        # Quick summary
        print(f"\nSummary: {result['status'].upper()} - {result['duration']}s")
        if result['coverage'] > 0:
            print(f"Coverage: {result['coverage']}%")
        if result['errors']:
            print(f"Errors: {result['errors'][0]}")
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST ANALYSIS REPORT")
    print("=" * 80)
    
    # Overall statistics
    total_tests = len(results)
    passed = sum(1 for r in results if r['status'] == 'passed')
    failed = sum(1 for r in results if r['status'] == 'failed')
    timeout = sum(1 for r in results if r['timeout'])
    needs_simplification = sum(1 for r in results if r['simplification_needed'])
    
    print(f"\nOverall Statistics:")
    print(f"  Total test files: {total_tests}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Timeout: {timeout}")
    print(f"  Needs simplification (>20s): {needs_simplification}")
    
    # Detailed results table
    print("\nDetailed Results:")
    print(f"{'Test File':<50} {'Status':<10} {'Time(s)':<10} {'Cov%':<8} {'P/F/S':<10}")
    print("-" * 100)
    
    for result in sorted(results, key=lambda x: x['duration'], reverse=True):
        test_name = Path(result['test_file']).name
        status = result['status']
        duration = f"{result['duration']:.1f}"
        coverage = f"{result['coverage']}%" if result['coverage'] > 0 else "N/A"
        pfs = f"{result['passed']}/{result['failed']}/{result['skipped']}"
        
        # Color code based on status
        if result['timeout']:
            status = "TIMEOUT!"
        elif result['simplification_needed']:
            status = "SLOW"
            
        print(f"{test_name:<50} {status:<10} {duration:<10} {coverage:<8} {pfs:<10}")
    
    # Tests needing simplification
    print(f"\n{'=' * 80}")
    print("TESTS REQUIRING SIMPLIFICATION")
    print("=" * 80)
    
    slow_tests = [r for r in results if r['simplification_needed']]
    for result in sorted(slow_tests, key=lambda x: x['duration'], reverse=True):
        print(f"\n{result['test_file']} ({result['duration']}s)")
        
        # Propose simplifications based on test name and duration
        suggestions = []
        
        if 'intense' in result['test_file']:
            suggestions.extend([
                "- Reduce n_shuffles (current: stage1=100, stage2=1000 → try: 20/100)",
                "- Increase downsampling factor (ds=5 → ds=10)",
                "- Reduce number of timepoints (10000 → 2000)",
                "- Disable parallelization for single tests",
                "- Use fixture caching for data generation"
            ])
        
        if 'dr' in result['test_file'] or 'dim_reduction' in result['test_file']:
            suggestions.extend([
                "- Reduce sample size (1000 → 300)",
                "- Use faster DR methods for testing (skip neural networks)",
                "- Add @pytest.mark.slow for comprehensive tests",
                "- Cache test data with session fixtures",
                "- Reduce parameter sweeps"
            ])
            
        if 'manifold' in result['test_file']:
            suggestions.extend([
                "- Reduce number of neurons (100 → 30)",
                "- Reduce simulation duration (300s → 100s)",
                "- Increase downsampling",
                "- Use simpler manifold generators for tests"
            ])
            
        if 'integration' in result['test_file']:
            suggestions.extend([
                "- Use smaller test datasets",
                "- Mock expensive operations",
                "- Split into smaller focused tests",
                "- Use pre-computed results where possible"
            ])
            
        if result['timeout']:
            suggestions.insert(0, "⚠️  CRITICAL: Add timeout handling or split into smaller tests")
            
        for suggestion in suggestions:
            print(f"  {suggestion}")
    
    # Failed tests
    if failed > 0:
        print(f"\n{'=' * 80}")
        print("FAILED TESTS")
        print("=" * 80)
        
        for result in results:
            if result['status'] == 'failed':
                print(f"\n{result['test_file']}")
                for error in result['errors']:
                    print(f"  ERROR: {error}")
    
    # Save detailed results
    with open('test_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print("Detailed results saved to: test_analysis_results.json")
    print("=" * 80)

if __name__ == '__main__':
    main()