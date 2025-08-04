#!/usr/bin/env python
"""Batch coverage runner that respects pytest session fixtures.

This runner groups test files by module and runs them together to:
1. Leverage session-scoped fixtures for performance
2. Avoid torch/numba import conflicts
3. Get accurate coverage reporting
"""

import subprocess
import json
import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import tempfile
import shutil


class BatchCoverageRunner:
    """Run coverage tests in batches to leverage fixtures."""
    
    def __init__(self):
        self.test_root = Path("tests")
        self.project_root = Path.cwd()
        
    def find_test_modules(self) -> Dict[str, List[Path]]:
        """Find all test files grouped by module."""
        modules = {}
        unit_dir = self.test_root / "unit"
        
        if not unit_dir.exists():
            return modules
            
        for module_dir in unit_dir.iterdir():
            if module_dir.is_dir() and not module_dir.name.startswith("__"):
                test_files = list(module_dir.glob("test_*.py"))
                if test_files:
                    modules[module_dir.name] = test_files
                    
                # Also check for subdirectories (like experiment/synthetic)
                for subdir in module_dir.iterdir():
                    if subdir.is_dir() and not subdir.name.startswith("__"):
                        sub_test_files = list(subdir.glob("test_*.py"))
                        if sub_test_files:
                            modules[f"{module_dir.name}/{subdir.name}"] = sub_test_files
                            
        return modules
    
    def run_module_batch(self, module_name: str, test_files: List[Path]) -> Dict[str, any]:
        """Run all tests for a module in a single batch."""
        print(f"\nRunning {module_name} module ({len(test_files)} files)...")
        
        # Determine coverage module
        base_module = module_name.split('/')[0]
        cov_module = f"driada.{base_module}"
        
        # Create test runner script that handles imports
        runner_script = '''
import sys
import os

# Set environment for stable execution
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache_batch'

# Pre-import problematic libraries
try:
    import numba
except:
    pass

try:
    import pynndescent
except:
    pass

try:
    import torch
    torch.tensor([1.0])
except:
    pass

try:
    import ssqueezepy
except:
    pass

# Now run pytest
import pytest

# Change to project root for conftest discovery
os.chdir('{project_root}')

# Run tests
sys.exit(pytest.main([
    {test_files},
    '--cov={cov_module}',
    '--cov-report=json',
    '--cov-report=term-missing',
    '-v',
    '--tb=short'
]))
'''.format(
            project_root=self.project_root,
            test_files=', '.join(f'"{f}"' for f in test_files),
            cov_module=cov_module
        )
        
        # Write script to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(runner_script)
            script_path = f.name
            
        try:
            # Run the script
            cmd = ["conda", "run", "-n", "driada", "python", script_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse results
            output = result.stdout + result.stderr
            passed = result.returncode == 0
            
            # Extract coverage data
            coverage_data = self._parse_coverage(output)
            
            # Count test results
            test_counts = self._parse_test_counts(output)
            
            # Save output for debugging
            debug_file = f"coverage_debug_{module_name.replace('/', '_')}.txt"
            with open(debug_file, 'w') as f:
                f.write(output)
            
            return {
                "module": module_name,
                "passed": passed,
                "test_files": len(test_files),
                "coverage": coverage_data,
                "test_counts": test_counts,
                "returncode": result.returncode,
                "debug_output": debug_file
            }
        finally:
            # Clean up temp file
            os.unlink(script_path)
    
    def _parse_coverage(self, output: str) -> Dict[str, float]:
        """Parse coverage from output."""
        import re
        coverage = {}
        
        # Debug: check if we have coverage output
        if "------ coverage:" in output or "Name" in output and "Stmts" in output:
            # Look for coverage table section
            lines = output.split('\n')
            in_coverage = False
            
            for line in lines:
                # Start of coverage table
                if "Name" in line and "Stmts" in line:
                    in_coverage = True
                    continue
                    
                # End of coverage table
                if in_coverage and ("=" in line or "TOTAL" in line or not line.strip()):
                    if "TOTAL" in line:
                        # Parse TOTAL line
                        match = re.search(r'TOTAL\s+\d+\s+\d+(?:\s+\d+\s+\d+)?\s+(\d+)%', line)
                        if match:
                            coverage['TOTAL'] = float(match.group(1))
                    if "=" in line:
                        break
                        
                # Parse coverage lines
                if in_coverage and line.strip():
                    # Match lines like: src/driada/network/net_base.py    123    45    63%
                    # The percentage can have decimals: 96.20%
                    match = re.match(r'\s*(?:src/)?driada/(\S+?)\s+\d+\s+\d+(?:\s+\d+\s+\d+)?\s+(\d+(?:\.\d+)?)%', line)
                    if match:
                        module_path = match.group(1).replace('/', '.').replace('.py', '')
                        percentage = float(match.group(2))
                        
                        # Store full path coverage
                        coverage[f"driada.{module_path}"] = percentage
                        
                        # Also extract top-level module
                        parts = module_path.split('.')
                        module_name = parts[0]
                        if module_name not in coverage or percentage > coverage[module_name]:
                            coverage[module_name] = percentage
        
        return coverage
    
    def _parse_test_counts(self, output: str) -> Dict[str, int]:
        """Parse test pass/fail counts."""
        import re
        
        counts = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0
        }
        
        # Look for pytest summary
        summary_match = re.search(
            r'=+ ([\d\s\w,]+) in [\d.]+s',
            output
        )
        
        if summary_match:
            summary = summary_match.group(1)
            
            passed_match = re.search(r'(\d+) passed', summary)
            if passed_match:
                counts["passed"] = int(passed_match.group(1))
                
            failed_match = re.search(r'(\d+) failed', summary)
            if failed_match:
                counts["failed"] = int(failed_match.group(1))
                
            skipped_match = re.search(r'(\d+) skipped', summary)
            if skipped_match:
                counts["skipped"] = int(skipped_match.group(1))
                
            error_match = re.search(r'(\d+) error', summary)
            if error_match:
                counts["errors"] = int(error_match.group(1))
                
        return counts
    
    def run_all_modules(self) -> Dict[str, any]:
        """Run all test modules."""
        modules = self.find_test_modules()
        results = {}
        
        print(f"Found {len(modules)} test modules")
        
        for module_name, test_files in sorted(modules.items()):
            result = self.run_module_batch(module_name, test_files)
            results[module_name] = result
            
        return results
    
    def generate_report(self, results: Dict[str, any]) -> None:
        """Generate coverage report."""
        print("\n" + "="*70)
        print("BATCH COVERAGE REPORT")
        print("="*70)
        
        # Aggregate coverage
        all_coverage = {}
        module_specific_coverage = {}  # Track coverage only for files in each module
        total_tests_passed = 0
        total_tests_failed = 0
        
        for module_name, result in results.items():
            if result["passed"]:
                # Collect coverage for files specifically in this module
                module_files = []
                module_statement_data = {}  # Track statements for weighted average
                
                # Parse statement counts from coverage output
                if "debug_output" in result:
                    debug_file = result["debug_output"]
                    if os.path.exists(debug_file):
                        with open(debug_file, 'r') as f:
                            debug_content = f.read()
                            
                        # Extract statement counts from coverage table
                        lines = debug_content.split('\n')
                        for line in lines:
                            if f"driada/{module_name.split('/')[0]}/" in line:
                                # Parse line like: src/driada/network/net_base.py    379    179    168     36  50.09%
                                match = re.match(r'\s*(?:src/)?driada/\S+/(\S+?)\s+(\d+)\s+(\d+)', line)
                                if match:
                                    file_name = match.group(1)
                                    statements = int(match.group(2))
                                    missed = int(match.group(3))
                                    module_statement_data[file_name] = {
                                        'statements': statements,
                                        'covered': statements - missed
                                    }
                
                for cov_path, percent in result["coverage"].items():
                    if cov_path.startswith(f"driada.{module_name.split('/')[0]}."):
                        module_files.append((cov_path, percent))
                        
                # Calculate different coverage metrics
                if module_files:
                    # Simple average
                    simple_avg = sum(pct for _, pct in module_files) / len(module_files)
                    
                    # Weighted average and true coverage
                    weighted_avg = simple_avg  # Default to simple if no statement data
                    true_coverage = simple_avg
                    
                    if module_statement_data:
                        total_statements = sum(data['statements'] for data in module_statement_data.values())
                        total_covered = sum(data['covered'] for data in module_statement_data.values())
                        
                        if total_statements > 0:
                            # True coverage
                            true_coverage = (total_covered / total_statements) * 100
                            
                            # Weighted average
                            weighted_sum = 0
                            for file_path, percent in module_files:
                                file_name = file_path.split('.')[-1] + '.py'
                                if file_name in module_statement_data:
                                    weight = module_statement_data[file_name]['statements'] / total_statements
                                    weighted_sum += percent * weight
                                    
                            if weighted_sum > 0:
                                weighted_avg = weighted_sum
                    
                    module_specific_coverage[module_name] = {
                        "average": simple_avg,
                        "weighted_average": weighted_avg,
                        "true_coverage": true_coverage,
                        "files": module_files,
                        "statement_data": module_statement_data
                    }
                
                # Still track all individual file coverage
                for cov_module, percent in result["coverage"].items():
                    if cov_module != "TOTAL" and "." in cov_module:
                        all_coverage[cov_module] = percent
            
            # Sum test counts
            counts = result.get("test_counts", {})
            total_tests_passed += counts.get("passed", 0)
            total_tests_failed += counts.get("failed", 0) + counts.get("errors", 0)
        
        # Display module coverage
        print("\nMODULE COVERAGE:")
        print("-" * 70)
        
        modules_at_target = 0
        for module_name, data in sorted(module_specific_coverage.items()):
            true_cov = data.get("true_coverage", data["average"])
            simple_avg = data["average"]
            status = "✅" if true_cov >= 85 else "⚠️" if true_cov >= 70 else "❌"
            
            # Show true coverage with simple average for comparison
            if abs(true_cov - simple_avg) > 0.1:
                print(f"{status} {module_name:20s}: {true_cov:5.1f}% (simple avg: {simple_avg:5.1f}%, {len(data['files'])} files)")
            else:
                print(f"{status} {module_name:20s}: {true_cov:5.1f}% ({len(data['files'])} files)")
                
            if true_cov >= 85:
                modules_at_target += 1
                
            # Show file breakdown with statement counts
            for file_path, pct in sorted(data["files"], key=lambda x: x[1], reverse=True):
                file_name = file_path.split('.')[-1]
                status = "✅" if pct >= 85 else "⚠️" if pct >= 70 else "❌"
                
                # Add statement count if available
                stmt_info = ""
                if data.get("statement_data"):
                    file_key = file_name + '.py'
                    if file_key in data["statement_data"]:
                        stmts = data["statement_data"][file_key]["statements"]
                        stmt_info = f" ({stmts} stmts)"
                
                print(f"  {status} {file_name:25s}: {pct:5.1f}%{stmt_info}")
        
        # Overall stats
        if module_specific_coverage:
            # Use true coverage for overall calculation
            overall_coverage = sum(data.get("true_coverage", data["average"]) for data in module_specific_coverage.values()) / len(module_specific_coverage)
        else:
            overall_coverage = 0
            
        print("\n" + "-" * 50)
        print(f"Overall Coverage: {overall_coverage:.1f}%")
        print(f"Modules at 85% target: {modules_at_target}/{len(module_specific_coverage)}")
        print(f"Total tests passed: {total_tests_passed}")
        print(f"Total tests failed: {total_tests_failed}")
        
        # Show failing modules
        failing_modules = [name for name, result in results.items() if not result["passed"]]
        if failing_modules:
            print("\nFAILING MODULES:")
            print("-" * 50)
            for module in failing_modules:
                print(f"  ❌ {module}")
        
        # Save detailed results
        with open("batch_coverage_results.json", "w") as f:
            json.dump({
                "summary": {
                    "overall_coverage": overall_coverage,
                    "modules_at_target": modules_at_target,
                    "total_modules": len(all_coverage),
                    "module_coverage": all_coverage,
                    "tests_passed": total_tests_passed,
                    "tests_failed": total_tests_failed
                },
                "detailed_results": results
            }, f, indent=2)
            
        print("\n✅ Detailed results saved to batch_coverage_results.json")


def main():
    """Run batch coverage analysis."""
    print("🚀 Starting Batch Coverage Runner")
    print("This runner groups tests by module to leverage session fixtures\n")
    
    runner = BatchCoverageRunner()
    
    if len(sys.argv) > 1:
        # Run specific module
        module_name = sys.argv[1]
        modules = runner.find_test_modules()
        
        if module_name in modules:
            result = runner.run_module_batch(module_name, modules[module_name])
            runner.generate_report({module_name: result})
        else:
            print(f"Module '{module_name}' not found")
            print(f"Available modules: {', '.join(sorted(modules.keys()))}")
    else:
        # Run all modules
        results = runner.run_all_modules()
        runner.generate_report(results)


if __name__ == "__main__":
    main()