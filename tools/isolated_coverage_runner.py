#!/usr/bin/env python
"""Isolated coverage runner to avoid torch import conflicts.

This tool runs pytest coverage in isolated processes to prevent the torch
'_has_torch_function' already has a docstring error that occurs when torch
is imported multiple times in the same process.
"""

import subprocess
import json
import sys
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import shutil


class IsolatedCoverageRunner:
    """Run coverage tests in isolated processes to avoid import conflicts."""
    
    def __init__(self):
        self.test_root = Path("tests")
        self.coverage_data = {}
        self.temp_dir = None
        
    def find_test_files(self, pattern: str = "**/test_*.py") -> List[Path]:
        """Find all test files matching pattern."""
        return list(self.test_root.glob(pattern))
    
    def run_single_test_file(self, test_file: Path, module: Optional[str] = None) -> Dict[str, any]:
        """Run a single test file in isolation and collect coverage."""
        # Determine which module to collect coverage for
        if module is None:
            # Try to infer module from test path
            parts = test_file.parts
            if "unit" in parts:
                idx = parts.index("unit")
                if idx + 1 < len(parts):
                    module = f"driada.{parts[idx + 1]}"
            else:
                module = "driada"
        
        # Use the isolated test runner script
        script_path = Path(__file__).parent / "run_isolated_test.py"
        
        cmd = [
            "conda", "run", "-n", "driada",
            "python", str(script_path),
            str(test_file),
            module
        ]
        
        # Create a temporary directory for this test's coverage data
        with tempfile.TemporaryDirectory() as tmpdir:
            cov_file = Path(tmpdir) / "coverage.json"
            env = os.environ.copy()
            env['COVERAGE_FILE'] = str(Path(tmpdir) / ".coverage")
            
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            
            # Try to find the coverage.json file
            coverage_json = Path.cwd() / "coverage.json"
            test_result = {
                "file": str(test_file),
                "module": module,
                "returncode": result.returncode,
                "passed": result.returncode == 0,
                "coverage": {},
                "error": None
            }
            
            # Capture error if test failed
            if result.returncode != 0:
                error_output = result.stderr + result.stdout
                if "RuntimeError: function '_has_torch_function'" in error_output:
                    test_result["error"] = "Torch import conflict"
                elif "ModuleNotFoundError" in error_output:
                    test_result["error"] = "Module not found"
                elif "ERROR" in error_output:
                    # Extract first error line
                    error_lines = [line for line in error_output.split('\n') if 'ERROR' in line]
                    test_result["error"] = error_lines[0] if error_lines else "Unknown error"
                else:
                    test_result["error"] = "Test failed"
            
            if coverage_json.exists():
                try:
                    with open(coverage_json) as f:
                        cov_data = json.load(f)
                        
                    # Extract file-level coverage
                    if "files" in cov_data:
                        for file_path, file_data in cov_data["files"].items():
                            if "driada" in file_path:
                                # Normalize path
                                normalized = file_path.replace("src/", "").replace("/", ".")
                                if normalized.endswith(".py"):
                                    normalized = normalized[:-3]
                                
                                percent = file_data["summary"]["percent_covered"]
                                test_result["coverage"][normalized] = percent
                    
                    # Extract totals
                    if "totals" in cov_data:
                        test_result["coverage"]["TOTAL"] = cov_data["totals"]["percent_covered"]
                    
                    # Clean up
                    coverage_json.unlink()
                    
                except Exception as e:
                    print(f"Error reading coverage.json: {e}")
            
            # Fallback: parse text output if JSON parsing failed
            if not test_result["coverage"] and result.stdout:
                test_result["coverage"] = self._parse_text_coverage(result.stdout + result.stderr)
            
            return test_result
    
    def _parse_text_coverage(self, output: str) -> Dict[str, float]:
        """Parse coverage from text output as fallback."""
        coverage = {}
        
        # Look for coverage lines
        patterns = [
            r'(?:src/)?driada/(\S+?)\s+\d+\s+\d+\s+(\d+)%',
            r'(?:src/)?driada/(\S+?)\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)%',
            r'(?:src/)?driada/(\S+?)\s+.*?\s+(\d+)%'
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, output):
                module_path = match.group(1).replace("/", ".").replace(".py", "")
                percentage = float(match.group(2))
                coverage[module_path] = percentage
        
        # Look for TOTAL
        total_match = re.search(r'TOTAL\s+.*?\s+(\d+)%', output)
        if total_match:
            coverage["TOTAL"] = float(total_match.group(1))
            
        return coverage
    
    def run_module_tests(self, module_name: str) -> Dict[str, any]:
        """Run all tests for a specific module."""
        module_test_dir = self.test_root / "unit" / module_name
        
        if not module_test_dir.exists():
            return {
                "module": module_name,
                "error": f"Test directory not found: {module_test_dir}",
                "coverage": {}
            }
        
        test_files = list(module_test_dir.glob("test_*.py"))
        results = []
        combined_coverage = {}
        
        print(f"\nRunning tests for module: {module_name}")
        print(f"Found {len(test_files)} test files")
        
        for test_file in test_files:
            print(f"  Running {test_file.name}...", end="", flush=True)
            result = self.run_single_test_file(test_file, f"driada.{module_name}")
            
            if result["passed"]:
                print(" ✓")
            else:
                error_msg = f" ({result.get('error', 'Unknown error')})" if result.get('error') else ""
                print(f" ✗{error_msg}")
            
            results.append(result)
            
            # Merge coverage data (take maximum)
            for module, cov in result["coverage"].items():
                if module not in combined_coverage or cov > combined_coverage[module]:
                    combined_coverage[module] = cov
        
        return {
            "module": module_name,
            "test_files": len(test_files),
            "passed": sum(1 for r in results if r["passed"]),
            "failed": sum(1 for r in results if not r["passed"]),
            "coverage": combined_coverage,
            "results": results
        }
    
    def run_all_modules(self) -> Dict[str, any]:
        """Run tests for all modules."""
        # Find all module directories
        unit_dir = self.test_root / "unit"
        module_dirs = [d for d in unit_dir.iterdir() if d.is_dir() and not d.name.startswith("__")]
        
        all_results = {}
        
        for module_dir in sorted(module_dirs):
            module_name = module_dir.name
            result = self.run_module_tests(module_name)
            all_results[module_name] = result
        
        return all_results
    
    def generate_report(self, results: Dict[str, any]) -> None:
        """Generate a coverage report from results."""
        print("\n" + "="*70)
        print("ISOLATED COVERAGE REPORT")
        print("="*70)
        
        # Aggregate coverage across all modules
        all_coverage = {}
        total_tests = 0
        total_passed = 0
        
        for module_name, module_result in results.items():
            total_tests += module_result.get("test_files", 0)
            total_passed += module_result.get("passed", 0)
            
            # Merge coverage
            for cov_module, percent in module_result.get("coverage", {}).items():
                if cov_module != "TOTAL":
                    key = cov_module.split(".")[1] if "." in cov_module else cov_module
                    if key not in all_coverage or percent > all_coverage[key]:
                        all_coverage[key] = percent
        
        # Module coverage summary
        print("\nMODULE COVERAGE:")
        print("-" * 50)
        
        modules_at_target = 0
        for module, coverage in sorted(all_coverage.items()):
            status = "✅" if coverage >= 85 else "⚠️" if coverage >= 70 else "❌"
            print(f"{status} {module:20s}: {coverage:5.1f}%")
            if coverage >= 85:
                modules_at_target += 1
        
        # Calculate overall coverage (average of module coverages)
        if all_coverage:
            overall_coverage = sum(all_coverage.values()) / len(all_coverage)
        else:
            overall_coverage = 0
        
        print("\n" + "-" * 50)
        print(f"Overall Coverage: {overall_coverage:.1f}%")
        print(f"Modules at 85% target: {modules_at_target}/{len(all_coverage)}")
        print(f"Test files passed: {total_passed}/{total_tests}")
        
        # Show modules needing work
        modules_below_target = [(m, c) for m, c in all_coverage.items() if c < 85]
        if modules_below_target:
            print("\nMODULES NEEDING COVERAGE IMPROVEMENT:")
            print("-" * 50)
            for module, coverage in sorted(modules_below_target, key=lambda x: x[1]):
                gap = 85 - coverage
                print(f"  {module}: {coverage:.1f}% (needs +{gap:.1f}%)")
        
        # Save detailed results
        with open("isolated_coverage_results.json", "w") as f:
            json.dump({
                "summary": {
                    "overall_coverage": overall_coverage,
                    "modules_at_target": modules_at_target,
                    "total_modules": len(all_coverage),
                    "module_coverage": all_coverage
                },
                "detailed_results": results
            }, f, indent=2)
        
        print("\n✅ Detailed results saved to isolated_coverage_results.json")


def main():
    """Main entry point."""
    print("🚀 Starting Isolated Coverage Runner")
    print("This tool runs each test file in isolation to avoid torch import conflicts\n")
    
    runner = IsolatedCoverageRunner()
    
    if len(sys.argv) > 1:
        # Run specific module
        module_name = sys.argv[1]
        result = runner.run_module_tests(module_name)
        runner.generate_report({module_name: result})
    else:
        # Run all modules
        results = runner.run_all_modules()
        runner.generate_report(results)


if __name__ == "__main__":
    main()