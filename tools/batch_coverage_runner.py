#!/usr/bin/env python
"""Batch coverage runner v2 with improved aggregation.

This runner:
1. Collects best coverage for each file across all test runs
2. Uses weighted averages based on file size (lines of code)
3. Properly handles distributed module files
4. Never shows 0% for files that have coverage

IMPORTANT: Pytest timeouts are disabled in this runner to ensure
all tests complete. The PYTEST_TIMEOUT environment variable is set to '0'
which disables timeouts completely. This ensures accurate coverage
reporting even for long-running tests.
"""

import subprocess
import json
import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import tempfile


class BatchCoverageRunnerV2:
    """Run coverage tests in batches with improved aggregation."""

    def __init__(self):
        self.test_root = Path("tests")
        self.project_root = Path.cwd()

        # Module file mappings - MUST follow folder structure exactly
        self.module_file_mappings = {
            "intense": [
                "driada/intense/",
            ],
            "experiment": [
                "driada/experiment/",
            ],
            "utils": [
                "driada/utils/",
            ],
            "integration": [
                "driada/integration/",
            ],
            "information": [
                "driada/information/",
            ],
            "network": [
                "driada/network/",
            ],
            "rsa": [
                "driada/rsa/",
            ],
            "gdrive": [
                "driada/gdrive/",
            ],
            "dim_reduction": [
                "driada/dim_reduction/",
            ],
            "dimensionality": [
                "driada/dimensionality/",
            ],
        }

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

                # Also check for subdirectories
                for subdir in module_dir.iterdir():
                    if subdir.is_dir() and not subdir.name.startswith("__"):
                        sub_test_files = list(subdir.glob("test_*.py"))
                        if sub_test_files:
                            modules[f"{module_dir.name}/{subdir.name}"] = sub_test_files

        return modules

    def check_file_belongs_to_module(self, file_path: str, module_name: str) -> bool:
        """Check if a file belongs to a specific module based on patterns."""
        base_module = module_name.split("/")[0]
        patterns = self.module_file_mappings.get(
            base_module, [f"driada/{base_module}/"]
        )

        # First check exclusions
        for pattern in patterns:
            if pattern.startswith("!"):
                exclude_pattern = pattern[1:]
                if exclude_pattern in file_path:
                    return False

        # Then check inclusions
        for pattern in patterns:
            if not pattern.startswith("!") and pattern in file_path:
                return True

        return False

    def run_module_batch(
        self, module_name: str, test_files: List[Path]
    ) -> Dict[str, any]:
        """Run all tests for a module in a single batch."""
        print(f"\nRunning {module_name} module ({len(test_files)} files)...")

        # Determine coverage module
        base_module = module_name.split("/")[0]
        cov_module = f"driada.{base_module}"

        # Create test runner script
        runner_script = """
import sys
import os

# Set environment for stable execution
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache_batch'

# IMPORTANT: Disable pytest timeouts to ensure all tests complete
os.environ['PYTEST_TIMEOUT'] = '0'  # Disable timeouts completely

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

# Change to project root
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
""".format(
            project_root=self.project_root,
            test_files=", ".join(f'"{f}"' for f in test_files),
            cov_module=cov_module,
        )

        # Write and run script
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(runner_script)
            script_path = f.name

        try:
            # Check if we're in a conda environment
            if os.environ.get("CONDA_DEFAULT_ENV") == "driada":
                cmd = ["conda", "run", "-n", "driada", "python", script_path]
            else:
                # Use system python (for CI environments)
                cmd = [sys.executable, script_path]
            result = subprocess.run(cmd, capture_output=True, text=True)

            output = result.stdout + result.stderr

            # Save debug output
            debug_file = f"coverage_debug_{module_name.replace('/', '_')}.txt"
            with open(debug_file, "w") as f:
                f.write(output)

            # Parse coverage data
            coverage_data = self._parse_coverage_from_output(output)

            return {
                "module": module_name,
                "passed": result.returncode == 0,
                "test_files": len(test_files),
                "coverage_data": coverage_data,
                "test_counts": self._parse_test_counts(output),
                "returncode": result.returncode,
                "debug_output": debug_file,
            }
        finally:
            os.unlink(script_path)

    def _parse_coverage_from_output(self, output: str) -> Dict[str, dict]:
        """Parse coverage data including file sizes from output."""
        coverage_data = {}
        in_coverage = False

        for line in output.split("\n"):
            # Start of coverage table
            if "Name" in line and "Stmts" in line:
                in_coverage = True
                continue

            # End of coverage table
            if in_coverage and ("=" in line or "TOTAL" in line):
                if "TOTAL" in line:
                    # Parse TOTAL line if needed
                    pass
                if "=" in line:
                    break

            # Parse coverage lines
            if in_coverage and line.strip():
                # Match lines with coverage data
                # Examples:
                # src/driada/network/net_base.py    379    179    168     36  50.09%
                # driada/utils/visual.py             341    328    190      0   2.45%
                match = re.match(
                    r"\s*(?:src/)?(driada/[\w/]+\.py)\s+(\d+)\s+(\d+)(?:\s+\d+\s+\d+)?\s+([\d.]+)%",
                    line,
                )
                if match:
                    file_path = match.group(1)  # driada/... path (src/ prefix is non-capturing)
                    statements = int(match.group(2))
                    missed = int(match.group(3))
                    coverage_pct = float(match.group(4))

                    # Create module path without duplication
                    module_path = file_path.replace("/", ".").replace(".py", "")

                    coverage_data[module_path] = {
                        "file_path": file_path,
                        "statements": statements,
                        "missed": missed,
                        "covered": statements - missed,
                        "coverage_pct": coverage_pct,
                    }

        return coverage_data

    def _parse_test_counts(self, output: str) -> Dict[str, int]:
        """Parse test pass/fail counts."""
        counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}

        # Look for pytest summary
        summary_match = re.search(r"=+ ([\d\s\w,]+) in [\d.]+s", output)

        if summary_match:
            summary = summary_match.group(1)

            for key, pattern in [
                ("passed", r"(\d+) passed"),
                ("failed", r"(\d+) failed"),
                ("skipped", r"(\d+) skipped"),
                ("errors", r"(\d+) error"),
            ]:
                match = re.search(pattern, summary)
                if match:
                    counts[key] = int(match.group(1))

        return counts

    def aggregate_coverage_data(self, all_results: Dict[str, any]) -> Dict[str, dict]:
        """Aggregate coverage data from all test runs, keeping best coverage."""
        aggregated = {}

        for module_name, result in all_results.items():
            if "coverage_data" in result:
                for module_path, data in result["coverage_data"].items():
                    if (
                        module_path not in aggregated
                        or data["coverage_pct"]
                        > aggregated[module_path]["coverage_pct"]
                    ):
                        # Keep the best coverage seen
                        aggregated[module_path] = data.copy()
                        aggregated[module_path]["tested_by"] = module_name

        return aggregated

    def calculate_module_coverage(
        self, module_name: str, aggregated_data: Dict[str, dict]
    ) -> Optional[dict]:
        """Calculate weighted coverage for a module."""
        module_files = []
        total_statements = 0
        total_covered = 0

        for module_path, data in aggregated_data.items():
            if self.check_file_belongs_to_module(data["file_path"], module_name):
                module_files.append(
                    {
                        "path": module_path,
                        "file": data["file_path"],
                        "statements": data["statements"],
                        "covered": data["covered"],
                        "coverage_pct": data["coverage_pct"],
                        "tested_by": data.get("tested_by", "unknown"),
                    }
                )
                total_statements += data["statements"]
                total_covered += data["covered"]

        if not module_files:
            return None

        # Calculate metrics
        simple_avg = sum(f["coverage_pct"] for f in module_files) / len(module_files)

        # Weighted average by file size
        weighted_sum = sum(f["coverage_pct"] * f["statements"] for f in module_files)
        weighted_avg = weighted_sum / total_statements if total_statements > 0 else 0

        # True coverage (total covered / total statements)
        true_coverage = (
            (total_covered / total_statements * 100) if total_statements > 0 else 0
        )

        return {
            "files": module_files,
            "file_count": len(module_files),
            "total_statements": total_statements,
            "total_covered": total_covered,
            "simple_average": simple_avg,
            "weighted_average": weighted_avg,
            "true_coverage": true_coverage,
        }

    def generate_report(self, results: Dict[str, any]) -> None:
        """Generate comprehensive coverage report."""
        print("\n" + "=" * 70)
        print("BATCH COVERAGE REPORT V2")
        print("=" * 70)

        # First aggregate all coverage data
        aggregated_data = self.aggregate_coverage_data(results)

        # Calculate coverage for each module
        module_coverage = {}
        for module_name in results.keys():
            # Skip experiment/synthetic as it's a duplicate of experiment
            if module_name == "experiment/synthetic":
                continue
            coverage = self.calculate_module_coverage(module_name, aggregated_data)
            if coverage:
                module_coverage[module_name] = coverage

        # Display results
        print("\nMODULE COVERAGE (Weighted by File Size):")
        print("-" * 70)

        modules_at_target = 0
        total_tests_passed = 0
        total_tests_failed = 0

        for module_name, coverage in sorted(module_coverage.items()):
            weighted_cov = coverage["weighted_average"]
            status = "✅" if weighted_cov >= 85 else "⚠️" if weighted_cov >= 70 else "❌"

            print(
                f"{status} {module_name:20s}: {weighted_cov:5.1f}% "
                f"({coverage['file_count']} files, {coverage['total_statements']:,} lines)"
            )

            if weighted_cov >= 85:
                modules_at_target += 1

            # Show top uncovered files
            uncovered_files = sorted(
                coverage["files"], key=lambda x: x["coverage_pct"]
            )[:3]
            for f in uncovered_files:
                if f["coverage_pct"] < 85:
                    print(
                        f"  ❌ {f['file'].split('/')[-1]:30s}: {f['coverage_pct']:5.1f}% ({f['statements']} lines)"
                    )

        # Count tests
        for result in results.values():
            counts = result.get("test_counts", {})
            total_tests_passed += counts.get("passed", 0)
            total_tests_failed += counts.get("failed", 0) + counts.get("errors", 0)

        # Overall stats
        all_statements = sum(c["total_statements"] for c in module_coverage.values())
        all_covered = sum(c["total_covered"] for c in module_coverage.values())
        overall_coverage = (
            (all_covered / all_statements * 100) if all_statements > 0 else 0
        )

        print("\n" + "-" * 50)
        print(f"Overall Coverage: {overall_coverage:.1f}%")
        print(f"Total Lines: {all_statements:,}")
        print(f"Lines Covered: {all_covered:,}")
        print(f"Modules at 85% target: {modules_at_target}/{len(module_coverage)}")
        print(f"Total tests passed: {total_tests_passed}")
        print(f"Total tests failed: {total_tests_failed}")

        # Show failures
        failing_modules = [
            name for name, result in results.items() if not result["passed"]
        ]
        if failing_modules:
            print("\nFAILING MODULES:")
            for module in failing_modules:
                print(f"  ❌ {module}")

        # Save detailed results
        output = {
            "summary": {
                "overall_coverage": overall_coverage,
                "total_statements": all_statements,
                "total_covered": all_covered,
                "modules_at_target": modules_at_target,
                "total_modules": len(module_coverage),
                "tests_passed": total_tests_passed,
                "tests_failed": total_tests_failed,
            },
            "module_coverage": module_coverage,
            "aggregated_file_coverage": {
                k: v["coverage_pct"] for k, v in aggregated_data.items()
            },
            "detailed_results": results,
        }

        with open("batch_coverage_results_v2.json", "w") as f:
            json.dump(output, f, indent=2)

        print("\n✅ Detailed results saved to batch_coverage_results_v2.json")

    def run_all_modules(self) -> Dict[str, any]:
        """Run all test modules."""
        modules = self.find_test_modules()
        results = {}

        print(f"Found {len(modules)} test modules")

        for module_name, test_files in sorted(modules.items()):
            result = self.run_module_batch(module_name, test_files)
            results[module_name] = result

        return results


def main():
    """Run batch coverage analysis with improved aggregation."""
    print("🚀 Starting Batch Coverage Runner V2")
    print("Features: Best coverage aggregation, weighted by file size\n")

    runner = BatchCoverageRunnerV2()

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
