"""Comprehensive test analysis with batched execution to avoid timeouts."""

import subprocess
import time
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class TestResult:
    """Store test execution results."""
    category: str  # e.g., "unit", "integration", "jit"
    module: str    # e.g., "information", "intense", "utils"
    duration: float
    passed: int
    failed: int
    skipped: int
    errors: int
    coverage: Dict[str, float]  # module -> coverage%
    failures: List[str]
    skipped_tests: List[str]
    status: str
    
    @property
    def total_tests(self) -> int:
        return self.passed + self.failed + self.skipped + self.errors


class TestAnalyzer:
    """Analyze test suite comprehensively with batched execution."""
    
    def __init__(self, test_dir: str = "tests"):
        self.test_dir = Path(test_dir)
        self.results: List[TestResult] = []
        # Categories to run separately to avoid timeouts
        self.test_categories = {
            "unit_jit": {
                "pattern": "**/test_*jit*.py",
                "env_vars": {"NUMBA_DISABLE_JIT": "0"},  # Enable JIT
                "description": "JIT-compiled tests"
            },
            "unit_non_jit": {
                "pattern": "unit/**/test_*.py",
                "exclude": ["*jit*", "*test_intense_pipelines*"],  # Exclude JIT and slow pipeline tests
                "env_vars": {"NUMBA_DISABLE_JIT": "1"},  # Disable JIT
                "description": "Regular unit tests"
            },
            "unit_intense_pipelines": {
                "pattern": "unit/intense/test_intense_pipelines*.py",
                "env_vars": {"NUMBA_DISABLE_JIT": "1"},
                "description": "INTENSE pipeline tests"
            },
            "integration": {
                "pattern": "integration/**/test_*.py",
                "env_vars": {"NUMBA_DISABLE_JIT": "1"},
                "description": "Integration tests"
            },
            "synthetic": {
                "pattern": "unit/experiment/synthetic/test_*.py",
                "env_vars": {"NUMBA_DISABLE_JIT": "1"},
                "description": "Synthetic data tests"
            }
        }
        
    def find_test_files(self, pattern: str, exclude: Optional[List[str]] = None) -> List[Path]:
        """Find test files matching pattern."""
        files = list(self.test_dir.glob(pattern))
        
        if exclude:
            files = [f for f in files if not any(exc in str(f) for exc in exclude)]
        
        return sorted(files)
    
    def parse_pytest_output(self, output: str) -> Dict[str, Any]:
        """Parse pytest output for test statistics."""
        result = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "failures": [],
            "skipped_tests": []
        }
        
        # Parse test counts - look for summary line
        summary_pattern = r'(\d+) passed|(\d+) failed|(\d+) skipped|(\d+) error'
        for match in re.finditer(summary_pattern, output):
            groups = match.groups()
            if groups[0]: result["passed"] = int(groups[0])
            if groups[1]: result["failed"] = int(groups[1])
            if groups[2]: result["skipped"] = int(groups[2])
            if groups[3]: result["errors"] = int(groups[3])
        
        # Extract failed test names
        failure_pattern = r'FAILED (.*?) - '
        result["failures"] = re.findall(failure_pattern, output)
        
        # Extract skipped test names
        skip_pattern = r'SKIPPED \[(.*?)\] (.*?):'
        result["skipped_tests"] = re.findall(skip_pattern, output)
        
        return result
    
    def parse_coverage(self, output: str) -> Dict[str, float]:
        """Extract coverage percentages by module."""
        coverage = {}
        
        # Look for coverage lines - format can be either:
        # "src/driada/module/submodule.py   100    20    80%"
        # "driada/module/submodule.py   100    20    80%"
        coverage_pattern = r'(?:src/)?driada/(\S+?)\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+(?:\.\d+)?)%'
        for match in re.finditer(coverage_pattern, output):
            module_path = match.group(1)
            percentage = float(match.group(2))
            
            # Extract top-level module
            module = module_path.split('/')[0].replace('.py', '')
            if module not in coverage or coverage[module] < percentage:
                coverage[module] = percentage
        
        # Also look for TOTAL line - format: "TOTAL    1234    567    89%"
        total_match = re.search(r'TOTAL\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+(?:\.\d+)?)%', output)
        if total_match:
            coverage["TOTAL"] = float(total_match.group(1))
        
        return coverage
    
    def run_test_category(self, category: str, config: Dict[str, Any]) -> TestResult:
        """Run tests for a specific category."""
        print(f"\n{'='*70}")
        print(f"Running {category}: {config['description']}")
        print(f"{'='*70}")
        
        # Find test files
        exclude = config.get("exclude", None)
        test_files = self.find_test_files(config["pattern"], exclude)
        
        if not test_files:
            print(f"No test files found for {category}")
            return TestResult(
                category=category,
                module="N/A",
                duration=0.0,
                passed=0,
                failed=0,
                skipped=0,
                errors=0,
                coverage={},
                failures=[],
                skipped_tests=[],
                status="NO_TESTS"
            )
        
        print(f"Found {len(test_files)} test files")
        
        # Build command
        cmd = [
            sys.executable, "-m", "pytest",
            *[str(f) for f in test_files],
            "-v",
            "--tb=short",
            "--cov=driada",
            "--cov-report=term-missing:skip-covered",
            "--durations=10",
            "-p", "no:warnings"
        ]
        
        # Set environment variables
        env = os.environ.copy()
        env.update(config.get("env_vars", {}))
        
        start_time = time.time()
        try:
            # Run with explicit driada environment activation
            full_cmd = ["conda", "run", "-n", "driada"] + cmd
            
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=600  # 10 minute timeout per category
            )
            duration = time.time() - start_time
            
            # Parse output
            output = result.stdout + result.stderr
            stats = self.parse_pytest_output(output)
            coverage = self.parse_coverage(output)
            
            status = "PASSED" if result.returncode == 0 else "FAILED"
            
            # Print summary
            print(f"\nDuration: {duration:.2f}s")
            print(f"Tests: {stats['passed']} passed, {stats['failed']} failed, "
                  f"{stats['skipped']} skipped, {stats['errors']} errors")
            
            if coverage:
                print("\nCoverage by module:")
                for module, cov in sorted(coverage.items()):
                    if module != "TOTAL":
                        print(f"  {module}: {cov:.1f}%")
                if "TOTAL" in coverage:
                    print(f"\nTotal Coverage: {coverage['TOTAL']:.1f}%")
            
            print(f"Status: {status}")
            
            # Show failures if any
            if stats["failures"]:
                print(f"\nFailures ({len(stats['failures'])}):")
                for failure in stats["failures"][:5]:  # First 5
                    print(f"  - {failure}")
                if len(stats["failures"]) > 5:
                    print(f"  ... and {len(stats['failures']) - 5} more")
            
            return TestResult(
                category=category,
                module=config['description'],
                duration=duration,
                passed=stats["passed"],
                failed=stats["failed"],
                skipped=stats["skipped"],
                errors=stats["errors"],
                coverage=coverage,
                failures=stats["failures"],
                skipped_tests=stats["skipped_tests"],
                status=status
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"TIMEOUT after {duration:.2f}s")
            
            return TestResult(
                category=category,
                module=config['description'],
                duration=duration,
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                coverage={},
                failures=["TIMEOUT"],
                skipped_tests=[],
                status="TIMEOUT"
            )
        except Exception as e:
            print(f"ERROR: {e}")
            
            return TestResult(
                category=category,
                module=config['description'],
                duration=0.0,
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                coverage={},
                failures=[str(e)],
                skipped_tests=[],
                status="ERROR"
            )
    
    def run_all_tests(self):
        """Run all test categories."""
        print(f"Running tests in {len(self.test_categories)} categories")
        print("This approach uses batching to avoid timeouts and leverage fixtures\n")
        
        for category, config in self.test_categories.items():
            result = self.run_test_category(category, config)
            self.results.append(result)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report."""
        total_duration = sum(r.duration for r in self.results)
        total_tests = sum(r.total_tests for r in self.results)
        total_passed = sum(r.passed for r in self.results)
        total_failed = sum(r.failed for r in self.results)
        total_skipped = sum(r.skipped for r in self.results)
        total_errors = sum(r.errors for r in self.results)
        
        # Aggregate coverage by module - take MAXIMUM coverage achieved
        module_coverage_all = {}
        best_module_coverage = {}
        for result in self.results:
            for module, cov in result.coverage.items():
                if module != "TOTAL":
                    if module not in module_coverage_all:
                        module_coverage_all[module] = []
                    module_coverage_all[module].append((cov, result.category))
                    
                    # Track best coverage per module
                    if module not in best_module_coverage or cov > best_module_coverage[module]["coverage"]:
                        best_module_coverage[module] = {
                            "coverage": cov,
                            "category": result.category
                        }
        
        # Find the best overall coverage from individual test runs
        best_total_coverage = 0
        best_coverage_category = None
        for result in self.results:
            if "TOTAL" in result.coverage and result.coverage["TOTAL"] > best_total_coverage:
                best_total_coverage = result.coverage["TOTAL"]
                best_coverage_category = result.category
        
        # Find slow categories (>60s)
        slow_categories = [r for r in self.results if r.duration > 60]
        
        # Find timeout tests
        timeout_tests = [r for r in self.results if r.status == "TIMEOUT"]
        
        # Find categories with skips
        categories_with_skips = [r for r in self.results if r.skipped > 0]
        
        # Find failing categories
        failing_categories = [r for r in self.results if r.failed > 0 or r.errors > 0]
        
        # Extract just coverage values for summary
        best_coverage_values = {m: data["coverage"] for m, data in best_module_coverage.items()}
        
        report = {
            "summary": {
                "total_categories": len(self.results),
                "total_tests": total_tests,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "total_skipped": total_skipped,
                "total_errors": total_errors,
                "total_duration": total_duration,
                "best_total_coverage": best_total_coverage,
                "best_coverage_category": best_coverage_category,
                "best_module_coverage": best_coverage_values,
                "module_coverage_details": best_module_coverage,
                "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
                "modules_meeting_target": sum(1 for cov in best_coverage_values.values() if cov >= 90),
                "modules_below_target": [(m, cov) for m, cov in best_coverage_values.items() if cov < 90]
            },
            "slow_categories": [asdict(r) for r in sorted(slow_categories, key=lambda x: x.duration, reverse=True)],
            "timeout_tests": [asdict(r) for r in timeout_tests],
            "categories_with_skips": [asdict(r) for r in categories_with_skips],
            "failing_categories": [asdict(r) for r in failing_categories],
            "all_results": [asdict(r) for r in sorted(self.results, key=lambda x: x.duration, reverse=True)]
        }
        
        return report
    
    def print_summary(self):
        """Print human-readable summary."""
        report = self.generate_report()
        summary = report["summary"]
        
        print("\n" + "="*70)
        print("COMPREHENSIVE TEST ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"\nTotal Categories: {summary['total_categories']}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"  ✅ Passed: {summary['total_passed']}")
        print(f"  ❌ Failed: {summary['total_failed']}")
        print(f"  ⏭️  Skipped: {summary['total_skipped']}")
        print(f"  💥 Errors: {summary['total_errors']}")
        print(f"\nTotal Duration: {summary['total_duration']:.2f}s ({summary['total_duration']/60:.2f} minutes)")
        print(f"Best Overall Coverage: {summary['best_total_coverage']:.2f}% (from {summary['best_coverage_category']})")
        print(f"Success Rate: {summary['success_rate']:.2f}%")
        print(f"\nModules Meeting 90% Target: {summary['modules_meeting_target']}/{len(summary['best_module_coverage'])}")
        
        # Module coverage
        if summary["best_module_coverage"]:
            print("\n" + "-"*70)
            print("BEST MODULE COVERAGE (Maximum achieved across all test runs):")
            print("-"*70)
            for module, cov in sorted(summary["best_module_coverage"].items(), key=lambda x: x[1], reverse=True):
                status = "✅" if cov >= 90 else "⚠️" if cov >= 80 else "❌"
                details = summary["module_coverage_details"][module]
                print(f"  {status} {module}: {cov:.1f}% (from {details['category']})")
            
            # Show modules below target
            if summary["modules_below_target"]:
                print("\n" + "-"*70)
                print("MODULES REQUIRING ATTENTION (Below 90% target):")
                print("-"*70)
                for module, cov in sorted(summary["modules_below_target"], key=lambda x: x[1], reverse=True):
                    gap = 90 - cov
                    print(f"  ❌ {module}: {cov:.1f}% (needs {gap:.1f}% more)")
        
        # Slow categories
        if report["slow_categories"]:
            print("\n" + "-"*70)
            print("SLOW TEST CATEGORIES (>60s):")
            print("-"*70)
            for test in report["slow_categories"]:
                print(f"  {test['category']} ({test['module']}): {test['duration']:.2f}s")
        
        # Timeout tests
        if report["timeout_tests"]:
            print("\n" + "-"*70)
            print("TIMEOUT TESTS:")
            print("-"*70)
            for test in report["timeout_tests"]:
                print(f"  {test['category']}: TIMEOUT after {test['duration']:.2f}s")
        
        # Categories with skips
        if report["categories_with_skips"]:
            print("\n" + "-"*70)
            print("CATEGORIES WITH SKIPPED TESTS:")
            print("-"*70)
            for test in report["categories_with_skips"]:
                print(f"  {test['category']} ({test['module']}): {test['skipped']} skipped")
        
        # Failing categories
        if report["failing_categories"]:
            print("\n" + "-"*70)
            print("FAILING CATEGORIES:")
            print("-"*70)
            for test in report["failing_categories"]:
                print(f"  {test['category']} ({test['module']}): {test['failed']} failed, {test['errors']} errors")
                for failure in test["failures"][:3]:  # First 3 failures
                    print(f"    - {failure}")
                if len(test["failures"]) > 3:
                    print(f"    ... and {len(test['failures']) - 3} more")
        
        # Save detailed report
        with open("comprehensive_test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("\n✅ Detailed report saved to comprehensive_test_report.json")


def main():
    """Run comprehensive test analysis."""
    # Ensure we're in the driada environment
    env_check = subprocess.run(
        ["conda", "info", "--envs"],
        capture_output=True,
        text=True
    )
    if "driada" not in env_check.stdout:
        print("❌ ERROR: driada conda environment not found!")
        print("Please create it with: conda create -n driada python=3.8")
        sys.exit(1)
    
    analyzer = TestAnalyzer()
    
    print("🚀 Starting Comprehensive Test Analysis")
    print("="*70)
    print("This will run tests in batches to avoid timeouts while leveraging fixtures.")
    print("Categories: JIT tests, Regular unit tests, INTENSE pipelines, Integration, Synthetic")
    print("Expected duration: ~10-20 minutes\n")
    
    start_time = time.time()
    analyzer.run_all_tests()
    total_time = time.time() - start_time
    
    analyzer.print_summary()
    
    print(f"\n🏁 Analysis complete in {total_time:.2f}s ({total_time/60:.2f} minutes)")
    
    # Quick summary of critical metrics
    report = analyzer.generate_report()
    summary = report["summary"]
    
    print("\n" + "="*70)
    print("QUICK METRICS:")
    print("="*70)
    print(f"✅ Best Coverage: {summary['best_total_coverage']:.1f}% (Target: 90%)")
    print(f"✅ Modules at Target: {summary['modules_meeting_target']}/{len(summary['best_module_coverage'])}")
    print(f"✅ Success Rate: {summary['success_rate']:.1f}%")
    print(f"✅ Total Runtime: {summary['total_duration']:.1f}s")
    
    if summary['best_total_coverage'] < 90:
        gap = 90 - summary['best_total_coverage']
        print(f"\n⚠️  Overall coverage is {gap:.1f}% below target!")
        print(f"Focus on: {', '.join([f'{m} ({c:.1f}%)' for m, c in summary['modules_below_target'][:3]])}")


if __name__ == "__main__":
    main()