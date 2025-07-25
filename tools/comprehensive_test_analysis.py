"""Comprehensive test analysis with individual timing, coverage, and failure reporting."""

import subprocess
import time
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict


@dataclass
class TestResult:
    """Store test execution results."""
    file_path: str
    duration: float
    passed: int
    failed: int
    skipped: int
    errors: int
    coverage: float
    failures: List[str]
    skipped_tests: List[str]
    status: str
    
    @property
    def total_tests(self) -> int:
        return self.passed + self.failed + self.skipped + self.errors


class TestAnalyzer:
    """Analyze test suite comprehensively."""
    
    def __init__(self, test_dir: str = "tests"):
        self.test_dir = Path(test_dir)
        self.results: List[TestResult] = []
        
    def find_test_files(self) -> List[Path]:
        """Find all test files recursively."""
        return sorted(self.test_dir.rglob("test_*.py"))
    
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
        
        # Parse test counts
        count_match = re.search(
            r'(\d+) passed|(\d+) failed|(\d+) skipped|(\d+) error',
            output
        )
        if count_match:
            counts = count_match.groups()
            result["passed"] = int(counts[0] or 0)
            result["failed"] = int(counts[1] or 0)
            result["skipped"] = int(counts[2] or 0)
            result["errors"] = int(counts[3] or 0)
        
        # Extract failed test names
        failure_pattern = r'FAILED (.*?) - '
        result["failures"] = re.findall(failure_pattern, output)
        
        # Extract skipped test names
        skip_pattern = r'SKIPPED \[(.*?)\] (.*?):'
        result["skipped_tests"] = re.findall(skip_pattern, output)
        
        return result
    
    def parse_coverage(self, output: str) -> float:
        """Extract coverage percentage from output."""
        # Look for total coverage line
        match = re.search(r'TOTAL.*?(\d+)%', output)
        if match:
            return float(match.group(1))
        return 0.0
    
    def run_test_file(self, test_file: Path) -> TestResult:
        """Run a single test file and collect results."""
        print(f"\n{'='*70}")
        print(f"Running: {test_file}")
        print(f"{'='*70}")
        
        cmd = [
            "conda", "run", "-n", "driada",
            "python", "-m", "pytest",
            str(test_file),
            "-v",
            "--tb=short",
            "--cov=src/driada",
            "--cov-report=term",
            "--durations=10",
            "-p", "no:warnings"
        ]
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per file
            )
            duration = time.time() - start_time
            
            # Parse output
            output = result.stdout + result.stderr
            stats = self.parse_pytest_output(output)
            coverage = self.parse_coverage(output)
            
            status = "PASSED" if result.returncode == 0 else "FAILED"
            
            # Print summary
            print(f"Duration: {duration:.2f}s")
            print(f"Tests: {stats['passed']} passed, {stats['failed']} failed, "
                  f"{stats['skipped']} skipped")
            print(f"Coverage: {coverage}%")
            print(f"Status: {status}")
            
            return TestResult(
                file_path=str(test_file),
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
                file_path=str(test_file),
                duration=duration,
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                coverage=0.0,
                failures=["TIMEOUT"],
                skipped_tests=[],
                status="TIMEOUT"
            )
        except Exception as e:
            print(f"ERROR: {e}")
            
            return TestResult(
                file_path=str(test_file),
                duration=0.0,
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                coverage=0.0,
                failures=[str(e)],
                skipped_tests=[],
                status="ERROR"
            )
    
    def run_all_tests(self):
        """Run all test files individually."""
        test_files = self.find_test_files()
        print(f"Found {len(test_files)} test files")
        
        for test_file in test_files:
            result = self.run_test_file(test_file)
            self.results.append(result)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report."""
        total_duration = sum(r.duration for r in self.results)
        total_tests = sum(r.total_tests for r in self.results)
        total_passed = sum(r.passed for r in self.results)
        total_failed = sum(r.failed for r in self.results)
        total_skipped = sum(r.skipped for r in self.results)
        total_errors = sum(r.errors for r in self.results)
        
        # Calculate weighted coverage
        coverage_weights = [(r.coverage * r.total_tests) for r in self.results]
        avg_coverage = sum(coverage_weights) / total_tests if total_tests > 0 else 0
        
        # Find slow tests (>10s)
        slow_tests = [r for r in self.results if r.duration > 10]
        
        # Find timeout tests
        timeout_tests = [r for r in self.results if r.status == "TIMEOUT"]
        
        # Find tests with skips
        tests_with_skips = [r for r in self.results if r.skipped > 0]
        
        # Find failing tests
        failing_tests = [r for r in self.results if r.failed > 0 or r.errors > 0]
        
        report = {
            "summary": {
                "total_files": len(self.results),
                "total_tests": total_tests,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "total_skipped": total_skipped,
                "total_errors": total_errors,
                "total_duration": total_duration,
                "average_coverage": avg_coverage,
                "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0
            },
            "slow_tests": [asdict(r) for r in sorted(slow_tests, key=lambda x: x.duration, reverse=True)],
            "timeout_tests": [asdict(r) for r in timeout_tests],
            "tests_with_skips": [asdict(r) for r in tests_with_skips],
            "failing_tests": [asdict(r) for r in failing_tests],
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
        
        print(f"\nTotal Files: {summary['total_files']}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"  ✅ Passed: {summary['total_passed']}")
        print(f"  ❌ Failed: {summary['total_failed']}")
        print(f"  ⏭️  Skipped: {summary['total_skipped']}")
        print(f"  💥 Errors: {summary['total_errors']}")
        print(f"\nTotal Duration: {summary['total_duration']:.2f}s ({summary['total_duration']/60:.2f} minutes)")
        print(f"Average Coverage: {summary['average_coverage']:.2f}%")
        print(f"Success Rate: {summary['success_rate']:.2f}%")
        
        # Slow tests
        if report["slow_tests"]:
            print("\n" + "-"*70)
            print("SLOW TESTS (>10s):")
            print("-"*70)
            for test in report["slow_tests"][:10]:  # Top 10
                print(f"  {test['file_path']}: {test['duration']:.2f}s")
        
        # Timeout tests
        if report["timeout_tests"]:
            print("\n" + "-"*70)
            print("TIMEOUT TESTS:")
            print("-"*70)
            for test in report["timeout_tests"]:
                print(f"  {test['file_path']}: TIMEOUT after {test['duration']:.2f}s")
        
        # Tests with skips
        if report["tests_with_skips"]:
            print("\n" + "-"*70)
            print("TESTS WITH SKIPPED ITEMS:")
            print("-"*70)
            for test in report["tests_with_skips"]:
                print(f"  {test['file_path']}: {test['skipped']} skipped")
                for skip_info in test["skipped_tests"]:
                    if isinstance(skip_info, tuple) and len(skip_info) >= 2:
                        print(f"    - {skip_info[1]}: {skip_info[0]}")
        
        # Failing tests
        if report["failing_tests"]:
            print("\n" + "-"*70)
            print("FAILING TESTS:")
            print("-"*70)
            for test in report["failing_tests"]:
                print(f"  {test['file_path']}: {test['failed']} failed, {test['errors']} errors")
                for failure in test["failures"]:
                    print(f"    - {failure}")
        
        # Save detailed report
        with open("comprehensive_test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("\n✅ Detailed report saved to comprehensive_test_report.json")


def main():
    """Run comprehensive test analysis."""
    analyzer = TestAnalyzer()
    
    print("Starting comprehensive test analysis...")
    print("This will run each test file individually to collect detailed metrics.")
    print("Expected duration: ~30-60 minutes\n")
    
    start_time = time.time()
    analyzer.run_all_tests()
    total_time = time.time() - start_time
    
    analyzer.print_summary()
    
    print(f"\n🏁 Analysis complete in {total_time:.2f}s ({total_time/60:.2f} minutes)")


if __name__ == "__main__":
    main()