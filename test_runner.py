#!/usr/bin/env python3
"""Run tests and collect timing/coverage data."""

import subprocess
import time
import os

# First run overall coverage
print("=" * 80)
print("RUNNING OVERALL COVERAGE TEST")
print("=" * 80)

start = time.time()
cmd = ["pytest", "--cov=driada", "--cov-report=term-missing", "--cov-report=html", "-x"]
result = subprocess.run(cmd, capture_output=True, text=True)
duration = time.time() - start

# Save coverage output
with open("coverage_report.txt", "w") as f:
    f.write(result.stdout)
    f.write("\n\nERRORS:\n")
    f.write(result.stderr)

print(f"Coverage test completed in {duration:.1f}s")
print(f"Coverage report saved to coverage_report.txt and htmlcov/")

# Extract coverage percentage
for line in result.stdout.split("\n"):
    if "TOTAL" in line and "%" in line:
        print(f"Overall coverage: {line.strip()}")

# Now run tests individually to get timing
print("\n" + "=" * 80)
print("RUNNING INDIVIDUAL TEST FILES")
print("=" * 80)

test_files = [
    "tests/unit/test_api_imports.py",
    "tests/unit/utils/test_spatial.py", 
    "tests/unit/utils/test_utils_signals.py",
    "tests/unit/visualization/test_visual.py",
    "tests/unit/visualization/test_visual_utils.py",
    "tests/unit/gdrive/test_download.py",
    "tests/unit/network/test_network.py",
    "tests/unit/experiment/test_neuron.py",
    "tests/unit/experiment/test_calcium_dynamics.py",
    "tests/unit/experiment/test_duplicate_behavior.py",
    "tests/unit/experiment/test_exp.py",
    "tests/unit/experiment/test_spike_reconstruction.py",
    "tests/unit/experiment/test_spike_reconstruction_refactor.py",
    "tests/unit/experiment/synthetic/test_2d_spatial_manifold.py",
    "tests/unit/experiment/synthetic/test_3d_spatial_manifold.py", 
    "tests/unit/experiment/synthetic/test_circular_manifold.py",
    "tests/unit/experiment/synthetic/test_mixed_population.py",
    "tests/unit/information/test_entropy.py",
    "tests/unit/information/test_conditional_mi_and_interaction.py",
    "tests/unit/information/test_gcmi.py",
    "tests/unit/intense/test_stats.py",
    "tests/unit/intense/test_intense.py",
    "tests/unit/intense/test_disentanglement.py",
    "tests/unit/intense/test_intense_pipelines.py",
    "tests/unit/dim_reduction/test_correct_cov_spectrum.py",
    "tests/unit/dim_reduction/test_dr_defaults.py",
    "tests/unit/dim_reduction/test_eff_dim.py",
    "tests/unit/dim_reduction/test_eps_graph.py",
    "tests/unit/dim_reduction/test_linear.py",
    "tests/unit/dim_reduction/test_manifold_metrics.py",
    "tests/unit/dim_reduction/test_dr.py",
    "tests/unit/dim_reduction/test_intrinsic.py",
    "tests/unit/dim_reduction/test_dr_extended.py",
    "tests/unit/dim_reduction/test_ae_vae_advanced.py",
    "tests/performance/test_parallel.py",
    "tests/integration/test_integration.py",
    "tests/integration/test_selectivity_mapper.py",
    "tests/integration/test_manifold_neural_data.py",
]

results = []
total_tests = len(test_files)

for i, test_file in enumerate(test_files):
    print(f"\n[{i+1}/{total_tests}] Running: {test_file}")
    start = time.time()
    
    cmd = ["pytest", test_file, "-xvs", "-q"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    duration = time.time() - start
    
    # Count test results
    output = proc.stdout
    passed = output.count(" PASSED")
    failed = output.count(" FAILED") 
    skipped = output.count(" SKIPPED")
    
    status = "PASSED" if proc.returncode == 0 else "FAILED"
    
    results.append({
        "file": test_file,
        "duration": duration,
        "status": status,
        "passed": passed,
        "failed": failed,
        "skipped": skipped
    })
    
    print(f"  Status: {status}, Duration: {duration:.2f}s, P:{passed} F:{failed} S:{skipped}")

# Summary
print("\n" + "=" * 80)
print("TEST EXECUTION SUMMARY")
print("=" * 80)

# Sort by duration
results.sort(key=lambda x: x["duration"], reverse=True)

print("\nSlowest tests:")
for r in results[:10]:
    print(f"  {r['duration']:6.2f}s - {r['file']} ({r['status']})")

print("\nFailed tests:")
failed_count = 0
for r in results:
    if r["status"] == "FAILED":
        print(f"  {r['file']}")
        failed_count += 1
if failed_count == 0:
    print("  None")

print("\nTests with skips:")
skip_count = 0  
for r in results:
    if r["skipped"] > 0:
        print(f"  {r['file']} - {r['skipped']} skipped")
        skip_count += 1
if skip_count == 0:
    print("  None")

# Overall stats
total_duration = sum(r["duration"] for r in results)
total_passed = sum(r["passed"] for r in results)
total_failed = sum(r["failed"] for r in results)
total_skipped = sum(r["skipped"] for r in results)

print(f"\nOverall statistics:")
print(f"  Total test files: {len(results)}")
print(f"  Total duration: {total_duration:.1f}s")
print(f"  Total tests passed: {total_passed}")
print(f"  Total tests failed: {total_failed}")
print(f"  Total tests skipped: {total_skipped}")

# Save detailed results
with open("test_timing_report.txt", "w") as f:
    f.write("DRIADA Test Timing Report\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("Test file timing (sorted by duration):\n")
    for r in results:
        f.write(f"{r['duration']:6.2f}s - {r['file']} - {r['status']} (P:{r['passed']} F:{r['failed']} S:{r['skipped']})\n")
    
    f.write(f"\nTotal execution time: {total_duration:.1f}s\n")

print("\nDetailed timing report saved to test_timing_report.txt")