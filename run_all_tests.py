#!/usr/bin/env python3
"""Run all tests individually and collect timing/status data."""

import subprocess
import time
import json
from pathlib import Path

test_files = [
    "tests/integration/test_integration.py",
    "tests/integration/test_manifold_neural_data.py",
    "tests/integration/test_selectivity_mapper.py",
    "tests/performance/test_parallel.py",
    "tests/unit/dim_reduction/test_ae_vae_advanced.py",
    "tests/unit/dim_reduction/test_correct_cov_spectrum.py",
    "tests/unit/dim_reduction/test_dr.py",
    "tests/unit/dim_reduction/test_dr_defaults.py",
    "tests/unit/dim_reduction/test_dr_extended.py",
    "tests/unit/dim_reduction/test_eff_dim.py",
    "tests/unit/dim_reduction/test_eps_graph.py",
    "tests/unit/dim_reduction/test_intrinsic.py",
    "tests/unit/dim_reduction/test_linear.py",
    "tests/unit/dim_reduction/test_manifold_metrics.py",
    "tests/unit/experiment/synthetic/test_2d_spatial_manifold.py",
    "tests/unit/experiment/synthetic/test_3d_spatial_manifold.py",
    "tests/unit/experiment/synthetic/test_circular_manifold.py",
    "tests/unit/experiment/synthetic/test_mixed_population.py",
    "tests/unit/experiment/test_calcium_dynamics.py",
    "tests/unit/experiment/test_duplicate_behavior.py",
    "tests/unit/experiment/test_exp.py",
    "tests/unit/experiment/test_neuron.py",
    "tests/unit/experiment/test_spike_reconstruction.py",
    "tests/unit/experiment/test_spike_reconstruction_refactor.py",
    "tests/unit/gdrive/test_download.py",
    "tests/unit/information/test_conditional_mi_and_interaction.py",
    "tests/unit/information/test_entropy.py",
    "tests/unit/information/test_gcmi.py",
    "tests/unit/intense/test_disentanglement.py",
    "tests/unit/intense/test_intense.py",
    "tests/unit/intense/test_intense_pipelines.py",
    "tests/unit/intense/test_stats.py",
    "tests/unit/network/test_network.py",
    "tests/unit/test_api_imports.py",
    "tests/unit/utils/test_spatial.py",
    "tests/unit/utils/test_utils_signals.py",
    "tests/unit/visualization/test_visual.py",
    "tests/unit/visualization/test_visual_utils.py"
]

results = []

print("Running all tests individually...")
print("=" * 80)

for test_file in test_files:
    print(f"\nRunning: {test_file}")
    start_time = time.time()
    
    # Run test with pytest
    cmd = ["pytest", test_file, "-xvs", "--tb=short"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Parse output for test counts
    passed = proc.stdout.count(" PASSED")
    failed = proc.stdout.count(" FAILED")
    skipped = proc.stdout.count(" SKIPPED")
    
    # Check for overall status
    status = "PASSED" if proc.returncode == 0 else "FAILED"
    
    result = {
        "file": test_file,
        "duration": round(duration, 2),
        "status": status,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "return_code": proc.returncode
    }
    
    # Extract error info if failed
    if status == "FAILED" and "short test summary" in proc.stdout:
        summary_start = proc.stdout.index("short test summary")
        error_summary = proc.stdout[summary_start:summary_start+500]
        result["error_summary"] = error_summary
    
    results.append(result)
    
    print(f"  Status: {status}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Passed: {passed}, Failed: {failed}, Skipped: {skipped}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Sort by duration
results.sort(key=lambda x: x["duration"], reverse=True)

print("\nSlowest tests:")
for r in results[:10]:
    print(f"  {r['duration']:6.2f}s - {r['file']} ({r['status']})")

print("\nFailed tests:")
for r in results:
    if r["status"] == "FAILED":
        print(f"  {r['file']}")

print("\nTests with skips:")
for r in results:
    if r["skipped"] > 0:
        print(f"  {r['file']} - {r['skipped']} skipped")

# Save detailed results
with open("test_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDetailed results saved to test_results.json")

# Calculate totals
total_duration = sum(r["duration"] for r in results)
total_passed = sum(r["passed"] for r in results)
total_failed = sum(r["failed"] for r in results)
total_skipped = sum(r["skipped"] for r in results)

print(f"\nTotals:")
print(f"  Total duration: {total_duration:.2f}s")
print(f"  Total passed: {total_passed}")
print(f"  Total failed: {total_failed}")
print(f"  Total skipped: {total_skipped}")