"""
Benchmark script to verify MTS-MTS FFT speedup claims.

Tests various dimension combinations and measures speedup vs loop fallback.
"""
import numpy as np
import time
from driada.information.info_base import compute_mi_mts_mts_fft
from driada.information.gcmi import mi_gg


def benchmark_case(d1, d2, n, nsh, description):
    """Benchmark a specific dimension combination."""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {description}")
    print(f"  d1={d1}, d2={d2}, n={n}, nsh={nsh}")
    print(f"{'='*70}")

    # Generate random data
    np.random.seed(42)
    x1 = np.random.randn(d1, n)
    x2 = np.random.randn(d2, n)
    shifts = np.arange(nsh)

    # Benchmark FFT approach
    print("\nFFT approach...")
    start_fft = time.time()
    mi_fft = compute_mi_mts_mts_fft(x1, x2, shifts, biascorrect=True)
    time_fft = time.time() - start_fft
    print(f"  Time: {time_fft:.3f}s")

    # Benchmark loop fallback (sample fewer shifts if nsh is large)
    loop_sample_size = min(nsh, 50)  # Sample to avoid excessive runtime
    shifts_sample = shifts[:loop_sample_size]

    print(f"\nLoop fallback (sampling {loop_sample_size}/{nsh} shifts)...")
    start_loop = time.time()
    mi_loop_sample = np.zeros(loop_sample_size)
    for i, s in enumerate(shifts_sample):
        x2_shifted = np.roll(x2, int(s), axis=1)
        mi_loop_sample[i] = mi_gg(x1, x2_shifted, biascorrect=True, demeaned=False)
    time_loop_sample = time.time() - start_loop

    # Extrapolate full loop time
    time_loop_extrapolated = time_loop_sample * (nsh / loop_sample_size)
    print(f"  Time (sampled): {time_loop_sample:.3f}s")
    print(f"  Time (extrapolated for {nsh} shifts): {time_loop_extrapolated:.3f}s")

    # Calculate speedup
    speedup = time_loop_extrapolated / time_fft
    print(f"\n>>> SPEEDUP: {speedup:.1f}x")

    # Verify correctness at shift=0
    mi_ref = mi_gg(x1, x2, biascorrect=True, demeaned=False)
    error = abs(mi_fft[0] - mi_ref)
    print(f"    Correctness check (shift=0): error={error:.2e}")

    return {
        'description': description,
        'd1': d1,
        'd2': d2,
        'n': n,
        'nsh': nsh,
        'time_fft': time_fft,
        'time_loop': time_loop_extrapolated,
        'speedup': speedup,
        'error': error
    }


def main():
    """Run benchmark suite."""
    print("\n" + "="*70)
    print("MTS-MTS FFT ACCELERATION BENCHMARK")
    print("="*70)

    results = []

    # Test Case 1: d1=2, d2=2, typical neuroscience use case
    results.append(benchmark_case(
        d1=2, d2=2, n=10000, nsh=1000,
        description="Case 1: 2D x 2D, typical neuroscience (claimed ~300x)"
    ))

    # Test Case 2: d1=2, d2=3, asymmetric dimensions
    results.append(benchmark_case(
        d1=2, d2=3, n=5000, nsh=500,
        description="Case 2: 2D x 3D, asymmetric (claimed ~160x)"
    ))

    # Test Case 3: d1=3, d2=3, maximum dimensions
    results.append(benchmark_case(
        d1=3, d2=3, n=10000, nsh=1000,
        description="Case 3: 3D x 3D, max dimensions (claimed ~100x)"
    ))

    # Test Case 4: Small n, many shifts (stress test)
    results.append(benchmark_case(
        d1=2, d2=2, n=1000, nsh=2000,
        description="Case 4: Small n, many shifts (stress test)"
    ))

    # Test Case 5: Large n, fewer shifts
    results.append(benchmark_case(
        d1=2, d2=2, n=20000, nsh=200,
        description="Case 5: Large n, fewer shifts"
    ))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Description':<45} {'Speedup':>10} {'Correctness':>12}")
    print("-" * 70)
    for r in results:
        correctness = "PASS" if r['error'] < 1e-7 else "FAIL"
        print(f"{r['description']:<45} {r['speedup']:>9.1f}x {correctness:>11}")

    # Statistics
    speedups = [r['speedup'] for r in results]
    print("\n" + "="*70)
    print(f"Average speedup: {np.mean(speedups):.1f}x")
    print(f"Minimum speedup: {np.min(speedups):.1f}x")
    print(f"Maximum speedup: {np.max(speedups):.1f}x")
    print("="*70)

    # Verify all tests passed correctness check
    all_correct = all(r['error'] < 1e-7 for r in results)
    if all_correct:
        print("\n>>> All correctness checks PASSED")
    else:
        print("\n>>> Some correctness checks FAILED")

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
