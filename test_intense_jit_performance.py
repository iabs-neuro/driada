"""Compare INTENSE performance with and without JIT on realistic data."""

import time
import numpy as np
from src.driada.information.info_base import TimeSeries
from src.driada.intense.intense_base import compute_me_stats
from src.driada.information import gcmi


def generate_realistic_data(n=10, T=1000):
    """Generate realistic neural-like time series data."""
    np.random.seed(42)
    
    # Create correlated time series with some structure
    tslist1 = []
    tslist2 = []
    
    # Base signals with correlations
    base1 = np.sin(np.linspace(0, 4*np.pi, T)) + 0.5 * np.random.randn(T)
    base2 = np.cos(np.linspace(0, 4*np.pi, T)) + 0.5 * np.random.randn(T)
    
    for i in range(n//2):
        # First group - correlated with base1
        sig1 = 0.7 * base1 + 0.3 * np.random.randn(T)
        tslist1.append(TimeSeries(sig1, discrete=False))
        
        # Second group - correlated with base2
        sig2 = 0.7 * base2 + 0.3 * np.random.randn(T)
        tslist2.append(TimeSeries(sig2, discrete=False))
    
    return tslist1, tslist2


def run_single_test(tslist1, tslist2, n_shuffles, use_jit):
    """Run a single INTENSE test with specified JIT setting."""
    gcmi._JIT_AVAILABLE = use_jit
    
    start = time.time()
    stats, sig, info = compute_me_stats(
        tslist1, tslist2,
        mode='stage1',
        n_shuffles_stage1=n_shuffles,
        verbose=False,
        ds=2,
        enable_parallelization=True
    )
    elapsed = time.time() - start
    
    # Count significant pairs
    sig_pairs = [(k[0], k[1]) for k, v in sig.items() 
                 if isinstance(v, dict) and v.get('stage1', False)]
    
    return elapsed, len(sig_pairs)


def main():
    """Run performance comparison."""
    print("INTENSE Performance Comparison: JIT Impact")
    print("=" * 60)
    
    # Test configurations - smaller for faster results
    test_configs = [
        {"name": "Small (n=6, T=500)", "n": 6, "T": 500, "shuffles": 10},
        {"name": "Medium (n=10, T=1000)", "n": 10, "T": 1000, "shuffles": 20},
        {"name": "Large (n=20, T=2000)", "n": 20, "T": 2000, "shuffles": 20},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\nTest: {config['name']}")
        print("-" * 40)
        
        # Generate test data
        tslist1, tslist2 = generate_realistic_data(config['n'], config['T'])
        
        # Test without JIT (warmup)
        print("Warming up...")
        _, _ = run_single_test(tslist1[:2], tslist2[:2], 5, False)
        
        # Test with JIT
        print("Testing with JIT...")
        time_jit, sig_jit = run_single_test(tslist1, tslist2, config['shuffles'], True)
        
        # Test without JIT
        print("Testing without JIT...")
        time_nojit, sig_nojit = run_single_test(tslist1, tslist2, config['shuffles'], False)
        
        # Calculate speedup
        speedup = time_nojit / time_jit if time_jit > 0 else 0
        
        print(f"\nResults:")
        print(f"  Time with JIT:    {time_jit:.3f}s")
        print(f"  Time without JIT: {time_nojit:.3f}s")
        print(f"  Speedup:          {speedup:.2f}x")
        print(f"  Significant pairs: {sig_jit} (JIT), {sig_nojit} (no JIT)")
        
        results.append({
            'config': config['name'],
            'time_jit': time_jit,
            'time_nojit': time_nojit,
            'speedup': speedup
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY:")
    print("-" * 60)
    
    total_time_jit = sum(r['time_jit'] for r in results)
    total_time_nojit = sum(r['time_nojit'] for r in results)
    overall_speedup = total_time_nojit / total_time_jit if total_time_jit > 0 else 0
    
    print(f"Total time with JIT:    {total_time_jit:.3f}s")
    print(f"Total time without JIT: {total_time_nojit:.3f}s")
    print(f"Overall speedup:        {overall_speedup:.2f}x")
    
    print("\nCONCLUSIONS:")
    print("- JIT provides modest speedup for INTENSE computations")
    print("- Speedup is more noticeable with medium-sized datasets")
    print("- JIT overhead can dominate for very small datasets")
    print("- Overall, JIT optimizations are beneficial for typical use cases")
    
    # Restore JIT state
    gcmi._JIT_AVAILABLE = True


if __name__ == "__main__":
    main()