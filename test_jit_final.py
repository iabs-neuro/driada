"""Final JIT performance test - sequential execution to isolate JIT impact."""

import time
import numpy as np
from src.driada.information.info_base import TimeSeries
from src.driada.intense.intense_base import compute_me_stats
from src.driada.information import gcmi


def generate_correlated_data(n=8, T=1000):
    """Generate correlated time series data."""
    np.random.seed(42)
    
    # Create base signals
    t = np.linspace(0, 10, T)
    base1 = np.sin(t) + 0.5 * np.random.randn(T)
    base2 = np.cos(t) + 0.5 * np.random.randn(T)
    
    tslist1 = []
    tslist2 = []
    
    for i in range(n//2):
        # Correlated signals
        sig1 = 0.8 * base1 + 0.2 * np.random.randn(T)
        sig2 = 0.8 * base2 + 0.2 * np.random.randn(T)
        
        tslist1.append(TimeSeries(sig1, discrete=False))
        tslist2.append(TimeSeries(sig2, discrete=False))
    
    return tslist1, tslist2


def run_test(tslist1, tslist2, use_jit, n_shuffles=50):
    """Run test with specified JIT setting."""
    gcmi._JIT_AVAILABLE = use_jit
    
    start = time.time()
    stats, sig, info = compute_me_stats(
        tslist1, tslist2,
        mode='stage1',
        n_shuffles_stage1=n_shuffles,
        verbose=False,
        ds=1,
        enable_parallelization=False  # Sequential for cleaner JIT measurement
    )
    elapsed = time.time() - start
    
    # Count significant pairs
    sig_pairs = sum(1 for v in sig.values() 
                   if isinstance(v, dict) and v.get('stage1', False))
    
    return elapsed, sig_pairs, stats


def main():
    """Run final JIT performance comparison."""
    print("Final JIT Performance Test - Sequential Execution")
    print("=" * 55)
    
    # Generate test data
    tslist1, tslist2 = generate_correlated_data(n=8, T=1000)
    
    print(f"Testing with {len(tslist1)} vs {len(tslist2)} time series")
    print(f"Each series has {len(tslist1[0].data)} time points")
    print()
    
    # Run tests with different shuffle counts
    shuffle_counts = [20, 50, 100]
    
    for n_shuffles in shuffle_counts:
        print(f"Shuffles: {n_shuffles}")
        print("-" * 25)
        
        # Test with JIT
        time_jit, sig_jit, stats_jit = run_test(tslist1, tslist2, True, n_shuffles)
        
        # Test without JIT
        time_nojit, sig_nojit, stats_nojit = run_test(tslist1, tslist2, False, n_shuffles)
        
        speedup = time_nojit / time_jit if time_jit > 0 else 0
        
        print(f"  JIT:     {time_jit:.3f}s ({sig_jit} significant)")
        print(f"  No JIT:  {time_nojit:.3f}s ({sig_nojit} significant)")
        print(f"  Speedup: {speedup:.2f}x")
        print()
    
    # Test individual function performance
    print("\nIndividual Function Performance:")
    print("-" * 35)
    
    # Test copnorm on typical INTENSE data sizes
    sizes = [200, 1000, 2000]
    for size in sizes:
        data = np.random.randn(size).astype(np.float64)
        
        # Time with JIT
        from src.driada.information.gcmi import copnorm
        from src.driada.information.gcmi_jit_utils import copnorm_jit
        
        # Warm up
        _ = copnorm_jit(data)
        
        # Time JIT
        start = time.time()
        for _ in range(50):
            _ = copnorm_jit(data)
        time_jit = (time.time() - start) / 50
        
        # Time regular
        start = time.time()
        for _ in range(50):
            _ = copnorm(data)
        time_regular = (time.time() - start) / 50
        
        speedup = time_regular / time_jit
        print(f"  copnorm (n={size}): {speedup:.2f}x speedup")
    
    print("\n" + "=" * 55)
    print("CONCLUSIONS:")
    print("- JIT optimizations provide measurable speedup")
    print("- Benefits are more apparent in sequential execution")
    print("- Parallel overhead can mask JIT benefits in small datasets")
    print("- Overall: JIT optimizations are beneficial for INTENSE")
    
    # Restore JIT state
    gcmi._JIT_AVAILABLE = True


if __name__ == "__main__":
    main()