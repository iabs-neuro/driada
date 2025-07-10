"""Compare INTENSE performance with and without JIT optimizations."""

import time
import numpy as np
from src.driada.information.info_base import TimeSeries
from src.driada.intense.intense_base import compute_me_stats
from src.driada.information import gcmi

# Test data generation
def generate_test_data(n=20, T=2000):
    """Generate test time series data."""
    np.random.seed(42)
    
    # Create correlated time series
    C = np.eye(n)
    C[1, n-1] = 0.9
    C[2, n-2] = 0.8
    C[5, n-5] = 0.7
    C = (C + C.T)
    np.fill_diagonal(C, 1)
    
    signals = np.random.multivariate_normal(np.zeros(n), C, size=T).T
    signals += np.random.randn(n, T) * 0.2
    
    tslist1 = [TimeSeries(sig, discrete=False) for sig in signals[:n//2]]
    tslist2 = [TimeSeries(sig, discrete=False) for sig in signals[n//2:]]
    
    return tslist1, tslist2


def run_performance_test():
    """Run performance comparison."""
    print("INTENSE Performance Comparison: JIT vs No-JIT")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        {"name": "Small (n=10, T=1000)", "n": 10, "T": 1000, "shuffles": 20},
        {"name": "Medium (n=20, T=2000)", "n": 20, "T": 2000, "shuffles": 20},
        {"name": "Large (n=40, T=5000)", "n": 40, "T": 5000, "shuffles": 20},
    ]
    
    for config in test_configs:
        print(f"\nTest: {config['name']}")
        print("-" * 30)
        
        # Generate test data
        tslist1, tslist2 = generate_test_data(config['n'], config['T'])
        
        # Test with JIT enabled
        gcmi._JIT_AVAILABLE = True
        start = time.time()
        stats_jit, sig_jit, info_jit = compute_me_stats(
            tslist1, tslist2,
            mode='stage1',
            n_shuffles_stage1=config['shuffles'],
            verbose=False,
            ds=2
        )
        time_jit = time.time() - start
        
        # Test with JIT disabled
        gcmi._JIT_AVAILABLE = False
        start = time.time()
        stats_nojit, sig_nojit, info_nojit = compute_me_stats(
            tslist1, tslist2,
            mode='stage1',
            n_shuffles_stage1=config['shuffles'],
            verbose=False,
            ds=2
        )
        time_nojit = time.time() - start
        
        # Restore JIT state
        gcmi._JIT_AVAILABLE = True
        
        # Compare results
        print(f"Time with JIT:    {time_jit:.3f}s")
        print(f"Time without JIT: {time_nojit:.3f}s")
        print(f"Speedup:          {time_nojit/time_jit:.2f}x")
        
        # Verify results are consistent
        sig_pairs_jit = [(k[0], k[1]) for k, v in sig_jit.items() 
                         if isinstance(v, dict) and v.get('stage1', False)]
        sig_pairs_nojit = [(k[0], k[1]) for k, v in sig_nojit.items() 
                           if isinstance(v, dict) and v.get('stage1', False)]
        
        if set(sig_pairs_jit) == set(sig_pairs_nojit):
            print(f"Results match: {len(sig_pairs_jit)} significant pairs found")
        else:
            print("WARNING: Results differ between JIT and non-JIT!")
    
    # Test individual function performance
    print("\n\nIndividual Function Performance:")
    print("=" * 50)
    
    # Test copnorm performance
    from src.driada.information.gcmi import copnorm
    
    for size in [100, 1000, 10000]:
        data = np.random.randn(2, size)
        
        # With JIT
        gcmi._JIT_AVAILABLE = True
        start = time.time()
        for _ in range(100):
            _ = copnorm(data)
        time_jit = time.time() - start
        
        # Without JIT
        gcmi._JIT_AVAILABLE = False
        start = time.time()
        for _ in range(100):
            _ = copnorm(data)
        time_nojit = time.time() - start
        
        gcmi._JIT_AVAILABLE = True
        
        print(f"\ncopnorm (size={size}):")
        print(f"  JIT: {time_jit:.4f}s, No-JIT: {time_nojit:.4f}s, Speedup: {time_nojit/time_jit:.2f}x")


if __name__ == "__main__":
    run_performance_test()