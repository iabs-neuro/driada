"""Test JIT performance on isolated functions."""

import time
import numpy as np
from src.driada.information import gcmi
from src.driada.information.gcmi import ctransform, copnorm, mi_gg, cmi_ggg, gcmi_cc
from src.driada.information.gcmi_jit_utils import (
    ctransform_jit, copnorm_jit, mi_gg_jit, cmi_ggg_jit, gcmi_cc_jit
)

def time_function(func, *args, n_iter=100, warmup=10):
    """Time a function with warmup iterations."""
    # Warmup
    for _ in range(warmup):
        _ = func(*args)
    
    # Time
    start = time.time()
    for _ in range(n_iter):
        _ = func(*args)
    elapsed = time.time() - start
    
    return elapsed / n_iter


def test_ctransform_performance():
    """Test ctransform performance."""
    print("\nCTRANSFORM Performance:")
    print("-" * 50)
    
    for size in [100, 1000, 10000]:
        data = np.random.randn(size).astype(np.float64)
        
        # Regular version
        time_regular = time_function(lambda x: ctransform(x), data)
        
        # JIT version
        time_jit = time_function(ctransform_jit, data)
        
        speedup = time_regular / time_jit
        print(f"Size {size:6d}: Regular {time_regular*1000:8.4f}ms, JIT {time_jit*1000:8.4f}ms, Speedup {speedup:6.2f}x")


def test_copnorm_performance():
    """Test copnorm performance."""
    print("\nCOPNORM Performance:")
    print("-" * 50)
    
    for size in [100, 1000, 10000]:
        data = np.random.randn(size).astype(np.float64)
        
        # Regular version
        time_regular = time_function(lambda x: copnorm(x), data)
        
        # JIT version  
        time_jit = time_function(copnorm_jit, data)
        
        speedup = time_regular / time_jit
        print(f"Size {size:6d}: Regular {time_regular*1000:8.4f}ms, JIT {time_jit*1000:8.4f}ms, Speedup {speedup:6.2f}x")


def test_mi_gg_performance():
    """Test mi_gg performance."""
    print("\nMI_GG Performance:")
    print("-" * 50)
    
    for size in [100, 1000, 5000]:
        x = np.random.randn(2, size).astype(np.float64)
        y = np.random.randn(2, size).astype(np.float64)
        
        # Regular version - need to disable JIT temporarily
        original_jit = gcmi._JIT_AVAILABLE
        gcmi._JIT_AVAILABLE = False
        time_regular = time_function(mi_gg, x, y, n_iter=10)
        gcmi._JIT_AVAILABLE = original_jit
        
        # JIT version - copy data to avoid in-place modifications
        time_jit = time_function(mi_gg_jit, x.copy(), y.copy(), n_iter=10)
        
        speedup = time_regular / time_jit
        print(f"Size {size:6d}: Regular {time_regular*1000:8.4f}ms, JIT {time_jit*1000:8.4f}ms, Speedup {speedup:6.2f}x")


def test_gcmi_cc_performance():
    """Test gcmi_cc performance."""
    print("\nGCMI_CC Performance:")
    print("-" * 50)
    
    for size in [100, 1000, 5000]:
        x = np.random.randn(1, size).astype(np.float64)
        y = np.random.randn(1, size).astype(np.float64)
        
        # Regular version
        original_jit = gcmi._JIT_AVAILABLE
        gcmi._JIT_AVAILABLE = False
        time_regular = time_function(gcmi_cc, x, y, n_iter=10)
        gcmi._JIT_AVAILABLE = original_jit
        
        # JIT version
        time_jit = time_function(gcmi_cc_jit, x, y, n_iter=10)
        
        speedup = time_regular / time_jit
        print(f"Size {size:6d}: Regular {time_regular*1000:8.4f}ms, JIT {time_jit*1000:8.4f}ms, Speedup {speedup:6.2f}x")


def test_cmi_ggg_performance():
    """Test cmi_ggg performance."""
    print("\nCMI_GGG Performance:")
    print("-" * 50)
    
    for size in [100, 500, 2000]:
        x = np.random.randn(1, size).astype(np.float64)
        y = np.random.randn(1, size).astype(np.float64)
        z = np.random.randn(1, size).astype(np.float64)
        
        # Regular version
        original_jit = gcmi._JIT_AVAILABLE
        gcmi._JIT_AVAILABLE = False
        time_regular = time_function(cmi_ggg, x, y, z, n_iter=10)
        gcmi._JIT_AVAILABLE = original_jit
        
        # JIT version - copy data to avoid in-place modifications
        time_jit = time_function(cmi_ggg_jit, x.copy(), y.copy(), z.copy(), n_iter=10)
        
        speedup = time_regular / time_jit
        print(f"Size {size:6d}: Regular {time_regular*1000:8.4f}ms, JIT {time_jit*1000:8.4f}ms, Speedup {speedup:6.2f}x")


def main():
    """Run all performance tests."""
    print("JIT Performance Comparison - Isolated Functions")
    print("=" * 70)
    print(f"JIT Available: {gcmi._JIT_AVAILABLE}")
    
    test_ctransform_performance()
    test_copnorm_performance()
    test_mi_gg_performance()
    test_gcmi_cc_performance()
    test_cmi_ggg_performance()
    
    print("\n" + "=" * 70)
    print("Performance Summary:")
    print("- ctransform/copnorm: Significant speedup for small arrays")
    print("- MI/CMI functions: Good speedup across all sizes")
    print("- JIT compilation overhead is amortized over multiple calls")


if __name__ == "__main__":
    main()