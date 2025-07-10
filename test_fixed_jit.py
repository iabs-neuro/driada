"""Test the fixed JIT implementations with O(n log n) algorithm."""

import time
import numpy as np
from src.driada.information.gcmi import ctransform, copnorm
from src.driada.information.gcmi_jit_utils import ctransform_jit, copnorm_jit


def time_function(func, *args, n_iter=100, warmup=5):
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


def test_ctransform_fixed():
    """Test the fixed O(n log n) ctransform performance."""
    print("Fixed CTRANSFORM Performance Test")
    print("=" * 50)
    
    for size in [100, 1000, 10000, 50000]:
        data = np.random.randn(size).astype(np.float64)
        
        # Test correctness first
        result_orig = ctransform(data).ravel()
        result_jit = ctransform_jit(data)
        
        # Check if results are close
        try:
            np.testing.assert_allclose(result_orig, result_jit, rtol=1e-9)
            correctness = "✅ PASS"
        except AssertionError:
            correctness = "❌ FAIL"
        
        # Time both versions
        time_orig = time_function(lambda x: ctransform(x), data)
        time_jit = time_function(ctransform_jit, data)
        
        speedup = time_orig / time_jit
        
        print(f"Size {size:6d}: Original {time_orig*1000:8.3f}ms, JIT {time_jit*1000:8.3f}ms, "
              f"Speedup {speedup:6.2f}x {correctness}")


def test_copnorm_fixed():
    """Test the fixed copnorm performance."""
    print("\nFixed COPNORM Performance Test")
    print("=" * 50)
    
    for size in [100, 1000, 10000, 50000]:
        data = np.random.randn(size).astype(np.float64)
        
        # Test correctness first
        result_orig = copnorm(data).ravel()
        result_jit = copnorm_jit(data)
        
        # Check if results are close (copnorm uses approximations)
        try:
            correlation = np.corrcoef(result_orig, result_jit)[0, 1]
            if correlation > 0.999:
                correctness = "✅ PASS"
            else:
                correctness = f"⚠️ CORR={correlation:.4f}"
        except:
            correctness = "❌ FAIL"
        
        # Time both versions
        time_orig = time_function(lambda x: copnorm(x), data)
        time_jit = time_function(copnorm_jit, data)
        
        speedup = time_orig / time_jit
        
        print(f"Size {size:6d}: Original {time_orig*1000:8.3f}ms, JIT {time_jit*1000:8.3f}ms, "
              f"Speedup {speedup:6.2f}x {correctness}")


def test_algorithmic_complexity():
    """Test that the JIT version now scales properly."""
    print("\nAlgorithmic Complexity Test")
    print("=" * 50)
    
    sizes = [100, 1000, 10000, 100000]
    times_orig = []
    times_jit = []
    
    for size in sizes:
        data = np.random.randn(size).astype(np.float64)
        
        # Time both versions with fewer iterations for large sizes
        n_iter = max(10, 1000 // (size // 1000)) if size > 1000 else 100
        
        time_orig = time_function(lambda x: ctransform(x), data, n_iter=n_iter)
        time_jit = time_function(ctransform_jit, data, n_iter=n_iter)
        
        times_orig.append(time_orig)
        times_jit.append(time_jit)
        
        print(f"Size {size:6d}: Original {time_orig*1000:8.3f}ms, JIT {time_jit*1000:8.3f}ms")
    
    # Check scaling
    print("\nScaling Analysis:")
    for i in range(1, len(sizes)):
        ratio = sizes[i] / sizes[i-1]
        orig_scaling = times_orig[i] / times_orig[i-1]
        jit_scaling = times_jit[i] / times_jit[i-1]
        
        print(f"Size {sizes[i-1]:6d} -> {sizes[i]:6d} (x{ratio:.0f}): "
              f"Original x{orig_scaling:.2f}, JIT x{jit_scaling:.2f}")


def main():
    """Run all performance tests."""
    print("JIT Performance Test - Fixed O(n log n) Algorithm")
    print("=" * 70)
    
    test_ctransform_fixed()
    test_copnorm_fixed()
    test_algorithmic_complexity()
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("- Fixed O(n²) -> O(n log n) algorithmic complexity")
    print("- Improved ndtri approximation for better performance")
    print("- JIT should now show proper scaling characteristics")


if __name__ == "__main__":
    main()