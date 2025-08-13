"""Performance tests for RSA improvements."""

import time
import numpy as np
from scipy.spatial.distance import pdist, squareform

from driada.rsa.core import compute_rdm
from driada.rsa.core_jit import fast_correlation_distance, fast_euclidean_distance
from driada.dim_reduction.data import MVData
from driada.utils.jit import is_jit_enabled, jit_info


def time_function(func, *args, n_runs=5):
    """Time a function execution."""
    times = []
    for _ in range(n_runs):
        start = time.time()
        result = func(*args)
        times.append(time.time() - start)
    return np.mean(times), np.std(times), result


def test_correlation_performance():
    """Compare correlation distance computation methods."""
    print("\n=== Correlation Distance Performance ===")

    # Test different sizes
    sizes = [(10, 50), (50, 100), (100, 500), (200, 1000)]

    for n_items, n_features in sizes:
        print(f"\nPattern matrix size: {n_items} items × {n_features} features")
        patterns = np.random.randn(n_items, n_features)

        # Method 1: Original numpy corrcoef
        def original_correlation():
            corr_matrix = np.corrcoef(patterns)
            rdm = 1 - corr_matrix
            np.fill_diagonal(rdm, 0)
            return rdm

        # Method 2: scipy pdist
        def scipy_correlation():
            distances = pdist(patterns, metric="correlation")
            return squareform(distances)

        # Method 3: JIT-compiled (if enabled)
        if is_jit_enabled():
            # Warm up JIT
            _ = fast_correlation_distance(patterns[:5])

            mean_orig, std_orig, rdm_orig = time_function(original_correlation)
            mean_scipy, std_scipy, rdm_scipy = time_function(scipy_correlation)
            mean_jit, std_jit, rdm_jit = time_function(
                fast_correlation_distance, patterns
            )

            print(f"  NumPy corrcoef: {mean_orig:.4f} ± {std_orig:.4f} s")
            print(f"  SciPy pdist:    {mean_scipy:.4f} ± {std_scipy:.4f} s")
            print(f"  JIT compiled:   {mean_jit:.4f} ± {std_jit:.4f} s")
            print(
                f"  JIT speedup:    {mean_orig/mean_jit:.2f}x vs NumPy, {mean_scipy/mean_jit:.2f}x vs SciPy"
            )

            # Verify correctness
            assert np.allclose(
                rdm_orig, rdm_jit, rtol=1e-5
            ), "JIT result differs from NumPy"
        else:
            mean_orig, std_orig, _ = time_function(original_correlation)
            mean_scipy, std_scipy, _ = time_function(scipy_correlation)

            print(f"  NumPy corrcoef: {mean_orig:.4f} ± {std_orig:.4f} s")
            print(f"  SciPy pdist:    {mean_scipy:.4f} ± {std_scipy:.4f} s")
            print("  JIT disabled - skipping comparison")


def test_euclidean_performance():
    """Compare Euclidean distance computation methods."""
    print("\n=== Euclidean Distance Performance ===")

    sizes = [(10, 50), (50, 100), (100, 500), (200, 1000)]

    for n_items, n_features in sizes:
        print(f"\nPattern matrix size: {n_items} items × {n_features} features")
        patterns = np.random.randn(n_items, n_features)

        # Method 1: scipy pdist
        def scipy_euclidean():
            distances = pdist(patterns, metric="euclidean")
            return squareform(distances)

        # Method 2: JIT-compiled (if enabled)
        if is_jit_enabled():
            # Warm up JIT
            _ = fast_euclidean_distance(patterns[:5])

            mean_scipy, std_scipy, rdm_scipy = time_function(scipy_euclidean)
            mean_jit, std_jit, rdm_jit = time_function(
                fast_euclidean_distance, patterns
            )

            print(f"  SciPy pdist:  {mean_scipy:.4f} ± {std_scipy:.4f} s")
            print(f"  JIT compiled: {mean_jit:.4f} ± {std_jit:.4f} s")
            print(f"  JIT speedup:  {mean_scipy/mean_jit:.2f}x")

            # Verify correctness
            assert np.allclose(
                rdm_scipy, rdm_jit, rtol=1e-5
            ), "JIT result differs from SciPy"
        else:
            mean_scipy, std_scipy, _ = time_function(scipy_euclidean)
            print(f"  SciPy pdist:  {mean_scipy:.4f} ± {std_scipy:.4f} s")
            print("  JIT disabled - skipping comparison")


def test_mvdata_integration_performance():
    """Test performance of MVData integration."""
    print("\n=== MVData Integration Performance ===")

    n_items, n_features = 100, 500
    patterns = np.random.randn(n_items, n_features)

    # Method 1: Direct numpy array
    def direct_numpy():
        return compute_rdm(patterns, metric="correlation")

    # Method 2: Via MVData
    def via_mvdata():
        mvdata = MVData(patterns.T)
        return compute_rdm(mvdata, metric="correlation")

    mean_numpy, std_numpy, rdm_numpy = time_function(direct_numpy)
    mean_mvdata, std_mvdata, rdm_mvdata = time_function(via_mvdata)

    print(f"\nDirect NumPy: {mean_numpy:.4f} ± {std_numpy:.4f} s")
    print(f"Via MVData:   {mean_mvdata:.4f} ± {std_mvdata:.4f} s")
    print(f"Overhead:     {(mean_mvdata - mean_numpy)/mean_numpy * 100:.1f}%")

    # Verify correctness
    assert np.allclose(rdm_numpy, rdm_mvdata, rtol=1e-5), "MVData result differs"


def test_caching_performance():
    """Test performance impact of caching."""
    print("\n=== Caching Performance ===")

    from driada import generate_synthetic_exp

    # Create a synthetic experiment
    exp = generate_synthetic_exp(
        n_dfeats=3,
        n_cfeats=0,
        nneurons=20,  # Reduced from 50 to avoid shuffle mask issues
        duration=30,  # Increased from 10 to provide more data points
        seed=42,
    )

    # First computation (no cache)
    start = time.time()
    rdm1, labels1 = exp.compute_rdm("d_feat_0", use_cache=True)
    time_first = time.time() - start

    # Second computation (should use cache)
    start = time.time()
    rdm2, labels2 = exp.compute_rdm("d_feat_0", use_cache=True)
    time_cached = time.time() - start

    # Third computation (bypass cache)
    start = time.time()
    rdm3, labels3 = exp.compute_rdm("d_feat_0", use_cache=False)
    time_no_cache = time.time() - start

    print(f"\nFirst computation:    {time_first:.4f} s")
    print(f"Cached computation:   {time_cached:.4f} s")
    print(f"No-cache computation: {time_no_cache:.4f} s")
    print(f"Cache speedup:        {time_first/time_cached:.1f}x")

    # Verify all results are the same
    assert np.allclose(rdm1, rdm2), "Cached result differs"
    assert np.allclose(rdm1, rdm3), "No-cache result differs"


if __name__ == "__main__":
    print("RSA Performance Test Suite")
    print("==========================")

    # Print JIT status
    jit_info()

    # Run tests
    test_correlation_performance()
    test_euclidean_performance()
    test_mvdata_integration_performance()
    test_caching_performance()

    print("\n" + "=" * 50)
    print("Performance testing complete!")

    if not is_jit_enabled():
        print("\nNote: JIT compilation is disabled. To enable:")
        print("  - Ensure numba is installed: pip install numba")
        print("  - Set environment: export DRIADA_DISABLE_NUMBA=false")
