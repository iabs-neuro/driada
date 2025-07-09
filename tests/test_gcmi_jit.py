"""Test JIT performance improvements and correctness for GCMI functions."""

import pytest
import numpy as np
import time
from src.driada.information.gcmi import ctransform, copnorm, _JIT_AVAILABLE
from src.driada.information.gcmi_jit_utils import ctransform_jit, copnorm_jit, ctransform_2d_jit, copnorm_2d_jit


@pytest.mark.skipif(not _JIT_AVAILABLE, reason="JIT utilities not available")
class TestGCMIJIT:
    """Test JIT-compiled GCMI functions."""
    
    def test_ctransform_correctness_1d(self):
        """Test that JIT ctransform produces same results as original."""
        np.random.seed(42)
        
        for size in [10, 100, 1000]:
            data = np.random.randn(size)
            
            # Compare results
            result_orig = ctransform(data).ravel()
            result_jit = ctransform_jit(data)
            
            np.testing.assert_allclose(result_orig, result_jit, rtol=1e-9)
    
    def test_ctransform_correctness_2d(self):
        """Test that JIT ctransform works correctly for 2D arrays."""
        np.random.seed(42)
        
        for shape in [(2, 100), (5, 50), (10, 200)]:
            data = np.random.randn(*shape)
            
            # Compare results
            result_orig = ctransform(data)
            result_jit = ctransform_2d_jit(data)
            
            np.testing.assert_allclose(result_orig, result_jit, rtol=1e-9)
    
    def test_copnorm_correctness_1d(self):
        """Test that JIT copnorm produces similar results as original."""
        np.random.seed(42)
        
        for size in [10, 100, 1000]:
            data = np.random.randn(size)
            
            # Compare results - copnorm uses approximations so allow more tolerance
            result_orig = copnorm(data).ravel()
            result_jit = copnorm_jit(data)
            
            # Check correlation is very high (approximation is good)
            correlation = np.corrcoef(result_orig, result_jit)[0, 1]
            assert correlation > 0.999, f"Correlation {correlation} too low"
            
            # Check values are close
            np.testing.assert_allclose(result_orig, result_jit, rtol=1e-3, atol=1e-3)
    
    def test_copnorm_correctness_2d(self):
        """Test that JIT copnorm works correctly for 2D arrays."""
        np.random.seed(42)
        
        for shape in [(2, 100), (5, 50)]:
            data = np.random.randn(*shape)
            
            # Compare results
            result_orig = copnorm(data)
            result_jit = copnorm_2d_jit(data)
            
            # Check correlation is very high for each row
            for i in range(shape[0]):
                correlation = np.corrcoef(result_orig[i], result_jit[i])[0, 1]
                assert correlation > 0.999, f"Row {i} correlation {correlation} too low"
    
    def test_edge_cases(self):
        """Test edge cases for JIT functions."""
        # Test with constant array
        const_data = np.ones(10)
        result_ct = ctransform_jit(const_data)
        # For constant array, the copula transform breaks ties by index
        # So we get evenly spaced values between 0 and 1
        assert np.all(result_ct > 0) and np.all(result_ct < 1)
        
        # Test with sorted array
        sorted_data = np.arange(10).astype(float)
        result_ct = ctransform_jit(sorted_data)
        expected = (np.arange(10) + 1) / 11.0
        np.testing.assert_allclose(result_ct, expected)
    
    def test_performance_improvement(self):
        """Benchmark JIT vs regular implementations."""
        np.random.seed(42)
        sizes = [100, 1000]
        
        for size in sizes:
            data = np.random.randn(size)
            
            # Warm up JIT
            _ = ctransform_jit(data)
            _ = copnorm_jit(data)
            
            # Time ctransform
            n_iter = 100
            start = time.time()
            for _ in range(n_iter):
                _ = ctransform(data)
            time_regular = time.time() - start
            
            start = time.time()
            for _ in range(n_iter):
                _ = ctransform_jit(data)
            time_jit = time.time() - start
            
            # JIT version should not be significantly slower
            # (relaxed constraint since performance varies)
            assert time_jit < time_regular * 2.0, f"JIT too slow for size {size}"
    
    def test_integration_with_gcmi(self):
        """Test that gcmi functions automatically use JIT versions."""
        np.random.seed(42)
        
        # Test 1D input
        data_1d = np.random.randn(100)
        result = ctransform(data_1d)
        assert result.shape == (1, 100)  # Should be 2D
        
        # Test 2D input
        data_2d = np.random.randn(3, 100)
        result = ctransform(data_2d)
        assert result.shape == (3, 100)
        
        # Test copnorm
        result = copnorm(data_1d)
        assert result.shape == (1, 100)
        
        result = copnorm(data_2d)
        assert result.shape == (3, 100)


def test_jit_availability():
    """Test that JIT imports are working."""
    from src.driada.information import gcmi
    assert hasattr(gcmi, '_JIT_AVAILABLE')
    assert gcmi._JIT_AVAILABLE == True  # Should be available in test environment


@pytest.mark.skipif(not _JIT_AVAILABLE, reason="JIT utilities not available")
def test_mi_gg_jit_correctness():
    """Test that JIT mi_gg produces same results as original."""
    from src.driada.information.gcmi import mi_gg
    from src.driada.information.gcmi_jit_utils import mi_gg_jit
    
    np.random.seed(42)
    
    for shape in [(1, 100), (2, 100), (3, 50)]:
        x = np.random.randn(*shape)
        y = np.random.randn(*shape)
        
        # Test with and without bias correction
        for biascorrect in [True, False]:
            result_orig = mi_gg(x, y, biascorrect=biascorrect)
            result_jit = mi_gg_jit(x, y, biascorrect=biascorrect)
            
            np.testing.assert_allclose(result_orig, result_jit, rtol=1e-9)


@pytest.mark.skipif(not _JIT_AVAILABLE, reason="JIT utilities not available")
def test_cmi_ggg_jit_correctness():
    """Test that JIT cmi_ggg produces same results as original."""
    from src.driada.information.gcmi import cmi_ggg
    from src.driada.information.gcmi_jit_utils import cmi_ggg_jit
    
    np.random.seed(42)
    
    for shape in [(1, 100), (2, 50)]:
        x = np.random.randn(*shape)
        y = np.random.randn(*shape)
        z = np.random.randn(*shape)
        
        # Test CMI computation
        result_orig = cmi_ggg(x, y, z, biascorrect=True)
        result_jit = cmi_ggg_jit(x, y, z, biascorrect=True)
        
        # CMI can be negative, so check absolute difference
        np.testing.assert_allclose(result_orig, result_jit, rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(not _JIT_AVAILABLE, reason="JIT utilities not available")
def test_gcmi_cc_jit_correctness():
    """Test that JIT gcmi_cc produces correct results."""
    from src.driada.information.gcmi import gcmi_cc
    from src.driada.information.gcmi_jit_utils import gcmi_cc_jit
    
    np.random.seed(42)
    
    for shape in [(1, 100), (2, 100)]:
        x = np.random.randn(*shape)
        y = np.random.randn(*shape)
        
        # Add correlation
        if shape[0] == 2:
            y[0] = 0.7 * x[0] + 0.3 * np.random.randn(shape[1])
        else:
            y = 0.7 * x + 0.3 * np.random.randn(*shape)
        
        result_orig = gcmi_cc(x, y)
        result_jit = gcmi_cc_jit(x, y)
        
        # GCMI uses approximations, so allow some tolerance
        assert abs(result_orig - result_jit) < 0.01, f"Difference {abs(result_orig - result_jit)} too large"


@pytest.mark.skipif(not _JIT_AVAILABLE, reason="JIT utilities not available")
def test_interaction_information_jit_consistency():
    """Test that interaction information produces consistent results with JIT functions."""
    from src.driada.information.info_base import TimeSeries, interaction_information, get_mi, conditional_mi
    from src.driada.information.gcmi import _JIT_AVAILABLE
    
    # Temporarily disable JIT to get reference value
    import src.driada.information.gcmi as gcmi_module
    original_jit = gcmi_module._JIT_AVAILABLE
    
    np.random.seed(42)
    
    # Create test data
    n = 200
    x = np.random.randn(n)
    y = 0.8 * x + 0.2 * np.random.randn(n)
    z = 0.5 * x + 0.5 * y + 0.3 * np.random.randn(n)
    
    ts_x = TimeSeries(x, discrete=False)
    ts_y = TimeSeries(y, discrete=False)
    ts_z = TimeSeries(z, discrete=False)
    
    # Get reference without JIT
    gcmi_module._JIT_AVAILABLE = False
    ii_ref = interaction_information(ts_x, ts_y, ts_z)
    mi_xy_ref = get_mi(ts_x, ts_y)
    cmi_ref = conditional_mi(ts_x, ts_y, ts_z)
    
    # Get results with JIT
    gcmi_module._JIT_AVAILABLE = True
    ii_jit = interaction_information(ts_x, ts_y, ts_z)
    mi_xy_jit = get_mi(ts_x, ts_y)
    cmi_jit = conditional_mi(ts_x, ts_y, ts_z)
    
    # Restore original state
    gcmi_module._JIT_AVAILABLE = original_jit
    
    # Check consistency
    np.testing.assert_allclose(ii_ref, ii_jit, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(mi_xy_ref, mi_xy_jit, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(cmi_ref, cmi_jit, rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(not _JIT_AVAILABLE, reason="JIT utilities not available")
def test_conditional_mi_integration():
    """Test that conditional MI works with JIT functions."""
    from src.driada.information.info_base import TimeSeries, conditional_mi
    
    np.random.seed(42)
    
    # Create chain: X -> Y -> Z
    n = 200
    x = np.random.randn(n)
    y = 0.8 * x + 0.2 * np.random.randn(n)
    z = 0.8 * y + 0.2 * np.random.randn(n)
    
    ts_x = TimeSeries(x, discrete=False)
    ts_y = TimeSeries(y, discrete=False)
    ts_z = TimeSeries(z, discrete=False)
    
    # I(X;Z|Y) should be close to 0 for a chain
    cmi = conditional_mi(ts_x, ts_z, ts_y)
    assert cmi < 0.1, f"CMI too high for chain structure: {cmi}"