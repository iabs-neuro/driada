"""
Tests for JIT-compiled RSA functions.
"""

import pytest
import numpy as np
from driada.rsa.core_jit import (
    fast_correlation_distance,
    fast_average_patterns,
    fast_euclidean_distance,
    fast_manhattan_distance,
)
from driada.utils.jit import is_jit_enabled


class TestJITCorrelationDistance:
    """Test JIT-compiled correlation distance function."""
    
    def test_correlation_distance_basic(self):
        """Test basic correlation distance computation."""
        if not is_jit_enabled():
            pytest.skip("JIT compilation is disabled")
            
        # Create orthogonal patterns
        patterns = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        
        rdm = fast_correlation_distance(patterns)
        
        # Check shape and diagonal
        assert rdm.shape == (4, 4)
        assert np.allclose(np.diag(rdm), 0)
        
        # Check symmetry
        assert np.allclose(rdm, rdm.T)
        
        # One-hot patterns have correlation -1/3 with each other
        # So distance = 1 - (-1/3) = 4/3
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert np.isclose(rdm[i, j], 4/3), f"Distance between one-hot patterns {i} and {j} should be 4/3"
    
    def test_correlation_distance_identical(self):
        """Test correlation distance for identical patterns."""
        if not is_jit_enabled():
            pytest.skip("JIT compilation is disabled")
            
        patterns = np.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],  # Identical to first
            [2, 4, 6, 8],  # Perfectly correlated but different scale
            [4, 3, 2, 1],  # Anti-correlated
        ])
        
        rdm = fast_correlation_distance(patterns)
        
        # Identical patterns should have zero distance
        assert np.isclose(rdm[0, 1], 0, atol=1e-10)
        
        # Perfectly correlated patterns should have zero distance
        assert np.isclose(rdm[0, 2], 0, atol=1e-10)
        
        # Anti-correlated patterns should have distance 2
        assert np.isclose(rdm[0, 3], 2, atol=1e-6)
    
    def test_correlation_distance_zero_variance(self):
        """Test handling of patterns with zero variance."""
        if not is_jit_enabled():
            pytest.skip("JIT compilation is disabled")
            
        patterns = np.array([
            [1, 1, 1, 1],  # Zero variance
            [2, 2, 2, 2],  # Zero variance, different value
            [1, 2, 3, 4],  # Normal pattern
            [1, 1, 1, 1],  # Same as first
        ])
        
        rdm = fast_correlation_distance(patterns)
        
        # Zero variance patterns that are identical should have distance 0
        assert np.isclose(rdm[0, 3], 0)
        
        # Zero variance patterns with different values should have distance 1
        assert np.isclose(rdm[0, 1], 1)
        
        # Zero variance vs normal pattern should have distance 1
        assert np.isclose(rdm[0, 2], 1)
    
    def test_correlation_distance_numerical_stability(self):
        """Test numerical stability with extreme values."""
        if not is_jit_enabled():
            pytest.skip("JIT compilation is disabled")
            
        # Create patterns with very large and very small values
        patterns = np.array([
            [1e10, 1e-10, 1e10, 1e-10],
            [1e-10, 1e10, 1e-10, 1e10],
            [1e5, 1e5, 1e-5, 1e-5],
        ])
        
        rdm = fast_correlation_distance(patterns)
        
        # Should not have NaN or inf values
        assert np.all(np.isfinite(rdm))
        
        # Distance should be in valid range [0, 2]
        assert np.all(rdm >= 0)
        assert np.all(rdm <= 2)


class TestJITAveragePatterns:
    """Test JIT-compiled pattern averaging function."""
    
    def test_average_patterns_basic(self):
        """Test basic pattern averaging."""
        if not is_jit_enabled():
            pytest.skip("JIT compilation is disabled")
            
        # Create data with clear patterns
        data = np.array([
            [1, 1, 2, 2, 3, 3],  # Feature 1
            [0, 0, 1, 1, 0, 0],  # Feature 2
            [5, 5, 5, 5, 5, 5],  # Feature 3
        ])
        labels = np.array([0, 0, 1, 1, 2, 2])
        unique_labels = np.array([0, 1, 2])
        
        patterns = fast_average_patterns(data, labels, unique_labels)
        
        # Check shape
        assert patterns.shape == (3, 3)  # 3 conditions, 3 features
        
        # Check averages
        assert np.allclose(patterns[0], [1, 0, 5])  # Condition 0
        assert np.allclose(patterns[1], [2, 1, 5])  # Condition 1
        assert np.allclose(patterns[2], [3, 0, 5])  # Condition 2
    
    def test_average_patterns_unbalanced(self):
        """Test averaging with unbalanced conditions."""
        if not is_jit_enabled():
            pytest.skip("JIT compilation is disabled")
            
        # Create data with unbalanced conditions
        data = np.array([
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
        ])
        labels = np.array([0, 0, 0, 1, 2])  # Condition 0 has 3 samples
        unique_labels = np.array([0, 1, 2])
        
        patterns = fast_average_patterns(data, labels, unique_labels)
        
        # Check averages
        assert np.allclose(patterns[0], [2, 4])  # Mean of first 3 timepoints
        assert np.allclose(patterns[1], [4, 2])  # Single timepoint
        assert np.allclose(patterns[2], [5, 1])  # Single timepoint
    
    def test_average_patterns_missing_label(self):
        """Test with a label that doesn't appear in data."""
        if not is_jit_enabled():
            pytest.skip("JIT compilation is disabled")
            
        data = np.array([
            [1, 2, 3, 4],
            [4, 3, 2, 1],
        ])
        labels = np.array([0, 0, 2, 2])  # No label 1
        unique_labels = np.array([0, 1, 2])  # Include label 1
        
        patterns = fast_average_patterns(data, labels, unique_labels)
        
        # Missing label should have zero pattern
        assert np.allclose(patterns[1], [0, 0])


class TestJITDistanceFunctions:
    """Test other JIT-compiled distance functions."""
    
    def test_euclidean_distance_known_values(self):
        """Test Euclidean distance with known values."""
        if not is_jit_enabled():
            pytest.skip("JIT compilation is disabled")
            
        patterns = np.array([
            [0, 0],
            [3, 0],  # Distance 3 from origin
            [0, 4],  # Distance 4 from origin
            [3, 4],  # Distance 5 from origin
        ])
        
        rdm = fast_euclidean_distance(patterns)
        
        # Check known distances
        assert np.isclose(rdm[0, 1], 3.0)
        assert np.isclose(rdm[0, 2], 4.0)
        assert np.isclose(rdm[0, 3], 5.0)
        assert np.isclose(rdm[1, 2], 5.0)  # 3-4-5 triangle
        
        # Check properties
        assert np.allclose(np.diag(rdm), 0)
        assert np.allclose(rdm, rdm.T)
    
    def test_manhattan_distance_known_values(self):
        """Test Manhattan distance with known values."""
        if not is_jit_enabled():
            pytest.skip("JIT compilation is disabled")
            
        patterns = np.array([
            [0, 0],
            [3, 0],  # Manhattan distance 3
            [0, 4],  # Manhattan distance 4
            [3, 4],  # Manhattan distance 7
        ])
        
        rdm = fast_manhattan_distance(patterns)
        
        # Check known distances
        assert np.isclose(rdm[0, 1], 3.0)
        assert np.isclose(rdm[0, 2], 4.0)
        assert np.isclose(rdm[0, 3], 7.0)
        assert np.isclose(rdm[1, 2], 7.0)  # |3-0| + |0-4| = 7
        
        # Check properties
        assert np.allclose(np.diag(rdm), 0)
        assert np.allclose(rdm, rdm.T)
    
    def test_distance_functions_high_dimensional(self):
        """Test distance functions with high-dimensional data."""
        if not is_jit_enabled():
            pytest.skip("JIT compilation is disabled")
            
        # Create random high-dimensional patterns
        np.random.seed(42)
        patterns = np.random.randn(20, 100)  # 20 items, 100 features
        
        # Test all distance functions
        rdm_euclidean = fast_euclidean_distance(patterns)
        rdm_manhattan = fast_manhattan_distance(patterns)
        rdm_correlation = fast_correlation_distance(patterns)
        
        # Check basic properties for all
        for rdm in [rdm_euclidean, rdm_manhattan, rdm_correlation]:
            assert rdm.shape == (20, 20)
            assert np.allclose(np.diag(rdm), 0)
            assert np.allclose(rdm, rdm.T)
            assert np.all(rdm >= 0)
            assert np.all(np.isfinite(rdm))
    
    def test_distance_functions_edge_cases(self):
        """Test distance functions with edge cases."""
        if not is_jit_enabled():
            pytest.skip("JIT compilation is disabled")
            
        # Single feature
        patterns_1d = np.array([[1], [2], [3]])
        rdm_euclidean = fast_euclidean_distance(patterns_1d)
        assert rdm_euclidean[0, 1] == 1.0
        assert rdm_euclidean[0, 2] == 2.0
        
        # Single pattern
        patterns_single = np.array([[1, 2, 3]])
        rdm_manhattan = fast_manhattan_distance(patterns_single)
        assert rdm_manhattan.shape == (1, 1)
        assert rdm_manhattan[0, 0] == 0.0
        
        # All zeros
        patterns_zeros = np.zeros((5, 10))
        rdm_correlation = fast_correlation_distance(patterns_zeros)
        # All patterns identical, all distances should be 0
        assert np.allclose(rdm_correlation, 0)


class TestJITPerformance:
    """Test performance characteristics of JIT functions."""
    
    def test_jit_warmup(self):
        """Test that JIT functions compile and run correctly."""
        if not is_jit_enabled():
            pytest.skip("JIT compilation is disabled")
            
        # Small data for warmup
        patterns = np.random.randn(5, 10)
        data = np.random.randn(10, 50)
        labels = np.random.randint(0, 3, 50)
        unique_labels = np.array([0, 1, 2])
        
        # Call each function to trigger compilation
        rdm1 = fast_euclidean_distance(patterns)
        rdm2 = fast_manhattan_distance(patterns)
        rdm3 = fast_correlation_distance(patterns)
        avg_patterns = fast_average_patterns(data, labels, unique_labels)
        
        # Basic checks
        assert rdm1.shape == (5, 5)
        assert rdm2.shape == (5, 5)
        assert rdm3.shape == (5, 5)
        assert avg_patterns.shape == (3, 10)
    
    def test_jit_consistency_with_numpy(self):
        """Test that JIT functions produce consistent results with numpy."""
        if not is_jit_enabled():
            pytest.skip("JIT compilation is disabled")
            
        # Create test data
        np.random.seed(42)
        patterns = np.random.randn(10, 20)
        
        # Compute Euclidean distance with JIT
        rdm_jit = fast_euclidean_distance(patterns)
        
        # Compute with numpy
        from scipy.spatial.distance import pdist, squareform
        rdm_numpy = squareform(pdist(patterns, metric='euclidean'))
        
        # Should be very close
        assert np.allclose(rdm_jit, rdm_numpy, rtol=1e-10)
        
        # Test Manhattan distance
        rdm_jit_manhattan = fast_manhattan_distance(patterns)
        rdm_numpy_manhattan = squareform(pdist(patterns, metric='cityblock'))
        assert np.allclose(rdm_jit_manhattan, rdm_numpy_manhattan, rtol=1e-10)