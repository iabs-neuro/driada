"""Test automatic alpha selection for LNC correction in KSG estimator."""

import numpy as np
import pytest

from driada.information.ksg import get_lnc_alpha, nonparam_mi_cc


class TestLNCAlphaSelection:
    """Test automatic alpha selection for Local Non-uniformity Correction."""
    
    def test_exact_alpha_lookup(self):
        """Test exact lookups from the alpha table."""
        # Test some exact values from the table
        assert abs(get_lnc_alpha(2, 3) - 0.182224) < 1e-6
        assert abs(get_lnc_alpha(3, 10) - 0.489972) < 1e-6
        assert abs(get_lnc_alpha(5, 15) - 0.498458) < 1e-6
        assert abs(get_lnc_alpha(10, 20) - 0.398082) < 1e-6
        
    def test_interpolation(self):
        """Test interpolation for dimensions not in table."""
        # Test interpolation for d values between table entries
        # For k=2, we have entries for d=3,4,5,6... Let's check if d=23 interpolates
        # since table only goes up to d=20
        
        # First verify exact lookups work
        alpha_d3 = get_lnc_alpha(2, 3)
        alpha_d4 = get_lnc_alpha(2, 4)
        assert abs(alpha_d3 - 0.182224) < 1e-6
        assert abs(alpha_d4 - 0.284370) < 1e-6
        
        # For d > 20, should use d=20 value (no interpolation, just clamping)
        alpha_d25 = get_lnc_alpha(2, 25)
        alpha_d20 = get_lnc_alpha(2, 20)
        assert alpha_d25 == alpha_d20
        
        # Test that float d values raise error (no interpolation possible)
        with pytest.raises(ValueError, match="d must be a positive integer"):
            get_lnc_alpha(5, 14.5)
        
    def test_extrapolation_low_d(self):
        """Test behavior for dimensions below available range."""
        # For k=5, minimum d is 6
        alpha_d2 = get_lnc_alpha(5, 2)
        alpha_d6 = get_lnc_alpha(5, 6)
        assert alpha_d2 == alpha_d6  # Should use minimum available
        
    def test_extrapolation_high_d(self):
        """Test behavior for dimensions above available range."""
        # For k=5, maximum d is 20
        alpha_d25 = get_lnc_alpha(5, 25)
        alpha_d20 = get_lnc_alpha(5, 20)
        assert alpha_d25 == alpha_d20  # Should use maximum available
        
    def test_nearest_k(self):
        """Test nearest k selection for unavailable k values."""
        # k=4 is not in table, should use nearest k
        # Available k values are: 2, 3, 5, 10
        # k=4 is equidistant from k=3 and k=5, min() will pick k=3
        alpha_k4 = get_lnc_alpha(4, 10)
        alpha_k3 = get_lnc_alpha(3, 10)
        alpha_k5 = get_lnc_alpha(5, 10)
        
        # k=4 should use k=3 (due to min() behavior with equal distances)
        assert alpha_k4 == alpha_k3
        
        # Test other k values
        # k=1 should use k=2 (nearest)
        assert get_lnc_alpha(1, 10) == get_lnc_alpha(2, 10)
        
        # k=7 should use k=5 (nearer than k=10)
        assert get_lnc_alpha(7, 10) == get_lnc_alpha(5, 10)
        
        # k=8 should use k=10 (nearer than k=5)
        assert get_lnc_alpha(8, 10) == get_lnc_alpha(10, 10)
        
    def test_mi_with_auto_alpha(self):
        """Test mutual information estimation with automatic alpha selection."""
        np.random.seed(42)
        
        # Create data where k > d and alpha is meaningful
        n = 500
        x = np.random.randn(n, 10)  # 10D
        y = x[:, 0:5] + 0.5 * np.random.randn(n, 5)  # 5D, correlated with first 5 dims of x
        
        # Use k=20 to ensure k > d=15
        k = 20
        
        # For k=20, closest available is k=10
        d = x.shape[1] + y.shape[1]  # Total dimensionality = 15
        expected_alpha = get_lnc_alpha(k, d)
        
        # Test with auto alpha
        np.random.seed(42)
        mi_auto = nonparam_mi_cc(x, y, k=k, alpha="auto")
        
        # Test with manual alpha=0 (no LNC correction)
        np.random.seed(42)
        mi_no_lnc = nonparam_mi_cc(x, y, k=k, alpha=0)
        
        # Test with specific alpha
        np.random.seed(42)
        mi_manual = nonparam_mi_cc(x, y, k=k, alpha=expected_alpha)
        
        
        # Auto should match manual with same alpha
        assert abs(mi_auto - mi_manual) < 1e-10
        
        # With k=20, d=15, we should get alpha for (10,15) = 0.100471
        assert expected_alpha > 0.1  # Meaningful alpha
        # There should be a noticeable difference with LNC correction
        assert abs(mi_auto - mi_no_lnc) > 0.01  # Significant difference
            
    def test_conditional_mi_with_auto_alpha(self):
        """Test conditional MI with automatic alpha selection."""
        np.random.seed(42)
        
        # Create data with conditional dependence
        n = 500
        z = np.random.randn(n, 1)
        x = z + 0.3 * np.random.randn(n, 1)
        y = z + 0.3 * np.random.randn(n, 1)
        
        # Test conditional MI with auto alpha
        cmi_auto = nonparam_mi_cc(x, y, z=z, k=3, alpha="auto")
        
        # Test with no LNC
        cmi_no_lnc = nonparam_mi_cc(x, y, z=z, k=3, alpha=0)
        
        # For low dimensional case (d=3), alpha should be small
        d = x.shape[1] + y.shape[1] + z.shape[1]  # 3
        expected_alpha = get_lnc_alpha(3, d)
        
        # Verify it runs without error
        # Note: CMI can be negative due to estimation bias
        assert abs(cmi_auto) < 1  # Should be small since X and Y are nearly conditionally independent given Z
        
        # The difference might be very small for low dimensions
        if expected_alpha > 0.1:
            assert abs(cmi_auto - cmi_no_lnc) > 1e-10


class TestAlphaBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def test_default_alpha_changed(self):
        """Test that default alpha is now 'auto' instead of 0."""
        np.random.seed(42)
        # Use data where k > d and alpha is meaningful
        n = 200
        x = np.random.randn(n, 8)
        y = x[:, :4] + 0.5 * np.random.randn(n, 4)
        
        # Use k=15 to ensure k > d=12
        k = 15
        
        # With no alpha specified, should use auto
        mi_default = nonparam_mi_cc(x, y, k=k)
        
        # With alpha=0
        mi_no_lnc = nonparam_mi_cc(x, y, k=k, alpha=0)
        
        # Get expected alpha
        d = x.shape[1] + y.shape[1]  # 12
        expected_alpha = get_lnc_alpha(k, d)
        
        # For k=15, d=12, get small but meaningful alpha
        assert expected_alpha > 0.01  # Small but meaningful alpha
        # Should see a measurable difference with LNC correction
        assert abs(mi_default - mi_no_lnc) > 1e-5  # Clear difference
    
    def test_explicit_alpha_still_works(self):
        """Test that explicitly setting alpha still works."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n, 2)
        y = x[:, 0:1] + np.random.randn(n, 1)
        
        # Test with explicit alpha value
        mi_explicit = nonparam_mi_cc(x, y, k=5, alpha=0.25)
        
        # Should run without error
        assert mi_explicit > 0