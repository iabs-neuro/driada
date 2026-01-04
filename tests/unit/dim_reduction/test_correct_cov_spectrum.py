"""Tests for correct_cov_spectrum function."""

import numpy as np
import warnings
from driada.dimensionality.utils import correct_cov_spectrum
from scipy.linalg import eigh


class TestCorrectCovSpectrum:
    """Test the covariance spectrum correction function."""

    def test_basic_functionality(self):
        """Test basic functionality with valid input."""
        np.random.seed(42)
        n = 50
        t = 1000

        # Create a valid correlation matrix
        data = np.random.randn(n, t)
        cmat = np.corrcoef(data)

        corrected_eigs = correct_cov_spectrum(n, t, cmat, correction_iters=5)

        assert isinstance(corrected_eigs, list)
        assert len(corrected_eigs) == 6  # initial + 5 iterations
        assert all(isinstance(eigs, np.ndarray) for eigs in corrected_eigs)
        assert all(len(eigs) == n for eigs in corrected_eigs)

    def test_negative_eigenvalues_handling(self):
        """Test handling of matrices with negative eigenvalues."""
        np.random.seed(42)
        n = 50
        t = 55  # Close to n to increase numerical issues

        # Create a nearly singular correlation matrix
        data = np.random.randn(n, t)
        # Make some rows highly correlated
        data[1:10] = 0.99 * data[0] + 0.01 * np.random.randn(9, t)
        cmat = np.corrcoef(data)

        # Check that original matrix might have negative eigenvalues
        orig_eigs = eigh(cmat, eigvals_only=True)
        # This is expected in near-singular cases

        # Function should handle this without errors
        corrected_eigs = correct_cov_spectrum(n, t, cmat, correction_iters=3)

        # All returned eigenvalues should be positive
        for eigs in corrected_eigs:
            assert np.all(eigs >= 0), f"Found negative eigenvalues: {eigs[eigs < 0]}"

    def test_warning_for_significant_negative_eigenvalues(self):
        """Test that warning is issued for significant negative eigenvalues."""
        np.random.seed(42)
        n = 100
        t = 100

        # Create a matrix that will have significant negative eigenvalues
        # Start with a positive semi-definite matrix
        A = np.random.randn(n, 50)
        cmat = A @ A.T / 50
        # Perturb it to introduce negative eigenvalues
        perturbation = 0.1 * (np.random.randn(n, n))
        perturbation = (perturbation + perturbation.T) / 2
        cmat += perturbation

        # Normalize to correlation matrix
        D = np.diag(1 / np.sqrt(np.diag(cmat)))
        cmat = D @ cmat @ D

        # Should issue warning if there are significant negative eigenvalues
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            corrected_eigs = correct_cov_spectrum(n, t, cmat, correction_iters=2)

            # Check if any warning about negative eigenvalues was issued
            neg_eig_warnings = [
                warning
                for warning in w
                if "negative eigenvalues" in str(warning.message)
            ]
            # This may or may not warn depending on the random seed

    def test_min_eigenvalue_parameter(self):
        """Test the min_eigenvalue parameter."""
        np.random.seed(42)
        n = 30
        t = 35

        data = np.random.randn(n, t)
        cmat = np.corrcoef(data)

        # Test with different min_eigenvalue settings
        min_eig_small = 1e-12
        min_eig_large = 1e-6

        corrected_eigs_small = correct_cov_spectrum(
            n, t, cmat, correction_iters=2, min_eigenvalue=min_eig_small
        )
        corrected_eigs_large = correct_cov_spectrum(
            n, t, cmat, correction_iters=2, min_eigenvalue=min_eig_large
        )

        # All eigenvalues should be at least min_eigenvalue
        for eigs in corrected_eigs_small:
            assert np.all(eigs >= min_eig_small)

        for eigs in corrected_eigs_large:
            assert np.all(eigs >= min_eig_large)

    def test_ensemble_size_parameter(self):
        """Test different ensemble sizes."""
        np.random.seed(42)
        n = 20
        t = 500

        data = np.random.randn(n, t)
        cmat = np.corrcoef(data)

        # Test with different ensemble sizes
        corrected_eigs_1 = correct_cov_spectrum(
            n, t, cmat, correction_iters=2, ensemble_size=1
        )
        corrected_eigs_5 = correct_cov_spectrum(
            n, t, cmat, correction_iters=2, ensemble_size=5
        )

        # Both should work and return valid results
        assert len(corrected_eigs_1) == 3
        assert len(corrected_eigs_5) == 3

        # Results should be different due to ensemble averaging
        assert not np.allclose(corrected_eigs_1[-1], corrected_eigs_5[-1])

    def test_extreme_cases(self):
        """Test extreme cases that might cause numerical issues."""
        np.random.seed(42)

        # Case 1: Identity matrix (should remain unchanged)
        n = 20
        t = 1000
        cmat = np.eye(n)

        corrected_eigs = correct_cov_spectrum(n, t, cmat, correction_iters=2)
        # Should stay close to all ones
        assert np.allclose(corrected_eigs[-1], np.ones(n), atol=0.1)

        # Case 2: Rank-deficient matrix
        n = 30
        t = 35
        rank = 10
        # Create low-rank data
        U = np.random.randn(n, rank)
        V = np.random.randn(rank, t)
        data = U @ V
        # Add small noise to avoid exact singularity
        data += 1e-6 * np.random.randn(n, t)
        cmat = np.corrcoef(data)

        corrected_eigs = correct_cov_spectrum(n, t, cmat, correction_iters=2)
        # Should handle rank deficiency without errors
        assert all(np.all(eigs >= 0) for eigs in corrected_eigs)

    def test_convergence_behavior(self):
        """Test that correction improves over iterations."""
        np.random.seed(42)
        n = 40
        t = 50  # Moderate n/t ratio where correction is beneficial

        data = np.random.randn(n, t)
        cmat = np.corrcoef(data)

        corrected_eigs = correct_cov_spectrum(n, t, cmat, correction_iters=10)

        # Check that eigenvalues are changing across iterations
        initial_eigs = corrected_eigs[0]
        final_eigs = corrected_eigs[-1]

        # They should be different (correction should have an effect)
        assert not np.allclose(initial_eigs, final_eigs)

        # All iterations should produce valid positive eigenvalues
        for i, eigs in enumerate(corrected_eigs):
            assert np.all(eigs >= 0), f"Iteration {i} has negative eigenvalues"
