"""Tests for matrix utilities."""

import numpy as np
from numpy import linalg as la
from driada.utils.matrix import nearestPD, is_positive_definite


class TestIsPositiveDefinite:
    """Test the is_positive_definite function."""

    def test_positive_definite_matrix(self):
        """Test with a known positive definite matrix."""
        # Create a positive definite matrix
        A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        assert is_positive_definite(A) is True

    def test_identity_matrix(self):
        """Test with identity matrix (always positive definite)."""
        I = np.eye(5)
        assert is_positive_definite(I) is True

    def test_not_positive_definite_matrix(self):
        """Test with a non-positive definite matrix."""
        # This matrix has a negative eigenvalue
        A = np.array([[1, 2], [2, 1]])
        assert is_positive_definite(A) is False

    def test_singular_matrix(self):
        """Test with a singular matrix (not positive definite)."""
        A = np.array([[1, 1], [1, 1]])
        assert is_positive_definite(A) is False

    def test_random_spd_matrix(self):
        """Test with randomly generated SPD matrix."""
        # Generate random SPD matrix
        n = 4
        A = np.random.randn(n, n)
        A = np.dot(A, A.T)  # A'A is always positive semi-definite
        A = A + np.eye(n) * 0.1  # Make it positive definite
        assert is_positive_definite(A) is True


class TestNearestPD:
    """Test the nearestPD function."""

    def test_already_pd_matrix(self):
        """Test that PD matrix is returned unchanged."""
        A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        A_pd = nearestPD(A)
        np.testing.assert_array_almost_equal(A, A_pd)
        assert is_positive_definite(A_pd)

    def test_symmetric_not_pd(self):
        """Test with symmetric but not PD matrix."""
        A = np.array([[1, 2], [2, 1]])
        A_pd = nearestPD(A)
        assert is_positive_definite(A_pd)
        # Check it's still symmetric
        np.testing.assert_array_almost_equal(A_pd, A_pd.T)

    def test_non_symmetric_matrix(self):
        """Test with non-symmetric matrix."""
        A = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]])
        A_pd = nearestPD(A)
        assert is_positive_definite(A_pd)
        # Result should be symmetric
        np.testing.assert_array_almost_equal(A_pd, A_pd.T)

    def test_negative_eigenvalues(self):
        """Test with matrix having negative eigenvalues."""
        # Create matrix with known negative eigenvalue
        A = np.array([[-1, 0, 0], [0, 2, 0], [0, 0, 3]])
        A_pd = nearestPD(A)
        assert is_positive_definite(A_pd)
        # All eigenvalues should be positive
        eigvals = la.eigvals(A_pd)
        assert np.all(eigvals > 0)

    def test_near_singular_matrix(self):
        """Test with near-singular matrix."""
        # Create near-singular matrix
        A = np.array([[1e-10, 0], [0, 1]])
        A_pd = nearestPD(A)
        assert is_positive_definite(A_pd)

    def test_frobenius_norm_minimization(self):
        """Test that the result is close to original in Frobenius norm."""
        # Random non-PD symmetric matrix
        n = 5
        A = np.random.randn(n, n)
        A = (A + A.T) / 2
        # Make it non-PD by setting some eigenvalues negative
        eigvals, eigvecs = la.eigh(A)
        eigvals[0] = -0.1
        A = eigvecs @ np.diag(eigvals) @ eigvecs.T

        A_pd = nearestPD(A)
        assert is_positive_definite(A_pd)

        # The PD approximation should be reasonably close
        frobenius_dist = la.norm(A - A_pd, "fro")
        assert frobenius_dist < la.norm(A, "fro")

    def test_iterative_correction(self):
        """Test the iterative correction for difficult cases."""
        # Create a matrix that needs iterative correction
        A = np.array([[1, 0.9, 0.8], [0.9, 1, 0.95], [0.8, 0.95, 1]])
        # Perturb to make it slightly non-PD
        A[0, 0] = 0.5

        A_pd = nearestPD(A)
        assert is_positive_definite(A_pd)
        # Check symmetry is preserved
        np.testing.assert_array_almost_equal(A_pd, A_pd.T)
