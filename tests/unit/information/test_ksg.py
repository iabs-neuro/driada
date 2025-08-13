"""Tests for KSG mutual information estimator utilities."""

import numpy as np
import pytest
from sklearn.neighbors import KDTree, BallTree
from driada.information.ksg import (
    add_noise,
    query_neighbors,
    count_neighbors,
    build_tree,
    avgdigamma,
    DEFAULT_NN,
)


class TestKSGUtilities:
    """Test KSG estimator utility functions."""

    def test_add_noise(self):
        """Test noise addition to break degeneracy."""
        # Test with 1D data
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x_noisy = add_noise(x.copy())

        # Should be slightly different
        assert not np.array_equal(x, x_noisy)
        # But very close
        assert np.allclose(x, x_noisy, rtol=1e-5)

        # Test with 2D data
        x_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        x_2d_noisy = add_noise(x_2d.copy())
        assert not np.array_equal(x_2d, x_2d_noisy)
        assert np.allclose(x_2d, x_2d_noisy, rtol=1e-5)

    def test_add_noise_amplitude(self):
        """Test noise addition with custom amplitude."""
        x = np.ones((10, 2))

        # Small amplitude
        x_small = add_noise(x.copy(), ampl=1e-12)
        diff_small = np.max(np.abs(x - x_small))

        # Larger amplitude
        x_large = add_noise(x.copy(), ampl=1e-6)
        diff_large = np.max(np.abs(x - x_large))

        assert diff_large > diff_small
        assert diff_small < 1e-11
        assert diff_large < 1e-5

    def test_build_tree_low_dim(self):
        """Test tree building for low-dimensional data."""
        # Low dimensional data (< 20 dims) should use KDTree
        points = np.random.randn(100, 5)
        tree = build_tree(points)
        assert isinstance(tree, KDTree)

        points_2d = np.random.randn(100, 2)
        tree_2d = build_tree(points_2d)
        assert isinstance(tree_2d, KDTree)

    def test_build_tree_high_dim(self):
        """Test tree building for high-dimensional data."""
        # High dimensional data (>= 20 dims) should use BallTree
        points = np.random.randn(100, 20)
        tree = build_tree(points)
        assert isinstance(tree, BallTree)

        points_30d = np.random.randn(100, 30)
        tree_30d = build_tree(points_30d)
        assert isinstance(tree_30d, BallTree)

    def test_build_tree_leaf_size(self):
        """Test tree building with custom leaf size."""
        points = np.random.randn(100, 3)
        tree = build_tree(points, lf=10)
        assert isinstance(tree, KDTree)
        # Leaf size is set in tree construction

    def test_query_neighbors(self):
        """Test k-nearest neighbor queries."""
        # Create simple 2D points
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
        tree = build_tree(points)

        k = 2  # Find 2 nearest neighbors
        distances = query_neighbors(tree, points, k)

        # Check shape
        assert distances.shape == (5,)

        # Check that distances are positive (except for query point itself)
        assert np.all(distances > 0)

        # For point [0.5, 0.5], the 3rd nearest should be at distance ~0.5
        center_dist = query_neighbors(tree, np.array([[0.5, 0.5]]), k)
        assert 0.4 < center_dist[0] < 0.6

    def test_count_neighbors(self):
        """Test radius neighbor counting."""
        # Create grid points
        points = np.array([[i, j] for i in range(3) for j in range(3)])
        tree = build_tree(points)

        # Count neighbors within radius 1.5 for each point
        radii = np.ones(9) * 1.5
        counts = count_neighbors(tree, points, radii)

        assert counts.shape == (9,)
        # Corner points should have 4 neighbors (including self)
        assert counts[0] == 4  # Point (0,0)
        assert counts[2] == 4  # Point (0,2)
        assert counts[6] == 4  # Point (2,0)
        assert counts[8] == 4  # Point (2,2)

        # Center point should have all 9
        assert counts[4] == 9  # Point (1,1)

    def test_count_neighbors_varying_radii(self):
        """Test neighbor counting with different radii."""
        points = np.random.randn(50, 2)
        tree = build_tree(points)

        # Different radius for each point
        radii = np.linspace(0.1, 2.0, 50)
        counts = count_neighbors(tree, points, radii)

        assert counts.shape == (50,)
        # Larger radii should generally have more neighbors
        assert counts[-1] > counts[0]

    def test_avgdigamma_basic(self):
        """Test average digamma calculation."""
        # Simple 2D points
        points = np.random.randn(100, 2)
        # dvec should be distances/radii for each point
        k = 5
        tree = build_tree(points)
        distances = query_neighbors(tree, points, k)

        result = avgdigamma(points, distances)

        # Should return a scalar
        assert isinstance(result, (float, np.floating))
        # Digamma can be negative for small arguments
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_avgdigamma_with_tree(self):
        """Test average digamma with pre-built tree."""
        points = np.random.randn(100, 2)
        k = 5
        tree = build_tree(points)
        distances = query_neighbors(tree, points, k)

        result = avgdigamma(points, distances, tree=tree)

        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)

    def test_avgdigamma_single_dimension(self):
        """Test avgdigamma with single dimension."""
        points = np.random.randn(100, 1)
        k = 5
        tree = build_tree(points)
        distances = query_neighbors(tree, points, k)

        result = avgdigamma(points, distances)

        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)

    def test_default_nn_constant(self):
        """Test that DEFAULT_NN is properly defined."""
        assert DEFAULT_NN == 5
        assert isinstance(DEFAULT_NN, int)
        assert DEFAULT_NN > 0


class TestKSGEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_points(self):
        """Test with empty point sets."""
        # This might raise an error in tree building
        points = np.array([]).reshape(0, 2)
        with pytest.raises((ValueError, IndexError)):
            tree = build_tree(points)

    def test_single_point(self):
        """Test with single point."""
        points = np.array([[1.0, 2.0]])
        tree = build_tree(points)

        # Query neighbors might behave differently
        with pytest.raises((ValueError, IndexError)):
            distances = query_neighbors(tree, points, k=2)

    def test_duplicate_points(self):
        """Test with duplicate points."""
        # All points are the same
        points = np.ones((10, 2))
        # Add small noise to break degeneracy
        points_noisy = add_noise(points)
        tree = build_tree(points_noisy)

        distances = query_neighbors(tree, points_noisy, k=2)
        # Distances should be very small
        assert np.all(distances < 1e-8)
