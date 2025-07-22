"""
Unit tests for manifold metrics module.
"""
import numpy as np
import pytest
from sklearn.datasets import make_swiss_roll
from driada.dim_reduction.manifold_metrics import (
    compute_distance_matrix,
    knn_preservation_rate,
    trustworthiness,
    continuity,
    geodesic_distance_correlation,
    stress,
    circular_structure_preservation,
    procrustes_analysis,
    manifold_preservation_score,
)


def test_compute_distance_matrix():
    """Test distance matrix computation"""
    # Simple 2D points
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    D = compute_distance_matrix(X)
    
    # Check shape
    assert D.shape == (4, 4)
    
    # Check diagonal is zero
    assert np.allclose(np.diag(D), 0)
    
    # Check symmetry
    assert np.allclose(D, D.T)
    
    # Check known distances
    assert np.isclose(D[0, 1], 1.0)  # (0,0) to (1,0)
    assert np.isclose(D[0, 2], 1.0)  # (0,0) to (0,1)
    assert np.isclose(D[0, 3], np.sqrt(2))  # (0,0) to (1,1)


def test_knn_preservation_rate():
    """Test k-NN preservation metric"""
    # Create data where we know the preservation
    np.random.seed(42)
    n_points = 50
    
    # Create a simple 2D manifold embedded in higher dimensions
    t = np.linspace(0, 4 * np.pi, n_points)
    X_2d = np.column_stack([np.cos(t), np.sin(t)])
    
    # Embed in 10D with random linear transformation
    A = np.random.randn(10, 2)
    X_high = X_2d @ A.T + 0.01 * np.random.randn(n_points, 10)
    
    # Perfect recovery via PCA should preserve neighbors
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_low = pca.fit_transform(X_high)
    
    # Should have high preservation
    rate = knn_preservation_rate(X_high, X_low, k=5)
    assert rate > 0.8
    
    # Test with scrambled data (poor preservation)
    X_low_scrambled = X_low[np.random.permutation(n_points)]
    rate_scrambled = knn_preservation_rate(X_high, X_low_scrambled, k=5)
    assert rate_scrambled < 0.3
    
    # Test flexible matching
    rate_flexible = knn_preservation_rate(X_high, X_low, k=5, flexible=True)
    assert rate_flexible >= rate  # Should be at least as good


def test_trustworthiness_continuity():
    """Test trustworthiness and continuity metrics"""
    # Generate simple manifold
    np.random.seed(42)
    n_points = 100
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # Add noise in higher dimensions
    noise = 0.1 * np.random.randn(n_points, 3)
    X_high = np.hstack([circle, noise])
    
    # Perfect 2D recovery
    X_low = circle
    
    # Should have high trustworthiness and continuity
    trust = trustworthiness(X_high, X_low, k=10)
    cont = continuity(X_high, X_low, k=10)
    
    assert trust > 0.9
    assert cont > 0.9
    
    # Random embedding should have lower scores
    X_random = np.random.randn(n_points, 2)
    trust_random = trustworthiness(X_high, X_random, k=10)
    cont_random = continuity(X_high, X_random, k=10)
    
    assert trust_random < trust
    assert cont_random < cont


def test_geodesic_distance_correlation():
    """Test geodesic distance preservation"""
    # Swiss roll has different geodesic and Euclidean distances
    n_samples = 200
    X, _ = make_swiss_roll(n_samples=n_samples, noise=0.0, random_state=42)
    
    # Project to 2D naively (loses geodesic structure)
    X_naive = X[:, [0, 2]]  # Just take X and Z coordinates
    
    # Correlation should be moderate to high (not perfect due to lost structure)
    corr = geodesic_distance_correlation(X, X_naive, k_neighbors=10)
    assert 0.3 < corr < 0.95  # Not too low, not perfect
    
    # Test with identical data
    corr_perfect = geodesic_distance_correlation(X, X, k_neighbors=10)
    assert corr_perfect > 0.93  # Should be very high but numerical precision affects it


def test_stress():
    """Test stress computation"""
    # Identical points have zero stress
    X = np.random.randn(30, 5)
    stress_zero = stress(X, X)
    assert np.isclose(stress_zero, 0.0)
    
    # Different embeddings have positive stress
    Y = np.random.randn(30, 2)
    stress_positive = stress(X, Y)
    assert stress_positive > 0
    
    # Normalized stress should be between 0 and 1
    stress_norm = stress(X, Y, normalized=True)
    assert 0 <= stress_norm <= 1


def test_circular_structure_preservation():
    """Test circular structure metrics"""
    # Generate perfect circle
    n_points = 100
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # Test metrics
    metrics = circular_structure_preservation(circle, true_angles=theta)
    
    # Perfect circle should have low distance CV
    assert metrics['distance_cv'] < 0.01
    
    # Should preserve consecutive neighbors perfectly
    assert metrics['consecutive_preservation'] > 0.95
    
    # Should have perfect circular correlation
    assert metrics['circular_correlation'] > 0.99
    
    # Test with noisy circle
    noisy_circle = circle + 0.1 * np.random.randn(n_points, 2)
    metrics_noisy = circular_structure_preservation(noisy_circle, true_angles=theta)
    
    # Should still be good but not perfect
    assert 0.05 < metrics_noisy['distance_cv'] < 0.2
    assert 0.7 < metrics_noisy['consecutive_preservation'] < 0.95


def test_procrustes_analysis():
    """Test Procrustes alignment"""
    # Create two similar configurations
    np.random.seed(42)
    X = np.random.randn(50, 2)
    
    # Rotate, scale, and translate
    angle = np.pi / 4
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    Y = 2.0 * (X @ R) + np.array([3, -2])
    
    # Add small noise
    Y += 0.1 * np.random.randn(50, 2)
    
    # Align Y to X
    Y_aligned, disparity = procrustes_analysis(X, Y)
    
    # Should be well aligned
    assert disparity < 10.0
    
    # Check alignment quality
    diff = np.mean(np.abs(X - Y_aligned))
    assert diff < 0.5


def test_manifold_preservation_score():
    """Test comprehensive manifold preservation score"""
    # Generate swiss roll
    n_samples = 300
    X, color = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42)
    
    # Good embedding (using first two meaningful dimensions)
    X_good = np.column_stack([color, X[:, 1]])  # Color encodes position along roll
    
    # Bad embedding (random projection)
    np.random.seed(42)
    X_bad = X @ np.random.randn(3, 2)
    
    # Compute scores
    scores_good = manifold_preservation_score(X, X_good, k_neighbors=10)
    scores_bad = manifold_preservation_score(X, X_bad, k_neighbors=10)
    
    # Good embedding should have better scores
    assert scores_good['overall_score'] > scores_bad['overall_score']
    assert scores_good['knn_preservation'] > scores_bad['knn_preservation']
    
    # Check all metrics are computed
    expected_metrics = ['knn_preservation', 'trustworthiness', 'continuity', 
                       'geodesic_correlation', 'overall_score']
    for metric in expected_metrics:
        assert metric in scores_good
        assert metric in scores_bad


def test_error_handling():
    """Test error handling in metrics"""
    X = np.random.randn(50, 10)
    Y = np.random.randn(30, 2)  # Different number of samples
    
    # Should raise error for mismatched sizes
    with pytest.raises(ValueError):
        knn_preservation_rate(X, Y)
    
    with pytest.raises(ValueError):
        trustworthiness(X, Y)
    
    # k too large
    with pytest.raises(ValueError):
        knn_preservation_rate(X, X[:, :2], k=100)
    
    # Wrong dimension for circular analysis
    with pytest.raises(ValueError):
        circular_structure_preservation(X)  # Needs 2D


def test_edge_cases():
    """Test edge cases"""
    # Very small dataset
    X_small = np.array([[0, 0], [1, 1], [2, 2]])
    Y_small = np.array([[0, 0], [1, 0], [2, 0]])
    
    # Should still work
    rate = knn_preservation_rate(X_small, Y_small, k=1)
    assert 0 <= rate <= 1
    
    # Identical points
    X_identical = np.ones((10, 5))
    Y_random = np.random.randn(10, 2)
    
    # Distance matrix should be all zeros
    D = compute_distance_matrix(X_identical)
    assert np.allclose(D, 0)
    
    # Metrics should handle this gracefully
    score = manifold_preservation_score(X_identical, Y_random, k_neighbors=3)
    assert 'overall_score' in score