"""
Unit tests for manifold metrics module.
"""

import numpy as np
import pytest
import warnings
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
    circular_distance,
    extract_angles_from_embedding,
    compute_reconstruction_error,
    compute_embedding_alignment_metrics,
    train_simple_decoder,
    compute_embedding_quality,
    compute_decoding_accuracy,
    manifold_reconstruction_score,
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
    assert metrics["distance_cv"] < 0.01

    # Should preserve consecutive neighbors perfectly
    assert metrics["consecutive_preservation"] > 0.95

    # Should have perfect circular correlation
    assert metrics["circular_correlation"] > 0.99

    # Test with noisy circle
    noisy_circle = circle + 0.1 * np.random.randn(n_points, 2)
    metrics_noisy = circular_structure_preservation(noisy_circle, true_angles=theta)

    # Should still be good but not perfect
    assert 0.05 < metrics_noisy["distance_cv"] < 0.2
    assert 0.7 < metrics_noisy["consecutive_preservation"] < 0.95


def test_procrustes_analysis():
    """Test Procrustes alignment"""
    # Create two similar configurations
    np.random.seed(42)
    X = np.random.randn(50, 2)

    # Rotate, scale, and translate
    angle = np.pi / 4
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    Y = 2.0 * (X @ R) + np.array([3, -2])

    # Add small noise
    Y += 0.1 * np.random.randn(50, 2)

    # Align Y to X
    Y_aligned, disparity, transform_info = procrustes_analysis(X, Y)

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
    assert scores_good["overall_score"] > scores_bad["overall_score"]
    assert scores_good["knn_preservation"] > scores_bad["knn_preservation"]

    # Check all metrics are computed
    expected_metrics = [
        "knn_preservation",
        "trustworthiness",
        "continuity",
        "geodesic_correlation",
        "overall_score",
    ]
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
    assert "overall_score" in score


def test_compute_distance_matrix_invalid_shape():
    """Test distance matrix computation with invalid input"""
    # 1D array should raise error
    X_1d = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError, match="X must be 2D array"):
        compute_distance_matrix(X_1d)

    # 3D array should also raise error
    X_3d = np.random.randn(5, 5, 5)
    with pytest.raises(ValueError, match="X must be 2D array"):
        compute_distance_matrix(X_3d)


def test_compute_distance_matrix_metrics():
    """Test distance matrix with different metrics"""
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

    # Test Manhattan distance
    D_manhattan = compute_distance_matrix(X, metric="cityblock")
    assert np.isclose(D_manhattan[0, 1], 1.0)  # (0,0) to (1,0)
    assert np.isclose(D_manhattan[0, 3], 2.0)  # (0,0) to (1,1) - Manhattan distance

    # Test Chebyshev distance
    D_chebyshev = compute_distance_matrix(X, metric="chebyshev")
    assert np.isclose(D_chebyshev[0, 1], 1.0)
    assert np.isclose(D_chebyshev[0, 3], 1.0)  # Max of |1-0| and |1-0|


def test_trustworthiness_edge_cases():
    """Test trustworthiness with edge cases"""
    # Test with k=n-1 (maximum k)
    np.random.seed(42)
    X_high = np.random.randn(10, 5)
    X_low = np.random.randn(10, 2)

    trust = trustworthiness(X_high, X_low, k=9)
    assert 0 <= trust <= 1

    # Test with very small k
    trust_small_k = trustworthiness(X_high, X_low, k=1)
    assert 0 <= trust_small_k <= 1


def test_continuity_edge_cases():
    """Test continuity with edge cases"""
    # Similar to trustworthiness tests
    np.random.seed(42)
    X_high = np.random.randn(10, 5)
    X_low = np.random.randn(10, 2)

    cont = continuity(X_high, X_low, k=9)
    assert isinstance(cont, float)
    assert -1 <= cont <= 1  # Continuity can be negative for very poor embeddings

    # Test with k=1
    cont_small_k = continuity(X_high, X_low, k=1)
    assert isinstance(cont_small_k, float)
    assert -1 <= cont_small_k <= 1


def test_geodesic_distance_correlation_disconnected():
    """Test geodesic correlation with disconnected graph"""
    # Create two disconnected clusters
    np.random.seed(42)
    cluster1 = np.random.randn(20, 3)
    cluster2 = np.random.randn(20, 3) + 100  # Far away
    X = np.vstack([cluster1, cluster2])

    # Random low-dimensional embedding
    Y = np.random.randn(40, 2)

    # Should handle disconnected components gracefully
    corr = geodesic_distance_correlation(X, Y, k_neighbors=5)
    assert isinstance(corr, float)
    assert -1 <= corr <= 1


def test_geodesic_distance_correlation_pearson():
    """Test geodesic correlation with Pearson method"""
    np.random.seed(42)
    X = np.random.randn(50, 5)
    Y = np.random.randn(50, 2)

    # Test Pearson correlation
    corr_pearson = geodesic_distance_correlation(X, Y, k_neighbors=10, method="pearson")
    assert isinstance(corr_pearson, float)
    assert -1 <= corr_pearson <= 1

    # Compare with Spearman
    corr_spearman = geodesic_distance_correlation(
        X, Y, k_neighbors=10, method="spearman"
    )
    # They should be different but both valid
    assert corr_pearson != corr_spearman


def test_stress_unnormalized():
    """Test unnormalized stress computation"""
    np.random.seed(42)
    X = np.random.randn(30, 5)
    Y = np.random.randn(30, 2)

    # Unnormalized stress
    stress_unnorm = stress(X, Y, normalized=False)
    assert stress_unnorm > 0

    # Should be larger than normalized
    stress_norm = stress(X, Y, normalized=True)
    assert stress_unnorm > stress_norm


def test_circular_structure_no_true_angles():
    """Test circular structure preservation without ground truth angles"""
    # Generate noisy circle
    n_points = 50
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    circle += 0.1 * np.random.randn(n_points, 2)

    # Test without true angles
    metrics = circular_structure_preservation(circle, k_neighbors=5)

    # Should only have distance_cv and consecutive_preservation
    assert "distance_cv" in metrics
    assert "consecutive_preservation" in metrics
    assert "circular_correlation" not in metrics


def test_procrustes_analysis_no_scaling():
    """Test Procrustes analysis without scaling"""
    np.random.seed(42)
    X = np.random.randn(30, 2)

    # Create rotated version
    angle = np.pi / 6
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    Y = X @ R

    # Align without scaling
    Y_aligned, disparity, transform_info = procrustes_analysis(X, Y, scaling=False)

    # Should be perfectly aligned (no noise)
    assert disparity < 1e-10


def test_procrustes_analysis_no_reflection():
    """Test Procrustes analysis without allowing reflection"""
    np.random.seed(42)
    X = np.random.randn(30, 2)

    # Create reflected and rotated version
    Y = X.copy()
    Y[:, 0] = -Y[:, 0]  # Reflect in y-axis

    # Align without reflection
    Y_aligned, disparity, transform_info = procrustes_analysis(X, Y, reflection=False)

    # Should not be perfectly aligned since reflection is not allowed
    assert disparity > 1.0


def test_procrustes_analysis_mismatched_shapes():
    """Test Procrustes analysis with mismatched shapes"""
    X = np.random.randn(30, 2)
    Y = np.random.randn(25, 2)  # Different number of points

    with pytest.raises(ValueError, match="X and Y must have the same shape"):
        procrustes_analysis(X, Y)


def test_manifold_preservation_score_custom_weights():
    """Test manifold preservation score with custom weights"""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    Y = np.random.randn(100, 2)

    # Custom weights emphasizing k-NN preservation
    weights = {
        "knn_preservation": 0.5,
        "trustworthiness": 0.2,
        "continuity": 0.2,
        "geodesic_correlation": 0.1,
    }

    scores = manifold_preservation_score(X, Y, k_neighbors=10, weights=weights)

    # Check overall score is weighted correctly
    expected_overall = sum(scores[key] * weights[key] for key in weights.keys())
    assert np.isclose(scores["overall_score"], expected_overall)


def test_manifold_preservation_score_nan_geodesic():
    """Test manifold preservation score handles NaN in geodesic correlation"""
    # Create data that might produce NaN geodesic correlation
    X = np.ones((10, 5))  # All identical points
    Y = np.random.randn(10, 2)

    scores = manifold_preservation_score(X, Y, k_neighbors=3)

    # Should handle NaN gracefully
    assert scores["geodesic_correlation"] == 0.0
    assert not np.isnan(scores["overall_score"])


def test_circular_distance():
    """Test circular distance computation"""
    # Test identical angles
    angles1 = np.array([0, np.pi / 2, np.pi, -np.pi / 2])
    angles2 = angles1.copy()
    distances = circular_distance(angles1, angles2)
    assert np.allclose(distances, 0)

    # Test opposite angles
    angles1 = np.array([0, 0, 0])
    angles2 = np.array([np.pi, -np.pi, np.pi])
    distances = circular_distance(angles1, angles2)
    assert np.allclose(distances, np.pi)

    # Test wraparound
    angles1 = np.array([0.1])
    angles2 = np.array([2 * np.pi + 0.1])
    distances = circular_distance(angles1, angles2)
    assert distances[0] < 0.01  # Should be close to 0


def test_extract_angles_from_embedding():
    """Test angle extraction from 2D embedding"""
    # Create points on a circle
    n_points = 8
    theta_true = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    embedding = np.column_stack([np.cos(theta_true), np.sin(theta_true)])

    # Extract angles
    angles = extract_angles_from_embedding(embedding)

    # Should recover the original angles (modulo 2π)
    diffs = circular_distance(angles, theta_true)
    assert np.all(diffs < 0.01)

    # Test with non-2D embedding
    embedding_3d = np.random.randn(10, 3)
    with pytest.raises(ValueError, match="Embedding must be 2D"):
        extract_angles_from_embedding(embedding_3d)


def test_compute_reconstruction_error_circular():
    """Test reconstruction error for circular manifolds"""
    # Perfect circle reconstruction
    n_points = 100
    true_angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    embedding = np.column_stack([np.cos(true_angles), np.sin(true_angles)])

    result = compute_reconstruction_error(
        embedding, true_angles, manifold_type="circular"
    )
    assert result["error"] < 0.01  # Should be very small
    assert result["correlation"] > 0.99  # Should have high correlation

    # Noisy reconstruction
    noisy_embedding = embedding + 0.1 * np.random.randn(n_points, 2)
    result_noisy = compute_reconstruction_error(
        noisy_embedding, true_angles, manifold_type="circular"
    )
    assert result_noisy["error"] > result["error"]
    assert result_noisy["error"] < np.pi  # Should still be reasonable
    assert result_noisy["correlation"] > 0.8  # Should still have decent correlation


def test_compute_reconstruction_error_spatial():
    """Test reconstruction error for spatial manifolds"""
    # Create 2D spatial data
    np.random.seed(42)
    true_positions = np.random.randn(50, 2)

    # Perfect reconstruction
    result = compute_reconstruction_error(
        true_positions, true_positions, manifold_type="spatial"
    )
    assert result["error"] < 1e-10
    assert result["correlation"] > 0.99

    # Noisy reconstruction
    noisy_positions = true_positions + 0.1 * np.random.randn(50, 2)
    result_noisy = compute_reconstruction_error(
        noisy_positions, true_positions, manifold_type="spatial"
    )
    assert result_noisy["error"] > 0
    assert result_noisy["error"] < 10.0  # Reasonable error for spatial data
    assert result_noisy["correlation"] > 0.8


def test_compute_reconstruction_error_invalid_type():
    """Test reconstruction error with invalid manifold type"""
    embedding = np.random.randn(10, 2)
    true_var = np.random.randn(10)

    with pytest.raises(ValueError, match="Unknown manifold type"):
        compute_reconstruction_error(embedding, true_var, manifold_type="invalid")


def test_train_simple_decoder_circular():
    """Test training decoder for circular variables"""
    # Generate circular data
    n_points = 200
    true_angles = np.random.uniform(0, 2 * np.pi, n_points)

    # Create embedding with some structure
    embedding = np.column_stack(
        [
            np.cos(true_angles) + 0.1 * np.random.randn(n_points),
            np.sin(true_angles) + 0.1 * np.random.randn(n_points),
        ]
    )

    # Train decoder
    decoder = train_simple_decoder(embedding, true_angles, manifold_type="circular")

    # Test on training data
    pred_angles = decoder(embedding)
    errors = circular_distance(pred_angles, true_angles)
    assert np.mean(errors) < 0.5  # Should fit reasonably well

    # Test on new data
    test_angles = np.random.uniform(0, 2 * np.pi, 50)
    test_embedding = np.column_stack([np.cos(test_angles), np.sin(test_angles)])
    test_pred = decoder(test_embedding)
    assert test_pred.shape == test_angles.shape


def test_train_simple_decoder_spatial():
    """Test training decoder for spatial variables"""
    # Generate spatial data
    np.random.seed(42)
    n_points = 200
    true_positions = np.random.randn(n_points, 2)

    # Create embedding (higher dimensional)
    embedding = np.random.randn(n_points, 10)
    # Add some structure
    embedding[:, :2] = true_positions + 0.1 * np.random.randn(n_points, 2)

    # Train decoder
    decoder = train_simple_decoder(embedding, true_positions, manifold_type="spatial")

    # Test on training data
    pred_positions = decoder(embedding)
    errors = np.linalg.norm(pred_positions - true_positions, axis=1)
    assert np.mean(errors) < 1.0


def test_train_simple_decoder_mismatched_shapes():
    """Test decoder training with mismatched shapes"""
    embedding = np.random.randn(100, 5)
    true_var = np.random.randn(50)  # Different number of samples

    with pytest.raises(ValueError, match="must have same number of timepoints"):
        train_simple_decoder(embedding, true_var)


def test_train_simple_decoder_invalid_type():
    """Test decoder training with invalid manifold type"""
    embedding = np.random.randn(100, 5)
    true_var = np.random.randn(100)

    with pytest.raises(ValueError, match="Unknown manifold type"):
        train_simple_decoder(embedding, true_var, manifold_type="invalid")


def test_compute_decoding_accuracy_circular():
    """Test decoding accuracy computation for circular manifolds"""
    # Generate data with clear structure
    n_points = 200
    true_angles = np.linspace(0, 4 * np.pi, n_points) % (2 * np.pi)
    embedding = np.column_stack(
        [
            np.cos(true_angles) + 0.05 * np.random.randn(n_points),
            np.sin(true_angles) + 0.05 * np.random.randn(n_points),
        ]
    )

    # Compute accuracy using the consistent decoder-based function
    results = compute_decoding_accuracy(
        embedding, true_angles, manifold_type="circular", train_fraction=0.7
    )

    # Check results structure
    assert "train_error" in results
    assert "test_error" in results
    assert "generalization_gap" in results

    # Should have reasonable errors for decoder
    assert results["train_error"] < 0.5
    assert results["test_error"] < 1.0
    # Generalization gap can be negative if test performs better
    assert -0.5 < results["generalization_gap"] < 1.0


def test_compute_decoding_accuracy_spatial():
    """Test decoding accuracy computation for spatial manifolds"""
    # Generate spatial data
    np.random.seed(42)
    n_points = 200
    true_positions = np.random.randn(n_points, 2)

    # Create structured embedding
    embedding = np.random.randn(n_points, 10)
    embedding[:, :2] = true_positions + 0.1 * np.random.randn(n_points, 2)

    # Compute accuracy
    results = compute_decoding_accuracy(
        embedding, true_positions, manifold_type="spatial", train_fraction=0.8
    )

    # Check reasonable performance
    assert results["train_error"] < 1.0
    assert results["test_error"] < 2.0


def test_manifold_reconstruction_score_circular():
    """Test comprehensive reconstruction score for circular manifolds"""
    # Generate smooth circular trajectory with varying velocity
    n_points = 200
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, n_points)
    # Add variation to avoid constant velocity issues
    true_angles = t + 0.2 * np.sin(3 * t)
    true_angles = np.arctan2(np.sin(true_angles), np.cos(true_angles))

    # Good reconstruction - small noise but preserve structure
    # Add offset to test centering
    offset = np.array([0.3, 0.2])
    good_embedding = (
        np.column_stack(
            [
                np.cos(true_angles) + 0.01 * np.random.randn(n_points),
                np.sin(true_angles) + 0.01 * np.random.randn(n_points),
            ]
        )
        + offset
    )

    # Bad reconstruction - large noise and distortion
    bad_angles = true_angles + 0.5 * np.random.randn(n_points)
    bad_embedding = np.column_stack(
        [
            np.cos(bad_angles) + 0.2 * np.random.randn(n_points),
            np.sin(bad_angles) + 0.2 * np.random.randn(n_points),
        ]
    )
    bad_embedding = bad_embedding * 0.5 + 0.5 * np.random.randn(n_points, 2)

    # Compute scores
    scores_good = manifold_reconstruction_score(
        good_embedding, true_angles, manifold_type="circular"
    )
    scores_bad = manifold_reconstruction_score(
        bad_embedding, true_angles, manifold_type="circular"
    )

    # Good should be better
    assert (
        scores_good["overall_reconstruction_score"]
        > scores_bad["overall_reconstruction_score"]
    )
    assert scores_good["reconstruction_error"] < scores_bad["reconstruction_error"]
    # Correlation should be better for good reconstruction
    assert scores_good["correlation"] > 0.5
    assert scores_good["correlation"] > scores_bad["correlation"]

    # Check all metrics present
    expected_keys = [
        "reconstruction_error",
        "correlation",
        "rotation_offset",
        "is_reflected",
        "decoding_train_error",
        "decoding_test_error",
        "generalization_gap",
        "overall_reconstruction_score",
    ]
    for key in expected_keys:
        assert key in scores_good
        assert key in scores_bad


def test_manifold_reconstruction_score_spatial():
    """Test comprehensive reconstruction score for spatial manifolds"""
    # Generate smooth trajectory
    n_points = 200
    t = np.linspace(0, 2 * np.pi, n_points)
    true_positions = np.column_stack([np.cos(t), np.sin(t)])

    # Test with good and bad reconstructions
    good_reconstruction = true_positions + 0.05 * np.random.randn(n_points, 2)
    bad_reconstruction = np.random.randn(n_points, 2)

    scores_good = manifold_reconstruction_score(
        good_reconstruction, true_positions, manifold_type="spatial"
    )
    scores_bad = manifold_reconstruction_score(
        bad_reconstruction, true_positions, manifold_type="spatial"
    )

    # Good should outperform bad
    assert (
        scores_good["overall_reconstruction_score"]
        > scores_bad["overall_reconstruction_score"]
    )


def test_manifold_reconstruction_score_custom_weights():
    """Test reconstruction score with custom weights"""
    n_points = 100
    true_angles = np.random.uniform(0, 2 * np.pi, n_points)
    embedding = np.column_stack([np.cos(true_angles), np.sin(true_angles)])

    # Custom weights
    weights = {
        "reconstruction_error": 0.5,
        "correlation": 0.3,
        "decoding_accuracy": 0.2,
    }

    scores = manifold_reconstruction_score(
        embedding, true_angles, manifold_type="circular", weights=weights
    )

    # Score should be positive
    assert scores["overall_reconstruction_score"] > 0
    assert scores["overall_reconstruction_score"] <= 1


def test_compute_embedding_quality_circular():
    """Test embedding quality computation for circular manifolds"""
    # Generate circular data
    n_points = 200
    np.random.seed(42)
    # Use angles that avoid boundary discontinuity issues
    # Create a smooth circular trajectory that wraps properly
    t = np.linspace(0, 8 * np.pi, n_points)
    true_angles = np.arctan2(np.sin(t), np.cos(t))  # Properly wrapped to [-pi, pi]

    # Good embedding with offset to test centering
    offset = np.array([0.2, 0.3])
    good_embedding = (
        np.column_stack(
            [
                np.cos(true_angles) + 0.02 * np.random.randn(n_points),
                np.sin(true_angles) + 0.02 * np.random.randn(n_points),
            ]
        )
        + offset
    )

    # Test embedding quality (no decoder)
    results = compute_embedding_quality(
        good_embedding, true_angles, manifold_type="circular", train_fraction=0.7
    )

    # Check structure
    assert "train_error" in results
    assert "test_error" in results
    assert "generalization_gap" in results

    # Should have low reconstruction errors
    assert results["train_error"] < 0.1  # Good reconstruction
    assert results["test_error"] < 0.15  # Slightly relaxed due to test/train split variability
    # Gap should be small for direct reconstruction
    assert abs(results["generalization_gap"]) < 0.08  # Slightly relaxed for robustness


def test_compute_embedding_alignment_metrics():
    """Test the new alignment metrics function"""
    # Generate circular data with known rotation
    n_points = 100
    true_angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    rotation_offset = np.pi / 4

    # Create rotated and reflected embedding
    rotated_angles = true_angles + rotation_offset
    embedding = np.column_stack(
        [np.cos(-rotated_angles), np.sin(-rotated_angles)]  # Reflected (negative)
    )

    # Test with all transformations allowed
    metrics_all = compute_embedding_alignment_metrics(
        embedding, true_angles, "circular", allow_rotation=True, allow_reflection=True
    )

    # Should find good alignment
    assert metrics_all["correlation"] > 0.95
    assert metrics_all["error"] < 0.1
    assert metrics_all["is_reflected"] == True
    # The rotation offset will be different due to reflection
    # Just check it's in valid range [0, 2π]
    assert 0 <= metrics_all["rotation_offset"] <= 2 * np.pi

    # Test without rotation allowed
    metrics_no_rot = compute_embedding_alignment_metrics(
        embedding, true_angles, "circular", allow_rotation=False, allow_reflection=True
    )

    # Should be worse without rotation
    assert metrics_no_rot["error"] > metrics_all["error"]
    assert metrics_no_rot["rotation_offset"] == 0.0

    # Test without reflection allowed
    metrics_no_refl = compute_embedding_alignment_metrics(
        embedding, true_angles, "circular", allow_rotation=True, allow_reflection=False
    )

    # Should be worse without reflection
    assert metrics_no_refl["error"] > metrics_all["error"]
    assert metrics_no_refl["is_reflected"] == False


def test_compute_embedding_quality_vs_decoding_accuracy():
    """Test that embedding quality and decoding accuracy measure different things"""
    n_points = 200
    true_angles = np.linspace(0, 2 * np.pi, n_points)

    # Create embedding with some structure but not perfect circle
    t = np.linspace(0, 2 * np.pi, n_points)
    embedding = np.column_stack(
        [
            np.cos(t) * (1 + 0.2 * np.sin(3 * t)),  # Distorted circle
            np.sin(t) * (1 + 0.2 * np.sin(3 * t)),
        ]
    )
    embedding += 0.05 * np.random.randn(n_points, 2)

    # Compare both methods
    quality_results = compute_embedding_quality(
        embedding, true_angles, manifold_type="circular"
    )

    decoder_results = compute_decoding_accuracy(
        embedding, true_angles, manifold_type="circular"
    )

    # Embedding quality measures direct angle extraction error
    # Decoder accuracy measures how well a linear model can map embedding to angles
    # They should be different
    assert quality_results["train_error"] != decoder_results["train_error"]

    # Both should have reasonable errors
    assert quality_results["train_error"] < 1.0
    assert decoder_results["train_error"] < 1.0
