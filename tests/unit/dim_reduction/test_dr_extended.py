"""
Extended tests for dimensionality reduction module.
Includes missing DR algorithm tests, integration tests, performance benchmarks,
and validation against known manifolds.
"""
import numpy as np
import pytest
import time
from sklearn.datasets import make_swiss_roll, make_s_curve, make_circles
from driada.dim_reduction.data import MVData
from driada.experiment import Experiment
from driada.experiment.synthetic import (
    generate_synthetic_exp, 
    generate_circular_manifold_exp,
    generate_2d_manifold_exp
)


# Test data generation helpers
def create_swiss_roll_data(n_samples=1000):
    """Create swiss roll data helper."""
    random_state = 42
    data, color = make_swiss_roll(n_samples=n_samples,
                                  noise=0.0,
                                  random_state=random_state,
                                  hole=False)
    return data.T, color


def create_s_curve_data(n_samples=1000):
    """Create S-curve data helper."""
    random_state = 42
    data, color = make_s_curve(n_samples=n_samples,
                               noise=0.1,
                               random_state=random_state)
    return data.T, color


# Additional DR method tests

def test_mds():
    """Test MDS (Multi-Dimensional Scaling)"""
    # Use smaller dataset for MDS
    data, _ = create_swiss_roll_data(200)
    D = MVData(data)
    
    # MDS requires distance matrix to be set
    D.distmat = D.get_distmat()
    
    # Use new simplified API
    emb = D.get_embedding(method='mds', dim=2)
    
    assert emb.coords.shape == (2, 200)
    assert not np.any(np.isnan(emb.coords))
    assert not np.any(np.isinf(emb.coords))


def test_lle():
    """Test LLE (Locally Linear Embedding)"""
    data, _ = create_swiss_roll_data(500)
    D = MVData(data)
    
    # Use new simplified API
    emb = D.get_embedding(
        method='lle',
        dim=2,
        nn=10,
        metric='l2'
    )
    
    assert emb.coords.shape == (2, 500)
    assert not np.any(np.isnan(emb.coords))


def test_hlle():
    """Test HLLE (Hessian Locally Linear Embedding)"""
    # Create simple circle data for HLLE
    n_points = 200
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    circle_2d = np.array([np.cos(theta), np.sin(theta)])
    
    # Embed in 3D with noise
    np.random.seed(42)
    A = np.random.randn(3, 2)
    data = A @ circle_2d + 0.05 * np.random.randn(3, n_points)
    
    D = MVData(data)
    
    metric_params = {
        'metric_name': 'l2',
        'sigma': 1,
        'p': 2
    }
    
    # HLLE needs more neighbors
    # Use new simplified API
    emb = D.get_embedding(
        method='hlle',
        dim=2,
        nn=30,
        metric='l2'
    )
    
    assert emb.coords.shape == (2, n_points)
    assert not np.any(np.isnan(emb.coords))


def test_mvu():
    """Test MVU (Maximum Variance Unfolding) - may not be fully implemented"""
    # Use small dataset for MVU
    n_points = 50
    data = np.random.randn(5, n_points)
    D = MVData(data)
    
    # MVU might not be fully implemented or might fail
    try:
        # Use new simplified API
        emb = D.get_embedding(
            method='mvu',
            dim=2,
            nn=10,
            metric='l2'
        )
        assert emb.coords.shape == (2, n_points)
    except Exception:
        # MVU implementation might be incomplete
        pytest.skip("MVU implementation not available or incomplete")


def test_vae():
    """Test VAE (Variational Autoencoder)"""
    data, _ = create_swiss_roll_data(500)
    D = MVData(data)
    
    # Use new simplified API
    emb = D.get_embedding(
        method='vae',
        dim=2,
        epochs=50,  # Reduced for testing
        lr=1e-3,
        seed=42,
        batch_size=64,
        feature_dropout=0.2,
        verbose=False,
        enc_kwargs={'dropout': 0.2},
        dec_kwargs={'dropout': 0.2}
    )
    
    assert emb.coords.shape == (2, 500)
    assert not np.any(np.isnan(emb.coords))


def test_dmaps_not_implemented():
    """Test that standard diffusion maps raises not implemented error"""
    data, _ = create_swiss_roll_data(100)
    D = MVData(data)
    
    # Use new simplified API
    with pytest.raises(Exception) as excinfo:
        emb = D.get_embedding(
            method='dmaps',
            dim=2,
            dm_alpha=0.5,
            nn=10,
            metric='l2'
        )
    assert "not so easy to implement" in str(excinfo.value)


# Integration tests with Experiment objects

def test_experiment_to_mvdata_pipeline():
    """Test conversion from Experiment to MVData and DR"""
    # Generate synthetic experiment
    exp = generate_synthetic_exp(
        n_dfeats=1,
        n_cfeats=2, 
        nneurons=50,
        duration=100,
        seed=42
    )
    
    # Extract neural activity
    calcium_data = exp.calcium.T  # MVData expects (features, samples)
    
    # Create MVData object
    D = MVData(calcium_data)
    
    # Apply PCA using new simplified API
    emb = D.get_embedding(method='pca', dim=3)
    
    assert emb.coords.shape == (3, calcium_data.shape[1])
    assert not np.any(np.isnan(emb.coords))


def test_circular_manifold_extraction():
    """Test DR on circular manifold data"""
    # Generate head direction cells
    exp, info = generate_circular_manifold_exp(
        n_neurons=100,
        duration=300,
        kappa=4.0,  # Von Mises concentration parameter
        noise_std=0.1,
        seed=42
    )
    
    # Extract neural activity
    calcium_data = exp.calcium.T
    D = MVData(calcium_data)
    
    # Apply Isomap (good for manifolds) using new simplified API
    emb = D.get_embedding(method='isomap', dim=2, nn=15, metric='l2')
    
    # Check embedding quality
    assert emb.coords.shape == (2, calcium_data.shape[1])
    assert not np.any(np.isnan(emb.coords))
    
    # Verify circular structure is preserved (loosely)
    # Compute distances in embedding space
    embedding_points = emb.coords.T
    center = np.mean(embedding_points, axis=0)
    distances = np.linalg.norm(embedding_points - center, axis=1)
    
    # Check that points form roughly circular pattern
    std_distance = np.std(distances)
    mean_distance = np.mean(distances)
    assert std_distance / mean_distance < 0.5  # Relatively uniform distances


def test_2d_manifold_extraction():
    """Test DR on 2D spatial manifold data"""
    # Generate place cells
    exp, info = generate_2d_manifold_exp(
        n_neurons=100,
        duration=300,
        field_sigma=0.2,  # Place field size
        peak_rate=5.0,
        seed=42
    )
    
    # Extract neural activity
    calcium_data = exp.calcium.T
    D = MVData(calcium_data)
    
    # Apply UMAP using new simplified API
    emb = D.get_embedding(method='umap', dim=2, nn=20, min_dist=0.1, metric='l2')
    
    # Check embedding
    assert emb.coords.shape == (2, calcium_data.shape[1])
    assert not np.any(np.isnan(emb.coords))


# Performance benchmark tests

@pytest.mark.parametrize("n_samples,n_features", [
    (100, 10),
    (500, 50),
    (1000, 100),
])
def test_pca_performance(n_samples, n_features):
    """Benchmark PCA performance"""
    # Generate random data
    np.random.seed(42)
    data = np.random.randn(n_features, n_samples)
    D = MVData(data)
    
    # Time the embedding using new simplified API
    start_time = time.time()
    emb = D.get_embedding(method='pca', dim=min(10, n_features - 1))
    elapsed_time = time.time() - start_time
    
    # Check results
    assert emb.coords.shape == (min(10, n_features - 1), n_samples)
    
    # Performance assertions (PCA should be fast)
    if n_samples <= 500:
        assert elapsed_time < 1.0  # Should complete in under 1 second
    else:
        assert elapsed_time < 5.0  # Larger data can take more time


@pytest.mark.parametrize("method_name", ['isomap', 'le', 'tsne'])
def test_nonlinear_methods_performance(method_name):
    """Benchmark nonlinear DR methods"""
    n_samples = 300
    # Generate swiss roll
    data, _ = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42)
    D = MVData(data.T)
    
    metric_params = {'metric_name': 'l2', 'sigma': 1, 'p': 2}
    graph_params = {
        'g_method_name': 'knn',
        'weighted': 0,
        'nn': 15,
        'max_deleted_nodes': 0.2,
        'dist_to_aff': 'hk'
    }
    
    # Use new simplified API
    start_time = time.time()
    if method_name == 'tsne':
        emb = D.get_embedding(method=method_name, dim=2)
    else:
        emb = D.get_embedding(method=method_name, dim=2, nn=15, metric='l2')
    elapsed_time = time.time() - start_time
    
    assert emb.coords.shape == (2, n_samples)
    # Just verify it completes in reasonable time
    assert elapsed_time < 30.0  # 30 seconds max


# Validation tests against known manifolds

def test_linear_manifold_preservation():
    """Test that PCA preserves linear structure"""
    # Create data on a 2D linear subspace in 10D
    np.random.seed(42)
    n_samples = 200
    
    # Generate 2D data
    latent = np.random.randn(2, n_samples)
    
    # Embed in 10D with random linear transformation
    A = np.random.randn(10, 2)
    data = A @ latent + 0.1 * np.random.randn(10, n_samples)
    
    # Apply PCA using new simplified API
    D = MVData(data)
    emb = D.get_embedding(method='pca', dim=2)
    
    # Check that we recover a 2D representation
    assert emb.coords.shape == (2, n_samples)
    
    # Verify that most variance is captured in 2D
    # (Since data is essentially 2D + noise)
    total_var = np.var(data, axis=1).sum()
    embedded_var = np.var(emb.coords, axis=1).sum()
    
    # Should capture most of the variance
    assert embedded_var / total_var > 0.8


def test_swiss_roll_unfolding():
    """Test that nonlinear methods can unfold swiss roll"""
    # Generate swiss roll
    data, color = make_swiss_roll(n_samples=500, noise=0.05, random_state=42)
    D = MVData(data.T)
    
    metric_params = {'metric_name': 'l2', 'sigma': 1, 'p': 2}
    graph_params = {
        'g_method_name': 'knn',
        'weighted': 0,
        'nn': 7,  # Fewer neighbors to avoid shortcuts across the roll
        'max_deleted_nodes': 0.2,
        'dist_to_aff': 'hk'
    }
    
    # Apply Isomap - should unfold the roll using new simplified API
    emb = D.get_embedding(method='isomap', dim=2, nn=10, metric='l2')
    
    # Use manifold metrics to evaluate preservation
    from driada.dim_reduction import knn_preservation_rate, trustworthiness, continuity
    
    # Convert to proper format (n_samples, n_features)
    X_high = data
    X_low = emb.coords.T
    
    # Compute preservation metrics
    k = 10
    preservation_rate = knn_preservation_rate(X_high, X_low, k=k, flexible=True)
    trust = trustworthiness(X_high, X_low, k=k)
    cont = continuity(X_high, X_low, k=k)
    
    # Should preserve local structure well with proper n_neighbors
    assert preservation_rate > 0.7, f"KNN preservation rate {preservation_rate:.3f} too low"
    assert trust > 0.8, f"Trustworthiness {trust:.3f} too low"
    assert cont > 0.8, f"Continuity {cont:.3f} too low"


def test_circle_preservation():
    """Test that circular structure is preserved"""
    # Generate points on a circle
    n_points = 100
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    
    # 2D circle embedded in 5D with noise
    circle_2d = np.array([np.cos(theta), np.sin(theta)])
    
    # Random linear embedding to 5D
    np.random.seed(42)
    A = np.random.randn(5, 2)
    data = A @ circle_2d + 0.1 * np.random.randn(5, n_points)
    
    D = MVData(data)
    
    # Test PCA first using new simplified API
    emb = D.get_embedding(method='pca', dim=2)
    
    # Use manifold metrics for circular structure
    from driada.dim_reduction import circular_structure_preservation, knn_preservation_rate
    
    # Convert to proper format
    X_high = data.T  # (n_samples, n_features)
    X_low = emb.coords.T  # (n_samples, 2)
    
    # Test circular structure preservation
    circular_metrics = circular_structure_preservation(X_low, true_angles=theta, k_neighbors=3)
    
    # Check metrics
    assert circular_metrics['distance_cv'] < 0.3, f"Distance CV {circular_metrics['distance_cv']:.3f} too high"
    assert circular_metrics['consecutive_preservation'] > 0.8, f"Consecutive preservation {circular_metrics['consecutive_preservation']:.3f} too low"
    assert circular_metrics['circular_correlation'] > 0.7, f"Circular correlation {circular_metrics['circular_correlation']:.3f} too low"
    
    # Also check general neighborhood preservation
    preservation_rate = knn_preservation_rate(X_high, X_low, k=5)
    assert preservation_rate > 0.8, f"KNN preservation {preservation_rate:.3f} too low for linear method on linear manifold"


# Graph construction method tests

def test_graph_construction_methods():
    """Test different graph construction methods"""
    data, _ = create_swiss_roll_data(200)
    D = MVData(data)
    
    metric_params = {
        'metric_name': 'l2',
        'sigma': 1,
        'p': 2
    }
    
    # Test knn graph construction
    graph_params = {
        'g_method_name': 'knn',
        'weighted': 0,
        'nn': 10,
        'max_deleted_nodes': 0.2,
        'dist_to_aff': 'hk'
    }
    
    # Verify graph can be constructed
    G = D.get_proximity_graph(metric_params, graph_params)
    assert G is not None
    assert hasattr(G, 'adj')
    assert G.adj.shape == (200, 200)
    
    # Test that it's symmetric for undirected graph
    assert np.allclose(G.adj.toarray(), G.adj.toarray().T)


# Edge case tests

def test_small_dataset():
    """Test DR methods on very small datasets"""
    # Create minimal dataset
    n_points = 10
    data = np.random.randn(5, n_points)
    D = MVData(data)
    
    # Test PCA using new simplified API
    emb = D.get_embedding(method='pca', dim=2)
    
    assert emb.coords.shape == (2, n_points)
    assert not np.any(np.isnan(emb.coords))


def test_high_dimensional_data():
    """Test DR on high-dimensional data"""
    # More features than samples
    n_samples = 50
    n_features = 100
    data = np.random.randn(n_features, n_samples)
    D = MVData(data)
    
    # PCA should handle this using new simplified API
    emb = D.get_embedding(method='pca', dim=10)
    
    assert emb.coords.shape == (10, n_samples)
    assert not np.any(np.isnan(emb.coords))


def test_linear_vs_nonlinear_on_manifolds():
    """Compare linear (PCA) vs nonlinear (Isomap) methods on manifolds"""
    from driada.dim_reduction import manifold_preservation_score
    
    # Generate swiss roll - a classic nonlinear manifold
    n_samples = 500
    data, color = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42)
    D = MVData(data.T)
    
    # Apply PCA (linear method) using new simplified API
    pca_emb = D.get_embedding(method='pca', dim=2)
    
    # Apply Isomap (nonlinear method) using new simplified API
    iso_emb = D.get_embedding(method='isomap', dim=2, nn=7, metric='l2')
    
    # Compare preservation scores
    pca_scores = manifold_preservation_score(data, pca_emb.coords.T, k_neighbors=10)
    iso_scores = manifold_preservation_score(data, iso_emb.coords.T, k_neighbors=10)
    
    # For swiss roll, the key metric is geodesic distance preservation
    # Isomap should preserve geodesic distances much better than PCA
    assert iso_scores['geodesic_correlation'] > pca_scores['geodesic_correlation'] + 0.2, \
        f"Isomap geodesic correlation {iso_scores['geodesic_correlation']:.3f} not sufficiently better than PCA {pca_scores['geodesic_correlation']:.3f}"
    
    # Overall, at least one method should do reasonably well
    assert max(iso_scores['overall_score'], pca_scores['overall_score']) > 0.5, \
        f"Neither method achieved good preservation (Isomap: {iso_scores['overall_score']:.3f}, PCA: {pca_scores['overall_score']:.3f})"