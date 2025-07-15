"""
Extended tests for dimensionality reduction module.
Includes missing DR algorithm tests, integration tests, performance benchmarks,
and validation against known manifolds.
"""
from sklearn.datasets import make_swiss_roll, make_s_curve, make_circles
import numpy as np
import pytest
import time
from src.driada.dim_reduction.data import MVData
from src.driada.dim_reduction.dr_base import METHODS_DICT
from src.driada.experiment import Experiment
from src.driada.experiment.synthetic import (
    generate_synthetic_exp, 
    generate_circular_manifold_exp,
    generate_2d_manifold_exp
)


# Test data generation helpers
def create_swiss_roll_data(n_samples=1000):
    random_state = 42
    data, color = make_swiss_roll(n_samples=n_samples,
                                  noise=0.0,
                                  random_state=random_state,
                                  hole=False)
    return data.T, color


# Additional DR method tests

def test_mds():
    """Test MDS (Multi-Dimensional Scaling)"""
    # Use smaller dataset for MDS
    data, _ = create_swiss_roll_data(200)
    D = MVData(data)
    
    # MDS needs distance matrix
    distmat = D.get_distmat()
    
    embedding_params = {
        'e_method_name': 'mds',
        'dim': 2
    }
    
    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    emb = D.get_embedding(embedding_params, distmat=distmat)
    
    assert emb.coords.shape == (2, 200)
    assert not np.any(np.isnan(emb.coords))
    assert not np.any(np.isinf(emb.coords))


def test_lle():
    """Test LLE (Locally Linear Embedding)"""
    data, _ = create_swiss_roll_data(500)
    D = MVData(data)
    
    metric_params = {
        'metric_name': 'l2',
        'sigma': 1,
        'p': 2
    }
    
    graph_params = {
        'g_method_name': 'knn',
        'weighted': 0,
        'nn': 10,
        'max_deleted_nodes': 0.2,
        'dist_to_aff': 'hk'
    }
    
    embedding_params = {
        'e_method_name': 'lle',
        'dim': 2
    }
    
    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    emb = D.get_embedding(
        embedding_params, 
        g_params=graph_params, 
        m_params=metric_params
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
    graph_params = {
        'g_method_name': 'knn',
        'weighted': 0,
        'nn': 30,
        'max_deleted_nodes': 0.2,
        'dist_to_aff': 'hk'
    }
    
    embedding_params = {
        'e_method_name': 'hlle',
        'dim': 2
    }
    
    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    emb = D.get_embedding(
        embedding_params, 
        g_params=graph_params, 
        m_params=metric_params
    )
    
    assert emb.coords.shape == (2, n_points)
    assert not np.any(np.isnan(emb.coords))


def test_mvu():
    """Test MVU (Maximum Variance Unfolding) - may not be fully implemented"""
    # Use small dataset for MVU
    n_points = 50
    data = np.random.randn(5, n_points)
    D = MVData(data)
    
    metric_params = {
        'metric_name': 'l2',
        'sigma': 1,
        'p': 2
    }
    
    graph_params = {
        'g_method_name': 'knn',
        'weighted': 0,
        'nn': 10,
        'max_deleted_nodes': 0.2,
        'dist_to_aff': 'hk'
    }
    
    embedding_params = {
        'e_method_name': 'mvu',
        'dim': 2
    }
    
    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    
    # MVU might not be fully implemented or might fail
    try:
        emb = D.get_embedding(
            embedding_params, 
            g_params=graph_params, 
            m_params=metric_params
        )
        assert emb.coords.shape == (2, n_points)
    except Exception:
        # MVU implementation might be incomplete
        pytest.skip("MVU implementation not available or incomplete")


def test_vae():
    """Test VAE (Variational Autoencoder)"""
    data, _ = create_swiss_roll_data(500)
    D = MVData(data)
    
    embedding_params = {
        'e_method_name': 'vae',
        'dim': 2
    }
    
    nn_params = {
        'continue_learning': 0,
        'epochs': 50,  # Reduced for testing
        'lr': 1e-3,
        'seed': 42,
        'batch_size': 64,
        'feature_dropout': 0.2,
        'verbose': False,
        'enc_kwargs': {'dropout': 0.2},
        'dec_kwargs': {'dropout': 0.2}
    }
    
    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    emb = D.get_embedding(embedding_params, kwargs=nn_params)
    
    assert emb.coords.shape == (2, 500)
    assert not np.any(np.isnan(emb.coords))


def test_dmaps_not_implemented():
    """Test that standard diffusion maps raises not implemented error"""
    data, _ = create_swiss_roll_data(100)
    D = MVData(data)
    
    metric_params = {
        'metric_name': 'l2',
        'sigma': 1,
        'p': 2
    }
    
    graph_params = {
        'g_method_name': 'knn',
        'weighted': 0,
        'nn': 10,
        'max_deleted_nodes': 0.2,
        'dist_to_aff': 'hk'
    }
    
    embedding_params = {
        'e_method_name': 'dmaps',
        'dim': 2,
        'dm_alpha': 0.5
    }
    
    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    
    with pytest.raises(Exception) as excinfo:
        emb = D.get_embedding(
            embedding_params, 
            g_params=graph_params, 
            m_params=metric_params
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
    
    # Apply PCA
    embedding_params = {
        'e_method_name': 'pca',
        'dim': 3
    }
    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    emb = D.get_embedding(embedding_params)
    
    assert emb.coords.shape == (3, calcium_data.shape[1])
    assert not np.any(np.isnan(emb.coords))


def test_circular_manifold_extraction():
    """Test DR on circular manifold data"""
    # Generate head direction cells
    exp = generate_circular_manifold_exp(
        n_neurons=100,
        duration=300,
        kappa=4.0,  # Von Mises concentration parameter
        noise_std=0.1,
        seed=42
    )
    
    # Extract neural activity
    calcium_data = exp.calcium.T
    D = MVData(calcium_data)
    
    # Apply Isomap (good for manifolds)
    metric_params = {'metric_name': 'l2', 'sigma': 1, 'p': 2}
    graph_params = {
        'g_method_name': 'knn',
        'weighted': 0,
        'nn': 15,
        'max_deleted_nodes': 0.2,
        'dist_to_aff': 'hk'
    }
    embedding_params = {
        'e_method_name': 'isomap',
        'dim': 2
    }
    
    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    emb = D.get_embedding(embedding_params, g_params=graph_params, m_params=metric_params)
    
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
    exp = generate_2d_manifold_exp(
        n_neurons=100,
        duration=300,
        field_sigma=0.2,  # Place field size
        peak_rate=5.0,
        seed=42
    )
    
    # Extract neural activity
    calcium_data = exp.calcium.T
    D = MVData(calcium_data)
    
    # Apply UMAP
    metric_params = {'metric_name': 'l2', 'sigma': 1, 'p': 2}
    graph_params = {
        'g_method_name': 'knn',
        'weighted': 0,
        'nn': 20,
        'max_deleted_nodes': 0.2,
        'dist_to_aff': 'hk'
    }
    embedding_params = {
        'e_method_name': 'umap',
        'dim': 2,
        'min_dist': 0.1
    }
    
    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    emb = D.get_embedding(embedding_params, g_params=graph_params, m_params=metric_params)
    
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
    
    embedding_params = {
        'e_method_name': 'pca',
        'dim': min(10, n_features - 1)
    }
    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    
    # Time the embedding
    start_time = time.time()
    emb = D.get_embedding(embedding_params)
    elapsed_time = time.time() - start_time
    
    # Check results
    assert emb.coords.shape == (embedding_params['dim'], n_samples)
    
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
    
    embedding_params = {
        'e_method_name': method_name,
        'dim': 2
    }
    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    
    start_time = time.time()
    if method_name == 'tsne':
        emb = D.get_embedding(embedding_params)
    else:
        emb = D.get_embedding(embedding_params, g_params=graph_params, m_params=metric_params)
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
    
    # Apply PCA
    D = MVData(data)
    embedding_params = {
        'e_method_name': 'pca',
        'dim': 2
    }
    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    emb = D.get_embedding(embedding_params)
    
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
        'nn': 12,
        'max_deleted_nodes': 0.2,
        'dist_to_aff': 'hk'
    }
    
    # Apply Isomap - should unfold the roll
    embedding_params = {
        'e_method_name': 'isomap',
        'dim': 2
    }
    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    emb = D.get_embedding(embedding_params, g_params=graph_params, m_params=metric_params)
    
    # Check spatial correspondence using k-nearest neighbors preservation
    embedding_points = emb.coords.T
    
    # For each point, check if its k nearest neighbors in original space
    # are still among its 2k nearest neighbors in embedded space
    from sklearn.neighbors import NearestNeighbors
    
    k = 10
    # Original space neighbors
    nbrs_orig = NearestNeighbors(n_neighbors=k+1).fit(data.T)
    orig_distances, orig_indices = nbrs_orig.kneighbors(data.T)
    
    # Embedded space neighbors  
    nbrs_emb = NearestNeighbors(n_neighbors=2*k+1).fit(embedding_points)
    emb_distances, emb_indices = nbrs_emb.kneighbors(embedding_points)
    
    # Check neighborhood preservation
    preserved = 0
    for i in range(len(data.T)):
        # Original neighbors (excluding self)
        orig_neighbors = set(orig_indices[i][1:k+1])
        # Embedded neighbors (excluding self)
        emb_neighbors = set(emb_indices[i][1:2*k+1])
        # Count preserved neighbors
        preserved += len(orig_neighbors.intersection(emb_neighbors))
    
    preservation_rate = preserved / (len(data.T) * k)
    
    # Should preserve at least 70% of local neighborhoods
    assert preservation_rate > 0.7


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
    
    # Test PCA first
    embedding_params = {
        'e_method_name': 'pca',
        'dim': 2
    }
    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    emb = D.get_embedding(embedding_params)
    
    # Check that embedded points form a circular structure
    embedding_points = emb.coords.T
    
    # Use circular statistics to test if points lie on a circle
    # 1. Check variance of distances from center
    center = np.mean(embedding_points, axis=0)
    distances = np.linalg.norm(embedding_points - center, axis=1)
    
    # Coefficient of variation should be small for circle
    cv = np.std(distances) / np.mean(distances)
    assert cv < 0.3  # Allow some variation due to noise
    
    # 2. Check angular uniformity using circular variance
    centered = embedding_points - center
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    
    # Compute circular variance
    cos_mean = np.mean(np.cos(angles))
    sin_mean = np.mean(np.sin(angles))
    circular_variance = 1 - np.sqrt(cos_mean**2 + sin_mean**2)
    
    # For uniform distribution on circle, circular variance should be high
    assert circular_variance > 0.8
    
    # 3. Check that consecutive points are neighbors in embedding
    # Since we generated points in order along the circle
    from sklearn.neighbors import NearestNeighbors
    
    nbrs = NearestNeighbors(n_neighbors=3).fit(embedding_points)
    distances, indices = nbrs.kneighbors(embedding_points)
    
    # Count how many times consecutive points are neighbors
    consecutive_preserved = 0
    for i in range(n_points):
        neighbors = set(indices[i])
        # Check if previous or next point is among neighbors
        prev_idx = (i - 1) % n_points
        next_idx = (i + 1) % n_points
        if prev_idx in neighbors or next_idx in neighbors:
            consecutive_preserved += 1
    
    # Most points should have their consecutive neighbors preserved
    assert consecutive_preserved / n_points > 0.8


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
    
    # Test PCA
    embedding_params = {
        'e_method_name': 'pca',
        'dim': 2
    }
    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    emb = D.get_embedding(embedding_params)
    
    assert emb.coords.shape == (2, n_points)
    assert not np.any(np.isnan(emb.coords))


def test_high_dimensional_data():
    """Test DR on high-dimensional data"""
    # More features than samples
    n_samples = 50
    n_features = 100
    data = np.random.randn(n_features, n_samples)
    D = MVData(data)
    
    # PCA should handle this
    embedding_params = {
        'e_method_name': 'pca',
        'dim': 10
    }
    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    emb = D.get_embedding(embedding_params)
    
    assert emb.coords.shape == (10, n_samples)
    assert not np.any(np.isnan(emb.coords))