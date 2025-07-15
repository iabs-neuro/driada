"""
Spatial Correspondence Metrics for Manifold Preservation Evaluation

This module provides comprehensive metrics for evaluating how well dimensionality
reduction methods preserve manifold structure.

Key metric categories:
---------------------
1. Neighborhood preservation: How well are local neighborhoods preserved?
2. Distance preservation: How well are geodesic distances preserved?
3. Topology preservation: How well is the global structure preserved?
4. Shape matching: How similar are the shapes after optimal alignment?
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from scipy.linalg import orthogonal_procrustes
from typing import Optional, Tuple, Union


def compute_distance_matrix(X: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """
    Compute pairwise distance matrix.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features)
    metric : str
        Distance metric to use (default: 'euclidean')
        
    Returns
    -------
    np.ndarray
        Symmetric distance matrix of shape (n_samples, n_samples)
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape {X.shape}")
    
    distances = pdist(X, metric=metric)
    return squareform(distances)


def knn_preservation_rate(
    X_high: np.ndarray, 
    X_low: np.ndarray, 
    k: int = 10,
    flexible: bool = False,
    flexibility_factor: float = 2.0
) -> float:
    """
    Compute k-nearest neighbor preservation rate.
    
    This metric measures what fraction of k nearest neighbors in the original
    high-dimensional space are preserved in the low-dimensional embedding.
    
    Parameters
    ----------
    X_high : np.ndarray
        Original high-dimensional data (n_samples, n_features_high)
    X_low : np.ndarray
        Low-dimensional embedding (n_samples, n_features_low)
    k : int
        Number of nearest neighbors to consider
    flexible : bool
        If True, check if k-NN are within (k * flexibility_factor)-NN in embedding
    flexibility_factor : float
        Factor to multiply k for flexible matching (default: 2.0)
        
    Returns
    -------
    float
        Preservation rate between 0 and 1
    """
    if X_high.shape[0] != X_low.shape[0]:
        raise ValueError("X_high and X_low must have same number of samples")
    
    n_samples = X_high.shape[0]
    if k >= n_samples:
        raise ValueError(f"k={k} must be less than n_samples={n_samples}")
    
    # Find k-NN in original space
    nbrs_high = NearestNeighbors(n_neighbors=k+1).fit(X_high)
    _, indices_high = nbrs_high.kneighbors(X_high)
    indices_high = indices_high[:, 1:]  # Remove self
    
    # Find k-NN (or flexible k-NN) in embedded space
    k_low = int(k * flexibility_factor) if flexible else k
    nbrs_low = NearestNeighbors(n_neighbors=min(k_low+1, n_samples)).fit(X_low)
    _, indices_low = nbrs_low.kneighbors(X_low)
    indices_low = indices_low[:, 1:]  # Remove self
    
    # Count preserved neighbors
    preserved = 0
    for i in range(n_samples):
        high_neighbors = set(indices_high[i])
        low_neighbors = set(indices_low[i][:k_low])
        preserved += len(high_neighbors.intersection(low_neighbors))
    
    return preserved / (n_samples * k)


def trustworthiness(
    X_high: np.ndarray,
    X_low: np.ndarray,
    k: int = 10
) -> float:
    """
    Compute trustworthiness of the embedding.
    
    Trustworthiness measures how much we can trust that points nearby in the
    embedding are truly neighbors in the original space.
    
    Parameters
    ----------
    X_high : np.ndarray
        Original high-dimensional data (n_samples, n_features_high)
    X_low : np.ndarray
        Low-dimensional embedding (n_samples, n_features_low)
    k : int
        Number of nearest neighbors to consider
        
    Returns
    -------
    float
        Trustworthiness score between 0 and 1
    """
    if X_high.shape[0] != X_low.shape[0]:
        raise ValueError("X_high and X_low must have same number of samples")
        
    n_samples = X_high.shape[0]
    if k >= n_samples:
        raise ValueError(f"k={k} must be less than n_samples={n_samples}")
    
    # Compute distance matrices
    dist_high = compute_distance_matrix(X_high)
    dist_low = compute_distance_matrix(X_low)
    
    # Get k-NN in embedded space
    nbrs_low = NearestNeighbors(n_neighbors=k+1).fit(X_low)
    _, indices_low = nbrs_low.kneighbors(X_low)
    indices_low = indices_low[:, 1:]  # Remove self
    
    # Compute ranks in original space
    ranks_high = np.argsort(np.argsort(dist_high, axis=1), axis=1)
    
    # Compute trustworthiness
    trust = 0.0
    for i in range(n_samples):
        for j in indices_low[i]:
            rank = ranks_high[i, j]
            if rank > k:
                trust += (rank - k)
    
    # Normalize
    max_trust = (n_samples - k - 1) * k * n_samples / 2
    if max_trust > 0:
        trust = 1 - (2 * trust / max_trust)
    else:
        trust = 1.0
        
    return trust


def continuity(
    X_high: np.ndarray,
    X_low: np.ndarray,
    k: int = 10
) -> float:
    """
    Compute continuity of the embedding.
    
    Continuity measures how well the embedding preserves the neighborhoods
    from the original space.
    
    Parameters
    ----------
    X_high : np.ndarray
        Original high-dimensional data (n_samples, n_features_high)
    X_low : np.ndarray
        Low-dimensional embedding (n_samples, n_features_low)
    k : int
        Number of nearest neighbors to consider
        
    Returns
    -------
    float
        Continuity score between 0 and 1
    """
    if X_high.shape[0] != X_low.shape[0]:
        raise ValueError("X_high and X_low must have same number of samples")
        
    n_samples = X_high.shape[0]
    if k >= n_samples:
        raise ValueError(f"k={k} must be less than n_samples={n_samples}")
    
    # Compute distance matrices
    dist_high = compute_distance_matrix(X_high)
    dist_low = compute_distance_matrix(X_low)
    
    # Get k-NN in original space
    nbrs_high = NearestNeighbors(n_neighbors=k+1).fit(X_high)
    _, indices_high = nbrs_high.kneighbors(X_high)
    indices_high = indices_high[:, 1:]  # Remove self
    
    # Compute ranks in embedded space
    ranks_low = np.argsort(np.argsort(dist_low, axis=1), axis=1)
    
    # Compute continuity
    cont = 0.0
    for i in range(n_samples):
        for j in indices_high[i]:
            rank = ranks_low[i, j]
            if rank > k:
                cont += (rank - k)
    
    # Normalize
    max_cont = (n_samples - k - 1) * k * n_samples / 2
    if max_cont > 0:
        cont = 1 - (2 * cont / max_cont)
    else:
        cont = 1.0
        
    return cont


def geodesic_distance_correlation(
    X_high: np.ndarray,
    X_low: np.ndarray,
    k_neighbors: int = 10,
    method: str = 'spearman'
) -> float:
    """
    Compute correlation between geodesic distances on the manifold and
    Euclidean distances in the embedding.
    
    Uses k-NN graph to approximate geodesic distances via shortest paths.
    
    Parameters
    ----------
    X_high : np.ndarray
        Original high-dimensional data (n_samples, n_features_high)
    X_low : np.ndarray
        Low-dimensional embedding (n_samples, n_features_low)
    k_neighbors : int
        Number of neighbors for graph construction
    method : str
        Correlation method ('spearman' or 'pearson')
        
    Returns
    -------
    float
        Correlation coefficient between -1 and 1
    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import shortest_path
    
    # Build k-NN graph for geodesic approximation
    graph = kneighbors_graph(X_high, n_neighbors=k_neighbors, mode='distance')
    
    # Compute geodesic distances via shortest paths
    geodesic_dist = shortest_path(graph, directed=False)
    
    # Handle disconnected components
    if np.any(np.isinf(geodesic_dist)):
        # Use only finite distances
        mask = np.isfinite(geodesic_dist)
        geodesic_flat = geodesic_dist[mask]
    else:
        geodesic_flat = geodesic_dist[np.triu_indices_from(geodesic_dist, k=1)]
    
    # Compute Euclidean distances in embedding
    euclidean_dist = compute_distance_matrix(X_low)
    
    if np.any(np.isinf(geodesic_dist)):
        euclidean_flat = euclidean_dist[mask]
    else:
        euclidean_flat = euclidean_dist[np.triu_indices_from(euclidean_dist, k=1)]
    
    # Compute correlation
    if method == 'spearman':
        corr, _ = spearmanr(geodesic_flat, euclidean_flat)
    else:  # pearson
        corr = np.corrcoef(geodesic_flat, euclidean_flat)[0, 1]
    
    return corr


def stress(
    X_high: np.ndarray,
    X_low: np.ndarray,
    normalized: bool = True
) -> float:
    """
    Compute stress (sum of squared differences in distances).
    
    Parameters
    ----------
    X_high : np.ndarray
        Original high-dimensional data (n_samples, n_features_high)
    X_low : np.ndarray
        Low-dimensional embedding (n_samples, n_features_low)
    normalized : bool
        If True, normalize by sum of squared distances
        
    Returns
    -------
    float
        Stress value (lower is better)
    """
    # Compute distance matrices
    dist_high = compute_distance_matrix(X_high)
    dist_low = compute_distance_matrix(X_low)
    
    # Compute stress
    diff = dist_high - dist_low
    stress_val = np.sum(diff ** 2)
    
    if normalized:
        stress_val /= np.sum(dist_high ** 2)
    
    return stress_val


def circular_structure_preservation(
    X_low: np.ndarray,
    true_angles: Optional[np.ndarray] = None,
    k_neighbors: int = 3
) -> dict:
    """
    Evaluate preservation of circular structure in embedding.
    
    Parameters
    ----------
    X_low : np.ndarray
        Low-dimensional embedding (n_samples, 2)
    true_angles : np.ndarray, optional
        True angles if known (for synthetic data)
    k_neighbors : int
        Number of neighbors for consecutive preservation
        
    Returns
    -------
    dict
        Dictionary containing various circular preservation metrics
    """
    if X_low.shape[1] != 2:
        raise ValueError("Circular analysis requires 2D embedding")
    
    n_samples = X_low.shape[0]
    
    # Center the embedding
    center = np.mean(X_low, axis=0)
    centered = X_low - center
    
    # Compute distances from center
    distances = np.linalg.norm(centered, axis=1)
    
    # Coefficient of variation of distances (should be small for circle)
    cv_distances = np.std(distances) / np.mean(distances)
    
    # Compute angles
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    
    # Sort by angle to check consecutive preservation
    angle_order = np.argsort(angles)
    
    # Check consecutive neighbor preservation
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1).fit(X_low)
    _, indices = nbrs.kneighbors(X_low)
    
    consecutive_preserved = 0
    for i in range(n_samples):
        pos_in_order = np.where(angle_order == i)[0][0]
        prev_idx = angle_order[(pos_in_order - 1) % n_samples]
        next_idx = angle_order[(pos_in_order + 1) % n_samples]
        
        neighbors = set(indices[i, 1:])  # Exclude self
        if prev_idx in neighbors or next_idx in neighbors:
            consecutive_preserved += 1
    
    results = {
        'distance_cv': cv_distances,
        'consecutive_preservation': consecutive_preserved / n_samples
    }
    
    # If true angles provided, compute angular correlation
    if true_angles is not None:
        # Unwrap angles to handle discontinuity
        angle_diff = angles - true_angles
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        
        # Circular correlation
        circular_corr = 1 - np.mean(np.abs(angle_diff)) / np.pi
        results['circular_correlation'] = circular_corr
        
    return results


def procrustes_analysis(
    X: np.ndarray,
    Y: np.ndarray,
    scaling: bool = True,
    reflection: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Perform Procrustes analysis to find optimal alignment.
    
    Parameters
    ----------
    X : np.ndarray
        Reference configuration (n_samples, n_features)
    Y : np.ndarray
        Configuration to be aligned (n_samples, n_features)
    scaling : bool
        Whether to allow scaling
    reflection : bool
        Whether to allow reflection
        
    Returns
    -------
    Y_aligned : np.ndarray
        Aligned version of Y
    disparity : float
        Procrustes distance after alignment
    """
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")
    
    # Center configurations
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    
    # Compute optimal rotation
    R, scale = orthogonal_procrustes(Y_centered, X_centered)
    
    # Apply transformation
    Y_aligned = Y_centered @ R
    
    if scaling:
        # Compute optimal scaling
        norm_X = np.linalg.norm(X_centered)
        norm_Y = np.linalg.norm(Y_aligned)
        if norm_Y > 0:
            scale_factor = norm_X / norm_Y
            Y_aligned *= scale_factor
    
    if not reflection:
        # Check if R includes reflection
        if np.linalg.det(R) < 0:
            # Remove reflection by flipping one axis
            Y_aligned[:, -1] *= -1
    
    # Compute disparity
    disparity = np.sqrt(np.sum((X_centered - Y_aligned) ** 2))
    
    # Return aligned points (with original center)
    Y_aligned += np.mean(X, axis=0)
    
    return Y_aligned, disparity


def manifold_preservation_score(
    X_high: np.ndarray,
    X_low: np.ndarray,
    k_neighbors: int = 10,
    weights: Optional[dict] = None
) -> dict:
    """
    Compute comprehensive manifold preservation score.
    
    Combines multiple metrics into an overall assessment of how well
    the embedding preserves manifold structure.
    
    Parameters
    ----------
    X_high : np.ndarray
        Original high-dimensional data (n_samples, n_features_high)
    X_low : np.ndarray
        Low-dimensional embedding (n_samples, n_features_low)
    k_neighbors : int
        Number of neighbors for local metrics
    weights : dict, optional
        Weights for combining metrics (default: equal weights)
        
    Returns
    -------
    dict
        Dictionary containing individual metrics and overall score
    """
    if weights is None:
        weights = {
            'knn_preservation': 0.25,
            'trustworthiness': 0.25,
            'continuity': 0.25,
            'geodesic_correlation': 0.25
        }
    
    # Compute individual metrics
    metrics = {
        'knn_preservation': knn_preservation_rate(X_high, X_low, k=k_neighbors),
        'trustworthiness': trustworthiness(X_high, X_low, k=k_neighbors),
        'continuity': continuity(X_high, X_low, k=k_neighbors),
        'geodesic_correlation': geodesic_distance_correlation(
            X_high, X_low, k_neighbors=k_neighbors
        )
    }
    
    # Handle potential NaN in geodesic correlation
    if np.isnan(metrics['geodesic_correlation']):
        metrics['geodesic_correlation'] = 0.0
    
    # Compute weighted average
    overall_score = sum(
        metrics[key] * weights.get(key, 0)
        for key in metrics
    )
    
    metrics['overall_score'] = overall_score
    
    return metrics


# =============================================================================
# MANIFOLD RECONSTRUCTION VALIDATION
# =============================================================================

def circular_distance(angles1: np.ndarray, angles2: np.ndarray) -> np.ndarray:
    """Compute circular distance between two sets of angles
    
    Parameters:
    -----------
    angles1, angles2 : np.ndarray
        Arrays of angles in radians
        
    Returns:
    --------
    np.ndarray
        Circular distances between corresponding angles
    """
    diff = angles1 - angles2
    return np.abs(np.arctan2(np.sin(diff), np.cos(diff)))


def extract_angles_from_embedding(embedding: np.ndarray) -> np.ndarray:
    """Extract angular information from 2D embedding
    
    Parameters:
    -----------
    embedding : np.ndarray
        2D embedding with shape (n_timepoints, 2)
        
    Returns:
    --------
    np.ndarray
        Extracted angles in radians
    """
    if embedding.shape[1] != 2:
        raise ValueError("Embedding must be 2D for angle extraction")
    
    # Center the embedding
    centered = embedding - np.mean(embedding, axis=0)
    
    # Extract angles
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    
    return angles


def compute_reconstruction_error(
    embedding: np.ndarray, 
    true_variable: np.ndarray, 
    manifold_type: str = 'circular'
) -> float:
    """Compute reconstruction error between embedding and ground truth
    
    Parameters:
    -----------
    embedding : np.ndarray
        Low-dimensional embedding
    true_variable : np.ndarray
        Ground truth variable (angles or positions)
    manifold_type : str
        Type of manifold ('circular' or 'spatial')
        
    Returns:
    --------
    float
        Reconstruction error
    """
    if manifold_type == 'circular':
        # Extract angles from embedding
        reconstructed_angles = extract_angles_from_embedding(embedding)
        
        # Compute circular distance
        distances = circular_distance(reconstructed_angles, true_variable)
        return np.mean(distances)
        
    elif manifold_type == 'spatial':
        # For spatial manifolds, we need to align the embedding with true positions
        # Use Procrustes analysis for optimal alignment
        aligned_embedding, _ = procrustes_analysis(true_variable, embedding)
        
        # Compute distances
        distances = np.linalg.norm(aligned_embedding - true_variable, axis=1)
        return np.mean(distances)
    
    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")


def compute_temporal_consistency(
    embedding: np.ndarray, 
    true_variable: np.ndarray, 
    manifold_type: str = 'circular'
) -> float:
    """Compute temporal consistency between embedding and ground truth
    
    Parameters:
    -----------
    embedding : np.ndarray
        Low-dimensional embedding
    true_variable : np.ndarray
        Ground truth variable (angles or positions)
    manifold_type : str
        Type of manifold ('circular' or 'spatial')
        
    Returns:
    --------
    float
        Temporal consistency score (correlation)
    """
    if manifold_type == 'circular':
        # Extract angles from embedding
        reconstructed_angles = extract_angles_from_embedding(embedding)
        
        # Compute temporal derivatives
        true_velocity = np.diff(true_variable)
        reconstructed_velocity = np.diff(reconstructed_angles)
        
        # Handle circular wrapping
        true_velocity = np.arctan2(np.sin(true_velocity), np.cos(true_velocity))
        reconstructed_velocity = np.arctan2(np.sin(reconstructed_velocity), np.cos(reconstructed_velocity))
        
        # Compute correlation
        correlation = np.corrcoef(true_velocity, reconstructed_velocity)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
        
    elif manifold_type == 'spatial':
        # Use Procrustes analysis for optimal alignment
        aligned_embedding, _ = procrustes_analysis(true_variable, embedding)
        
        # Compute velocity vectors
        true_velocity = np.diff(true_variable, axis=0)
        reconstructed_velocity = np.diff(aligned_embedding, axis=0)
        
        # Compute correlation of velocity magnitudes
        true_speed = np.linalg.norm(true_velocity, axis=1)
        reconstructed_speed = np.linalg.norm(reconstructed_velocity, axis=1)
        
        correlation = np.corrcoef(true_speed, reconstructed_speed)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")


def train_simple_decoder(embedding: np.ndarray, true_variable: np.ndarray, manifold_type: str = 'circular'):
    """Train a simple decoder from embedding to ground truth variable
    
    Parameters:
    -----------
    embedding : np.ndarray
        Low-dimensional embedding with shape (n_timepoints, n_features)
    true_variable : np.ndarray
        Ground truth variable (angles or positions)
    manifold_type : str
        Type of manifold ('circular' or 'spatial')
        
    Returns:
    --------
    callable
        Trained decoder function
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    # Ensure embedding has correct shape
    if embedding.shape[0] != true_variable.shape[0]:
        raise ValueError(f"Embedding and true_variable must have same number of timepoints. "
                        f"Got embedding: {embedding.shape}, true_variable: {true_variable.shape}")
    
    # Standardize embedding
    scaler = StandardScaler()
    embedding_scaled = scaler.fit_transform(embedding)
    
    if manifold_type == 'circular':
        # For circular variables, predict sin and cos components
        sin_component = np.sin(true_variable)
        cos_component = np.cos(true_variable)
        
        # Train separate regressors for sin and cos
        sin_regressor = LinearRegression().fit(embedding_scaled, sin_component)
        cos_regressor = LinearRegression().fit(embedding_scaled, cos_component)
        
        def decoder(new_embedding):
            new_embedding_scaled = scaler.transform(new_embedding)
            pred_sin = sin_regressor.predict(new_embedding_scaled)
            pred_cos = cos_regressor.predict(new_embedding_scaled)
            return np.arctan2(pred_sin, pred_cos)
            
    elif manifold_type == 'spatial':
        # For spatial variables, direct regression
        regressor = LinearRegression().fit(embedding_scaled, true_variable)
        
        def decoder(new_embedding):
            new_embedding_scaled = scaler.transform(new_embedding)
            return regressor.predict(new_embedding_scaled)
    
    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")
    
    return decoder


def compute_decoding_accuracy(
    embedding: np.ndarray, 
    true_variable: np.ndarray, 
    manifold_type: str = 'circular',
    train_fraction: float = 0.8
) -> dict:
    """Compute decoding accuracy using train/test split
    
    Parameters:
    -----------
    embedding : np.ndarray
        Low-dimensional embedding
    true_variable : np.ndarray
        Ground truth variable (angles or positions)
    manifold_type : str
        Type of manifold ('circular' or 'spatial')
    train_fraction : float
        Fraction of data to use for training
        
    Returns:
    --------
    dict
        Dictionary containing training and testing errors
    """
    n_samples = embedding.shape[0]
    n_train = int(n_samples * train_fraction)
    
    # Split data
    train_embedding = embedding[:n_train]
    test_embedding = embedding[n_train:]
    train_variable = true_variable[:n_train]
    test_variable = true_variable[n_train:]
    
    # Train decoder
    decoder = train_simple_decoder(train_embedding, train_variable, manifold_type)
    
    # Compute training error
    train_predictions = decoder(train_embedding)
    train_error = compute_reconstruction_error(
        train_embedding, train_variable, manifold_type
    ) if manifold_type == 'circular' else np.mean(
        np.linalg.norm(train_predictions - train_variable, axis=1)
    )
    
    # Compute testing error
    test_predictions = decoder(test_embedding)
    if manifold_type == 'circular':
        test_error = np.mean(circular_distance(test_predictions, test_variable))
    else:
        test_error = np.mean(np.linalg.norm(test_predictions - test_variable, axis=1))
    
    return {
        'train_error': train_error,
        'test_error': test_error,
        'generalization_gap': test_error - train_error
    }


def manifold_reconstruction_score(
    embedding: np.ndarray,
    true_variable: np.ndarray,
    manifold_type: str = 'circular',
    weights: Optional[dict] = None
) -> dict:
    """Compute comprehensive manifold reconstruction score
    
    Parameters:
    -----------
    embedding : np.ndarray
        Low-dimensional embedding
    true_variable : np.ndarray
        Ground truth variable (angles or positions)
    manifold_type : str
        Type of manifold ('circular' or 'spatial')
    weights : dict, optional
        Weights for combining metrics
        
    Returns:
    --------
    dict
        Dictionary containing reconstruction metrics
    """
    if weights is None:
        weights = {
            'reconstruction_error': 0.4,
            'temporal_consistency': 0.3,
            'decoding_accuracy': 0.3
        }
    
    # Compute metrics
    reconstruction_error = compute_reconstruction_error(embedding, true_variable, manifold_type)
    temporal_consistency = compute_temporal_consistency(embedding, true_variable, manifold_type)
    decoding_results = compute_decoding_accuracy(embedding, true_variable, manifold_type)
    
    # Normalize reconstruction error (lower is better, so invert)
    max_error = np.pi if manifold_type == 'circular' else 1.0  # Normalized for spatial
    normalized_error = 1.0 - min(reconstruction_error / max_error, 1.0)
    
    # Normalize decoding accuracy (lower test error is better)
    max_decode_error = np.pi if manifold_type == 'circular' else 1.0
    normalized_decode = 1.0 - min(decoding_results['test_error'] / max_decode_error, 1.0)
    
    # Ensure temporal consistency is positive
    temporal_consistency = max(temporal_consistency, 0.0)
    
    # Compute weighted score
    overall_score = (
        weights['reconstruction_error'] * normalized_error +
        weights['temporal_consistency'] * temporal_consistency +
        weights['decoding_accuracy'] * normalized_decode
    )
    
    return {
        'reconstruction_error': reconstruction_error,
        'temporal_consistency': temporal_consistency,
        'decoding_train_error': decoding_results['train_error'],
        'decoding_test_error': decoding_results['test_error'],
        'generalization_gap': decoding_results['generalization_gap'],
        'overall_reconstruction_score': overall_score
    }