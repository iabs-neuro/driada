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

KNN-based Metrics Comparison:
----------------------------
This module provides three complementary k-nearest neighbor metrics:

1. **knn_preservation_rate**: Simple intersection-based metric
   - Measures: |neighbors_original ∩ neighbors_embedding| / k
   - Symmetric: Treats false positives and false negatives equally
   - Use when: You want a simple, interpretable overall score

2. **trustworthiness**: Focuses on avoiding false neighbors
   - Measures: How much can we trust that embedded neighbors are true neighbors?
   - Penalizes: Points that appear close in embedding but were far in original
   - Use when: False patterns in embedding would be problematic (e.g., clustering)

3. **continuity**: Focuses on preserving true neighbors  
   - Measures: How well are original neighborhoods preserved in embedding?
   - Penalizes: True neighbors that become separated in embedding
   - Use when: Losing connections would miss important structure (e.g., manifolds)

For comprehensive evaluation, use all three metrics or combine trustworthiness
and continuity, as they capture complementary aspects of embedding quality.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from scipy.linalg import orthogonal_procrustes
from typing import Optional, Tuple, Dict, Any


def compute_distance_matrix(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
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
        
    Raises
    ------
    ValueError
        If X is not a 2D array
        
    Notes
    -----
    For empty arrays, returns a (1, 1) matrix due to scipy's squareform behavior.    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape {X.shape}")

    distances = pdist(X, metric=metric)
    return squareform(distances)


def knn_preservation_rate(
    X_high: np.ndarray,
    X_low: np.ndarray,
    k: int = 10,
    flexible: bool = False,
    flexibility_factor: float = 2.0,
) -> float:
    """
    Compute k-nearest neighbor preservation rate.

    This metric measures what fraction of k nearest neighbors in the original
    high-dimensional space are preserved in the low-dimensional embedding.
    It provides a simple, symmetric measure of neighborhood preservation.

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
        Preservation rate between 0 and 1. Higher values indicate better
        neighborhood preservation.

    Notes
    -----
    This metric differs from trustworthiness and continuity in that it:
    - Treats false positives and false negatives equally
    - Uses exact neighborhood matching (or flexible matching if enabled)
    - Does not consider the ranking of points beyond the k-th neighbor
    
    Use this metric when:
    - You want a simple, interpretable measure of neighborhood preservation
    - Both types of errors (missing neighbors and false neighbors) are equally important
    - You don't need to distinguish between different types of embedding errors
    
    Mathematical formulation:
        preservation_rate = |N_k(i, high) ∩ N_k(i, low)| / k
    where N_k(i, space) is the set of k nearest neighbors of point i in that space.
    
    See Also
    --------
    ~driada.dim_reduction.manifold_metrics.trustworthiness : Focuses on avoiding false neighbors in the embedding
    ~driada.dim_reduction.manifold_metrics.continuity : Focuses on preserving true neighbors from the original space    """
    if X_high.shape[0] != X_low.shape[0]:
        raise ValueError("X_high and X_low must have same number of samples")

    n_samples = X_high.shape[0]
    if k >= n_samples:
        raise ValueError(f"k={k} must be less than n_samples={n_samples}")

    # Find k-NN in original space
    nbrs_high = NearestNeighbors(n_neighbors=k + 1).fit(X_high)
    _, indices_high = nbrs_high.kneighbors(X_high)
    indices_high = indices_high[:, 1:]  # Remove self

    # Find k-NN (or flexible k-NN) in embedded space
    k_low = int(k * flexibility_factor) if flexible else k
    nbrs_low = NearestNeighbors(n_neighbors=min(k_low + 1, n_samples)).fit(X_low)
    _, indices_low = nbrs_low.kneighbors(X_low)
    indices_low = indices_low[:, 1:]  # Remove self

    # Count preserved neighbors
    preserved = 0
    for i in range(n_samples):
        high_neighbors = set(indices_high[i])
        low_neighbors = set(indices_low[i][:k_low])
        preserved += len(high_neighbors.intersection(low_neighbors))

    return preserved / (n_samples * k)


def trustworthiness(X_high: np.ndarray, X_low: np.ndarray, k: int = 10) -> float:
    """
    Compute trustworthiness of the embedding.

    Trustworthiness measures how much we can trust that points nearby in the
    embedding are truly neighbors in the original space. It penalizes "false
    neighbors" - points that appear close in the embedding but were far apart
    in the original space.

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
        Trustworthiness score between 0 and 1. Higher values indicate that
        neighbors in the embedding can be trusted (few false neighbors).

    Notes
    -----
    Trustworthiness focuses on precision: are the neighbors we see in the
    embedding actually neighbors in the original space? This is important
    when false neighbors could lead to incorrect interpretations.
    
    Use trustworthiness when:
    - You want to avoid spurious patterns in the embedding
    - False neighbors (points incorrectly appearing close) are problematic
    - You're using the embedding for neighbor-based analysis or clustering
    
    Mathematical formulation:
        T(k) = 1 - (2/N*k*(2N-3k-1)) * Σᵢ Σⱼ∈Uₖ(i) (r(i,j) - k)
    where:
    - Uₖ(i) is the set of points that are among k-NN of i in embedding but not in original
    - r(i,j) is the rank of j as neighbor of i in the original space
    - The penalty (r(i,j) - k) increases with how far j was from i originally
    
    See Also
    --------
    ~driada.dim_reduction.manifold_metrics.continuity : Complementary metric focusing on preserving true neighbors
    ~driada.dim_reduction.manifold_metrics.knn_preservation_rate : Simple symmetric measure of neighborhood preservation
    
    References
    ----------
    Venna, J., & Kaski, S. (2006). Local multidimensional scaling. 
    Neural Networks, 19(6-7), 889-899.    """
    if X_high.shape[0] != X_low.shape[0]:
        raise ValueError("X_high and X_low must have same number of samples")

    n_samples = X_high.shape[0]
    if k >= n_samples:
        raise ValueError(f"k={k} must be less than n_samples={n_samples}")

    # Compute distance matrices
    dist_high = compute_distance_matrix(X_high)
    dist_low = compute_distance_matrix(X_low)

    # Get k-NN in embedded space
    nbrs_low = NearestNeighbors(n_neighbors=k + 1).fit(X_low)
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
                trust += rank - k

    # Normalize - correct formula for maximum possible penalty
    # Maximum penalty occurs when all k neighbors in embedding were 
    # the furthest points in original space
    max_trust = n_samples * k * (n_samples - 1 - k)
    if max_trust > 0:
        trust = 1 - trust / max_trust
    else:
        trust = 1.0

    return max(0.0, min(1.0, trust))  # Ensure result is in [0, 1]


def continuity(X_high: np.ndarray, X_low: np.ndarray, k: int = 10) -> float:
    """
    Compute continuity of the embedding.

    Continuity measures how well the embedding preserves the neighborhoods
    from the original space. It penalizes "missing neighbors" - points that
    were close in the original space but are far apart in the embedding.

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
        Continuity score between 0 and 1. Higher values indicate that
        original neighbors are preserved (few missing neighbors).

    Notes
    -----
    Continuity focuses on recall: are the true neighbors from the original
    space preserved in the embedding? This is important when losing
    important connections would miss critical structure.
    
    Use continuity when:
    - You want to preserve all important relationships from the original data
    - Missing neighbors (losing true connections) is problematic
    - You're studying the continuity of manifolds or connected structures
    
    Mathematical formulation:
        C(k) = 1 - (2/N*k*(2N-3k-1)) * Σᵢ Σⱼ∈Vₖ(i) (r'(i,j) - k)
    where:
    - Vₖ(i) is the set of points that are among k-NN of i in original but not in embedding
    - r'(i,j) is the rank of j as neighbor of i in the embedding space
    - The penalty (r'(i,j) - k) increases with how far j is from i in embedding
    
    Together with trustworthiness:
    - High trustworthiness + High continuity = Excellent embedding
    - High trustworthiness + Low continuity = Embedding compresses neighborhoods
    - Low trustworthiness + High continuity = Embedding creates false neighborhoods
    - Low trustworthiness + Low continuity = Poor embedding quality
    
    See Also
    --------
    ~driada.dim_reduction.manifold_metrics.trustworthiness : Complementary metric focusing on avoiding false neighbors
    ~driada.dim_reduction.manifold_metrics.knn_preservation_rate : Simple symmetric measure of neighborhood preservation
    
    References
    ----------
    Venna, J., & Kaski, S. (2006). Local multidimensional scaling. 
    Neural Networks, 19(6-7), 889-899.    """
    if X_high.shape[0] != X_low.shape[0]:
        raise ValueError("X_high and X_low must have same number of samples")

    n_samples = X_high.shape[0]
    if k >= n_samples:
        raise ValueError(f"k={k} must be less than n_samples={n_samples}")

    # Compute distance matrices
    dist_high = compute_distance_matrix(X_high)
    dist_low = compute_distance_matrix(X_low)

    # Get k-NN in original space
    nbrs_high = NearestNeighbors(n_neighbors=k + 1).fit(X_high)
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
                cont += rank - k

    # Normalize - correct formula for maximum possible penalty
    # Maximum penalty occurs when all k neighbors in original were
    # the furthest points in embedding
    max_cont = n_samples * k * (n_samples - 1 - k)
    if max_cont > 0:
        cont = 1 - cont / max_cont
    else:
        cont = 1.0

    return max(0.0, min(1.0, cont))  # Ensure result is in [0, 1]


def geodesic_distance_correlation(
    X_high: np.ndarray,
    X_low: np.ndarray,
    k_neighbors: int = 10,
    method: str = "spearman",
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
        Correlation coefficient between -1 and 1. Returns 0.0 if correlation
        cannot be computed (e.g., all distances are infinite).
        
    Raises
    ------
    ValueError
        If X_high and X_low have different number of samples
        If k_neighbors >= n_samples    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import shortest_path
    
    # Input validation
    if X_high.shape[0] != X_low.shape[0]:
        raise ValueError("X_high and X_low must have same number of samples")
    
    n_samples = X_high.shape[0]
    if k_neighbors >= n_samples:
        raise ValueError(f"k_neighbors={k_neighbors} must be less than n_samples={n_samples}")

    # Build k-NN graph for geodesic approximation
    graph = kneighbors_graph(X_high, n_neighbors=k_neighbors, mode="distance")

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

    # Check if we have valid data for correlation
    if len(geodesic_flat) == 0 or len(euclidean_flat) == 0:
        return 0.0
    
    # Compute correlation
    if method == "spearman":
        corr, _ = spearmanr(geodesic_flat, euclidean_flat)
    else:  # pearson
        corr = np.corrcoef(geodesic_flat, euclidean_flat)[0, 1]

    # Handle NaN (can occur if all values are identical)
    if np.isnan(corr):
        return 0.0
        
    return corr


def stress(X_high: np.ndarray, X_low: np.ndarray, normalized: bool = True) -> float:
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
        Stress value (lower is better).
        
    Raises
    ------
    ValueError
        If X_high and X_low have different number of samples.
        If normalized=True and all distances in X_high are zero (degenerate data).
        
    Notes
    -----
    Stress = Σ(d_ij^high - d_ij^low)² / Σ(d_ij^high)² if normalized,
    otherwise just Σ(d_ij^high - d_ij^low)²    """
    # Validate input shapes
    if X_high.shape[0] != X_low.shape[0]:
        raise ValueError("X_high and X_low must have same number of samples")
    
    # Compute distance matrices
    dist_high = compute_distance_matrix(X_high)
    dist_low = compute_distance_matrix(X_low)

    # Compute stress
    diff = dist_high - dist_low
    stress_val = np.sum(diff**2)

    if normalized:
        sum_dist_squared = np.sum(dist_high**2)
        if sum_dist_squared == 0:
            raise ValueError("All distances in high-dimensional data are zero. "
                           "This indicates degenerate data (all points identical).")
        stress_val /= sum_dist_squared

    return stress_val


def circular_structure_preservation(
    X_low: np.ndarray, true_angles: Optional[np.ndarray] = None, k_neighbors: int = 3
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
        Dictionary containing various circular preservation metrics:
        - distance_cv: coefficient of variation of distances from center
        - consecutive_preservation: fraction with circular neighbors preserved
        - circular_correlation (if true_angles provided)
        
    Raises
    ------
    ValueError
        If X_low is not 2D (shape[1] != 2)
        If k_neighbors >= n_samples
        If all points are at the center (degenerate circle)    """
    if X_low.shape[1] != 2:
        raise ValueError("Circular analysis requires 2D embedding")

    n_samples = X_low.shape[0]
    if k_neighbors >= n_samples:
        raise ValueError(f"k_neighbors={k_neighbors} must be less than n_samples={n_samples}")

    # Center the embedding
    center = np.mean(X_low, axis=0)
    centered = X_low - center

    # Compute distances from center
    distances = np.linalg.norm(centered, axis=1)

    # Check for degenerate data before computing coefficient of variation
    mean_dist = np.mean(distances)
    if mean_dist == 0:
        raise ValueError("All points are at the center. This indicates degenerate data.")
    
    # Coefficient of variation of distances (should be small for circle)
    cv_distances = np.std(distances) / mean_dist

    # Compute angles
    angles = np.arctan2(centered[:, 1], centered[:, 0])

    # Sort by angle to check consecutive preservation
    angle_order = np.argsort(angles)

    # Check consecutive neighbor preservation
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X_low)
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
        "distance_cv": cv_distances,
        "consecutive_preservation": consecutive_preserved / n_samples,
    }

    # If true angles provided, compute angular correlation
    if true_angles is not None:
        # Unwrap angles to handle discontinuity
        angle_diff = angles - true_angles
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        # Circular correlation
        circular_corr = 1 - np.mean(np.abs(angle_diff)) / np.pi
        results["circular_correlation"] = circular_corr

    return results


def procrustes_analysis(
    X: np.ndarray, Y: np.ndarray, scaling: bool = True, reflection: bool = True
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
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
    transform_info : dict
        Dictionary containing transformation parameters:
        - 'scale_factor': float, the scaling factor applied
        - 'is_reflected': bool, whether reflection was detected
        - 'rotation_matrix': np.ndarray, the rotation matrix R
        
    Notes
    -----
    When reflection=False and a reflection is detected, the rotation matrix
    is corrected using SVD decomposition to remove the reflection component.    """
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")

    # Center configurations
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)

    # Compute optimal rotation
    R, scale = orthogonal_procrustes(Y_centered, X_centered)

    # Apply transformation
    Y_aligned = Y_centered @ R

    # Track transformation parameters
    scale_factor = 1.0
    is_reflected = np.linalg.det(R) < 0
    
    if scaling:
        # Compute optimal scaling
        norm_X = np.linalg.norm(X_centered)
        norm_Y = np.linalg.norm(Y_aligned)
        if norm_Y > 0:
            scale_factor = norm_X / norm_Y
            Y_aligned *= scale_factor

    if not reflection and is_reflected:
        # Remove reflection by recomputing without it
        # SVD decomposition of R
        U, s, Vt = np.linalg.svd(R)
        # Flip the sign of the smallest singular value
        s[-1] *= -1
        # Reconstruct rotation without reflection
        R = U @ np.diag(s) @ Vt
        # Recompute aligned Y
        Y_aligned = Y_centered @ R
        if scaling:
            Y_aligned *= scale_factor
        is_reflected = False

    # Compute disparity after all transformations
    disparity = np.sqrt(np.sum((X_centered - Y_aligned) ** 2))

    # Return aligned points (with original center)
    Y_aligned += np.mean(X, axis=0)
    
    transform_info = {
        'scale_factor': scale_factor,
        'is_reflected': is_reflected,
        'rotation_matrix': R
    }

    return Y_aligned, disparity, transform_info


def manifold_preservation_score(
    X_high: np.ndarray,
    X_low: np.ndarray,
    k_neighbors: int = 10,
    weights: Optional[dict] = None,
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
        
    Notes
    -----
    Input validation is performed by the individual metric functions.
    NaN values in geodesic_correlation are replaced with 0.0.    """
    if weights is None:
        weights = {
            "knn_preservation": 0.25,
            "trustworthiness": 0.25,
            "continuity": 0.25,
            "geodesic_correlation": 0.25,
        }

    # Compute individual metrics
    metrics = {
        "knn_preservation": knn_preservation_rate(X_high, X_low, k=k_neighbors),
        "trustworthiness": trustworthiness(X_high, X_low, k=k_neighbors),
        "continuity": continuity(X_high, X_low, k=k_neighbors),
        "geodesic_correlation": geodesic_distance_correlation(
            X_high, X_low, k_neighbors=k_neighbors
        ),
    }

    # Handle potential NaN in geodesic correlation
    if np.isnan(metrics["geodesic_correlation"]):
        metrics["geodesic_correlation"] = 0.0

    # Compute weighted average
    overall_score = sum(metrics[key] * weights.get(key, 0) for key in metrics)

    metrics["overall_score"] = overall_score

    return metrics


# =============================================================================
# MANIFOLD RECONSTRUCTION VALIDATION
# =============================================================================


def circular_distance(angles1: np.ndarray, angles2: np.ndarray) -> np.ndarray:
    """Compute circular distance between two sets of angles.
    
    Calculates the shortest angular distance between angles on a circle,
    accounting for the circular nature where 0 and 2π are the same point.
    The result is always in [0, π].

    Parameters
    ----------
    angles1 : np.ndarray
        First array of angles in radians.
    angles2 : np.ndarray
        Second array of angles in radians. Must be broadcastable with angles1.

    Returns
    -------
    np.ndarray
        Circular distances between corresponding angles, in range [0, π].
        
    Notes
    -----
    Uses the formula: |arctan2(sin(θ₁-θ₂), cos(θ₁-θ₂))| to ensure
    the result is the shortest distance on the circle.    """
    diff = angles1 - angles2
    return np.abs(np.arctan2(np.sin(diff), np.cos(diff)))


def circular_diff(angles: np.ndarray) -> np.ndarray:
    """Compute differences between consecutive angles, handling circular wrapping.
    
    Calculates angle[i+1] - angle[i] for each consecutive pair, ensuring
    the result represents the shortest angular distance with proper sign.

    Parameters
    ----------
    angles : np.ndarray
        1D array of angles in radians.

    Returns
    -------
    np.ndarray
        Array of wrapped differences in [-π, π]. Length is len(angles) - 1.
        Positive values indicate counter-clockwise movement.
        
    Raises
    ------
    ValueError
        If angles is empty or not 1D.
        
    Examples
    --------
    >>> angles = np.array([0, 3*np.pi/2, np.pi/4])  # 0°, 270°, 45°
    >>> diffs = circular_diff(angles)
    >>> # First diff: 270° - 0° = -90° (shortest path)
    >>> # Second diff: 45° - 270° = 135° (forward wrap)    """
    # Input validation
    if angles.ndim != 1:
        raise ValueError(f"angles must be 1D array, got shape {angles.shape}")
    if len(angles) == 0:
        raise ValueError("angles array is empty")
    if len(angles) == 1:
        return np.array([])
        
    diffs = np.diff(angles)
    # Wrap differences to [-pi, pi]
    return np.arctan2(np.sin(diffs), np.cos(diffs))


def extract_angles_from_embedding(embedding: np.ndarray) -> np.ndarray:
    """Extract angular positions from a 2D embedding.
    
    Computes the angle (in radians) from the centroid to each point
    in a 2D embedding. Useful for detecting circular structure.

    Parameters
    ----------
    embedding : np.ndarray
        2D embedding with shape (n_samples, 2) where each row is a point
        in 2D space.

    Returns
    -------
    np.ndarray
        Array of angles in radians [-π, π] from the centroid to each point.
        Uses atan2 convention where 0 is along positive x-axis.
        
    Raises
    ------
    ValueError
        If embedding does not have exactly 2 dimensions (columns)
        If embedding is empty    """
    if embedding.ndim != 2 or embedding.shape[1] != 2:
        raise ValueError(f"Embedding must be 2D with shape (n_samples, 2), got shape {embedding.shape}")
    if embedding.shape[0] == 0:
        raise ValueError("Embedding is empty")

    # Center the embedding
    centered = embedding - np.mean(embedding, axis=0)

    # Extract angles
    angles = np.arctan2(centered[:, 1], centered[:, 0])

    return angles


def find_optimal_circular_alignment(
    true_angles: np.ndarray,
    reconstructed_angles: np.ndarray,
    allow_rotation: bool = True,
    allow_reflection: bool = True,
) -> Tuple[float, float, bool]:
    """Find optimal rotation and reflection to align circular data.

    Parameters
    ----------
    true_angles : np.ndarray
        Ground truth angles in radians
    reconstructed_angles : np.ndarray
        Reconstructed angles in radians
    allow_rotation : bool
        Whether to allow arbitrary rotation offset
    allow_reflection : bool
        Whether to allow reflection (chirality flip)

    Returns
    -------
    optimal_offset : float
        Optimal rotation offset in radians
    min_error : float
        Minimum mean circular distance after alignment
    is_reflected : bool
        Whether reflection was applied
        
    Notes
    -----
    Uses grid search with 1-degree resolution for rotation offset.
    For higher precision, consider using scipy.optimize.minimize_scalar.    """

    def compute_mean_circular_distance(angles1, angles2, offset=0):
        """Compute mean circular distance between two sets of angles.
        
        Calculates the average absolute circular distance between corresponding
        angles, with an optional rotation offset applied to the second set.
        Uses circular arithmetic to handle angle wrapping correctly.
        
        Parameters
        ----------
        angles1 : array-like
            First set of angles in radians.
        angles2 : array-like
            Second set of angles in radians. Must have same shape as angles1.
        offset : float, default=0
            Rotation offset in radians to add to angles2 before comparison.
            
        Returns
        -------
        float
            Mean absolute circular distance in radians, range [0, π].
            Returns NaN if input arrays are empty.
            
        Notes
        -----
        The circular distance is computed as:
        1. diff = angles1 - (angles2 + offset)
        2. Wrapped to [-π, π] using arctan2(sin(diff), cos(diff))
        3. Absolute value taken and averaged
        
        This metric is symmetric and rotation-invariant when offset is optimized.
        No input validation is performed - arrays must have compatible shapes.        """
        diff = angles1 - (angles2 + offset)
        return np.mean(np.abs(np.arctan2(np.sin(diff), np.cos(diff))))

    best_offset = 0.0
    best_error = np.inf
    best_reflected = False

    # Try both orientations if allowed
    orientations = [False]
    if allow_reflection:
        orientations.append(True)

    for reflected in orientations:
        test_angles = -reconstructed_angles if reflected else reconstructed_angles

        if allow_rotation:
            # Find optimal rotation offset
            # Use optimization or grid search
            offsets = np.linspace(0, 2 * np.pi, 360, endpoint=False)
            errors = [
                compute_mean_circular_distance(true_angles, test_angles, offset)
                for offset in offsets
            ]
            min_idx = np.argmin(errors)
            offset = offsets[min_idx]
            error = errors[min_idx]
        else:
            offset = 0.0
            error = compute_mean_circular_distance(true_angles, test_angles)

        if error < best_error:
            best_error = error
            best_offset = offset
            best_reflected = reflected

    return best_offset, best_error, best_reflected


def compute_circular_correlation(
    angles1: np.ndarray, angles2: np.ndarray, offset: float = 0.0
) -> float:
    """Compute circular correlation coefficient between two angular datasets.
    
    Measures the similarity between two sets of circular data using complex
    representation. The result is invariant to common rotation.

    Parameters
    ----------
    angles1 : np.ndarray
        First array of angles in radians.
    angles2 : np.ndarray
        Second array of angles in radians. Must have same length as angles1.
    offset : float, default=0.0
        Rotation offset to apply to angles2 before correlation.

    Returns
    -------
    float
        Circular correlation coefficient in [0, 1]. Higher values indicate
        stronger circular correlation.
        
    Raises
    ------
    ValueError
        If angles1 and angles2 have different lengths.
        If either array is empty.
        If there is no variation in the data (all values identical).
        
    Notes
    -----
    Uses complex representation z = exp(i*angle) and computes correlation
    in complex plane. Result is the absolute value of complex correlation.    """
    # Input validation
    if len(angles1) != len(angles2):
        raise ValueError(f"angles1 and angles2 must have same length, got {len(angles1)} and {len(angles2)}")
    if len(angles1) == 0:
        raise ValueError("Input arrays are empty")
        
    # Convert to unit complex numbers
    z1 = np.exp(1j * angles1)
    z2 = np.exp(1j * (angles2 + offset))

    # Compute circular correlation
    mean_z1 = np.mean(z1)
    mean_z2 = np.mean(z2)

    # Centered complex numbers
    z1_centered = z1 - mean_z1
    z2_centered = z2 - mean_z2

    # Circular correlation
    numerator = np.abs(np.sum(z1_centered * np.conj(z2_centered)))
    denominator = np.sqrt(
        np.sum(np.abs(z1_centered) ** 2) * np.sum(np.abs(z2_centered) ** 2)
    )

    if denominator == 0:
        raise ValueError("Cannot compute correlation: no variation in the data (all values identical)")
        
    return numerator / denominator


def compute_reconstruction_error(
    embedding: np.ndarray,
    true_variable: np.ndarray,
    manifold_type: str = "circular",
    allow_rotation: bool = True,
    allow_reflection: bool = True,
    allow_scaling: bool = True,
) -> dict:
    """Compute reconstruction error between embedding and ground truth.

    Parameters
    ----------
    embedding : np.ndarray
        Low-dimensional embedding
    true_variable : np.ndarray
        Ground truth variable (angles or positions)
    manifold_type : str
        Type of manifold ('circular' or 'spatial')
    allow_rotation : bool
        Whether to allow rotation/translation
    allow_reflection : bool
        Whether to allow reflection
    allow_scaling : bool
        Whether to allow scaling

    Returns
    -------
    dict
        Dictionary containing:
        - error: reconstruction error
        - correlation: correlation after alignment
        - rotation_offset: optimal rotation (for circular)
        - is_reflected: whether reflection was applied
        - scale_factor: optimal scale (if applicable)
        
    Raises
    ------
    ValueError
        If manifold_type is not 'circular' or 'spatial'.    """
    if manifold_type == "circular":
        # Extract angles from embedding
        reconstructed_angles = extract_angles_from_embedding(embedding)

        # Find optimal alignment
        offset, error, reflected = find_optimal_circular_alignment(
            true_variable,
            reconstructed_angles,
            allow_rotation=allow_rotation,
            allow_reflection=allow_reflection,
        )

        # Apply optimal transformation
        aligned_angles = reconstructed_angles
        if reflected:
            aligned_angles = -aligned_angles
        aligned_angles = aligned_angles + offset
        
        # Wrap angles to [-π, π] after transformation
        aligned_angles = np.arctan2(np.sin(aligned_angles), np.cos(aligned_angles))

        # Compute correlation after alignment
        correlation = compute_circular_correlation(true_variable, aligned_angles)

        return {
            "error": error,
            "correlation": correlation,
            "rotation_offset": offset,
            "is_reflected": reflected,
            "scale_factor": 1.0,  # No scaling for circular data
        }

    elif manifold_type == "spatial":
        # For spatial manifolds, use Procrustes analysis
        aligned_embedding, disparity, transform_info = procrustes_analysis(
            true_variable, embedding, scaling=allow_scaling, reflection=allow_reflection
        )

        # Compute error and correlation
        error = np.mean(np.linalg.norm(aligned_embedding - true_variable, axis=1))
        correlation = np.corrcoef(aligned_embedding.flatten(), true_variable.flatten())[
            0, 1
        ]

        return {
            "error": error,
            "correlation": correlation,
            "rotation_offset": 0.0,  # Not applicable for spatial
            "is_reflected": transform_info['is_reflected'],
            "scale_factor": transform_info['scale_factor'],
        }

    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")


def compute_embedding_alignment_metrics(
    embedding: np.ndarray,
    true_variable: np.ndarray,
    manifold_type: str = "circular",
    allow_rotation: bool = True,
    allow_reflection: bool = True,
    allow_scaling: bool = True,
) -> Dict[str, float]:
    """Compute comprehensive alignment metrics between embedding and true variable.

    Parameters
    ----------
    embedding : np.ndarray
        The embedding to evaluate. Shape: (n_samples, n_dims).
    true_variable : np.ndarray
        The true variable.
    manifold_type : str, optional
        Type of manifold: 'circular' or 'spatial'. Default is 'circular'.
    allow_rotation : bool, optional
        Whether to allow rotation when finding alignment. Default is True.
    allow_reflection : bool, optional
        Whether to allow reflection when finding alignment. Default is True.
    allow_scaling : bool, optional
        Whether to allow scaling when finding alignment. Default is True.

    Returns
    -------
    Dict[str, float]
        Dictionary containing all metrics from compute_reconstruction_error,
        plus for circular manifolds:
        - 'velocity_correlation': Correlation between angular velocities
        - 'variance_ratio': Ratio of embedding to true variance

    Raises
    ------
    ValueError
        For circular manifolds:
        - If fewer than 3 points (cannot compute velocity correlation)
        - If velocity correlation is NaN (no variation in velocities)
        - If true variable has zero circular variance (all points at same angle)    """
    # Get reconstruction metrics with alignment
    metrics = compute_reconstruction_error(
        embedding,
        true_variable,
        manifold_type,
        allow_rotation=allow_rotation,
        allow_reflection=allow_reflection,
        allow_scaling=allow_scaling,
    )

    if manifold_type == "circular":
        # Add circular-specific metrics
        reconstructed_angles = extract_angles_from_embedding(embedding)

        # Apply optimal transformation
        if metrics["is_reflected"]:
            reconstructed_angles = -reconstructed_angles
        reconstructed_angles = reconstructed_angles + metrics["rotation_offset"]

        # Compute velocity correlation
        true_vel = circular_diff(true_variable)
        recon_vel = circular_diff(reconstructed_angles)

        if len(true_vel) <= 1:
            raise ValueError(
                f"Cannot compute velocity correlation with {len(true_variable)} points. "
                "Need at least 3 points to compute meaningful velocities."
            )
        
        vel_corr = np.corrcoef(true_vel, recon_vel)[0, 1]
        if np.isnan(vel_corr):
            raise ValueError(
                "Velocity correlation is NaN. This typically indicates no variation "
                "in velocities (constant or near-constant angular velocities)."
            )
        metrics["velocity_correlation"] = vel_corr

        # Add circular variance preservation
        true_var = 1 - np.abs(np.mean(np.exp(1j * true_variable)))
        if true_var == 0:
            raise ValueError(
                "True variable has zero circular variance (all points at same angle). "
                "Cannot compute meaningful variance ratio."
            )
        recon_var = 1 - np.abs(np.mean(np.exp(1j * reconstructed_angles)))
        metrics["variance_ratio"] = recon_var / true_var

    return metrics


def train_simple_decoder(
    embedding: np.ndarray, true_variable: np.ndarray, manifold_type: str = "circular"
) -> callable:
    """Train a simple decoder to reconstruct true variable from embedding.

    Parameters
    ----------
    embedding : np.ndarray
        The embedding features. Shape: (n_samples, n_dims).
    true_variable : np.ndarray
        The true variable to reconstruct. For circular manifolds, should be
        angles in radians. For spatial manifolds, assumes 1D array.
    manifold_type : str, optional
        Type of manifold: 'circular' or 'spatial'. Default is 'circular'.

    Returns
    -------
    callable
        A decoder function that takes an embedding and returns reconstructed variable.
        The decoder expects input of the same shape as the training embedding.

    Raises
    ------
    ValueError
        If embedding and true_variable have different numbers of samples,
        or if manifold_type is not 'circular' or 'spatial'.

    Notes
    -----
    For circular manifolds, trains separate regressors for sin and cos components.
    For spatial manifolds, performs direct regression. The embedding is always
    standardized before training.    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    # Ensure embedding has correct shape
    if embedding.shape[0] != true_variable.shape[0]:
        raise ValueError(
            f"Embedding and true_variable must have same number of timepoints. "
            f"Got embedding: {embedding.shape}, true_variable: {true_variable.shape}"
        )

    # Standardize embedding
    scaler = StandardScaler()
    embedding_scaled = scaler.fit_transform(embedding)

    if manifold_type == "circular":
        # For circular variables, predict sin and cos components
        sin_component = np.sin(true_variable)
        cos_component = np.cos(true_variable)

        # Train separate regressors for sin and cos
        sin_regressor = LinearRegression().fit(embedding_scaled, sin_component)
        cos_regressor = LinearRegression().fit(embedding_scaled, cos_component)

        def decoder(new_embedding):
            """Decode embedding features to circular angles.
            
            Uses trained linear regressors to predict sin and cos components
            from embedding features, then combines them to recover angles.
            This approach naturally handles the circular topology.
            
            Parameters
            ----------
            new_embedding : np.ndarray
                Embedding features of shape (n_samples, n_features).
                Must have same number of features as training embedding.
                Single samples should be reshaped to (1, n_features).
                
            Returns
            -------
            np.ndarray
                Predicted angles in radians, shape (n_samples,).
                Range [-π, π] due to arctan2 output.
                
            Raises
            ------
            ValueError
                If new_embedding has wrong number of features for scaler.
                
            Notes
            -----
            Uses captured variables from training:
            - scaler: StandardScaler for feature normalization
            - sin_regressor: LinearRegression for sin component
            - cos_regressor: LinearRegression for cos component
            
            The two-component prediction avoids discontinuities at ±π.            """
            new_embedding_scaled = scaler.transform(new_embedding)
            pred_sin = sin_regressor.predict(new_embedding_scaled)
            pred_cos = cos_regressor.predict(new_embedding_scaled)
            return np.arctan2(pred_sin, pred_cos)

    elif manifold_type == "spatial":
        # For spatial variables, direct regression
        regressor = LinearRegression().fit(embedding_scaled, true_variable)

        def decoder(new_embedding):
            """Decode embedding features to spatial coordinates.
            
            Uses trained linear regression to directly predict spatial
            coordinates from embedding features. Works for any dimensional
            target space determined during training.
            
            Parameters
            ----------
            new_embedding : np.ndarray
                Embedding features of shape (n_samples, n_features).
                Must have same number of features as training embedding.
                Single samples should be reshaped to (1, n_features).
                
            Returns
            -------
            np.ndarray
                Predicted spatial coordinates of shape (n_samples, n_dims)
                where n_dims matches the training target dimensionality.
                For 1D targets, shape is (n_samples,).
                
            Raises
            ------
            ValueError
                If new_embedding has wrong number of features for scaler.
                
            Notes
            -----
            Uses captured variables from training:
            - scaler: StandardScaler for feature normalization
            - regressor: LinearRegression for direct prediction
            
            Assumes linear relationship between embedding and coordinates.
            Quality depends on how well this assumption holds.            """
            new_embedding_scaled = scaler.transform(new_embedding)
            return regressor.predict(new_embedding_scaled)

    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")

    return decoder


def compute_embedding_quality(
    embedding: np.ndarray,
    true_variable: np.ndarray,
    manifold_type: str = "circular",
    train_fraction: float = 0.8,
    allow_rotation: bool = True,
    allow_reflection: bool = True,
    allow_scaling: bool = True,
    random_state: int = 42,
) -> Dict[str, float]:
    """Evaluate embedding quality using train/test split.

    Parameters
    ----------
    embedding : np.ndarray
        The embedding to evaluate. Shape: (n_samples, n_dims).
    true_variable : np.ndarray
        The true variable.
    manifold_type : str, optional
        Type of manifold: 'circular' or 'spatial'. Default is 'circular'.
    train_fraction : float, optional
        Fraction of data to use for training. Default is 0.8.
    allow_rotation : bool, optional
        Whether to allow rotation when finding alignment. Default is True.
    allow_reflection : bool, optional
        Whether to allow reflection when finding alignment. Default is True.
    allow_scaling : bool, optional
        Whether to allow scaling when finding alignment. Default is True.
    random_state : int, optional
        Random seed for train/test split reproducibility. Default is 42.

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'train_error': Reconstruction error on training set
        - 'test_error': Reconstruction error on test set
        - 'generalization_gap': Difference between test and train error
          (can be negative if test error is lower)

    Notes
    -----
    Uses random (not sequential) split of data to avoid domain shift issues
    that can occur with temporal/spatial data. The alignment is computed
    separately for train and test sets.    """
    from sklearn.model_selection import train_test_split

    # Split data randomly to avoid domain shift
    train_embedding, test_embedding, train_variable, test_variable = train_test_split(
        embedding, true_variable, test_size=1-train_fraction, random_state=random_state
    )

    # Compute reconstruction errors with alignment
    train_metrics = compute_reconstruction_error(
        train_embedding,
        train_variable,
        manifold_type,
        allow_rotation=allow_rotation,
        allow_reflection=allow_reflection,
        allow_scaling=allow_scaling,
    )
    test_metrics = compute_reconstruction_error(
        test_embedding,
        test_variable,
        manifold_type,
        allow_rotation=allow_rotation,
        allow_reflection=allow_reflection,
        allow_scaling=allow_scaling,
    )

    train_error = train_metrics["error"]
    test_error = test_metrics["error"]

    return {
        "train_error": train_error,
        "test_error": test_error,
        "generalization_gap": test_error - train_error,
    }


def compute_decoding_accuracy(
    embedding: np.ndarray,
    true_variable: np.ndarray,
    manifold_type: str = "circular",
    train_fraction: float = 0.8,
    random_state: int = 42,
) -> dict:
    """Compute decoding accuracy using simple linear decoder.

    This function trains a decoder to map from the embedding space to
    the true variables and measures how well it generalizes.

    Parameters
    ----------
    embedding : np.ndarray
        Low-dimensional embedding. Shape: (n_samples, n_dims).
    true_variable : np.ndarray
        Ground truth variable (angles for circular, positions for spatial).
        Shape: (n_samples,) for circular or 1D spatial.
    manifold_type : str, optional
        Type of manifold ('circular' or 'spatial'). Default is 'circular'.
    train_fraction : float, optional
        Fraction of data to use for training. Default is 0.8.
        Must be between 0 and 1.
    random_state : int, optional
        Random seed for train/test split reproducibility. Default is 42.

    Returns
    -------
    dict
        Dictionary containing:
        - 'train_error': float, training reconstruction error
        - 'test_error': float, testing reconstruction error
        - 'test_r2': float, proper R² score on test set
        - 'generalization_gap': float, difference (test_error - train_error)

    Notes
    -----
    Uses random (not sequential) split of data to avoid domain shift issues
    that can occur with temporal/spatial data. For reproducibility, the
    random_state parameter controls the split.    """
    from sklearn.model_selection import train_test_split

    # Split data randomly to avoid domain shift
    train_embedding, test_embedding, train_variable, test_variable = train_test_split(
        embedding, true_variable, test_size=1-train_fraction, random_state=random_state
    )

    # Train decoder
    decoder = train_simple_decoder(train_embedding, train_variable, manifold_type)

    # Compute training error using decoder predictions
    train_predictions = decoder(train_embedding)
    if manifold_type == "circular":
        train_error = np.mean(circular_distance(train_predictions, train_variable))
    else:
        train_error = np.mean(
            np.linalg.norm(train_predictions - train_variable, axis=1)
        )

    # Compute testing error using decoder predictions
    test_predictions = decoder(test_embedding)
    if manifold_type == "circular":
        test_error = np.mean(circular_distance(test_predictions, test_variable))
        # For circular data, compute R² using circular distances
        # Convert circular distances to "residuals" for R² calculation
        residuals = circular_distance(test_predictions, test_variable)
        # Total sum of squares for circular data using mean direction
        mean_direction = np.arctan2(np.mean(np.sin(test_variable)), np.mean(np.cos(test_variable)))
        total_residuals = circular_distance(test_variable, mean_direction)

        # R² calculation for circular data
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum(total_residuals**2)
        test_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        test_error = np.mean(np.linalg.norm(test_predictions - test_variable, axis=1))
        # For spatial data, compute standard R²
        if test_variable.ndim == 1:
            # 1D case
            ss_res = np.sum((test_variable - test_predictions)**2)
            ss_tot = np.sum((test_variable - np.mean(test_variable))**2)
            test_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            # Multi-dimensional case - compute R² per dimension and average
            r2_per_dim = []
            for dim in range(test_variable.shape[1]):
                y_true_dim = test_variable[:, dim]
                y_pred_dim = test_predictions[:, dim]
                ss_res_dim = np.sum((y_true_dim - y_pred_dim)**2)
                ss_tot_dim = np.sum((y_true_dim - np.mean(y_true_dim))**2)
                r2_dim = 1 - (ss_res_dim / ss_tot_dim) if ss_tot_dim > 0 else 0.0
                r2_per_dim.append(r2_dim)
            test_r2 = np.mean(r2_per_dim)

    return {
        "train_error": train_error,
        "test_error": test_error,
        "test_r2": test_r2,
        "generalization_gap": test_error - train_error,
    }


def manifold_reconstruction_score(
    embedding: np.ndarray,
    true_variable: np.ndarray,
    manifold_type: str = "circular",
    weights: Optional[dict] = None,
    allow_rotation: bool = True,
    allow_reflection: bool = True,
    allow_scaling: bool = True,
    random_state: int = 42,
) -> dict:
    """Compute comprehensive manifold reconstruction score.

    Parameters
    ----------
    embedding : np.ndarray
        Low-dimensional embedding. Shape: (n_samples, n_dims).
    true_variable : np.ndarray
        Ground truth variable (angles or positions).
    manifold_type : str, optional
        Type of manifold ('circular' or 'spatial'). Default is 'circular'.
    weights : dict, optional
        Weights for combining metrics. Keys should be 'reconstruction_error',
        'correlation', and 'decoding_accuracy'. If None, uses default weights
        that sum to 1.0.
    allow_rotation : bool, optional
        Whether to allow rotation when finding alignment. Default is True.
    allow_reflection : bool, optional
        Whether to allow reflection when finding alignment. Default is True.
    allow_scaling : bool, optional
        Whether to allow scaling when finding alignment. Default is True.
    random_state : int, optional
        Random seed for train/test split reproducibility. Default is 42.

    Returns
    -------
    dict
        Dictionary containing:
        - 'reconstruction_error': float, reconstruction error after alignment
        - 'correlation': float, correlation after alignment
        - 'rotation_offset': float, rotation offset applied
        - 'is_reflected': bool, whether reflection was applied
        - 'decoding_train_error': float, decoder training error
        - 'decoding_test_error': float, decoder test error
        - 'generalization_gap': float, decoder generalization gap
        - 'overall_reconstruction_score': float, weighted combination of metrics

    Notes
    -----
    For spatial manifolds, assumes data is normalized to unit scale for
    error normalization. Negative correlations are treated as 0 in scoring.
    The overall score is normalized to [0, 1] range where 1 is perfect.    """
    # Validate weights if provided
    if weights is not None:
        required_keys = {"reconstruction_error", "correlation", "decoding_accuracy"}
        if not all(key in weights for key in required_keys):
            raise ValueError(
                f"weights must contain keys {required_keys}, got {set(weights.keys())}"
            )
        if any(w < 0 for w in weights.values()):
            raise ValueError("All weights must be non-negative")
    
    if weights is None:
        weights = {
            "reconstruction_error": 0.4,
            "correlation": 0.3,
            "decoding_accuracy": 0.3,
        }

    # Compute metrics with proper alignment
    alignment_metrics = compute_embedding_alignment_metrics(
        embedding,
        true_variable,
        manifold_type,
        allow_rotation=allow_rotation,
        allow_reflection=allow_reflection,
        allow_scaling=allow_scaling,
    )

    reconstruction_error = alignment_metrics["error"]
    correlation = alignment_metrics["correlation"]

    # Use decoder-based accuracy for consistency
    decoding_results = compute_decoding_accuracy(
        embedding, true_variable, manifold_type, random_state=random_state
    )

    # Normalize reconstruction error (lower is better, so invert)
    max_error = np.pi if manifold_type == "circular" else 1.0  # Normalized for spatial
    normalized_error = 1.0 - min(reconstruction_error / max_error, 1.0)

    # Normalize decoding accuracy (lower test error is better)
    max_decode_error = np.pi if manifold_type == "circular" else 1.0
    normalized_decode = 1.0 - min(
        decoding_results["test_error"] / max_decode_error, 1.0
    )

    # Ensure correlation is positive for scoring
    correlation_score = max(correlation, 0.0)

    # Compute weighted score
    overall_score = (
        weights["reconstruction_error"] * normalized_error
        + weights["correlation"] * correlation_score
        + weights["decoding_accuracy"] * normalized_decode
    )

    return {
        "reconstruction_error": reconstruction_error,
        "correlation": correlation,
        "rotation_offset": alignment_metrics.get("rotation_offset", 0.0),
        "is_reflected": alignment_metrics.get("is_reflected", False),
        "decoding_train_error": decoding_results["train_error"],
        "decoding_test_error": decoding_results["test_error"],
        "generalization_gap": decoding_results["generalization_gap"],
        "overall_reconstruction_score": overall_score,
    }
