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
from typing import Optional, Tuple


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
    flexibility_factor: float = 2.0,
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

    # Normalize
    max_trust = (n_samples - k - 1) * k * n_samples / 2
    if max_trust > 0:
        trust = 1 - (2 * trust / max_trust)
    else:
        trust = 1.0

    return trust


def continuity(X_high: np.ndarray, X_low: np.ndarray, k: int = 10) -> float:
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
        Correlation coefficient between -1 and 1
    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import shortest_path

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

    # Compute correlation
    if method == "spearman":
        corr, _ = spearmanr(geodesic_flat, euclidean_flat)
    else:  # pearson
        corr = np.corrcoef(geodesic_flat, euclidean_flat)[0, 1]

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
        Stress value (lower is better)
    """
    # Compute distance matrices
    dist_high = compute_distance_matrix(X_high)
    dist_low = compute_distance_matrix(X_low)

    # Compute stress
    diff = dist_high - dist_low
    stress_val = np.sum(diff**2)

    if normalized:
        stress_val /= np.sum(dist_high**2)

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
    """
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


def circular_diff(angles: np.ndarray) -> np.ndarray:
    """Compute differences between consecutive angles, handling wrapping

    Parameters:
    -----------
    angles : np.ndarray
        Array of angles in radians

    Returns:
    --------
    np.ndarray
        Wrapped differences in [-pi, pi]
    """
    diffs = np.diff(angles)
    # Wrap differences to [-pi, pi]
    return np.arctan2(np.sin(diffs), np.cos(diffs))


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
    """

    def compute_mean_circular_distance(angles1, angles2, offset=0):
        """Compute mean circular distance with optional offset."""
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
    """Compute circular correlation coefficient.

    Parameters
    ----------
    angles1, angles2 : np.ndarray
        Arrays of angles in radians
    offset : float
        Rotation offset to apply to angles2

    Returns
    -------
    float
        Circular correlation coefficient in [-1, 1]
    """
    # Convert to unit complex numbers
    z1 = np.exp(1j * angles1)
    z2 = np.exp(1j * (angles2 + offset))

    # Compute circular correlation
    n = len(z1)
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

    if denominator > 0:
        return numerator / denominator
    else:
        return 0.0


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
    """
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
        aligned_embedding, disparity = procrustes_analysis(
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
            "is_reflected": False,  # TODO: detect from Procrustes
            "scale_factor": 1.0,  # TODO: extract from Procrustes
        }

    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")


def compute_temporal_consistency(
    embedding: np.ndarray, true_variable: np.ndarray, manifold_type: str = "circular"
) -> float:
    """[DEPRECATED] Use compute_embedding_alignment_metrics instead.

    This function is deprecated because it doesn't properly handle arbitrary
    rotations and reflections in circular data. Use compute_embedding_alignment_metrics
    which properly accounts for all allowed transformations.

    Parameters
    ----------
    embedding : np.ndarray
        Low-dimensional embedding
    true_variable : np.ndarray
        Ground truth variable (angles or positions)
    manifold_type : str
        Type of manifold ('circular' or 'spatial')

    Returns
    -------
    float
        Temporal consistency score (correlation)
    """
    import warnings

    warnings.warn(
        "compute_temporal_consistency is deprecated and will be removed in a future version. "
        "Use compute_embedding_alignment_metrics instead, which properly handles "
        "rotation and reflection transformations.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Call the new function and extract velocity correlation
    metrics = compute_embedding_alignment_metrics(
        embedding,
        true_variable,
        manifold_type,
        allow_rotation=True,
        allow_reflection=True,
        allow_scaling=True,
    )

    # Return velocity correlation if available, otherwise correlation
    return metrics.get("velocity_correlation", metrics.get("correlation", 0.0))


def compute_embedding_alignment_metrics(
    embedding: np.ndarray,
    true_variable: np.ndarray,
    manifold_type: str = "circular",
    allow_rotation: bool = True,
    allow_reflection: bool = True,
    allow_scaling: bool = True,
) -> dict:
    """Compute comprehensive metrics after optimal alignment.

    This function replaces the deprecated compute_temporal_consistency with a more
    general approach that properly handles transformations.

    Parameters
    ----------
    embedding : np.ndarray
        Low-dimensional embedding
    true_variable : np.ndarray
        Ground truth variable
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
        Dictionary containing all alignment metrics
    """
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

        if len(true_vel) > 1:
            vel_corr = np.corrcoef(true_vel, recon_vel)[0, 1]
            metrics["velocity_correlation"] = (
                vel_corr if not np.isnan(vel_corr) else 0.0
            )
        else:
            metrics["velocity_correlation"] = 0.0

        # Add circular variance preservation
        true_var = 1 - np.abs(np.mean(np.exp(1j * true_variable)))
        recon_var = 1 - np.abs(np.mean(np.exp(1j * reconstructed_angles)))
        metrics["variance_ratio"] = recon_var / true_var if true_var > 0 else 1.0

    return metrics


def train_simple_decoder(
    embedding: np.ndarray, true_variable: np.ndarray, manifold_type: str = "circular"
):
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
            new_embedding_scaled = scaler.transform(new_embedding)
            pred_sin = sin_regressor.predict(new_embedding_scaled)
            pred_cos = cos_regressor.predict(new_embedding_scaled)
            return np.arctan2(pred_sin, pred_cos)

    elif manifold_type == "spatial":
        # For spatial variables, direct regression
        regressor = LinearRegression().fit(embedding_scaled, true_variable)

        def decoder(new_embedding):
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
) -> dict:
    """Compute embedding quality metrics without using decoders

    This function directly measures how well the embedding preserves the
    structure of the manifold by extracting angles (for circular) or
    using Procrustes alignment (for spatial).

    Parameters:
    -----------
    embedding : np.ndarray
        Low-dimensional embedding
    true_variable : np.ndarray
        Ground truth variable (angles or positions)
    manifold_type : str
        Type of manifold ('circular' or 'spatial')
    train_fraction : float
        Fraction of data to use for training set

    Returns:
    --------
    dict
        Dictionary containing reconstruction errors for train/test splits
    """
    n_samples = embedding.shape[0]
    n_train = int(n_samples * train_fraction)

    # Split data
    train_embedding = embedding[:n_train]
    test_embedding = embedding[n_train:]
    train_variable = true_variable[:n_train]
    test_variable = true_variable[n_train:]

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
) -> dict:
    """Compute decoding accuracy using simple linear decoder

    This function trains a decoder to map from the embedding space to
    the true variables and measures how well it generalizes.

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
    else:
        test_error = np.mean(np.linalg.norm(test_predictions - test_variable, axis=1))

    return {
        "train_error": train_error,
        "test_error": test_error,
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
        embedding, true_variable, manifold_type
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
