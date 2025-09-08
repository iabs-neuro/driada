"""
Core RSA functions for computing and comparing RDMs.
"""
from __future__ import annotations

import numpy as np
import logging
from typing import Dict, Tuple, Optional, Union, TYPE_CHECKING
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from ..dim_reduction.data import MVData
from ..utils.jit import is_jit_enabled
from .core_jit import (
    fast_euclidean_distance,
    fast_manhattan_distance,
    fast_average_patterns,
)

if TYPE_CHECKING:
    from ..experiment import Experiment


def compute_rdm(
    patterns: Union[np.ndarray, MVData],
    metric: str = "correlation",
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Compute representational dissimilarity matrix from patterns.

    Parameters
    ----------
    patterns : np.ndarray or MVData
        Pattern matrix of shape (n_items, n_features) if np.ndarray (will be transposed automatically)
        or MVData object
        Each row is a pattern/item, each column is a feature
    metric : str, default 'correlation'
        Distance metric: 'correlation', 'euclidean', 'cosine', 'manhattan'
    logger : logging.Logger, optional
        Logger instance for debugging (currently unused)

    Returns
    -------
    rdm : np.ndarray
        Representational dissimilarity matrix (n_items, n_items)    """
    # Convert to MVData if needed
    if isinstance(patterns, np.ndarray):
        # MVData expects (n_features, n_items) so transpose
        # RSA doesn't require non-zero columns since we're computing distances
        mvdata = MVData(patterns.T, allow_zero_columns=True)
    else:
        mvdata = patterns

    # Check if we can use JIT-compiled functions for raw numpy arrays
    if (
        isinstance(patterns, np.ndarray)
        and is_jit_enabled()
        and metric in ["euclidean", "manhattan"]
    ):
        # Use JIT only for euclidean and manhattan where it's simpler
        if metric == "euclidean":
            rdm = fast_euclidean_distance(patterns)
        elif metric == "manhattan":
            rdm = fast_manhattan_distance(patterns)
    elif isinstance(patterns, np.ndarray):
        # Use scipy for numpy arrays
        distances = pdist(patterns, metric=metric)
        rdm = squareform(distances)
    else:
        # Use MVData's built-in methods
        if metric == "correlation":
            # We need correlation between items (columns), so use axis=1
            corr_mat = mvdata.corr_mat(axis=1)
            rdm = 1 - corr_mat
        else:
            # get_distmat computes distances between columns (items)
            rdm = mvdata.get_distmat(metric)

    # Ensure diagonal is zero
    np.fill_diagonal(rdm, 0)

    # Ensure no negative values due to numerical errors
    rdm = np.maximum(rdm, 0)
    
    # Check for NaN or inf values
    if np.any(np.isnan(rdm)) or np.any(np.isinf(rdm)):
        import warnings
        warnings.warn(
            "RDM contains NaN or infinite values. This may indicate "
            "constant features or numerical instability.",
            RuntimeWarning
        )

    return rdm


def compute_rdm_from_timeseries_labels(
    data: np.ndarray,
    labels: np.ndarray,
    metric: str = "correlation",
    average_method: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RDM from time series data using behavioral variable labels.

    Parameters
    ----------
    data : np.ndarray
        Data array of shape (n_features, n_timepoints)
    labels : np.ndarray
        Label for each timepoint, shape (n_timepoints,)
        Each unique label defines a condition/item
    metric : str, default 'correlation'
        Distance metric for RDM computation
    average_method : str, default 'mean'
        How to average within conditions: 'mean' or 'median'

    Returns
    -------
    rdm : np.ndarray
        Representational dissimilarity matrix
    unique_labels : np.ndarray
        The unique labels in order as they appear in the RDM    """
    # Get unique labels (conditions)
    unique_labels = np.unique(labels)

    # Use JIT-compiled averaging if available and using mean
    if is_jit_enabled() and average_method == "mean":
        pattern_matrix = fast_average_patterns(data, labels, unique_labels)
    else:
        # Extract patterns for each condition
        patterns = []
        for label in unique_labels:
            mask = labels == label
            condition_data = data[:, mask]

            if average_method == "mean":
                pattern = np.mean(condition_data, axis=1)
            elif average_method == "median":
                pattern = np.median(condition_data, axis=1)
            else:
                raise ValueError(f"Unknown average method: {average_method}")

            patterns.append(pattern)

        # Stack into pattern matrix (n_conditions, n_features)
        pattern_matrix = np.array(patterns)

    # Compute RDM
    rdm = compute_rdm(pattern_matrix, metric=metric)

    return rdm, unique_labels


def compute_rdm_from_trials(
    data: np.ndarray,
    trial_starts: np.ndarray,
    trial_labels: np.ndarray,
    trial_duration: Optional[int] = None,
    metric: str = "correlation",
    average_method: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RDM from time series data using explicit trial structure.

    Parameters
    ----------
    data : np.ndarray
        Data array of shape (n_features, n_timepoints)
    trial_starts : np.ndarray
        Start indices for each trial
    trial_labels : np.ndarray
        Label for each trial (same length as trial_starts)
    trial_duration : int, optional
        Fixed duration for each trial. If None, uses time until next trial
    metric : str, default 'correlation'
        Distance metric for RDM computation
    average_method : str, default 'mean'
        How to average within trial: 'mean' or 'median'

    Returns
    -------
    rdm : np.ndarray
        Representational dissimilarity matrix
    unique_labels : np.ndarray
        The unique trial labels in order as they appear in the RDM    """
    # Check if trial_starts are sorted
    if len(trial_starts) > 1 and not np.all(np.diff(trial_starts) > 0):
        raise ValueError("trial_starts must be sorted in ascending order")
    
    n_timepoints = data.shape[1]
    n_trials = len(trial_starts)

    # Extract data for each trial
    trial_patterns = []
    trial_labels_list = []

    for i, (start, label) in enumerate(zip(trial_starts, trial_labels)):
        # Validate trial start
        if start < 0 or start >= n_timepoints:
            raise ValueError(
                f"Trial {i} start index {start} is out of bounds [0, {n_timepoints})"
            )
        
        # Determine trial end
        if trial_duration is not None:
            end = min(start + trial_duration, n_timepoints)
        else:
            # Use next trial start or end of data
            if i < n_trials - 1:
                end = trial_starts[i + 1]
                # Validate next trial start
                if end < 0 or end > n_timepoints:
                    raise ValueError(
                        f"Trial {i+1} start index {end} is out of bounds [0, {n_timepoints}]"
                    )
            else:
                end = n_timepoints

        # Extract trial data
        trial_data = data[:, start:end]

        if trial_data.shape[1] > 0:  # Only process non-empty trials
            if average_method == "mean":
                pattern = np.mean(trial_data, axis=1)
            elif average_method == "median":
                pattern = np.median(trial_data, axis=1)
            else:
                raise ValueError(f"Unknown average method: {average_method}")

            trial_patterns.append(pattern)
            trial_labels_list.append(label)

    # Get unique labels and average patterns for each condition
    unique_labels = np.unique(trial_labels_list)
    condition_patterns = []

    for label in unique_labels:
        # Find all patterns for this label
        label_patterns = [
            pattern
            for pattern, l in zip(trial_patterns, trial_labels_list)
            if l == label
        ]

        # Average across repetitions of the same condition
        if len(label_patterns) > 0:
            if average_method == "mean":
                avg_pattern = np.mean(label_patterns, axis=0)
            elif average_method == "median":
                avg_pattern = np.median(label_patterns, axis=0)
            else:
                # This should never happen due to earlier validation
                raise ValueError(f"Unknown average method: {average_method}")
            condition_patterns.append(avg_pattern)

    # Stack into pattern matrix
    pattern_matrix = np.array(condition_patterns)

    # Compute RDM
    rdm = compute_rdm(pattern_matrix, metric=metric)

    return rdm, unique_labels


def compare_rdms(rdm1: np.ndarray, rdm2: np.ndarray, method: str = "spearman") -> float:
    """
    Compare two representational dissimilarity matrices.
    
    Quantifies the similarity between two RDMs using correlation or
    cosine similarity metrics. Only the upper triangular portion
    (excluding diagonal) is compared since RDMs are symmetric.

    Parameters
    ----------
    rdm1 : np.ndarray
        First RDM, square symmetric matrix of shape (n_items, n_items).
    rdm2 : np.ndarray
        Second RDM, must have the same shape as rdm1.
    method : str, default 'spearman'
        Comparison method:
        - 'spearman': Spearman rank correlation (robust to monotonic transforms)
        - 'pearson': Pearson correlation (assumes linear relationship)
        - 'kendall': Kendall's tau (robust but slower, O(n²) complexity)
        - 'cosine': Cosine similarity (angle between RDM vectors)

    Returns
    -------
    float
        Similarity score between RDMs:
        - Correlations ('spearman', 'pearson', 'kendall'): Range [-1, 1]
        - Cosine similarity: Range [0, 1] (NaN if either RDM has zero norm)
        - Returns NaN if correlation cannot be computed (e.g., constant RDMs)
        
    Raises
    ------
    ValueError
        If RDMs have different shapes.
        If method is not one of the supported options.
    RuntimeWarning
        If either RDM contains NaN values (via warnings.warn).
        
    Notes
    -----
    Only upper triangular values are used since RDMs are symmetric
    and diagonal is uninformative (always 0).
    
    P-values from statistical tests are computed internally but not
    returned. Use bootstrap_rdm_comparison for statistical inference.
    
    Kendall's tau is more robust than Spearman but has O(n²) complexity
    in the number of unique values, making it slow for large RDMs.
    
    For cosine similarity, if either RDM vector has zero norm (all values
    identical), the function returns NaN and issues a warning.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create two similar RDMs
    >>> rdm1 = np.array([[0, 0.5, 0.8], [0.5, 0, 0.3], [0.8, 0.3, 0]])
    >>> rdm2 = np.array([[0, 0.6, 0.7], [0.6, 0, 0.4], [0.7, 0.4, 0]])
    
    >>> # Compare using different methods
    >>> pearson_sim = compare_rdms(rdm1, rdm2, method='pearson')
    >>> print(f"Pearson correlation: {pearson_sim:.3f}")
    Pearson correlation: 0.954
    
    >>> spearman_sim = compare_rdms(rdm1, rdm2, method='spearman')
    >>> print(f"Spearman correlation: {spearman_sim:.3f}")
    Spearman correlation: 1.000
    
    >>> # Cosine similarity
    >>> cosine_sim = compare_rdms(rdm1, rdm2, method='cosine')
    >>> print(f"Cosine similarity: {cosine_sim:.3f}")
    Cosine similarity: 0.985
    
    See Also
    --------
    ~driada.rsa.core.compute_rdm : Compute RDMs from neural patterns
    ~driada.rsa.core.bootstrap_rdm_comparison : Statistical comparison with confidence intervals
    ~driada.rsa.core.rsa_compare : High-level interface for comparing datasets    """
    # Ensure RDMs are same shape
    if rdm1.shape != rdm2.shape:
        raise ValueError(
            f"RDMs must have the same shape. Got {rdm1.shape} and {rdm2.shape}"
        )

    # Extract upper triangular parts (excluding diagonal)
    mask = np.triu(np.ones_like(rdm1, dtype=bool), k=1)
    rdm1_vec = rdm1[mask]
    rdm2_vec = rdm2[mask]
    
    # Check for NaN values
    if np.any(np.isnan(rdm1_vec)) or np.any(np.isnan(rdm2_vec)):
        import warnings
        warnings.warn(
            "RDMs contain NaN values. Correlation may return NaN.",
            RuntimeWarning
        )

    # Compute similarity
    if method == "spearman":
        similarity, _ = stats.spearmanr(rdm1_vec, rdm2_vec)
    elif method == "pearson":
        similarity, _ = stats.pearsonr(rdm1_vec, rdm2_vec)
    elif method == "kendall":
        similarity, _ = stats.kendalltau(rdm1_vec, rdm2_vec)
    elif method == "cosine":
        # Cosine similarity
        dot_product = np.dot(rdm1_vec, rdm2_vec)
        norm1 = np.linalg.norm(rdm1_vec)
        norm2 = np.linalg.norm(rdm2_vec)
        
        if norm1 == 0 or norm2 == 0:
            import warnings
            warnings.warn(
                "One or both RDMs have zero norm (all values identical). "
                "Cosine similarity is undefined, returning NaN.",
                RuntimeWarning
            )
            similarity = np.nan
        else:
            similarity = dot_product / (norm1 * norm2)
    else:
        raise ValueError(f"Unknown comparison method: {method}")

    return similarity


def bootstrap_rdm_comparison(
    data1: np.ndarray,
    data2: np.ndarray,
    labels1: np.ndarray,
    labels2: np.ndarray,
    metric: str = "correlation",
    comparison_method: str = "spearman",
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Bootstrap test for RDM similarity between two datasets.
    
    Performs statistical inference on RDM similarity using within-condition
    resampling. This maintains the experimental design while estimating
    confidence intervals and assessing reliability of the similarity.

    Parameters
    ----------
    data1 : np.ndarray
        First dataset of shape (n_features1, n_timepoints). Features can
        be different between datasets (e.g., comparing V1 vs V2 neurons).
    data2 : np.ndarray
        Second dataset of shape (n_features2, n_timepoints). Must have
        the same number of timepoints as data1.
    labels1 : np.ndarray
        Condition labels for each timepoint in data1, shape (n_timepoints,).
    labels2 : np.ndarray
        Condition labels for each timepoint in data2, shape (n_timepoints,).
        Must contain the same unique values as labels1.
    metric : str, default 'correlation'
        Distance metric for RDM computation. See compute_rdm for options.
    comparison_method : str, default 'spearman'
        Method for comparing RDMs. See compare_rdms for options.
    n_bootstrap : int, default 1000
        Number of bootstrap iterations. Higher values give more stable
        estimates but take longer.
    random_state : int, optional
        Random seed for reproducibility. Creates a local RandomState
        to avoid affecting global random state.

    Returns
    -------
    dict
        Dictionary containing:
        - 'observed': float, observed RDM similarity between datasets
        - 'bootstrap_distribution': np.ndarray, bootstrap similarity values
        - 'p_value': float, two-tailed test of observed vs bootstrap mean
        - 'ci_lower': float, 2.5th percentile of bootstrap distribution
        - 'ci_upper': float, 97.5th percentile of bootstrap distribution
        - 'mean': float, mean of bootstrap distribution
        - 'std': float, standard deviation of bootstrap distribution
        
    Raises
    ------
    ValueError
        If datasets don't have the same unique condition labels.
        
    Notes
    -----
    The bootstrap procedure:
    1. Resamples timepoints within each condition independently
    2. Maintains the number of samples per condition
    3. Computes RDMs from resampled data
    4. Calculates similarity between resampled RDMs
    
    This within-condition resampling preserves the experimental design
    while capturing trial-by-trial variability.
    
    The p-value tests whether the observed similarity is extreme
    relative to the bootstrap distribution mean. This is NOT a
    standard null hypothesis test but rather a stability assessment.
    
    Uses a local RandomState to avoid modifying global numpy random
    state, ensuring thread safety and reproducibility.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create two datasets with different patterns but similar structure
    >>> np.random.seed(42)
    >>> n_features = 20
    >>> n_timepoints = 90
    >>> 
    >>> # 3 conditions, 30 samples each
    >>> labels = np.repeat(['A', 'B', 'C'], 30)
    >>> 
    >>> # Create data with condition-specific patterns
    >>> v1_data = np.random.randn(n_features, n_timepoints)
    >>> v2_data = np.random.randn(n_features, n_timepoints)
    >>> 
    >>> # Bootstrap will resample and compute similarity distribution
    >>> results = bootstrap_rdm_comparison(
    ...     v1_data, v2_data, labels, labels,
    ...     n_bootstrap=50, random_state=42
    ... )
    >>> 
    >>> # Check that results contain expected keys
    >>> print('Keys:', sorted(results.keys()))
    Keys: ['bootstrap_distribution', 'ci_lower', 'ci_upper', 'mean', 'observed', 'p_value', 'std']
    >>> 
    >>> # Random data should give low correlation
    >>> print(f"Observed between -1 and 1: {-1 <= results['observed'] <= 1}")
    Observed between -1 and 1: True
    
    See Also
    --------
    ~driada.rsa.core.compare_rdms : Direct RDM comparison without bootstrap
    ~driada.rsa.core.compute_rdm_from_timeseries_labels : Compute RDM from labeled data
    ~driada.rsa.core.rsa_compare : High-level interface with multiple data types    """
    # Create a local random number generator to avoid modifying global state
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random.RandomState()

    # Compute observed RDM similarity
    rdm1, _ = compute_rdm_from_timeseries_labels(data1, labels1, metric=metric)
    rdm2, _ = compute_rdm_from_timeseries_labels(data2, labels2, metric=metric)
    observed_similarity = compare_rdms(rdm1, rdm2, method=comparison_method)

    # Get unique conditions
    unique_conditions1 = np.unique(labels1)
    unique_conditions2 = np.unique(labels2)
    
    # Check that both datasets have same conditions
    if not np.array_equal(unique_conditions1, unique_conditions2):
        raise ValueError("Both datasets must have the same set of condition labels")
    
    # Bootstrap with within-condition resampling
    bootstrap_similarities = []

    for _ in range(n_bootstrap):
        # Resample within each condition to maintain balance
        idx1 = []
        idx2 = []
        
        for condition in unique_conditions1:
            # Get indices for this condition
            cond_idx1 = np.where(labels1 == condition)[0]
            cond_idx2 = np.where(labels2 == condition)[0]
            
            # Resample with replacement within condition
            n_samples1 = len(cond_idx1)
            n_samples2 = len(cond_idx2)
            
            resampled_idx1 = rng.choice(cond_idx1, size=n_samples1, replace=True)
            resampled_idx2 = rng.choice(cond_idx2, size=n_samples2, replace=True)
            
            idx1.extend(resampled_idx1)
            idx2.extend(resampled_idx2)
        
        # Convert to arrays and shuffle to mix conditions
        idx1 = np.array(idx1)
        idx2 = np.array(idx2)
        rng.shuffle(idx1)
        rng.shuffle(idx2)

        # Compute RDMs on resampled data
        rdm1_boot, _ = compute_rdm_from_timeseries_labels(
            data1[:, idx1], labels1[idx1], metric=metric
        )
        rdm2_boot, _ = compute_rdm_from_timeseries_labels(
            data2[:, idx2], labels2[idx2], metric=metric
        )

        # Compare
        sim_boot = compare_rdms(rdm1_boot, rdm2_boot, method=comparison_method)
        bootstrap_similarities.append(sim_boot)

    bootstrap_similarities = np.array(bootstrap_similarities)

    # Compute statistics
    mean_similarity = np.mean(bootstrap_similarities)
    std_similarity = np.std(bootstrap_similarities)
    
    # Compute p-value (two-tailed) - tests if observed is extreme relative to bootstrap mean
    p_value = np.mean(
        np.abs(bootstrap_similarities - mean_similarity)
        >= np.abs(observed_similarity - mean_similarity)
    )

    # Compute confidence intervals
    ci_lower = np.percentile(bootstrap_similarities, 2.5)
    ci_upper = np.percentile(bootstrap_similarities, 97.5)

    return {
        "observed": observed_similarity,
        "bootstrap_distribution": bootstrap_similarities,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "mean": mean_similarity,
        "std": std_similarity,
    }


# Unified API function
def compute_rdm_unified(
    data: Union[np.ndarray, MVData, Experiment],
    items: Optional[Union[np.ndarray, str, Dict]] = None,
    data_type: str = "calcium",
    metric: str = "correlation",
    average_method: str = "mean",
    trial_duration: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute RDM with automatic data type detection and dispatching.

    This function intelligently dispatches to the appropriate RDM computation
    method based on the input data type and items specification. It provides
    a unified interface for computing RDMs from various data structures
    (arrays, MVData, Experiments) and item definitions (pre-averaged patterns,
    timeseries labels, or trial structures).

    Parameters
    ----------
    data : np.ndarray, MVData, or Experiment
        The data to compute RDM from:
        - np.ndarray: Raw data matrix (n_features, n_timepoints)
        - MVData: MVData object
        - Experiment: DRIADA Experiment object
    items : np.ndarray, str, dict, or None
        How to define items/conditions:
        - None: Compute RDM directly from patterns (requires pre-averaged data)
        - np.ndarray: Condition labels for each timepoint
        - str: Name of dynamic feature (for Experiment objects)
        - dict: Trial structure with 'trial_starts' and 'trial_labels'
    data_type : str, default 'calcium'
        For Experiment objects, which data type to use ('calcium' or 'spikes')
    metric : str, default 'correlation'
        Distance metric for RDM computation. Options: 'correlation',
        'euclidean', 'cosine', 'manhattan'
    average_method : str, default 'mean'
        How to average within conditions ('mean' or 'median')
    trial_duration : int, optional
        For trial structure, fixed duration for each trial. If specified
        in both parameter and items dict, dict value takes precedence.

    Returns
    -------
    rdm : np.ndarray
        Representational dissimilarity matrix
    labels : np.ndarray or None
        The unique labels/conditions if items were specified
        
    Raises
    ------
    ValueError
        If items required but not provided for Experiment objects.
        If trial structure dict missing required keys.
        If metric is not one of the valid options.
        If MVData/Embedding used with trial structure.
        
    Notes
    -----
    Imports are performed inside the function to avoid circular
    dependencies. This has minimal performance impact as the function
    is typically called only a few times per analysis.
    
    When trial_duration is specified in both the items dict and as a
    parameter, the dict value takes precedence and a warning is issued.

    Examples
    --------
    >>> import numpy as np
    >>> # Direct pattern RDM (pre-averaged data)
    >>> patterns = np.random.randn(10, 50)  # 10 items, 50 features
    >>> rdm, _ = compute_rdm_unified(patterns)
    >>> print(f"RDM shape: {rdm.shape}")
    RDM shape: (10, 10)

    >>> # From time series with labels
    >>> data = np.random.randn(100, 1000)  # 100 features, 1000 timepoints
    >>> labels = np.repeat([0, 1, 2, 3], 250)
    >>> rdm, unique_labels = compute_rdm_unified(data, labels)
    >>> print(f"RDM shape: {rdm.shape}, unique labels: {unique_labels}")
    RDM shape: (4, 4), unique labels: [0 1 2 3]

    >>> # From MVData object
    >>> from driada.dim_reduction.data import MVData
    >>> mvdata = MVData(np.random.randn(50, 100))  # 50 features, 100 samples
    >>> rdm, _ = compute_rdm_unified(mvdata)
    >>> print(f"RDM shape: {rdm.shape}")
    RDM shape: (100, 100)
    
    See Also
    --------
    ~driada.rsa.core.compute_rdm : Direct RDM computation from patterns
    ~driada.rsa.core.compute_rdm_from_timeseries_labels : RDM from labeled timeseries
    ~driada.rsa.core.compute_rdm_from_trials : RDM from trial structure
    ~driada.rsa.integration.compute_experiment_rdm : RDM from Experiment objects    """
    # Import here to avoid circular dependency
    from ..experiment import Experiment
    from ..dim_reduction.embedding import Embedding
    from .integration import compute_experiment_rdm, compute_mvdata_rdm

    # Handle Embedding objects
    if isinstance(data, Embedding):
        # Convert embedding to MVData and process
        mvdata = data.to_mvdata()
        if items is None:
            # Direct pattern RDM - embedding already represents patterns
            return compute_rdm(mvdata, metric=metric), None
        else:
            # Items specified - use MVData processing
            if isinstance(items, dict):
                raise ValueError(
                    "Trial structure not supported for Embedding objects. Use labels array instead."
                )
            return compute_mvdata_rdm(
                mvdata, items, metric=metric, average_method=average_method
            )

    # Handle Experiment objects
    elif isinstance(data, Experiment):
        if items is None:
            raise ValueError("items must be specified for Experiment objects")
        return compute_experiment_rdm(
            data,
            items,
            data_type=data_type,
            metric=metric,
            average_method=average_method,
        )

    # Handle MVData objects
    elif isinstance(data, MVData):
        if items is None:
            # Direct pattern RDM - assume data is already averaged
            # MVData stores as (n_features, n_items)
            return compute_rdm(data, metric=metric), None
        else:
            if isinstance(items, dict):
                # Trial structure not directly supported for MVData
                raise ValueError(
                    "Trial structure not supported for MVData. Use labels array instead."
                )
            # Items should be labels array
            return compute_mvdata_rdm(
                data, items, metric=metric, average_method=average_method
            )

    # Handle numpy arrays
    else:
        # Validate metric
        valid_metrics = ["correlation", "euclidean", "cosine", "manhattan"]
        if metric not in valid_metrics:
            raise ValueError(
                f"Invalid metric '{metric}'. Must be one of {valid_metrics}"
            )
            
        if items is None:
            # Direct pattern RDM - assume rows are patterns
            return compute_rdm(data, metric=metric), None
        elif isinstance(items, dict):
            # Trial structure
            # Handle trial_duration conflict
            if "trial_duration" in items and trial_duration is not None:
                import warnings
                warnings.warn(
                    "trial_duration specified in both items dict and parameter. "
                    "Using value from items dict.",
                    UserWarning
                )
            return compute_rdm_from_trials(
                data,
                trial_starts=items["trial_starts"],
                trial_labels=items["trial_labels"],
                trial_duration=items.get("trial_duration", trial_duration),
                metric=metric,
                average_method=average_method,
            )
        else:
            # Labels array
            return compute_rdm_from_timeseries_labels(
                data, items, metric=metric, average_method=average_method
            )


# Simplified high-level API for common use case
def rsa_compare(
    data1: Union[np.ndarray, MVData, Experiment],
    data2: Union[np.ndarray, MVData, Experiment],
    items: Optional[Union[str, Dict]] = None,
    metric: str = "correlation",
    comparison: str = "spearman",
    data_type: str = "calcium",
    logger: Optional[logging.Logger] = None,
) -> float:
    """
    Compare neural representations using RSA.

    This is a simplified API for the most common RSA use case: comparing
    two sets of neural representations. It automatically handles different
    data types (arrays, MVData, Embeddings, Experiments) and computes the
    similarity between their representational geometries.

    Parameters
    ----------
    data1 : np.ndarray, MVData, or Experiment
        First dataset (n_items, n_features) if array, MVData object, or Experiment
    data2 : np.ndarray, MVData, or Experiment
        Second dataset (same n_items as data1)
    items : str, dict, or None
        How to define conditions (required for Experiment objects):
        - None: For arrays/MVData, assumes data is already averaged per item
        - str: Name of dynamic feature (e.g., 'stimulus_type')
        - dict: Trial structure with 'trial_starts' and 'trial_labels'
    metric : str, default 'correlation'
        Distance metric for RDM computation
    comparison : str, default 'spearman'
        Method for comparing RDMs ('spearman', 'pearson', 'kendall', 'cosine')
    data_type : str, default 'calcium'
        For Experiment objects, which data to use ('calcium' or 'spikes')
    logger : logging.Logger, optional
        Logger for debugging messages

    Returns
    -------
    similarity : float
        Similarity score between the two neural representations.
        Range depends on comparison method: [-1, 1] for correlations,
        [0, 1] for cosine similarity.
        
    Raises
    ------
    ValueError
        If items not specified for Experiment objects.
        If trying to compare Experiment with non-Experiment data.
        If RDMs have incompatible shapes (different numbers of items).
        
    Notes
    -----
    Imports are performed inside the function to avoid circular
    dependencies. Embedding objects are automatically converted to
    MVData for uniform processing.
    
    When comparing arrays or MVData without items specification,
    assumes the data is already averaged per condition (each row
    represents one item/condition).

    Examples
    --------
    >>> import numpy as np
    >>> # Compare two brain areas with structured data
    >>> np.random.seed(42)
    >>> n_stimuli = 5
    >>> n_neurons_v1, n_neurons_v2 = 20, 15
    >>> 
    >>> # Create orthogonal patterns for each stimulus
    >>> v1_data = np.zeros((n_stimuli, n_neurons_v1))
    >>> v2_data = np.zeros((n_stimuli, n_neurons_v2))
    >>> 
    >>> # Each stimulus activates different neurons in both areas
    >>> for i in range(n_stimuli):
    ...     # V1: each stimulus activates 4 specific neurons
    ...     v1_data[i, i*4:(i+1)*4] = 1.0
    ...     # V2: similar pattern with 3 neurons per stimulus
    ...     v2_data[i, i*3:(i+1)*3] = 1.0
    >>> 
    >>> # Add small noise for realism
    >>> v1_data += 0.1 * np.random.randn(n_stimuli, n_neurons_v1)
    >>> v2_data += 0.1 * np.random.randn(n_stimuli, n_neurons_v2)
    >>> 
    >>> # This creates similar RDM structure in both areas
    >>> similarity = rsa_compare(v1_data, v2_data, comparison='spearman')
    >>> print(f"RSA similarity: {similarity:.3f}")
    RSA similarity: 0.479

    >>> # Compare using compute_rdm_unified first
    >>> from driada.rsa import compute_rdm_unified
    >>> np.random.seed(123)
    >>> data1 = np.random.randn(50, 90)  # 50 features, 90 timepoints
    >>> data2 = np.random.randn(40, 90)  # 40 features, same timepoints
    >>> labels = np.repeat(['A', 'B', 'C'], 30)  # 30 samples per condition
    >>> # First compute RDMs with labels
    >>> rdm1, _ = compute_rdm_unified(data1, items=labels)
    >>> rdm2, _ = compute_rdm_unified(data2, items=labels) 
    >>> # Both RDMs now have shape (3, 3) for the 3 conditions
    >>> print(f"RDM shapes: {rdm1.shape}, {rdm2.shape}")
    RDM shapes: (3, 3), (3, 3)
    >>> # Compare the RDMs
    >>> from driada.rsa import compare_rdms
    >>> similarity = compare_rdms(rdm1, rdm2)
    >>> print(f"RSA similarity between -1 and 1: {-1 <= similarity <= 1}")
    RSA similarity between -1 and 1: True
    
    See Also
    --------
    ~driada.rsa.core.compute_rdm_unified : Unified RDM computation interface
    ~driada.rsa.core.compare_rdms : Direct comparison of RDM matrices
    ~driada.rsa.core.bootstrap_rdm_comparison : Statistical comparison with confidence intervals    """
    # Import here to avoid circular dependency
    from ..experiment import Experiment
    from ..dim_reduction.embedding import Embedding
    from .integration import rsa_between_experiments

    if logger is None:
        logger = logging.getLogger(__name__)

    # Handle Embedding objects
    if isinstance(data1, Embedding) or isinstance(data2, Embedding):
        # Convert embeddings to MVData for uniform processing
        if isinstance(data1, Embedding):
            data1 = data1.to_mvdata()
        if isinstance(data2, Embedding):
            data2 = data2.to_mvdata()
        logger.debug("Converted Embedding objects to MVData for RSA comparison")

    # Check if both are Experiment objects
    if isinstance(data1, Experiment) and isinstance(data2, Experiment):
        if items is None:
            raise ValueError(
                "items must be specified when comparing Experiment objects"
            )
        logger.debug("Comparing two Experiment objects using RSA")
        return rsa_between_experiments(
            data1,
            data2,
            items=items,
            data_type=data_type,
            metric=metric,
            comparison_method=comparison,
            average_method="mean",
        )

    # Check if mixed types
    elif isinstance(data1, Experiment) or isinstance(data2, Experiment):
        raise ValueError(
            "Cannot compare Experiment with non-Experiment data. Both inputs must be same type."
        )

    # Original behavior for arrays/MVData
    else:
        logger.debug("Computing RDMs for RSA comparison")

        # Compute RDMs
        rdm1 = compute_rdm(data1, metric=metric, logger=logger)
        rdm2 = compute_rdm(data2, metric=metric, logger=logger)
        
        # Check dimension compatibility
        if rdm1.shape != rdm2.shape:
            raise ValueError(
                f"RDMs have incompatible shapes: {rdm1.shape} vs {rdm2.shape}. "
                "Ensure both datasets have the same number of items/conditions."
            )

        # Compare RDMs
        similarity = compare_rdms(rdm1, rdm2, method=comparison)

        logger.debug(f"RSA comparison complete. Similarity: {similarity:.3f}")

        return similarity
