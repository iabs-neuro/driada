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

    Returns
    -------
    rdm : np.ndarray
        Representational dissimilarity matrix (n_items, n_items)
    """
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
        The unique labels in order as they appear in the RDM
    """
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
        The unique trial labels in order as they appear in the RDM
    """
    n_timepoints = data.shape[1]
    n_trials = len(trial_starts)

    # Extract data for each trial
    trial_patterns = []
    trial_labels_list = []

    for i, (start, label) in enumerate(zip(trial_starts, trial_labels)):
        # Determine trial end
        if trial_duration is not None:
            end = min(start + trial_duration, n_timepoints)
        else:
            # Use next trial start or end of data
            if i < n_trials - 1:
                end = trial_starts[i + 1]
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
            avg_pattern = np.mean(label_patterns, axis=0)
            condition_patterns.append(avg_pattern)

    # Stack into pattern matrix
    pattern_matrix = np.array(condition_patterns)

    # Compute RDM
    rdm = compute_rdm(pattern_matrix, metric=metric)

    return rdm, unique_labels


def compare_rdms(rdm1: np.ndarray, rdm2: np.ndarray, method: str = "spearman") -> float:
    """
    Compare two representational dissimilarity matrices.

    Parameters
    ----------
    rdm1 : np.ndarray
        First RDM (square matrix)
    rdm2 : np.ndarray
        Second RDM (same shape as rdm1)
    method : str, default 'spearman'
        Comparison method: 'spearman', 'pearson', 'kendall', 'cosine'

    Returns
    -------
    similarity : float
        Similarity score between RDMs
    """
    # Ensure RDMs are same shape
    if rdm1.shape != rdm2.shape:
        raise ValueError(
            f"RDMs must have the same shape. Got {rdm1.shape} and {rdm2.shape}"
        )

    # Extract upper triangular parts (excluding diagonal)
    mask = np.triu(np.ones_like(rdm1, dtype=bool), k=1)
    rdm1_vec = rdm1[mask]
    rdm2_vec = rdm2[mask]

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
        similarity = dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0
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
    
    Uses within-condition resampling to maintain balanced representation
    of all conditions while assessing the stability of the RDM similarity.

    Parameters
    ----------
    data1 : np.ndarray
        First dataset (n_features1, n_timepoints)
    data2 : np.ndarray
        Second dataset (n_features2, n_timepoints)
    labels1 : np.ndarray
        Condition labels for data1
    labels2 : np.ndarray
        Condition labels for data2
    metric : str, default 'correlation'
        Distance metric for RDM computation
    comparison_method : str, default 'spearman'
        Method for comparing RDMs
    n_bootstrap : int, default 1000
        Number of bootstrap iterations
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'observed': Observed RDM similarity
        - 'bootstrap_distribution': Bootstrap samples
        - 'p_value': Bootstrap p-value
        - 'ci_lower': 95% CI lower bound
        - 'ci_upper': 95% CI upper bound
        
    Notes
    -----
    The bootstrap procedure resamples trials within each condition
    independently, maintaining the balance of conditions. This provides
    confidence intervals for the RDM similarity that account for 
    trial-by-trial variability while preserving the experimental design.
    """
    if random_state is not None:
        np.random.seed(random_state)

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
            
            resampled_idx1 = np.random.choice(cond_idx1, size=n_samples1, replace=True)
            resampled_idx2 = np.random.choice(cond_idx2, size=n_samples2, replace=True)
            
            idx1.extend(resampled_idx1)
            idx2.extend(resampled_idx2)
        
        # Convert to arrays and shuffle to mix conditions
        idx1 = np.array(idx1)
        idx2 = np.array(idx2)
        np.random.shuffle(idx1)
        np.random.shuffle(idx2)

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

    # Compute p-value (two-tailed)
    p_value = np.mean(
        np.abs(bootstrap_similarities - np.mean(bootstrap_similarities))
        >= np.abs(observed_similarity - np.mean(bootstrap_similarities))
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
        "mean": np.mean(bootstrap_similarities),
        "std": np.std(bootstrap_similarities),
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
    Unified RDM computation with automatic data type detection.

    This function intelligently dispatches to the appropriate RDM computation
    method based on the input data type and items specification.

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
        Distance metric for RDM computation
    average_method : str, default 'mean'
        How to average within conditions ('mean' or 'median')
    trial_duration : int, optional
        For trial structure, fixed duration for each trial

    Returns
    -------
    rdm : np.ndarray
        Representational dissimilarity matrix
    labels : np.ndarray or None
        The unique labels/conditions if items were specified

    Examples
    --------
    # Direct pattern RDM (pre-averaged data)
    >>> patterns = np.random.randn(10, 50)  # 10 items, 50 features
    >>> rdm, _ = compute_rdm_unified(patterns)

    # From time series with labels
    >>> data = np.random.randn(100, 1000)  # 100 features, 1000 timepoints
    >>> labels = np.repeat([0, 1, 2, 3], 250)
    >>> rdm, unique_labels = compute_rdm_unified(data, labels)

    # From Experiment with behavioral variable
    >>> rdm, labels = compute_rdm_unified(exp, items='stimulus_type')

    # From Experiment with trial structure
    >>> trial_info = {'trial_starts': [0, 100, 200], 'trial_labels': ['A', 'B', 'A']}
    >>> rdm, labels = compute_rdm_unified(exp, items=trial_info)
    """
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
        if items is None:
            # Direct pattern RDM - assume rows are patterns
            return compute_rdm(data, metric=metric), None
        elif isinstance(items, dict):
            # Trial structure
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
    Compare neural representations between two datasets using RSA.

    This is a simplified API for the most common RSA use case: comparing
    two sets of neural representations.

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
        Method for comparing RDMs
    data_type : str, default 'calcium'
        For Experiment objects, which data to use ('calcium' or 'spikes')
    logger : logging.Logger, optional
        Logger for debugging

    Returns
    -------
    similarity : float
        Similarity score between the two neural representations

    Examples
    --------
    >>> # Compare V1 and V2 representations (arrays)
    >>> v1_data = np.random.randn(10, 100)  # 10 stimuli, 100 neurons
    >>> v2_data = np.random.randn(10, 150)  # 10 stimuli, 150 neurons
    >>> similarity = rsa_compare(v1_data, v2_data)

    >>> # Compare two experiments
    >>> similarity = rsa_compare(exp1, exp2, items='stimulus_type')

    >>> # Compare with trial structure
    >>> trials = {'trial_starts': [0, 100, 200], 'trial_labels': ['A', 'B', 'A']}
    >>> similarity = rsa_compare(exp1, exp2, items=trials)
    """
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

        # Compare RDMs
        similarity = compare_rdms(rdm1, rdm2, method=comparison)

        logger.debug(f"RSA comparison complete. Similarity: {similarity:.3f}")

        return similarity
