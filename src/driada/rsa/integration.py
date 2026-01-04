"""
Integration of RSA with DRIADA's data structures.
"""

import numpy as np
from typing import Dict, Union, Tuple
from ..experiment import Experiment
from ..dim_reduction.data import MVData
from .core import (
    compute_rdm_from_timeseries_labels,
    compute_rdm_from_trials,
    compare_rdms,
    bootstrap_rdm_comparison,
)


def compute_experiment_rdm(
    experiment: Experiment,
    items: Union[str, Dict[str, np.ndarray]],
    data_type: str = "calcium",
    metric: str = "correlation",
    average_method: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RDM from Experiment object using specified item definition.
    
    Extracts neural data from a DRIADA Experiment and computes the
    representational dissimilarity matrix based on either behavioral
    variables or explicit trial structure.

    Parameters
    ----------
    experiment : Experiment
        DRIADA Experiment object containing neural data
    items : str or dict
        How to define items/conditions:
        - str: name of dynamic feature to use as condition labels
        - dict with 'trial_starts' and 'trial_labels' for explicit trials
    data_type : str, default 'calcium'
        Type of data to use ('calcium' or 'spikes')
    metric : str, default 'correlation'
        Distance metric for RDM computation
    average_method : str, default 'mean'
        How to average within conditions ('mean' or 'median')

    Returns
    -------
    rdm : np.ndarray
        Representational dissimilarity matrix of shape (n_conditions, n_conditions)
    labels : np.ndarray
        The unique labels/conditions in the order they appear in RDM
        
    Raises
    ------
    ValueError
        If experiment has no data of the specified type.
        If dynamic feature name not found in experiment.
        If trial structure dict missing required keys.
        If items is not str or dict.
        
    Notes
    -----
    When using dynamic features, the function extracts the feature data
    and uses it as condition labels for each timepoint. When using trial
    structure, explicit trial boundaries and labels are provided.

    Examples
    --------
    See compute_rdm_from_timeseries_labels and compute_rdm_from_trials
    for the core functionality that this function wraps. The function
    extracts data from Experiment objects and delegates to those functions.
    
    Direct usage with Experiment objects requires creating complex data
    structures that are better demonstrated in the test files and tutorials.
    
    See Also
    --------
    ~driada.rsa.core.compute_rdm_from_timeseries_labels : Lower-level function for labeled data
    ~driada.rsa.core.compute_rdm_from_trials : Lower-level function for trial structure
    ~driada.rsa.core.compare_rdms : Compare two RDMs using correlation metrics    """
    # Get neural data
    if data_type == "calcium":
        if not hasattr(experiment, "calcium") or experiment.calcium is None:
            raise ValueError("Experiment has no calcium data")
        data = experiment.calcium.data
    elif data_type == "spikes":
        if not hasattr(experiment, "spikes") or experiment.spikes is None:
            raise ValueError("Experiment has no spike data")
        data = experiment.spikes.data
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    # Process based on items type
    if isinstance(items, str):
        # Option 2: Use behavioral variable as conditions
        if items not in experiment.dynamic_features:
            raise ValueError(
                f"Feature '{items}' not found in experiment dynamic features"
            )

        # Get labels from dynamic feature
        feature_data = experiment.dynamic_features[items]
        if hasattr(feature_data, "data"):
            labels = feature_data.data
        else:
            labels = np.array(feature_data)

        return compute_rdm_from_timeseries_labels(
            data, labels, metric=metric, average_method=average_method
        )

    elif isinstance(items, dict):
        # Option 3: Use explicit trial structure
        if "trial_starts" not in items or "trial_labels" not in items:
            raise ValueError(
                "Trial structure dict must contain 'trial_starts' and 'trial_labels'"
            )

        trial_starts = np.array(items["trial_starts"])
        trial_labels = np.array(items["trial_labels"])
        trial_duration = items.get("trial_duration", None)

        return compute_rdm_from_trials(
            data,
            trial_starts,
            trial_labels,
            trial_duration=trial_duration,
            metric=metric,
            average_method=average_method,
        )

    else:
        raise ValueError(
            "items must be a string (feature name) or dict (trial structure)"
        )


def compute_mvdata_rdm(
    mvdata: MVData,
    labels: np.ndarray,
    metric: str = "correlation",
    average_method: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RDM from MVData object with condition labels.
    
    Extracts data from MVData object and computes representational
    dissimilarity matrix based on provided condition labels.

    Parameters
    ----------
    mvdata : MVData
        MVData object containing data matrix of shape (n_features, n_timepoints)
    labels : np.ndarray
        Condition labels for each timepoint, shape (n_timepoints,)
    metric : str, default 'correlation'
        Distance metric for RDM computation ('correlation', 'euclidean',
        'cosine', 'manhattan')
    average_method : str, default 'mean'
        How to average within conditions ('mean' or 'median')

    Returns
    -------
    rdm : np.ndarray
        Representational dissimilarity matrix of shape (n_conditions, n_conditions)
    unique_labels : np.ndarray
        The unique labels in order as they appear in RDM
        
    Notes
    -----
    This is a thin wrapper around compute_rdm_from_timeseries_labels that
    extracts data from MVData objects. MVData stores data in the expected
    format (n_features, n_timepoints).
    
    Examples
    --------
    >>> import numpy as np
    >>> from driada.dim_reduction.data import MVData
    >>> 
    >>> # Create MVData and compute RDM
    >>> data = np.random.randn(100, 500)  # 100 features, 500 timepoints
    >>> mvdata = MVData(data)
    >>> labels = np.repeat(['A', 'B', 'C'], [150, 200, 150])
    >>> rdm, unique_labels = compute_mvdata_rdm(mvdata, labels)
    >>> print(f"RDM shape: {rdm.shape}, conditions: {unique_labels}")
    RDM shape: (3, 3), conditions: ['A' 'B' 'C']
    
    See Also
    --------
    ~driada.rsa.core.compute_rdm_from_timeseries_labels : Core function this wraps
    ~driada.rsa.core.compute_rdm_unified : Unified interface for all data types    """
    # MVData stores data as (n_features, n_timepoints)
    data = mvdata.data

    return compute_rdm_from_timeseries_labels(
        data, labels, metric=metric, average_method=average_method
    )


def rsa_between_experiments(
    exp1: Experiment,
    exp2: Experiment,
    items: Union[str, Dict[str, np.ndarray]],
    data_type: str = "calcium",
    metric: str = "correlation",
    comparison_method: str = "spearman",
    average_method: str = "mean",
    bootstrap: bool = False,
    n_bootstrap: int = 1000,
) -> Union[float, Dict]:
    """
    Perform RSA comparison between two experiments.
    
    Computes representational dissimilarity matrices for both experiments
    and quantifies their similarity. Optionally performs bootstrap analysis
    for statistical inference.

    Parameters
    ----------
    exp1 : Experiment
        First experiment
    exp2 : Experiment
        Second experiment  
    items : str or dict
        How to define items/conditions (must be same for both experiments):
        - str: name of dynamic feature to use as condition labels
        - dict: trial structure with 'trial_starts' and 'trial_labels'
    data_type : str, default 'calcium'
        Type of data to use ('calcium' or 'spikes')
    metric : str, default 'correlation'
        Distance metric for RDM computation
    comparison_method : str, default 'spearman'
        Method for comparing RDMs ('spearman', 'pearson', 'kendall', 'cosine')
    average_method : str, default 'mean'
        How to average within conditions ('mean' or 'median')
    bootstrap : bool, default False
        Whether to perform bootstrap significance testing
    n_bootstrap : int, default 1000
        Number of bootstrap iterations

    Returns
    -------
    similarity : float or dict
        If bootstrap=False: similarity score between RDMs
        If bootstrap=True: dict with keys:
        - 'observed': observed similarity
        - 'bootstrap_distribution': bootstrap values
        - 'p_value': two-tailed p-value
        - 'ci_lower', 'ci_upper': 95% confidence interval
        - 'mean', 'std': bootstrap statistics
        
    Raises
    ------
    ValueError
        If experiments have different condition labels.
    NotImplementedError
        If bootstrap requested with trial structure (not yet supported).
        
    Notes
    -----
    Both experiments must have the same conditions (same unique labels)
    for meaningful comparison. The function automatically extracts the
    appropriate data type and computes RDMs before comparing them.
    
    Bootstrap analysis uses within-condition resampling to maintain
    experimental design while estimating variability.

    Examples
    --------
    >>> import numpy as np
    >>> from driada.experiment.synthetic import generate_synthetic_exp
    >>> 
    >>> # Create two synthetic experiments
    >>> exp1 = generate_synthetic_exp(
    ...     n_dfeats=3, n_cfeats=0, nneurons=30,
    ...     duration=300, fps=1.0, seed=42, verbose=False
    ... )
    >>> exp2 = generate_synthetic_exp(
    ...     n_dfeats=3, n_cfeats=0, nneurons=25,
    ...     duration=300, fps=1.0, seed=43, verbose=False
    ... )
    >>> 
    >>> # Add simple integer conditions to avoid string issues
    >>> conditions = np.repeat([1, 2, 3], 100)
    >>> exp1.dynamic_features['stimulus'] = conditions
    >>> exp2.dynamic_features['stimulus'] = conditions
    >>> 
    >>> # Compare neural representations
    >>> similarity = rsa_between_experiments(
    ...     exp1, exp2, items='stimulus',
    ...     comparison_method='spearman'
    ... )
    >>> # Random synthetic data gives variable similarity
    >>> print(f"RSA similarity between -1 and 1: {-1 <= similarity <= 1}")
    RSA similarity between -1 and 1: True
    
    See Also
    --------
    ~driada.rsa.integration.compute_experiment_rdm : Compute RDM from single experiment
    ~driada.rsa.core.compare_rdms : Direct RDM comparison
    ~driada.rsa.core.bootstrap_rdm_comparison : Bootstrap analysis details    """
    # Compute RDMs for each experiment
    rdm1, labels1 = compute_experiment_rdm(
        exp1, items, data_type, metric, average_method
    )
    rdm2, labels2 = compute_experiment_rdm(
        exp2, items, data_type, metric, average_method
    )

    # Ensure same labels
    if not np.array_equal(labels1, labels2):
        raise ValueError(
            "Experiments must have the same condition labels for comparison"
        )

    if bootstrap:
        # Get the raw data and labels for bootstrap
        data1 = exp1.calcium.data if data_type == "calcium" else exp1.spikes.data
        data2 = exp2.calcium.data if data_type == "calcium" else exp2.spikes.data

        if isinstance(items, str):
            labels = exp1.dynamic_features[items].data
        else:
            # For trial structure, create continuous labels
            # This is a simplification - proper implementation would handle trials
            raise NotImplementedError(
                "Bootstrap not yet implemented for trial structure"
            )

        return bootstrap_rdm_comparison(
            data1,
            data2,
            labels,
            labels,
            metric=metric,
            comparison_method=comparison_method,
            n_bootstrap=n_bootstrap,
        )
    else:
        # Simple comparison
        return compare_rdms(rdm1, rdm2, method=comparison_method)
