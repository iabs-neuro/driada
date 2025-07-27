"""
Integration of RSA with DRIADA's data structures.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from ..experiment import Experiment
from ..dim_reduction.data import MVData
from .core import (
    compute_rdm, 
    compute_rdm_from_timeseries_labels,
    compute_rdm_from_trials,
    compare_rdms, 
    bootstrap_rdm_comparison
)


def compute_experiment_rdm(
    experiment: Experiment,
    items: Union[str, Dict[str, np.ndarray]],
    data_type: str = 'calcium',
    metric: str = 'correlation',
    average_method: str = 'mean'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RDM from Experiment object using specified item definition.
    
    Parameters
    ----------
    experiment : Experiment
        DRIADA Experiment object
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
        Representational dissimilarity matrix
    labels : np.ndarray
        The unique labels/conditions in the order they appear in RDM
    
    Examples
    --------
    # Option 1: Use behavioral variable as conditions
    rdm, labels = compute_experiment_rdm(exp, items='stimulus_type')
    
    # Option 2: Use explicit trial structure
    trial_info = {
        'trial_starts': [0, 100, 200, 300],
        'trial_labels': ['A', 'B', 'A', 'C']
    }
    rdm, labels = compute_experiment_rdm(exp, items=trial_info)
    """
    # Get neural data
    if data_type == 'calcium':
        if not hasattr(experiment, 'calcium') or experiment.calcium is None:
            raise ValueError("Experiment has no calcium data")
        data = experiment.calcium.data
    elif data_type == 'spikes':
        if not hasattr(experiment, 'spikes') or experiment.spikes is None:
            raise ValueError("Experiment has no spike data")
        data = experiment.spikes.data
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    # Process based on items type
    if isinstance(items, str):
        # Option 2: Use behavioral variable as conditions
        if items not in experiment.dynamic_features:
            raise ValueError(f"Feature '{items}' not found in experiment dynamic features")
        
        # Get labels from dynamic feature
        feature_data = experiment.dynamic_features[items]
        if hasattr(feature_data, 'data'):
            labels = feature_data.data
        else:
            labels = np.array(feature_data)
        
        return compute_rdm_from_timeseries_labels(
            data, labels, metric=metric, average_method=average_method
        )
        
    elif isinstance(items, dict):
        # Option 3: Use explicit trial structure
        if 'trial_starts' not in items or 'trial_labels' not in items:
            raise ValueError("Trial structure dict must contain 'trial_starts' and 'trial_labels'")
        
        trial_starts = np.array(items['trial_starts'])
        trial_labels = np.array(items['trial_labels'])
        trial_duration = items.get('trial_duration', None)
        
        return compute_rdm_from_trials(
            data, trial_starts, trial_labels, 
            trial_duration=trial_duration,
            metric=metric, 
            average_method=average_method
        )
        
    else:
        raise ValueError("items must be a string (feature name) or dict (trial structure)")


def compute_mvdata_rdm(
    mvdata: MVData,
    labels: np.ndarray,
    metric: str = 'correlation',
    average_method: str = 'mean'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RDM from MVData object with condition labels.
    
    Parameters
    ----------
    mvdata : MVData
        MVData object containing data matrix
    labels : np.ndarray
        Condition labels for each timepoint
    metric : str, default 'correlation'
        Distance metric for RDM computation
    average_method : str, default 'mean'
        How to average within conditions
        
    Returns
    -------
    rdm : np.ndarray
        Representational dissimilarity matrix
    unique_labels : np.ndarray
        The unique labels in order as they appear in RDM
    """
    # MVData stores data as (n_features, n_timepoints)
    data = mvdata.data
    
    return compute_rdm_from_timeseries_labels(
        data, labels, metric=metric, average_method=average_method
    )


def rsa_between_experiments(
    exp1: Experiment,
    exp2: Experiment,
    items: Union[str, Dict[str, np.ndarray]],
    data_type: str = 'calcium',
    metric: str = 'correlation',
    comparison_method: str = 'spearman',
    average_method: str = 'mean',
    bootstrap: bool = False,
    n_bootstrap: int = 1000
) -> Union[float, Dict]:
    """
    Perform RSA between two experiments.
    
    Parameters
    ----------
    exp1 : Experiment
        First experiment
    exp2 : Experiment
        Second experiment
    items : str or dict
        How to define items/conditions (must be same for both experiments)
    data_type : str, default 'calcium'
        Type of data to use
    metric : str, default 'correlation'
        Distance metric for RDM computation
    comparison_method : str, default 'spearman'
        Method for comparing RDMs
    average_method : str, default 'mean'
        How to average within conditions
    bootstrap : bool, default False
        Whether to perform bootstrap significance testing
    n_bootstrap : int, default 1000
        Number of bootstrap iterations
        
    Returns
    -------
    similarity : float or dict
        If bootstrap=False: similarity score
        If bootstrap=True: dict with bootstrap results
    """
    # Compute RDMs for each experiment
    rdm1, labels1 = compute_experiment_rdm(
        exp1, items, data_type, metric, average_method
    )
    rdm2, labels2 = compute_experiment_rdm(
        exp2, items, data_type, metric, average_method
    )
    
    # Ensure same labels
    if not np.array_equal(labels1, labels2):
        raise ValueError("Experiments must have the same condition labels for comparison")
    
    if bootstrap:
        # Get the raw data and labels for bootstrap
        data1 = exp1.calcium.data if data_type == 'calcium' else exp1.spikes.data
        data2 = exp2.calcium.data if data_type == 'calcium' else exp2.spikes.data
        
        if isinstance(items, str):
            labels = exp1.dynamic_features[items].data
        else:
            # For trial structure, create continuous labels
            # This is a simplification - proper implementation would handle trials
            raise NotImplementedError("Bootstrap not yet implemented for trial structure")
        
        return bootstrap_rdm_comparison(
            data1, data2, labels, labels,
            metric=metric,
            comparison_method=comparison_method,
            n_bootstrap=n_bootstrap
        )
    else:
        # Simple comparison
        return compare_rdms(rdm1, rdm2, method=comparison_method)


def rsa_between_mvdata(
    mvdata1: MVData,
    mvdata2: MVData,
    labels: np.ndarray,
    metric: str = 'correlation',
    comparison_method: str = 'spearman',
    average_method: str = 'mean'
) -> float:
    """
    Perform RSA between two MVData objects.
    
    Parameters
    ----------
    mvdata1 : MVData
        First MVData object
    mvdata2 : MVData
        Second MVData object
    labels : np.ndarray
        Condition labels (same length as timepoints in MVData)
    metric : str, default 'correlation'
        Distance metric for RDM computation
    comparison_method : str, default 'spearman'
        Method for comparing RDMs
    average_method : str, default 'mean'
        How to average within conditions
        
    Returns
    -------
    similarity : float
        Similarity score between the two RDMs
    """
    # Compute RDMs
    rdm1, _ = compute_mvdata_rdm(mvdata1, labels, metric, average_method)
    rdm2, _ = compute_mvdata_rdm(mvdata2, labels, metric, average_method)
    
    # Compare
    return compare_rdms(rdm1, rdm2, method=comparison_method)


def compute_embedding_rdm(
    experiment: Experiment,
    embedding_method: str,
    items: Union[str, Dict[str, np.ndarray]],
    data_type: str = 'calcium',
    metric: str = 'correlation',
    average_method: str = 'mean'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RDM from embedding stored in experiment.
    
    Parameters
    ----------
    experiment : Experiment
        Experiment with stored embeddings
    embedding_method : str
        Name of embedding method (e.g., 'pca', 'umap')
    items : str or dict
        How to define items/conditions
    data_type : str, default 'calcium'
        Data type used for embedding
    metric : str, default 'correlation'
        Distance metric for RDM computation
    average_method : str, default 'mean'
        How to average within conditions
        
    Returns
    -------
    rdm : np.ndarray
        RDM computed from embedding
    labels : np.ndarray
        The unique labels in order
    """
    # Get embedding
    embedding_dict = experiment.get_embedding(embedding_method, data_type)
    embedding_data = embedding_dict['data']  # (n_timepoints, n_components)
    
    # Handle downsampling if embedding was downsampled
    ds = embedding_dict.get('metadata', {}).get('ds', 1)
    
    # Process based on items type
    if isinstance(items, str):
        # Get labels and handle downsampling
        feature_data = experiment.dynamic_features[items]
        if hasattr(feature_data, 'data'):
            labels = feature_data.data[::ds]  # Downsample labels to match embedding
        else:
            labels = np.array(feature_data)[::ds]
        
        # Transpose embedding data to (n_components, n_timepoints) for consistency
        return compute_rdm_from_timeseries_labels(
            embedding_data.T, labels, metric=metric, average_method=average_method
        )
        
    elif isinstance(items, dict):
        # Adjust trial starts for downsampling
        trial_starts = np.array(items['trial_starts']) // ds
        trial_labels = np.array(items['trial_labels'])
        trial_duration = items.get('trial_duration', None)
        if trial_duration is not None:
            trial_duration = trial_duration // ds
        
        return compute_rdm_from_trials(
            embedding_data.T, trial_starts, trial_labels,
            trial_duration=trial_duration,
            metric=metric,
            average_method=average_method
        )
    else:
        raise ValueError("items must be a string (feature name) or dict (trial structure)")