"""
Analyze functional organization in neural manifolds.

This module provides functions to analyze how single-neuron selectivity
relates to population-level manifold structure.
"""

import logging
from typing import Dict, List, Optional
import numpy as np
from collections import defaultdict


def get_functional_organization(experiment, method_name: str, data_type: str = "calcium", 
                               intense_results: Optional[object] = None) -> Dict:
    """
    Analyze functional organization in the manifold.
    
    Examines how neurons contribute to different embedding components
    and identifies functional clusters based on selectivity patterns.

    Parameters
    ----------
    experiment : Experiment
        Experiment object with stored embeddings.
    method_name : str
        Name of the embedding method to analyze.
    data_type : str, optional
        Data type used for embedding ('calcium' or 'spikes'). Default is 'calcium'.
    intense_results : IntenseResults, optional
        IntenseResults object from compute_embedding_selectivity for this
        specific embedding method. Must be an instance of
        driada.intense.intense_base.IntenseResults. If not provided, only
        basic statistics (component importance) will be returned.

    Returns
    -------
    dict
        Dictionary containing:
        - 'component_importance': Variance explained by each component
        - 'n_components': Number of embedding components
        - 'n_neurons_used': Number of neurons in the embedding
        - 'neuron_indices': Indices of neurons used
        
        If intense_results provided, also includes:
        - 'neuron_participation': How many components each neuron contributes to
        - 'component_specialization': How selective each component is
        - 'functional_clusters': Groups of neurons with similar embedding selectivity
        - 'n_participating_neurons': Number of neurons with selectivity
        - 'mean_components_per_neuron': Average participation per neuron
        
    Raises
    ------
    KeyError
        If the specified embedding method is not found.
    ValueError
        If component variances sum to zero.
        
    Examples
    --------
    >>> # Analyze functional organization in PCA embedding
    >>> from driada.intense import compute_embedding_selectivity
    >>> results = compute_embedding_selectivity(exp, embedding_methods='pca')
    >>> intense_res = results['pca']['intense_results']
    >>> org = get_functional_organization(exp, 'pca', intense_results=intense_res)
    >>> print(f"Component importance: {org['component_importance']}")
    >>> print(f"Neurons participating: {org['n_participating_neurons']}")
    
    DOC_VERIFIED
    """
    # Get embedding and metadata
    embedding_dict = experiment.get_embedding(method_name, data_type)
    embedding = embedding_dict["data"]
    metadata = embedding_dict.get("metadata", {})
    neuron_indices = metadata.get(
        "neuron_indices", list(range(experiment.n_cells))
    )

    # Compute component importance (variance explained)
    component_var = np.var(embedding, axis=0)
    var_sum = np.sum(component_var)
    
    if var_sum == 0:
        raise ValueError("All embedding components have zero variance")
        
    component_importance = component_var / var_sum

    organization = {
        "component_importance": component_importance,
        "n_components": embedding.shape[1],
        "n_neurons_used": len(neuron_indices),
        "neuron_indices": neuron_indices,
    }

    # Check if we have selectivity results
    if intense_results is not None:
        # Validate it's an actual IntenseResults object
        from ..intense.intense_base import IntenseResults
        if not isinstance(intense_results, IntenseResults):
            raise TypeError(
                f"intense_results must be an IntenseResults object, got {type(intense_results).__name__}"
            )
        if not hasattr(intense_results, 'significance'):
            raise ValueError("intense_results must have 'significance' attribute")
        
        significance_data = intense_results.significance
        if significance_data is None:
            # No significance data available
            return organization
        # Analyze neuron participation across components
        neuron_participation = {}
        component_specialization = {}

        for comp_idx in range(embedding.shape[1]):
            feat_name = f"{method_name}_comp{comp_idx}"
            selective_neurons = []

            # Check which neurons are selective to this component
            if feat_name in significance_data:
                feat_sig = significance_data[feat_name]
                # Iterate only over neurons actually used in the embedding
                for idx, neuron_idx in enumerate(neuron_indices):
                    if neuron_idx in feat_sig and feat_sig[neuron_idx].get("stage2", False):
                        selective_neurons.append(neuron_idx)
                        
                        # Track neuron participation
                        if neuron_idx not in neuron_participation:
                            neuron_participation[neuron_idx] = []
                        neuron_participation[neuron_idx].append(comp_idx)

            component_specialization[comp_idx] = {
                "n_selective_neurons": len(selective_neurons),
                "selective_neurons": selective_neurons,
                "selectivity_rate": len(selective_neurons) / len(neuron_indices) if len(neuron_indices) > 0 else 0,
            }

        # Identify functional clusters (neurons selective to same components)
        cluster_map = defaultdict(list)

        for neuron_idx, components in neuron_participation.items():
            cluster_key = tuple(sorted(components))
            cluster_map[cluster_key].append(neuron_idx)

        functional_clusters = []
        for components, neurons in cluster_map.items():
            if len(neurons) > 1:  # Only keep clusters with multiple neurons
                functional_clusters.append(
                    {
                        "components": list(components),
                        "neurons": neurons,
                        "size": len(neurons),
                    }
                )

        # Sort clusters by size
        functional_clusters.sort(key=lambda x: x["size"], reverse=True)

        organization.update(
            {
                "neuron_participation": neuron_participation,
                "component_specialization": component_specialization,
                "functional_clusters": functional_clusters,
                "n_participating_neurons": len(neuron_participation),
                "mean_components_per_neuron": (
                    np.mean([len(comps) for comps in neuron_participation.values()])
                    if neuron_participation
                    else 0
                ),
            }
        )
    else:
        # No intense_results provided - return only basic statistics
        pass

    return organization


def compare_embeddings(experiment, method_names: List[str], data_type: str = "calcium",
                      intense_results_dict: Optional[Dict] = None) -> Dict:
    """
    Compare functional organization across different embedding methods.
    
    Analyzes and compares how different dimensionality reduction methods
    organize the neural population, including neuron participation overlap
    and clustering patterns.

    Parameters
    ----------
    experiment : Experiment
        Experiment object with stored embeddings.
    method_names : list of str
        List of embedding method names to compare.
    data_type : str, optional
        Data type used for embeddings ('calcium' or 'spikes'). Default is 'calcium'.
    intense_results_dict : dict, optional
        Dict mapping method names to IntenseResults objects (not the full
        compute_embedding_selectivity output). Each value must be an instance
        of driada.intense.intense_base.IntenseResults. If not provided, only
        basic comparison metrics will be returned.

    Returns
    -------
    dict
        Comparison metrics including:
        - 'methods': List of valid methods analyzed
        - 'n_components': Number of components per method
        - 'n_participating_neurons': Neurons with selectivity per method
        - 'mean_components_per_neuron': Average participation per method
        - 'n_functional_clusters': Number of clusters per method
        - 'participation_overlap': Pairwise overlap between methods
        
    Raises
    ------
    ValueError
        If no valid embeddings found to compare.
    TypeError
        If method_names is not a list.
        
    Examples
    --------
    >>> # Compare PCA and UMAP embeddings
    >>> from driada.intense import compute_embedding_selectivity
    >>> results = compute_embedding_selectivity(exp, embedding_methods=['pca', 'umap'])
    >>> # Extract IntenseResults objects
    >>> intense_dict = {method: results[method]['intense_results'] 
    ...                 for method in results}
    >>> comparison = compare_embeddings(exp, ['pca', 'umap'], intense_results_dict=intense_dict)
    >>> print(f"Overlap: {comparison['participation_overlap']['pca_vs_umap']:.2f}")
    
    DOC_VERIFIED
    """
    if not isinstance(method_names, list):
        raise TypeError("method_names must be a list")
        
    if len(method_names) == 0:
        raise ValueError("method_names cannot be empty")
        
    organizations = {}
    logger = logging.getLogger(__name__)
    
    # Validate intense_results_dict if provided
    if intense_results_dict:
        from ..intense.intense_base import IntenseResults
        for method, intense_res in intense_results_dict.items():
            if not isinstance(intense_res, IntenseResults):
                raise TypeError(
                    f"intense_results_dict['{method}'] must be an IntenseResults object, "
                    f"got {type(intense_res).__name__}"
                )
    
    for method in method_names:
        try:
            # Get IntenseResults for this method if available
            intense_res = None
            if intense_results_dict and method in intense_results_dict:
                intense_res = intense_results_dict[method]
                
            organizations[method] = get_functional_organization(
                experiment, method, data_type,
                intense_results=intense_res
            )
        except KeyError:
            logger.warning(f"No embedding found for method '{method}'")

    if len(organizations) == 0:
        raise ValueError("No valid embeddings found to compare")

    if len(organizations) == 1:
        # Special case: only one embedding exists
        logger.info(
            "Only one embedding found, returning individual statistics"
        )

    comparison = {
        "methods": list(organizations.keys()),
        "n_components": {
            m: org["n_components"] for m, org in organizations.items()
        },
        "n_participating_neurons": {
            m: org.get("n_participating_neurons", 0)
            for m, org in organizations.items()
        },
        "mean_components_per_neuron": {
            m: org.get("mean_components_per_neuron", 0)
            for m, org in organizations.items()
        },
        "n_functional_clusters": {
            m: len(org.get("functional_clusters", []))
            for m, org in organizations.items()
        },
    }

    # Compare neuron participation overlap (only if we have multiple methods)
    if len(organizations) > 1 and all("neuron_participation" in org for org in organizations.values()):
        from itertools import combinations
        
        method_pairs = list(combinations(organizations.keys(), 2))
        participation_overlap = {}

        for m1, m2 in method_pairs:
            neurons1 = set(organizations[m1]["neuron_participation"].keys())
            neurons2 = set(organizations[m2]["neuron_participation"].keys())

            if neurons1 or neurons2:
                overlap = len(neurons1 & neurons2) / len(neurons1 | neurons2)
            else:
                overlap = 0

            participation_overlap[f"{m1}_vs_{m2}"] = overlap

        comparison["participation_overlap"] = participation_overlap

    return comparison