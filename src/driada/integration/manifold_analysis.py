"""
Analyze functional organization in neural manifolds.

This module provides functions to analyze how single-neuron selectivity
relates to population-level manifold structure.
"""

import logging
from typing import Dict, List
import numpy as np
from collections import defaultdict


def get_functional_organization(experiment, method_name: str, data_type: str = "calcium", 
                               embedding_selectivity_results=None) -> Dict:
    """
    Analyze functional organization in the manifold.
    
    Examines how neurons contribute to different embedding components
    and identifies functional clusters based on selectivity patterns.

    Parameters
    ----------
    experiment : Experiment
        Experiment object with stored embeddings and selectivity analysis.
    method_name : str
        Name of the embedding method to analyze.
    data_type : str, optional
        Data type used for embedding ('calcium' or 'spikes'). Default is 'calcium'.
    embedding_selectivity_results : IntenseResults or dict, optional
        Results from compute_embedding_selectivity. Can be either:
        - IntenseResults object from a single embedding method
        - Dict mapping method names to IntenseResults objects
        If not provided, will check if results are stored in experiment's stats_tables.

    Returns
    -------
    dict
        Dictionary containing:
        - 'component_importance': Variance explained by each component
        - 'neuron_participation': How many components each neuron contributes to
        - 'component_specialization': How selective each component is
        - 'functional_clusters': Groups of neurons with similar embedding selectivity
        - 'n_components': Number of embedding components
        - 'n_neurons_used': Number of neurons in the embedding
        - 'neuron_indices': Indices of neurons used
        - Additional statistics if selectivity analysis available
        
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
    >>> org = get_functional_organization(exp, 'pca', embedding_selectivity_results=results['pca'])
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
    has_embedding_selectivity = False
    significance_data = None
    
    if embedding_selectivity_results is not None:
        # Check if it's an IntenseResults object or a dict of IntenseResults
        if hasattr(embedding_selectivity_results, 'significance'):
            # Single IntenseResults object
            has_embedding_selectivity = True
            significance_data = embedding_selectivity_results.significance
        elif isinstance(embedding_selectivity_results, dict) and method_name in embedding_selectivity_results:
            # Dict of results from compute_embedding_selectivity
            method_results = embedding_selectivity_results[method_name]
            if 'intense_results' in method_results and hasattr(method_results['intense_results'], 'significance'):
                # Extract IntenseResults object
                has_embedding_selectivity = True
                significance_data = method_results['intense_results'].significance
            elif 'significance' in method_results:
                # Fallback for older format
                has_embedding_selectivity = True
                significance_data = method_results['significance']
    else:
        # Fall back to checking stats_tables (for backward compatibility)
        stats_key = f"{method_name}_comp0"
        if (hasattr(experiment, "stats_tables")
            and experiment.stats_tables is not None
            and data_type in experiment.stats_tables
            and stats_key in experiment.stats_tables[data_type]):
            has_embedding_selectivity = True
            # Will use experiment.significance_tables

    if has_embedding_selectivity:
        # Analyze neuron participation across components
        neuron_participation = {}
        component_specialization = {}

        for comp_idx in range(embedding.shape[1]):
            feat_name = f"{method_name}_comp{comp_idx}"
            selective_neurons = []

            # Check which neurons are selective to this component
            if significance_data is not None:
                # Use provided IntenseResults significance data
                if feat_name in significance_data:
                    feat_sig = significance_data[feat_name]
                    for neuron_idx in range(experiment.n_cells):
                        if neuron_idx in feat_sig and feat_sig[neuron_idx].get("stage2", False):
                            selective_neurons.append(neuron_idx)
                            
                            # Track neuron participation
                            if neuron_idx not in neuron_participation:
                                neuron_participation[neuron_idx] = []
                            neuron_participation[neuron_idx].append(comp_idx)
            else:
                # Fall back to experiment tables
                if hasattr(experiment, "significance_tables") and experiment.significance_tables is not None:
                    sig_tables = experiment.significance_tables
                    if data_type in sig_tables and feat_name in sig_tables[data_type]:
                        feat_sig = sig_tables[data_type][feat_name]
                        for neuron_idx in range(experiment.n_cells):
                            if neuron_idx in feat_sig and feat_sig[neuron_idx].get("stage2", False):
                                selective_neurons.append(neuron_idx)
                                
                                # Track neuron participation
                                if neuron_idx not in neuron_participation:
                                    neuron_participation[neuron_idx] = []
                                neuron_participation[neuron_idx].append(comp_idx)

            component_specialization[comp_idx] = {
                "n_selective_neurons": len(selective_neurons),
                "selective_neurons": selective_neurons,
                "selectivity_rate": len(selective_neurons) / experiment.n_cells if experiment.n_cells > 0 else 0,
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

    return organization


def compare_embeddings(experiment, method_names: List[str], data_type: str = "calcium",
                      embedding_selectivity_results=None) -> Dict:
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
    embedding_selectivity_results : dict, optional
        Dict mapping method names to IntenseResults objects from compute_embedding_selectivity.
        If not provided, will check if results are stored in experiment's stats_tables.

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
    >>> results = compute_embedding_selectivity(exp, embedding_methods=None)  # All methods
    >>> comparison = compare_embeddings(exp, ['pca', 'umap'], embedding_selectivity_results=results)
    >>> print(f"Overlap: {comparison['participation_overlap']['pca_vs_umap']:.2f}")
    
    DOC_VERIFIED
    """
    if not isinstance(method_names, list):
        raise TypeError("method_names must be a list")
        
    if len(method_names) == 0:
        raise ValueError("method_names cannot be empty")
        
    organizations = {}
    logger = logging.getLogger(__name__)
    
    for method in method_names:
        try:
            # Pass the IntenseResults for this method if available
            method_results = None
            if embedding_selectivity_results and method in embedding_selectivity_results:
                method_results = embedding_selectivity_results[method]
                
            organizations[method] = get_functional_organization(
                experiment, method, data_type,
                embedding_selectivity_results=method_results
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