"""
Analyze functional organization in neural manifolds.

This module provides functions to analyze how single-neuron selectivity
relates to population-level manifold structure.
"""

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..experiment.exp_base import Experiment
    from ..intense.intense_base import IntenseResults
import numpy as np
from collections import defaultdict


def get_functional_organization(experiment: 'Experiment', method_name: str, data_type: str = "calcium", 
                               intense_results: Optional['IntenseResults'] = None) -> Dict:
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
    TypeError
        If intense_results is not an IntenseResults object when provided.
    ValueError
        If intense_results lacks the required 'significance' attribute.
        
    Notes
    -----
    This function reads embedding data from the experiment's stored embeddings.
    When intense_results is provided, it performs detailed selectivity analysis
    to identify functional clusters and neuron participation patterns.
        
    Examples
    --------
    Basic usage without selectivity analysis:
    
    >>> # Create a simple synthetic experiment with circular manifold
    >>> import numpy as np
    >>> from driada.experiment.synthetic import generate_circular_manifold_exp
    >>> np.random.seed(42)
    >>> 
    >>> # Generate experiment with head direction cells
    >>> exp = generate_circular_manifold_exp(
    ...     n_neurons=20,
    ...     duration=30,  # Short duration for example
    ...     fps=10.0,
    ...     kappa=4.0,  # Moderate tuning width
    ...     verbose=False
    ... )
    >>> 
    >>> # Create PCA embedding (automatically stores it)
    >>> embedding = exp.create_embedding('pca', n_components=5, verbose=False)
    >>> 
    >>> # Basic analysis without selectivity (just component importance)
    >>> org_basic = get_functional_organization(exp, 'pca')
    >>> print(f"Number of components: {org_basic['n_components']}")
    Number of components: 5
    >>> print(f"Component importance shape: {org_basic['component_importance'].shape}")
    Component importance shape: (5,)
    >>> print(f"Neurons used: {org_basic['n_neurons_used']}")
    Neurons used: 20
    
    Advanced usage with selectivity analysis (intensive computation):
    
    >>> # The following example shows how to use with selectivity analysis
    >>> # Note: This requires intensive computation and is skipped in doctests
    >>> from driada.intense import compute_embedding_selectivity  # doctest: +SKIP
    >>> 
    >>> # For real analysis, use longer experiments:
    >>> exp_long = generate_circular_manifold_exp(  # doctest: +SKIP
    ...     n_neurons=50, duration=300, verbose=False)
    >>> 
    >>> # Compute PCA embedding
    >>> exp_long.create_embedding('pca', n_components=5, verbose=False)  # doctest: +SKIP
    >>> 
    >>> # Compute selectivity with full parameters
    >>> results = compute_embedding_selectivity(  # doctest: +SKIP
    ...     exp_long, 
    ...     embedding_methods='pca',
    ...     n_shuffles_stage1=100,
    ...     n_shuffles_stage2=500,
    ...     mode='two_stage',
    ...     verbose=False
    ... )
    >>> 
    >>> # Extract the IntenseResults object
    >>> intense_res = results['pca']['intense_results']  # doctest: +SKIP
    >>> 
    >>> # Full functional organization analysis with selectivity
    >>> org_full = get_functional_organization(  # doctest: +SKIP
    ...     exp_long, 'pca', intense_results=intense_res)
    >>> 
    >>> # Access detailed selectivity information
    >>> print(f"Participating neurons: {org_full['n_participating_neurons']}")  # doctest: +SKIP
    >>> print(f"Mean components per neuron: {org_full['mean_components_per_neuron']:.2f}")  # doctest: +SKIP
    >>> print(f"Number of functional clusters: {len(org_full['functional_clusters'])}")  # doctest: +SKIP
    
    See Also
    --------
    ~driada.integration.manifold_analysis.compare_embeddings : Compare multiple embedding methods
    ~driada.intense.pipelines.compute_embedding_selectivity : Compute selectivity for embeddings    """
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


def compare_embeddings(experiment: 'Experiment', method_names: List[str], data_type: str = "calcium",
                      intense_results_dict: Optional[Dict[str, 'IntenseResults']] = None) -> Dict:
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
        If no valid embeddings found to compare or method_names is empty.
    TypeError
        If method_names is not a list or intense_results_dict contains
        non-IntenseResults values.
        
    Notes
    -----
    This function logs warnings when embeddings are not found for requested methods.
    When only one valid embedding is found, it returns statistics for that single
    embedding without computing overlaps.
        
    Examples
    --------
    Basic comparison without selectivity:
    
    >>> # Create a simple synthetic experiment for comparison
    >>> import numpy as np
    >>> from driada.experiment.synthetic import generate_circular_manifold_exp
    >>> np.random.seed(42)
    >>> 
    >>> # Generate experiment with 20 neurons, 500 frames
    >>> exp = generate_circular_manifold_exp(
    ...     n_neurons=20,
    ...     duration=50,  # 50 seconds at 10 fps = 500 frames
    ...     fps=10.0,
    ...     kappa=4.0,  # Moderate tuning width
    ...     verbose=False
    ... )
    >>> 
    >>> # Create PCA embedding
    >>> pca_embedding = exp.create_embedding('pca', n_components=3, verbose=False)
    >>> 
    >>> # Create t-SNE embedding  
    >>> tsne_embedding = exp.create_embedding('tsne', n_components=3, 
    ...                                       perplexity=10, random_state=42)
    >>> 
    >>> # Basic comparison without selectivity
    >>> comparison_basic = compare_embeddings(exp, ['pca', 'tsne'])
    >>> print(f"Methods compared: {sorted(comparison_basic['methods'])}")
    Methods compared: ['pca', 'tsne']
    >>> print(f"Components: PCA={comparison_basic['n_components']['pca']}, "
    ...       f"t-SNE={comparison_basic['n_components']['tsne']}")
    Components: PCA=3, t-SNE=3
    >>> print(f"Mean components per neuron (no selectivity): "
    ...       f"PCA={comparison_basic['mean_components_per_neuron']['pca']}, "
    ...       f"t-SNE={comparison_basic['mean_components_per_neuron']['tsne']}")
    Mean components per neuron (no selectivity): PCA=0, t-SNE=0
    
    Advanced comparison with selectivity analysis:
    
    >>> # The following example shows comparison with selectivity analysis
    >>> # Note: This requires intensive computation and is skipped in doctests
    >>> from driada.intense import compute_embedding_selectivity  # doctest: +SKIP
    >>> 
    >>> # Compute selectivity for both methods
    >>> results = compute_embedding_selectivity(  # doctest: +SKIP
    ...     exp, 
    ...     embedding_methods=['pca', 'tsne'],
    ...     n_shuffles_stage1=100,
    ...     n_shuffles_stage2=500,
    ...     mode='two_stage',
    ...     verbose=False
    ... )
    >>> 
    >>> # Extract IntenseResults objects (not the full dict)
    >>> intense_dict = {  # doctest: +SKIP
    ...     method: results[method]['intense_results'] 
    ...     for method in ['pca', 'tsne']
    ... }
    >>> 
    >>> # Full comparison with selectivity
    >>> comparison_full = compare_embeddings(  # doctest: +SKIP
    ...     exp, ['pca', 'tsne'], 
    ...     intense_results_dict=intense_dict
    ... )
    >>> 
    >>> # Access detailed comparison metrics
    >>> print(f"Participating neurons: PCA={comparison_full['n_participating_neurons']['pca']}, "  # doctest: +SKIP
    ...       f"t-SNE={comparison_full['n_participating_neurons']['tsne']}")
    >>> print(f"Participation overlap: {comparison_full['participation_overlap']['pca_vs_tsne']:.2f}")  # doctest: +SKIP
    
    Error handling:
    
    >>> # Test error handling
    >>> try:
    ...     compare_embeddings(exp, [])  # Empty list
    ... except ValueError as e:
    ...     print(f"Error: {str(e)}")
    Error: method_names cannot be empty
    >>> 
    >>> # Test with non-existent embedding (requesting a method not computed)
    >>> comparison_missing = compare_embeddings(exp, ['pca', 'umap'])  # umap not computed
    >>> print(f"Valid methods found: {sorted(comparison_missing['methods'])}")
    Valid methods found: ['pca']
    
    See Also
    --------
    ~driada.integration.manifold_analysis.get_functional_organization : Analyze individual embeddings
    ~driada.intense.pipelines.compute_embedding_selectivity : Compute selectivity for embeddings    """
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