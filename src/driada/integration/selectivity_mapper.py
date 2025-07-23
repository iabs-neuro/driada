"""
SelectivityManifoldMapper: Bridge between INTENSE selectivity analysis and dimensionality reduction.

This module provides tools for analyzing the relationship between single-neuron selectivity
profiles and population-level manifold structure.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

from ..experiment import Experiment
from ..dim_reduction.data import MVData
from ..intense.pipelines import compute_embedding_selectivity


class SelectivityManifoldMapper:
    """
    Analyzes relationships between neuronal selectivity and manifold structure.
    
    This class provides methods to:
    1. Create population embeddings and store them in the Experiment
    2. Compute selectivity of neurons to embedding components
    3. Analyze functional organization in the manifold
    
    Parameters
    ----------
    experiment : Experiment
        An Experiment object with computed selectivity results
    device : Optional[torch.device], default=None
        Device for computation (for future GPU support)
    logger : Optional[logging.Logger], default=None
        Logger for debugging and info messages
    config : Optional[Dict], default=None
        Configuration dictionary for custom parameters
        
    Examples
    --------
    >>> # Create mapper and generate embeddings
    >>> mapper = SelectivityManifoldMapper(exp)
    >>> 
    >>> # Create and store PCA embedding
    >>> mapper.create_embedding('pca', n_components=10, neuron_selection='significant')
    >>> 
    >>> # Analyze neuron selectivity to PCA components
    >>> results = mapper.analyze_embedding_selectivity('pca')
    >>> 
    >>> # Get functional organization summary
    >>> summary = mapper.get_functional_organization('pca')
    """
    
    def __init__(
        self,
        experiment: Experiment,
        device: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict] = None
    ):
        self.experiment = experiment
        self.device = device
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        
        # Validate experiment has required data
        if not hasattr(experiment, 'calcium') or experiment.calcium is None:
            raise ValueError("Experiment must have calcium data")
        
        # Check if selectivity analysis has been performed
        self.has_selectivity = hasattr(experiment, 'stats_tables') and experiment.stats_tables
        
        if self.logger:
            self.logger.info(
                f"Initialized {self.__class__.__name__} with {experiment.n_cells} neurons"
            )
    
    def create_embedding(
        self,
        method: str,
        n_components: int = 2,
        data_type: str = 'calcium',
        neuron_selection: Optional[Union[str, List[int]]] = None,
        **dr_kwargs
    ) -> np.ndarray:
        """
        Create dimensionality reduction embedding and store it in the experiment.
        
        Parameters
        ----------
        method : str
            DR method name ('pca', 'umap', 'isomap', etc.)
        n_components : int
            Number of embedding dimensions
        data_type : str
            Type of data to use ('calcium' or 'spikes')
        neuron_selection : str, list or None
            How to select neurons:
            - None or 'all': Use all neurons
            - 'significant': Use only significantly selective neurons
            - List of integers: Use specific neuron indices
        **dr_kwargs
            Additional arguments for the DR method
            
        Returns
        -------
        embedding : np.ndarray
            The embedding array, shape (n_timepoints, n_components)
        """
        # Select neurons
        if neuron_selection is None or neuron_selection == 'all':
            neuron_indices = np.arange(self.experiment.n_cells)
        elif neuron_selection == 'significant':
            if not self.has_selectivity:
                raise ValueError("Cannot select significant neurons without selectivity analysis")
            sig_neurons = self.experiment.get_significant_neurons()
            neuron_indices = np.array(list(sig_neurons.keys()))
            if len(neuron_indices) == 0:
                self.logger.warning("No significant neurons found, using all neurons")
                neuron_indices = np.arange(self.experiment.n_cells)
        else:
            neuron_indices = np.array(neuron_selection)
        
        # Get neural data
        if data_type == 'calcium':
            neural_data = self.experiment.calcium.data[neuron_indices, :]
        elif data_type == 'spikes':
            neural_data = self.experiment.spikes.data[neuron_indices, :]
        else:
            raise ValueError("data_type must be 'calcium' or 'spikes'")
        
        # Apply downsampling if requested
        ds = dr_kwargs.pop('ds', 1)  # Remove 'ds' from dr_kwargs and default to 1
        if ds > 1:
            neural_data = neural_data[:, ::ds]
            if self.logger:
                self.logger.info(f"Downsampling data by factor {ds}: {neural_data.shape[1]} timepoints")
        
        # Create MVData and compute embedding
        mvdata = MVData(data=neural_data)  # MVData expects (n_features, n_samples)
        
        # Prepare parameters for the new simplified API
        params = {'dim': n_components}
        
        # Handle method-specific parameters from dr_kwargs
        if 'n_neighbors' in dr_kwargs:
            params['n_neighbors'] = dr_kwargs['n_neighbors']
        if 'min_dist' in dr_kwargs:
            params['min_dist'] = dr_kwargs['min_dist']
        if 'perplexity' in dr_kwargs:
            params['perplexity'] = dr_kwargs['perplexity']
        if 'dm_alpha' in dr_kwargs:
            params['dm_alpha'] = dr_kwargs['dm_alpha']
        
        # Add any other parameters
        params.update(dr_kwargs)
        
        # Get embedding using simplified API
        embedding_obj = mvdata.get_embedding(method=method, **params)
        embedding = embedding_obj.coords.T  # Transpose to (n_timepoints, n_components)
        
        # Check if embedding has all timepoints (accounting for downsampling)
        expected_frames = self.experiment.n_frames // ds
        if embedding.shape[0] < expected_frames:
            n_missing = expected_frames - embedding.shape[0]
            raise ValueError(
                f"{method} embedding dropped {n_missing} timepoints due to graph disconnection. "
                f"This is not supported for INTENSE analysis. Try increasing n_neighbors or using a different method."
            )
        
        # Store metadata
        metadata = {
            'method': method,
            'n_components': n_components,
            'neuron_selection': neuron_selection,
            'neuron_indices': neuron_indices.tolist(),
            'n_neurons': len(neuron_indices),
            'dr_params': dr_kwargs,
            'data_type': data_type,
            'ds': ds  # Store downsampling factor
        }
        
        # Store in experiment
        self.experiment.store_embedding(embedding, method, data_type, metadata)
        
        if self.logger:
            self.logger.info(
                f"Created {method} embedding with {n_components} components "
                f"using {len(neuron_indices)} neurons"
            )
        
        return embedding
    
    def analyze_embedding_selectivity(
        self,
        embedding_methods: Optional[Union[str, List[str]]] = None,
        data_type: str = 'calcium',
        **intense_kwargs
    ) -> Dict:
        """
        Analyze how neurons are selective to embedding components.
        
        Parameters
        ----------
        embedding_methods : str, list or None
            Embedding methods to analyze. If None, analyzes all stored embeddings
        data_type : str
            Data type ('calcium' or 'spikes')
        **intense_kwargs
            Additional arguments for compute_embedding_selectivity
            
        Returns
        -------
        results : dict
            Results from compute_embedding_selectivity
        """
        results = compute_embedding_selectivity(
            self.experiment,
            embedding_methods=embedding_methods,
            data_type=data_type,
            **intense_kwargs
        )
        
        return results
    
    def get_functional_organization(
        self,
        method_name: str,
        data_type: str = 'calcium'
    ) -> Dict:
        """
        Analyze functional organization in the manifold.
        
        Parameters
        ----------
        method_name : str
            Name of the embedding method
        data_type : str
            Data type used for embedding
            
        Returns
        -------
        organization : dict
            Dictionary containing:
            - 'component_importance': Variance explained by each component
            - 'neuron_participation': How many components each neuron contributes to
            - 'component_specialization': How selective each component is
            - 'functional_clusters': Groups of neurons with similar embedding selectivity
        """
        # Get embedding and metadata
        embedding_dict = self.experiment.get_embedding(method_name, data_type)
        embedding = embedding_dict['data']
        metadata = embedding_dict.get('metadata', {})
        neuron_indices = metadata.get('neuron_indices', list(range(self.experiment.n_cells)))
        
        # Compute component importance (variance explained)
        component_var = np.var(embedding, axis=0)
        component_importance = component_var / np.sum(component_var)
        
        # Get selectivity results if available
        stats_key = f"{method_name}_comp0"
        has_embedding_selectivity = (
            hasattr(self.experiment, 'stats_tables') and 
            data_type in self.experiment.stats_tables and
            stats_key in self.experiment.stats_tables[data_type]
        )
        
        organization = {
            'component_importance': component_importance,
            'n_components': embedding.shape[1],
            'n_neurons_used': len(neuron_indices),
            'neuron_indices': neuron_indices
        }
        
        if has_embedding_selectivity:
            # Analyze neuron participation across components
            neuron_participation = {}
            component_specialization = {}
            
            for comp_idx in range(embedding.shape[1]):
                feat_name = f"{method_name}_comp{comp_idx}"
                selective_neurons = []
                
                # Check which neurons are selective to this component
                for neuron_idx in range(self.experiment.n_cells):
                    if (feat_name in self.experiment.significance_tables[data_type] and
                        neuron_idx in self.experiment.significance_tables[data_type][feat_name] and
                        self.experiment.significance_tables[data_type][feat_name][neuron_idx].get('stage2', False)):
                        selective_neurons.append(neuron_idx)
                        
                        # Track neuron participation
                        if neuron_idx not in neuron_participation:
                            neuron_participation[neuron_idx] = []
                        neuron_participation[neuron_idx].append(comp_idx)
                
                component_specialization[comp_idx] = {
                    'n_selective_neurons': len(selective_neurons),
                    'selective_neurons': selective_neurons,
                    'selectivity_rate': len(selective_neurons) / self.experiment.n_cells
                }
            
            # Identify functional clusters (neurons selective to same components)
            from collections import defaultdict
            cluster_map = defaultdict(list)
            
            for neuron_idx, components in neuron_participation.items():
                cluster_key = tuple(sorted(components))
                cluster_map[cluster_key].append(neuron_idx)
            
            functional_clusters = []
            for components, neurons in cluster_map.items():
                if len(neurons) > 1:  # Only keep clusters with multiple neurons
                    functional_clusters.append({
                        'components': list(components),
                        'neurons': neurons,
                        'size': len(neurons)
                    })
            
            # Sort clusters by size
            functional_clusters.sort(key=lambda x: x['size'], reverse=True)
            
            organization.update({
                'neuron_participation': neuron_participation,
                'component_specialization': component_specialization,
                'functional_clusters': functional_clusters,
                'n_participating_neurons': len(neuron_participation),
                'mean_components_per_neuron': np.mean([len(comps) for comps in neuron_participation.values()]) if neuron_participation else 0
            })
        
        return organization
    
    def compare_embeddings(
        self,
        method_names: List[str],
        data_type: str = 'calcium'
    ) -> Dict:
        """
        Compare functional organization across different embedding methods.
        
        Parameters
        ----------
        method_names : list
            List of embedding method names to compare
        data_type : str
            Data type used for embeddings
            
        Returns
        -------
        comparison : dict
            Comparison metrics between embeddings
        """
        organizations = {}
        for method in method_names:
            try:
                organizations[method] = self.get_functional_organization(method, data_type)
            except KeyError:
                self.logger.warning(f"No embedding found for method '{method}'")
        
        if len(organizations) < 2:
            raise ValueError("Need at least 2 embeddings to compare")
        
        comparison = {
            'methods': list(organizations.keys()),
            'n_components': {m: org['n_components'] for m, org in organizations.items()},
            'n_participating_neurons': {m: org.get('n_participating_neurons', 0) for m, org in organizations.items()},
            'mean_components_per_neuron': {m: org.get('mean_components_per_neuron', 0) for m, org in organizations.items()},
            'n_functional_clusters': {m: len(org.get('functional_clusters', [])) for m, org in organizations.items()}
        }
        
        # Compare neuron participation overlap
        if all('neuron_participation' in org for org in organizations.values()):
            method_pairs = [(m1, m2) for i, m1 in enumerate(method_names) for m2 in method_names[i+1:]]
            participation_overlap = {}
            
            for m1, m2 in method_pairs:
                neurons1 = set(organizations[m1]['neuron_participation'].keys())
                neurons2 = set(organizations[m2]['neuron_participation'].keys())
                
                if neurons1 or neurons2:
                    overlap = len(neurons1 & neurons2) / len(neurons1 | neurons2)
                else:
                    overlap = 0
                
                participation_overlap[f"{m1}_vs_{m2}"] = overlap
            
            comparison['participation_overlap'] = participation_overlap
        
        return comparison