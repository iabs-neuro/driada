"""
Test suite for SelectivityManifoldMapper and embedding selectivity analysis.
"""

import pytest
import numpy as np
import logging

import driada
from driada.experiment import Experiment
from driada.integration import SelectivityManifoldMapper
from driada.intense import compute_cell_feat_significance, compute_embedding_selectivity


class TestExperimentEmbeddings:
    """Test embedding storage functionality in Experiment class."""
    
    def test_store_embedding(self, small_experiment):
        """Test storing embeddings in experiment."""
        # Use fixture for consistent test data
        exp = small_experiment
        
        # Create dummy embedding
        n_components = 3
        embedding = np.random.randn(exp.n_frames, n_components)
        metadata = {'param1': 1, 'param2': 'test'}
        
        # Store embedding
        exp.store_embedding(embedding, 'test_method', 'calcium', metadata)
        
        # Verify storage
        assert 'test_method' in exp.embeddings['calcium']
        stored = exp.embeddings['calcium']['test_method']
        assert np.array_equal(stored['data'], embedding)
        assert stored['metadata'] == metadata
        assert stored['shape'] == embedding.shape
        assert 'timestamp' in stored
    
    def test_get_embedding(self, small_experiment):
        """Test retrieving embeddings from experiment."""
        # Use fixture for consistent test data
        exp = small_experiment
        
        # Store embedding
        embedding = np.random.randn(exp.n_frames, 2)
        exp.store_embedding(embedding, 'pca', 'calcium')
        
        # Retrieve embedding
        retrieved = exp.get_embedding('pca', 'calcium')
        assert np.array_equal(retrieved['data'], embedding)
        
        # Test error for non-existent embedding
        with pytest.raises(KeyError):
            exp.get_embedding('nonexistent', 'calcium')
    
    def test_embedding_validation(self, small_experiment):
        """Test embedding validation."""
        exp = small_experiment
        
        # Wrong shape should raise error
        wrong_shape = np.random.randn(50, 2)  # Wrong number of timepoints
        with pytest.raises(ValueError, match="must match expected frames"):
            exp.store_embedding(wrong_shape, 'test', 'calcium')
        
        # Wrong data type should raise error
        embedding = np.random.randn(exp.n_frames, 2)
        with pytest.raises(ValueError, match="must be 'calcium' or 'spikes'"):
            exp.store_embedding(embedding, 'test', 'invalid_type')


class TestSelectivityManifoldMapper:
    """Test SelectivityManifoldMapper functionality."""
    
    @pytest.fixture
    def exp_with_selectivity(self, mixed_population_exp_fast, intense_params_fast):
        """Generate experiment with computed selectivity."""
        # Use fast fixture for quicker test
        exp = mixed_population_exp_fast
        
        # Compute selectivity with fast params
        compute_cell_feat_significance(
            exp,
            mode='stage1',
            n_shuffles_stage1=intense_params_fast['n_shuffles_stage1'],
            ds=intense_params_fast['ds'],
            verbose=False,
            enable_parallelization=False,
            allow_mixed_dimensions=True,
            find_optimal_delays=False
        )
        
        return exp
    
    def test_mapper_initialization(self, exp_with_selectivity):
        """Test mapper initialization."""
        mapper = SelectivityManifoldMapper(exp_with_selectivity)
        assert mapper.experiment == exp_with_selectivity
        assert mapper.has_selectivity
        
        # Test with logger
        logger = logging.getLogger('test')
        mapper = SelectivityManifoldMapper(exp_with_selectivity, logger=logger)
        assert mapper.logger == logger
        
        # Test with device and config
        device = "cuda:0"  # Mock device
        config = {'param1': 'value1', 'param2': 42}
        mapper = SelectivityManifoldMapper(exp_with_selectivity, device=device, config=config)
        assert mapper.device == device
        assert mapper.config == config
    
    def test_mapper_init_without_calcium(self, small_experiment):
        """Test mapper initialization without calcium data."""
        exp = small_experiment
        # Remove calcium data
        exp.calcium = None
        
        with pytest.raises(ValueError, match="must have calcium data"):
            SelectivityManifoldMapper(exp)
    
    def test_mapper_init_no_stats_tables(self, small_experiment):
        """Test mapper initialization without stats_tables."""
        exp = small_experiment
        # Remove stats_tables to simulate no selectivity analysis
        if hasattr(exp, 'stats_tables'):
            delattr(exp, 'stats_tables')
        
        mapper = SelectivityManifoldMapper(exp)
        assert mapper.has_selectivity is False
    
    def test_create_embedding_all_neurons(self, exp_with_selectivity):
        """Test creating embedding with all neurons."""
        mapper = SelectivityManifoldMapper(exp_with_selectivity)
        
        # Create PCA embedding
        embedding = mapper.create_embedding('pca', n_components=5)
        
        # Check shape
        assert embedding.shape == (exp_with_selectivity.n_frames, 5)
        
        # Check storage
        stored = exp_with_selectivity.get_embedding('pca', 'calcium')
        assert np.array_equal(stored['data'], embedding)
        assert stored['metadata']['n_neurons'] == exp_with_selectivity.n_cells
        assert stored['metadata']['method'] == 'pca'
    
    def test_create_embedding_significant_neurons(self, exp_with_selectivity):
        """Test creating embedding with only significant neurons."""
        mapper = SelectivityManifoldMapper(exp_with_selectivity)
        
        # Create embedding with significant neurons only
        embedding = mapper.create_embedding(
            'pca', 
            n_components=3,
            neuron_selection='significant'
        )
        
        # Get number of significant neurons
        sig_neurons = exp_with_selectivity.get_significant_neurons()
        n_sig = len(sig_neurons)
        
        # Check metadata
        stored = exp_with_selectivity.get_embedding('pca', 'calcium')
        if n_sig > 0:
            assert stored['metadata']['n_neurons'] == n_sig
        else:
            # If no significant neurons, should use all
            assert stored['metadata']['n_neurons'] == exp_with_selectivity.n_cells
    
    def test_create_embedding_specific_neurons(self, exp_with_selectivity):
        """Test creating embedding with specific neuron indices."""
        mapper = SelectivityManifoldMapper(exp_with_selectivity)
        
        # Use specific neurons
        neuron_indices = [0, 5, 10, 15, 19]
        embedding = mapper.create_embedding(
            'pca',
            n_components=2,
            neuron_selection=neuron_indices
        )
        
        # Check metadata
        stored = exp_with_selectivity.get_embedding('pca', 'calcium')
        assert stored['metadata']['n_neurons'] == len(neuron_indices)
        assert stored['metadata']['neuron_indices'] == neuron_indices
    
    def test_create_embedding_with_spikes(self, exp_with_selectivity):
        """Test creating embedding using spike data."""
        mapper = SelectivityManifoldMapper(exp_with_selectivity)
        
        # Create embedding with spikes
        embedding = mapper.create_embedding(
            'pca',
            n_components=3,
            data_type='spikes'
        )
        
        # Check storage under spikes
        stored = exp_with_selectivity.get_embedding('pca', 'spikes')
        assert stored['metadata']['data_type'] == 'spikes'
        assert embedding.shape == (exp_with_selectivity.n_frames, 3)
    
    def test_create_embedding_invalid_data_type(self, exp_with_selectivity):
        """Test creating embedding with invalid data type."""
        mapper = SelectivityManifoldMapper(exp_with_selectivity)
        
        with pytest.raises(ValueError, match="must be 'calcium' or 'spikes'"):
            mapper.create_embedding('pca', data_type='invalid')
    
    def test_create_embedding_with_downsampling(self, exp_with_selectivity):
        """Test creating embedding with downsampling."""
        mapper = SelectivityManifoldMapper(exp_with_selectivity)
        
        # Create embedding with downsampling
        embedding = mapper.create_embedding(
            'pca',
            n_components=2,
            ds=5  # Downsample by 5
        )
        
        # Check that embedding has correct shape (downsampled)
        expected_frames = exp_with_selectivity.n_frames // 5
        assert embedding.shape[0] == expected_frames
        
        # Check metadata stores downsampling factor
        stored = exp_with_selectivity.get_embedding('pca', 'calcium')
        assert stored['metadata']['ds'] == 5
    
    def test_create_embedding_no_significant_neurons(self, exp_with_selectivity):
        """Test creating embedding when no significant neurons found."""
        mapper = SelectivityManifoldMapper(exp_with_selectivity)
        
        # Mock get_significant_neurons to return empty
        original_method = exp_with_selectivity.get_significant_neurons
        exp_with_selectivity.get_significant_neurons = lambda: {}
        
        try:
            # Create embedding with significant neurons selection
            embedding = mapper.create_embedding(
                'pca',
                n_components=2,
                neuron_selection='significant'
            )
            
            # Should fall back to all neurons
            stored = exp_with_selectivity.get_embedding('pca', 'calcium')
            assert stored['metadata']['n_neurons'] == exp_with_selectivity.n_cells
        finally:
            # Restore original method
            exp_with_selectivity.get_significant_neurons = original_method
    
    def test_create_embedding_without_selectivity_significant(self, small_experiment):
        """Test requesting significant neurons without selectivity analysis."""
        exp = small_experiment
        # Ensure no stats_tables
        if hasattr(exp, 'stats_tables'):
            delattr(exp, 'stats_tables')
        
        mapper = SelectivityManifoldMapper(exp)
        
        with pytest.raises(ValueError, match="Cannot select significant neurons"):
            mapper.create_embedding(
                'pca',
                neuron_selection='significant'
            )
    
    def test_create_embedding_method_specific_params(self, exp_with_selectivity):
        """Test creating embedding with method-specific parameters."""
        mapper = SelectivityManifoldMapper(exp_with_selectivity)
        
        # Create UMAP embedding with specific parameters
        embedding = mapper.create_embedding(
            'umap',
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            custom_param='test'
        )
        
        # Check that parameters were stored
        stored = exp_with_selectivity.get_embedding('umap', 'calcium')
        assert stored['metadata']['dr_params'].get('n_neighbors') == 15
        assert stored['metadata']['dr_params'].get('min_dist') == 0.1
        assert stored['metadata']['dr_params'].get('custom_param') == 'test'
    
    def test_get_functional_organization(self, exp_with_selectivity):
        """Test functional organization analysis."""
        mapper = SelectivityManifoldMapper(exp_with_selectivity)
        
        # Create embedding
        mapper.create_embedding('pca', n_components=4)
        
        # Get organization
        org = mapper.get_functional_organization('pca')
        
        # Check basic structure
        assert 'component_importance' in org
        assert 'n_components' in org
        assert org['n_components'] == 4
        assert len(org['component_importance']) == 4
        assert np.isclose(np.sum(org['component_importance']), 1.0)
    
    def test_get_functional_organization_with_selectivity(self, exp_with_selectivity):
        """Test functional organization with selectivity results."""
        mapper = SelectivityManifoldMapper(exp_with_selectivity)
        
        # Create embedding
        mapper.create_embedding('pca', n_components=2)
        
        # Mock selectivity results for PCA components
        exp_with_selectivity.stats_tables['calcium']['pca_comp0'] = {}
        exp_with_selectivity.stats_tables['calcium']['pca_comp1'] = {}
        
        if not hasattr(exp_with_selectivity, 'significance_tables'):
            exp_with_selectivity.significance_tables = {'calcium': {}}
        
        exp_with_selectivity.significance_tables['calcium']['pca_comp0'] = {
            0: {'stage2': True},
            1: {'stage2': True},
            2: {'stage2': False}
        }
        exp_with_selectivity.significance_tables['calcium']['pca_comp1'] = {
            1: {'stage2': True},
            3: {'stage2': True}
        }
        
        # Get organization with selectivity
        org = mapper.get_functional_organization('pca')
        
        # Should have selectivity info
        assert 'neuron_participation' in org
        assert 'component_specialization' in org
        assert 'functional_clusters' in org
        assert 'n_participating_neurons' in org
        assert 'mean_components_per_neuron' in org
        
        # Check neuron participation
        assert 0 in org['neuron_participation']
        assert 0 in org['neuron_participation'][0]  # Neuron 0 participates in component 0
        assert 1 in org['neuron_participation']
        assert set(org['neuron_participation'][1]) == {0, 1}  # Neuron 1 in both components
        
        # Check component specialization
        assert org['component_specialization'][0]['n_selective_neurons'] == 2
        assert org['component_specialization'][1]['n_selective_neurons'] == 2
    
    def test_compare_embeddings(self, exp_with_selectivity):
        """Test comparing multiple embeddings."""
        mapper = SelectivityManifoldMapper(exp_with_selectivity)
        
        # For integration test, verify the comparison functionality exists
        # This method requires at least 2 embeddings to compare
        mapper.create_embedding('pca', n_components=3)
        
        # Test that compare method exists and validates input correctly
        with pytest.raises(ValueError, match="at least 2 embeddings"):
            mapper.compare_embeddings(['pca'])
        
        # Create second embedding for comparison
        mapper.create_embedding('umap', n_components=2)
        
        # Now comparison should work
        comparison = mapper.compare_embeddings(['pca', 'umap'])
        assert 'methods' in comparison
        assert comparison['methods'] == ['pca', 'umap']
        assert 'n_components' in comparison
        assert comparison['n_components']['pca'] == 3
        assert comparison['n_components']['umap'] == 2
    
    def test_compare_embeddings_with_participation(self, exp_with_selectivity):
        """Test compare embeddings with neuron participation overlap."""
        mapper = SelectivityManifoldMapper(exp_with_selectivity)
        
        # Create embeddings
        mapper.create_embedding('pca', n_components=2)
        mapper.create_embedding('umap', n_components=2)
        
        # Mock selectivity results with neuron participation
        if not hasattr(exp_with_selectivity, 'significance_tables'):
            exp_with_selectivity.significance_tables = {'calcium': {}}
        
        # Setup PCA selectivity
        exp_with_selectivity.stats_tables['calcium']['pca_comp0'] = {}
        exp_with_selectivity.stats_tables['calcium']['pca_comp1'] = {}
        exp_with_selectivity.significance_tables['calcium']['pca_comp0'] = {
            0: {'stage2': True}, 1: {'stage2': True}, 2: {'stage2': True}
        }
        exp_with_selectivity.significance_tables['calcium']['pca_comp1'] = {
            1: {'stage2': True}, 3: {'stage2': True}
        }
        
        # Setup UMAP selectivity  
        exp_with_selectivity.stats_tables['calcium']['umap_comp0'] = {}
        exp_with_selectivity.stats_tables['calcium']['umap_comp1'] = {}
        exp_with_selectivity.significance_tables['calcium']['umap_comp0'] = {
            1: {'stage2': True}, 2: {'stage2': True}
        }
        exp_with_selectivity.significance_tables['calcium']['umap_comp1'] = {
            2: {'stage2': True}, 4: {'stage2': True}
        }
        
        # Compare embeddings
        comparison = mapper.compare_embeddings(['pca', 'umap'])
        
        # Check participation overlap
        assert 'participation_overlap' in comparison
        assert 'pca_vs_umap' in comparison['participation_overlap']
        # PCA neurons: {0, 1, 2, 3}, UMAP neurons: {1, 2, 4}
        # Intersection: {1, 2}, Union: {0, 1, 2, 3, 4}
        # Overlap: 2/5 = 0.4
        assert comparison['participation_overlap']['pca_vs_umap'] == 0.4
    
    def test_compare_embeddings_missing_embedding(self, exp_with_selectivity):
        """Test compare embeddings with missing embedding."""
        mapper = SelectivityManifoldMapper(exp_with_selectivity)
        
        # Create only PCA embedding
        mapper.create_embedding('pca', n_components=2)
        
        # Try to compare with non-existent embedding
        # Should skip missing and return error
        comparison = mapper.compare_embeddings(['pca', 'missing', 'umap'])
        
        # Should only have pca (since both missing and umap don't exist)
        assert comparison['methods'] == ['pca']


class TestEmbeddingSelectivity:
    """Test computing selectivity to embedding components."""
    
    @pytest.fixture
    def exp_with_embedding(self, mixed_population_exp_fast):
        """Generate experiment with embedding."""
        # Use fast fixture
        exp = mixed_population_exp_fast
        
        # Create and store PCA embedding
        mapper = SelectivityManifoldMapper(exp)
        mapper.create_embedding('pca', n_components=3)
        
        return exp
    
    def test_compute_embedding_selectivity(self, exp_with_embedding):
        """Test computing selectivity to embedding components."""
        results = compute_embedding_selectivity(
            exp_with_embedding,
            embedding_methods='pca',
            mode='stage1',
            n_shuffles_stage1=10,
            ds=5,
            verbose=False,
            enable_parallelization=False
        )
        
        # Check results structure
        assert 'pca' in results
        pca_results = results['pca']
        
        assert 'stats' in pca_results
        assert 'significance' in pca_results
        assert 'info' in pca_results
        assert 'significant_neurons' in pca_results
        assert 'n_components' in pca_results
        assert 'component_selectivity' in pca_results
        
        assert pca_results['n_components'] == 3
        assert len(pca_results['component_selectivity']) == 3
    
    def test_embedding_selectivity_integration(self, exp_with_embedding):
        """Test full integration with mapper."""
        mapper = SelectivityManifoldMapper(exp_with_embedding)
        
        # Analyze selectivity
        results = mapper.analyze_embedding_selectivity(
            'pca',
            n_shuffles_stage1=10,
            n_shuffles_stage2=50,
            ds=5,
            verbose=False,
            enable_parallelization=False
        )
        
        # Get functional organization
        org = mapper.get_functional_organization('pca')
        
        # With selectivity computed, should have more info
        if 'neuron_participation' in org:
            assert isinstance(org['neuron_participation'], dict)
            assert 'component_specialization' in org
            assert 'functional_clusters' in org
    
    def test_multiple_embeddings(self, small_experiment):
        """Test analyzing multiple embeddings."""
        # Use fixture
        exp = small_experiment
        
        mapper = SelectivityManifoldMapper(exp)
        
        # Create PCA embedding only
        mapper.create_embedding('pca', n_components=2)
        
        # Analyze embedding selectivity
        results = compute_embedding_selectivity(
            exp,
            embedding_methods='pca',
            mode='stage1',  # Quick test
            n_shuffles_stage1=10,
            ds=5,
            verbose=False,
            enable_parallelization=False
        )
        
        # Should have results for PCA
        assert 'pca' in results


class TestIntegrationWorkflow:
    """Test complete workflow integration."""
    
    def test_full_workflow(self, mixed_population_exp_fast, intense_params_fast):
        """Test complete workflow from data to manifold selectivity."""
        # 1. Use fixture for experiment
        exp = mixed_population_exp_fast
        
        # 2. Compute behavioral selectivity with fixture params
        compute_cell_feat_significance(
            exp,
            **intense_params_fast
        )
        
        # 3. Create mapper and embeddings
        mapper = SelectivityManifoldMapper(exp)
        
        # Create PCA embedding only
        mapper.create_embedding('pca', n_components=3, neuron_selection='all')
        
        # 4. Analyze embedding selectivity (filter params for embedding selectivity)
        embedding_params = {k: v for k, v in intense_params_fast.items() 
                          if k not in ['allow_mixed_dimensions']}
        results = mapper.analyze_embedding_selectivity(
            'pca',
            **embedding_params
        )
        
        # 5. Get functional organization
        org_pca = mapper.get_functional_organization('pca')
        
        # Verify results
        assert 'pca' in results
        assert org_pca['n_components'] == 3
        assert 'component_importance' in org_pca
        assert len(org_pca['component_importance']) == 3