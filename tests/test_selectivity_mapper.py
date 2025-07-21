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
    
    def test_store_embedding(self):
        """Test storing embeddings in experiment."""
        # Generate synthetic experiment
        exp = driada.generate_synthetic_exp(
            n_cfeats=2,
            nneurons=20,
            duration=100,
            seed=42
        )
        
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
    
    def test_get_embedding(self):
        """Test retrieving embeddings from experiment."""
        # Generate synthetic experiment
        exp = driada.generate_synthetic_exp(
            n_cfeats=2,
            nneurons=20,
            duration=100,
            seed=42
        )
        
        # Store embedding
        embedding = np.random.randn(exp.n_frames, 2)
        exp.store_embedding(embedding, 'pca', 'calcium')
        
        # Retrieve embedding
        retrieved = exp.get_embedding('pca', 'calcium')
        assert np.array_equal(retrieved['data'], embedding)
        
        # Test error for non-existent embedding
        with pytest.raises(KeyError):
            exp.get_embedding('nonexistent', 'calcium')
    
    def test_embedding_validation(self):
        """Test embedding validation."""
        exp = driada.generate_synthetic_exp(
            n_cfeats=2,
            nneurons=20,
            duration=100,
            seed=42
        )
        
        # Wrong shape should raise error
        wrong_shape = np.random.randn(50, 2)  # Wrong number of timepoints
        with pytest.raises(ValueError, match="must match experiment frames"):
            exp.store_embedding(wrong_shape, 'test', 'calcium')
        
        # Wrong data type should raise error
        embedding = np.random.randn(exp.n_frames, 2)
        with pytest.raises(ValueError, match="must be 'calcium' or 'spikes'"):
            exp.store_embedding(embedding, 'test', 'invalid_type')


class TestSelectivityManifoldMapper:
    """Test SelectivityManifoldMapper functionality."""
    
    @pytest.fixture
    def exp_with_selectivity(self):
        """Generate experiment with computed selectivity."""
        exp = driada.generate_synthetic_exp(
            n_cfeats=3,
            nneurons=50,
            duration=300,
            seed=42
        )
        
        # Compute selectivity
        compute_cell_feat_significance(
            exp,
            mode='two_stage',
            n_shuffles_stage1=20,
            n_shuffles_stage2=100,
            verbose=False
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
        neuron_indices = [0, 5, 10, 15, 20]
        embedding = mapper.create_embedding(
            'umap',
            n_components=2,
            neuron_selection=neuron_indices,
            n_neighbors=5  # DR parameter
        )
        
        # Check metadata
        stored = exp_with_selectivity.get_embedding('umap', 'calcium')
        assert stored['metadata']['n_neurons'] == len(neuron_indices)
        assert stored['metadata']['neuron_indices'] == neuron_indices
        assert stored['metadata']['dr_params']['n_neighbors'] == 5
    
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
    
    def test_compare_embeddings(self, exp_with_selectivity):
        """Test comparing multiple embeddings."""
        mapper = SelectivityManifoldMapper(exp_with_selectivity)
        
        # Create multiple embeddings
        mapper.create_embedding('pca', n_components=3)
        mapper.create_embedding('umap', n_components=3, n_neighbors=15)
        
        # Compare
        comparison = mapper.compare_embeddings(['pca', 'umap'])
        
        # Check structure
        assert 'methods' in comparison
        assert set(comparison['methods']) == {'pca', 'umap'}
        assert comparison['n_components']['pca'] == 3
        assert comparison['n_components']['umap'] == 3


class TestEmbeddingSelectivity:
    """Test computing selectivity to embedding components."""
    
    @pytest.fixture
    def exp_with_embedding(self):
        """Generate experiment with embedding."""
        exp = driada.generate_synthetic_exp(
            n_cfeats=2,
            nneurons=30,
            duration=200,
            seed=42
        )
        
        # Create and store PCA embedding
        mapper = SelectivityManifoldMapper(exp)
        mapper.create_embedding('pca', n_components=3)
        
        return exp
    
    def test_compute_embedding_selectivity(self, exp_with_embedding):
        """Test computing selectivity to embedding components."""
        results = compute_embedding_selectivity(
            exp_with_embedding,
            embedding_methods='pca',
            mode='two_stage',
            n_shuffles_stage1=10,
            n_shuffles_stage2=50,
            verbose=False
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
            verbose=False
        )
        
        # Get functional organization
        org = mapper.get_functional_organization('pca')
        
        # With selectivity computed, should have more info
        if 'neuron_participation' in org:
            assert isinstance(org['neuron_participation'], dict)
            assert 'component_specialization' in org
            assert 'functional_clusters' in org
    
    def test_multiple_embeddings(self):
        """Test analyzing multiple embeddings."""
        exp = driada.generate_synthetic_exp(
            n_cfeats=2,
            nneurons=20,
            duration=150,
            seed=42
        )
        
        mapper = SelectivityManifoldMapper(exp)
        
        # Create multiple embeddings
        mapper.create_embedding('pca', n_components=2)
        mapper.create_embedding('umap', n_components=2, n_neighbors=15)
        
        # Analyze all
        results = compute_embedding_selectivity(
            exp,
            mode='stage1',  # Quick test
            n_shuffles_stage1=10,
            verbose=False
        )
        
        # Should have results for both
        assert 'pca' in results
        assert 'umap' in results


class TestIntegrationWorkflow:
    """Test complete workflow integration."""
    
    def test_full_workflow(self):
        """Test complete workflow from data to manifold selectivity."""
        # 1. Generate experiment with spatial features
        exp = driada.generate_mixed_population_exp(
            n_neurons=50,
            manifold_fraction=0.6,  # 60% manifold neurons = 30 neurons
            manifold_type='2d_spatial',
            manifold_params={
                'field_sigma': 0.15, 
                'baseline_rate': 0.1, 
                'peak_rate': 1.0,
                'noise_std': 0.05,
                'decay_time': 2.0,
                'calcium_noise_std': 0.1
            },
            duration=300,
            seed=42
        )
        
        # 2. Compute behavioral selectivity
        compute_cell_feat_significance(
            exp,
            mode='two_stage',
            n_shuffles_stage1=20,
            n_shuffles_stage2=100,
            pval_thr=0.05,
            allow_mixed_dimensions=True,
            find_optimal_delays=False,  # Disable to avoid issues with MultiTimeSeries
            verbose=False
        )
        
        # 3. Create mapper and embeddings
        mapper = SelectivityManifoldMapper(exp)
        
        # Create embeddings with different neuron selections
        mapper.create_embedding('pca', n_components=5, neuron_selection='all')
        mapper.create_embedding('umap', n_components=5, neuron_selection='significant')
        
        # 4. Analyze embedding selectivity
        results = mapper.analyze_embedding_selectivity(
            ['pca', 'umap'],
            n_shuffles_stage1=10,
            n_shuffles_stage2=50,
            verbose=False
        )
        
        # 5. Compare organizations
        comparison = mapper.compare_embeddings(['pca', 'umap'])
        
        # Verify results
        assert 'pca' in results
        assert 'umap' in results
        assert 'participation_overlap' in comparison or 'n_participating_neurons' in comparison
        
        # Get functional organizations
        org_pca = mapper.get_functional_organization('pca')
        org_umap = mapper.get_functional_organization('umap')
        
        # Both should have valid organization info
        assert org_pca['n_components'] == 5
        assert org_umap['n_components'] == 5