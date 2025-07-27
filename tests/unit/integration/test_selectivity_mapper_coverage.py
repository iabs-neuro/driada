"""
Additional tests for SelectivityManifoldMapper to increase coverage.
Focuses on edge cases and untested paths.
"""

import pytest
import numpy as np
import logging
from unittest.mock import Mock, patch, MagicMock


class TestSelectivityMapperCoverage:
    """Test cases to increase coverage of SelectivityManifoldMapper."""
    
    @pytest.fixture
    def mock_experiment(self):
        """Create a mock experiment with minimal required attributes."""
        exp = Mock()
        exp.n_cells = 20
        exp.n_frames = 1000
        exp.calcium = Mock()
        exp.calcium.data = np.random.randn(20, 1000)
        exp.spikes = Mock()
        exp.spikes.data = np.random.randn(20, 1000) > 0
        exp.embeddings = {'calcium': {}, 'spikes': {}}
        exp.stats_tables = {'calcium': {}}
        exp.significance_tables = {'calcium': {}}
        
        # Mock methods
        exp.get_significant_neurons = Mock(return_value={0: {'features': ['feat1']}, 5: {'features': ['feat2']}})
        exp.store_embedding = Mock()
        exp.get_embedding = Mock()
        
        return exp
    
    @pytest.fixture
    def mock_mapper(self, mock_experiment):
        """Create a SelectivityManifoldMapper with mocked dependencies."""
        # Import here to avoid import errors
        from driada.integration import SelectivityManifoldMapper
        
        mapper = SelectivityManifoldMapper(mock_experiment)
        return mapper
    
    def test_mapper_init_without_calcium(self):
        """Test mapper initialization without calcium data."""
        from driada.integration import SelectivityManifoldMapper
        
        exp = Mock()
        exp.calcium = None
        
        with pytest.raises(ValueError, match="must have calcium data"):
            SelectivityManifoldMapper(exp)
    
    def test_mapper_init_no_stats_tables(self):
        """Test mapper initialization without stats_tables."""
        from driada.integration import SelectivityManifoldMapper
        
        exp = Mock()
        exp.calcium = Mock()
        exp.n_cells = 10
        # No stats_tables attribute
        
        mapper = SelectivityManifoldMapper(exp)
        assert mapper.has_selectivity is False
    
    def test_mapper_with_custom_config(self, mock_experiment):
        """Test mapper with custom config and device."""
        from driada.integration import SelectivityManifoldMapper
        
        device = Mock()
        config = {'param1': 'value1'}
        logger = logging.getLogger('test')
        
        mapper = SelectivityManifoldMapper(
            mock_experiment,
            device=device,
            logger=logger,
            config=config
        )
        
        assert mapper.device == device
        assert mapper.config == config
        assert mapper.logger == logger
    
    def test_create_embedding_with_spikes(self, mock_mapper):
        """Test creating embedding using spike data."""
        with patch('driada.integration.selectivity_mapper.MVData') as mock_mvdata:
            # Mock the embedding object
            mock_embedding = Mock()
            mock_embedding.coords = np.random.randn(3, 100).T  # Transposed
            
            mock_mvdata_instance = Mock()
            mock_mvdata_instance.get_embedding = Mock(return_value=mock_embedding)
            mock_mvdata.return_value = mock_mvdata_instance
            
            # Create embedding with spikes
            embedding = mock_mapper.create_embedding(
                'pca',
                n_components=3,
                data_type='spikes'
            )
            
            # Verify spikes data was used
            mock_mvdata.assert_called_once()
            call_args = mock_mvdata.call_args[1]
            assert 'data' in call_args
            # Should use spikes data
            assert call_args['data'].shape == (20, 1000)
    
    def test_create_embedding_invalid_data_type(self, mock_mapper):
        """Test creating embedding with invalid data type."""
        with pytest.raises(ValueError, match="must be 'calcium' or 'spikes'"):
            mock_mapper.create_embedding('pca', data_type='invalid')
    
    def test_create_embedding_with_downsampling(self, mock_mapper):
        """Test creating embedding with downsampling."""
        with patch('driada.integration.selectivity_mapper.MVData') as mock_mvdata:
            # Mock the embedding object
            mock_embedding = Mock()
            mock_embedding.coords = np.random.randn(2, 200).T  # Downsampled
            
            mock_mvdata_instance = Mock()
            mock_mvdata_instance.get_embedding = Mock(return_value=mock_embedding)
            mock_mvdata.return_value = mock_mvdata_instance
            
            # Create embedding with downsampling
            embedding = mock_mapper.create_embedding(
                'pca',
                n_components=2,
                ds=5  # Downsample by 5
            )
            
            # Verify downsampling happened
            mock_mvdata.assert_called_once()
            call_args = mock_mvdata.call_args[1]
            # Data should be downsampled: 1000 / 5 = 200
            assert call_args['data'].shape[1] == 200
    
    def test_create_embedding_no_significant_neurons(self, mock_mapper):
        """Test creating embedding when no significant neurons found."""
        # Mock get_significant_neurons to return empty
        mock_mapper.experiment.get_significant_neurons = Mock(return_value={})
        
        with patch('driada.integration.selectivity_mapper.MVData') as mock_mvdata:
            # Mock the embedding object
            mock_embedding = Mock()
            mock_embedding.coords = np.random.randn(2, 1000).T
            
            mock_mvdata_instance = Mock()
            mock_mvdata_instance.get_embedding = Mock(return_value=mock_embedding)
            mock_mvdata.return_value = mock_mvdata_instance
            
            # Create embedding with significant neurons selection
            embedding = mock_mapper.create_embedding(
                'pca',
                n_components=2,
                neuron_selection='significant'
            )
            
            # Should fall back to all neurons
            mock_mvdata.assert_called_once()
            call_args = mock_mvdata.call_args[1]
            # Should use all 20 neurons
            assert call_args['data'].shape[0] == 20
    
    def test_create_embedding_without_selectivity_significant(self, mock_mapper):
        """Test requesting significant neurons without selectivity analysis."""
        mock_mapper.has_selectivity = False
        
        with pytest.raises(ValueError, match="Cannot select significant neurons"):
            mock_mapper.create_embedding(
                'pca',
                neuron_selection='significant'
            )
    
    def test_create_embedding_method_specific_params(self, mock_mapper):
        """Test creating embedding with method-specific parameters."""
        with patch('driada.integration.selectivity_mapper.MVData') as mock_mvdata:
            # Mock the embedding object
            mock_embedding = Mock()
            mock_embedding.coords = np.random.randn(2, 1000).T
            
            mock_mvdata_instance = Mock()
            mock_mvdata_instance.get_embedding = Mock(return_value=mock_embedding)
            mock_mvdata.return_value = mock_mvdata_instance
            
            # Create embedding with various parameters
            embedding = mock_mapper.create_embedding(
                'umap',
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                custom_param='test'
            )
            
            # Verify parameters were passed
            get_embedding_call = mock_mvdata_instance.get_embedding.call_args
            assert get_embedding_call[1]['n_neighbors'] == 15
            assert get_embedding_call[1]['min_dist'] == 0.1
            assert get_embedding_call[1]['custom_param'] == 'test'
    
    def test_create_embedding_graph_disconnection_error(self, mock_mapper):
        """Test error when embedding drops timepoints due to graph disconnection."""
        with patch('driada.integration.selectivity_mapper.MVData') as mock_mvdata:
            # Mock embedding with fewer timepoints
            mock_embedding = Mock()
            mock_embedding.coords = np.random.randn(2, 800).T  # Missing 200 frames
            
            mock_mvdata_instance = Mock()
            mock_mvdata_instance.get_embedding = Mock(return_value=mock_embedding)
            mock_mvdata.return_value = mock_mvdata_instance
            
            with pytest.raises(ValueError, match="dropped .* timepoints due to graph disconnection"):
                mock_mapper.create_embedding('isomap', n_components=2)
    
    def test_get_functional_organization_no_selectivity(self, mock_mapper):
        """Test functional organization without selectivity results."""
        # Setup embedding in experiment
        mock_mapper.experiment.get_embedding = Mock(return_value={
            'data': np.random.randn(1000, 3),
            'metadata': {'neuron_indices': [0, 1, 2, 3, 4]}
        })
        
        org = mock_mapper.get_functional_organization('pca')
        
        # Should have basic info but not selectivity-related fields
        assert 'component_importance' in org
        assert 'n_components' in org
        assert org['n_components'] == 3
        assert 'neuron_participation' not in org
        assert 'functional_clusters' not in org
    
    def test_get_functional_organization_with_selectivity(self, mock_mapper):
        """Test functional organization with selectivity results."""
        # Setup embedding in experiment
        mock_mapper.experiment.get_embedding = Mock(return_value={
            'data': np.random.randn(1000, 2),
            'metadata': {}
        })
        
        # Setup selectivity results
        mock_mapper.experiment.stats_tables = {
            'calcium': {
                'pca_comp0': {},
                'pca_comp1': {}
            }
        }
        
        mock_mapper.experiment.significance_tables = {
            'calcium': {
                'pca_comp0': {
                    0: {'stage2': True},
                    1: {'stage2': True},
                    2: {'stage2': False}
                },
                'pca_comp1': {
                    1: {'stage2': True},
                    3: {'stage2': True}
                }
            }
        }
        
        org = mock_mapper.get_functional_organization('pca')
        
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
    
    def test_compare_embeddings_not_enough(self, mock_mapper):
        """Test compare embeddings with insufficient embeddings."""
        with pytest.raises(ValueError, match="Need at least 2 embeddings"):
            mock_mapper.compare_embeddings(['pca'])
    
    def test_compare_embeddings_missing_embedding(self, mock_mapper):
        """Test compare embeddings with missing embedding."""
        # Mock get_functional_organization to raise KeyError for missing
        def side_effect(method, data_type):
            if method == 'missing':
                raise KeyError("No embedding found")
            return {
                'n_components': 2,
                'n_participating_neurons': 5,
                'mean_components_per_neuron': 1.5,
                'functional_clusters': []
            }
        
        mock_mapper.get_functional_organization = Mock(side_effect=side_effect)
        
        # Should skip missing embedding
        comparison = mock_mapper.compare_embeddings(['pca', 'missing', 'umap'])
        
        # Should only have pca and umap
        assert comparison['methods'] == ['pca', 'umap']
    
    def test_compare_embeddings_with_participation(self, mock_mapper):
        """Test compare embeddings with neuron participation overlap."""
        # Mock get_functional_organization
        def mock_org(method, data_type):
            if method == 'pca':
                return {
                    'n_components': 2,
                    'n_participating_neurons': 5,
                    'mean_components_per_neuron': 1.5,
                    'functional_clusters': [{'size': 3}],
                    'neuron_participation': {0: [0], 1: [0, 1], 2: [1]}
                }
            else:  # umap
                return {
                    'n_components': 3,
                    'n_participating_neurons': 4,
                    'mean_components_per_neuron': 1.2,
                    'functional_clusters': [],
                    'neuron_participation': {1: [0], 2: [1], 3: [2]}
                }
        
        mock_mapper.get_functional_organization = Mock(side_effect=mock_org)
        
        comparison = mock_mapper.compare_embeddings(['pca', 'umap'])
        
        # Check structure
        assert comparison['methods'] == ['pca', 'umap']
        assert comparison['n_components']['pca'] == 2
        assert comparison['n_components']['umap'] == 3
        assert comparison['n_participating_neurons']['pca'] == 5
        assert comparison['n_functional_clusters']['pca'] == 1
        
        # Check overlap calculation
        assert 'participation_overlap' in comparison
        assert 'pca_vs_umap' in comparison['participation_overlap']
        # Overlap: intersection {1, 2} / union {0, 1, 2, 3} = 2/4 = 0.5
        assert comparison['participation_overlap']['pca_vs_umap'] == 0.5
    
    def test_compare_embeddings_empty_participation(self, mock_mapper):
        """Test compare embeddings with empty neuron participation."""
        # Mock get_functional_organization with empty participation
        def mock_org(method, data_type):
            return {
                'n_components': 2,
                'n_participating_neurons': 0,
                'mean_components_per_neuron': 0,
                'functional_clusters': [],
                'neuron_participation': {}
            }
        
        mock_mapper.get_functional_organization = Mock(side_effect=mock_org)
        
        comparison = mock_mapper.compare_embeddings(['pca', 'umap'])
        
        # Should handle empty participation
        assert comparison['participation_overlap']['pca_vs_umap'] == 0
    
    def test_analyze_embedding_selectivity_wrapper(self, mock_mapper):
        """Test analyze_embedding_selectivity method."""
        with patch('driada.integration.selectivity_mapper.compute_embedding_selectivity') as mock_compute:
            mock_compute.return_value = {'pca': {'stats': {}}}
            
            # Call with various parameters
            results = mock_mapper.analyze_embedding_selectivity(
                embedding_methods=['pca', 'umap'],
                data_type='spikes',
                n_shuffles=100,
                custom_param='test'
            )
            
            # Verify call was made correctly
            mock_compute.assert_called_once_with(
                mock_mapper.experiment,
                embedding_methods=['pca', 'umap'],
                data_type='spikes',
                n_shuffles=100,
                custom_param='test'
            )
            
            assert results == {'pca': {'stats': {}}}