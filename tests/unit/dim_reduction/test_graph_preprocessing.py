"""Test graph preprocessing parameter functionality."""

import pytest
import numpy as np
from driada.dim_reduction.dr_base import merge_params_with_defaults, METHODS_DICT
from driada.dim_reduction.graph import ProximityGraph
from driada.dim_reduction.data import MVData


class TestGraphPreprocessing:
    """Test suite for graph preprocessing parameter."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        # Create data with clear disconnected components
        n_samples = 100
        n_features = 10
        
        # Create two separate clusters
        data1 = np.random.randn(n_features, n_samples // 2) 
        data2 = np.random.randn(n_features, n_samples // 2) + 1000  # Very far away
        
        data = np.hstack([data1, data2])
        return data
    
    @pytest.fixture
    def mvdata(self, sample_data):
        """Create MVData object from sample data."""
        return MVData(data=sample_data, labels=None)
    
    def test_default_preprocessing_for_connected_methods(self):
        """Test that methods requiring connected graphs default to giant_cc."""
        connected_methods = ["le", "auto_le", "dmaps", "auto_dmaps", "isomap", "lle", "hlle", "mvu"]
        
        for method_name in connected_methods:
            params = merge_params_with_defaults(method_name)
            g_params = params["g_params"]
            
            assert g_params is not None, f"{method_name} should have graph params"
            assert g_params["graph_preprocessing"] == "giant_cc", \
                f"{method_name} should default to giant_cc preprocessing"
    
    def test_default_preprocessing_for_disconnected_methods(self):
        """Test that methods handling disconnected graphs default to None."""
        # Currently only UMAP handles disconnected graphs
        params = merge_params_with_defaults("umap")
        g_params = params["g_params"]
        
        assert g_params is not None, "umap should have graph params"
        assert g_params["graph_preprocessing"] is None, \
            "umap should default to None preprocessing"
    
    def test_user_override_preprocessing(self):
        """Test that user can override default preprocessing."""
        # Test overriding for a method that defaults to giant_cc
        user_params = {"g_params": {"graph_preprocessing": "remove_isolates"}}
        params = merge_params_with_defaults("le", user_params)
        assert params["g_params"]["graph_preprocessing"] == "remove_isolates"
        
        # Test overriding for a method that defaults to None
        user_params = {"g_params": {"graph_preprocessing": "giant_cc"}}
        params = merge_params_with_defaults("umap", user_params)
        assert params["g_params"]["graph_preprocessing"] == "giant_cc"
    
    def test_proximity_graph_uses_preprocessing(self, mvdata):
        """Test that ProximityGraph actually uses the preprocessing parameter."""
        # Test with default preprocessing
        params = merge_params_with_defaults("le")
        # Increase max_deleted_nodes to allow for our test data
        params["g_params"]["max_deleted_nodes"] = 0.6
        
        graph = ProximityGraph(
            d=mvdata.data,
            m_params=params["m_params"],
            g_params=params["g_params"],
            create_nx_graph=False
        )
        # Should have removed disconnected components
        assert graph.n < mvdata.data.shape[1], \
            "giant_cc preprocessing should remove isolated nodes"
        
        # Test with no preprocessing
        g_params_no_prep = params["g_params"].copy()
        g_params_no_prep["graph_preprocessing"] = None
        graph_no_prep = ProximityGraph(
            d=mvdata.data,
            m_params=params["m_params"],
            g_params=g_params_no_prep,
            create_nx_graph=False
        )
        # Should keep all nodes (minus self-loops)
        assert graph_no_prep.n == mvdata.data.shape[1], \
            "None preprocessing should keep all nodes"
    
    def test_preprocessing_options(self, mvdata):
        """Test different preprocessing options work correctly."""
        params = merge_params_with_defaults("le")
        # Increase max_deleted_nodes to allow for our test data
        params["g_params"]["max_deleted_nodes"] = 0.6
        
        preprocessing_options = [None, "remove_isolates", "giant_cc"]
        
        for prep in preprocessing_options:
            g_params = params["g_params"].copy()
            g_params["graph_preprocessing"] = prep
            
            # Should not raise an error
            graph = ProximityGraph(
                d=mvdata.data,
                m_params=params["m_params"],
                g_params=g_params,
                create_nx_graph=False
            )
            
            assert hasattr(graph, 'adj'), f"Graph should have adjacency matrix with {prep} preprocessing"
            assert hasattr(graph, 'n'), f"Graph should have node count with {prep} preprocessing"
    
    def test_handles_disconnected_property(self):
        """Test that handles_disconnected_graphs property is set correctly."""
        # Methods that should handle disconnected graphs
        assert METHODS_DICT["umap"].handles_disconnected_graphs == 1
        
        # Methods that should not handle disconnected graphs
        for method_name in ["le", "auto_le", "dmaps", "auto_dmaps", "isomap", "lle", "hlle", "mvu"]:
            assert METHODS_DICT[method_name].handles_disconnected_graphs == 0
    
    def test_backwards_compatibility(self):
        """Test that existing code without explicit preprocessing still works."""
        # Create params without any preprocessing specified
        base_params = {
            "g_method_name": "knn",
            "nn": 15,
            "weighted": 0,
            "max_deleted_nodes": 0.2,
            "dist_to_aff": "hk",
        }
        
        # For methods requiring connected graphs, should default to giant_cc
        params_le = merge_params_with_defaults("le", {"g_params": base_params})
        assert params_le["g_params"]["graph_preprocessing"] == "giant_cc"
        
        # For UMAP, should default to None
        params_umap = merge_params_with_defaults("umap", {"g_params": base_params})
        assert params_umap["g_params"]["graph_preprocessing"] is None