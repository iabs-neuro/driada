"""Comprehensive tests for parameter mapping in dimension reduction.

CRITICAL: These tests ensure that ALL parameter aliases and direct
parameters are correctly mapped to their appropriate locations.
"""

import pytest
import numpy as np
from driada.dim_reduction import MVData, METHODS_DICT
from driada.dim_reduction.dr_base import merge_params_with_defaults


class TestCriticalParameterMapping:
    """Test CRITICAL parameter mapping that was previously broken."""

    def test_nn_parameter_mapping(self):
        """Test that nn parameter is correctly mapped to graph params."""
        # This was BROKEN - nn was going to e_params instead of g_params
        params = merge_params_with_defaults("isomap", {"nn": 50})
        assert params["g_params"]["nn"] == 50, "nn parameter not mapped to graph params!"

    def test_k_parameter_mapping(self):
        """Test that k (common alias for nn) is mapped correctly."""
        params = merge_params_with_defaults("isomap", {"k": 30})
        assert params["g_params"]["nn"] == 30, "k parameter not mapped to nn!"

    def test_n_neighbors_still_works(self):
        """Test that n_neighbors (sklearn alias) still works."""
        params = merge_params_with_defaults("isomap", {"n_neighbors": 25})
        assert params["g_params"]["nn"] == 25

    def test_weighted_parameter_mapping(self):
        """Test that weighted goes to graph params."""
        params = merge_params_with_defaults("le", {"weighted": 1})
        assert params["g_params"]["weighted"] == 1
        assert "weighted" not in params["e_params"]

    def test_dist_to_aff_parameter_mapping(self):
        """Test that dist_to_aff goes to graph params."""
        params = merge_params_with_defaults("le", {"dist_to_aff": "gaussian"})
        assert params["g_params"]["dist_to_aff"] == "gaussian"
        assert "dist_to_aff" not in params["e_params"]

    def test_graph_preprocessing_parameter_mapping(self):
        """Test that graph_preprocessing goes to graph params."""
        params = merge_params_with_defaults("le", {"graph_preprocessing": None})
        assert params["g_params"]["graph_preprocessing"] is None
        assert "graph_preprocessing" not in params["e_params"]

    def test_metric_name_direct_parameter(self):
        """Test that metric_name can be passed directly."""
        params = merge_params_with_defaults("le", {"metric_name": "cosine"})
        assert params["m_params"]["metric_name"] == "cosine"

    def test_n_components_alias(self):
        """Test that n_components is mapped to dim."""
        params = merge_params_with_defaults("pca", {"n_components": 5})
        assert params["e_params"]["dim"] == 5
        assert "n_components" not in params["e_params"]


class TestParameterMappingIntegration:
    """Integration tests ensuring parameters work end-to-end."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return MVData(np.random.randn(20, 100))

    def test_nn_parameter_works_in_practice(self, sample_data):
        """Test that nn parameter actually affects the graph construction."""
        # Test with nn=15 (reasonable for 100 samples)
        emb1 = sample_data.get_embedding(method="isomap", nn=15, dim=2)
        assert emb1.graph.nn == 15, "nn parameter not propagated to graph!"

        # Test with nn=30
        emb2 = sample_data.get_embedding(method="isomap", nn=30, dim=2)
        assert emb2.graph.nn == 30, "nn parameter not propagated to graph!"

    def test_k_parameter_works_in_practice(self, sample_data):
        """Test that k parameter works as nn alias."""
        emb = sample_data.get_embedding(method="le", k=15, dim=2)
        assert emb.graph.nn == 15, "k parameter not working as nn alias!"

    def test_all_graph_methods_accept_nn(self, sample_data):
        """Test that ALL graph-based methods accept nn parameter."""
        graph_methods = ["le", "auto_le", "dmaps", "isomap", "lle",
                        "ltsa", "hessian_lle", "modified_lle"]

        for method in graph_methods:
            if method in METHODS_DICT:
                # Should not raise any errors (use nn=15 for stability)
                emb = sample_data.get_embedding(method=method, nn=15, dim=2)
                assert emb.graph.nn == 15, f"{method} doesn't accept nn parameter!"

    def test_n_components_works(self, sample_data):
        """Test that n_components works as dim alias."""
        emb = sample_data.get_embedding(method="pca", n_components=3)
        assert emb.coords.shape[0] == 3, "n_components not working as dim alias!"

    def test_multiple_aliases_priority(self, sample_data):
        """Test parameter priority when multiple aliases are given."""
        # If both nn and n_neighbors are given, last one should win
        # (dict order in Python 3.7+ is insertion order)
        emb = sample_data.get_embedding(method="isomap", nn=10, n_neighbors=20, dim=2)
        assert emb.graph.nn == 20, "Parameter priority not working correctly!"


class TestParameterMappingForAllMethods:
    """Ensure ALL methods handle parameters correctly."""

    @pytest.fixture
    def mvdata(self):
        """Create MVData for testing."""
        np.random.seed(42)
        return MVData(np.random.randn(30, 200))

    def test_pca_parameters(self, mvdata):
        """Test PCA-specific parameters."""
        emb = mvdata.get_embedding(method="pca", dim=3)
        assert emb.coords.shape[0] == 3
        
        emb = mvdata.get_embedding(method="pca", n_components=4)
        assert emb.coords.shape[0] == 4

    def test_umap_parameters(self, mvdata):
        """Test UMAP-specific parameters."""
        emb = mvdata.get_embedding(
            method="umap",
            nn=15,  # Should work now!
            min_dist=0.5,
            dim=2
        )
        assert emb.graph.nn == 15
        assert emb.min_dist == 0.5

    def test_tsne_parameters(self, mvdata):
        """Test t-SNE parameters."""
        emb = mvdata.get_embedding(
            method="tsne",
            perplexity=20,
            n_iter=250,
            dim=2
        )
        assert emb.coords.shape[0] == 2

    def test_lle_variants(self, mvdata):
        """Test LLE variant parameters."""
        for method in ["lle", "ltsa", "hessian_lle", "modified_lle"]:
            if method in METHODS_DICT:
                emb = mvdata.get_embedding(method=method, nn=15, dim=2)
                assert emb.graph.nn == 15, f"{method} not handling nn parameter!"


class TestRegressionPrevention:
    """Tests to prevent regression of parameter mapping bugs."""

    def test_all_parameter_combinations(self):
        """Test various parameter combinations to catch edge cases."""
        test_cases = [
            # (method, params, expected_g_nn)
            ("isomap", {"nn": 30}, 30),
            ("isomap", {"k": 25}, 25),
            ("isomap", {"n_neighbors": 20}, 20),
            ("le", {"nn": 15, "weighted": 1}, 15),
            ("dmaps", {"k": 10, "t": 5}, 10),
            ("umap", {"nn": 40, "min_dist": 0.3}, 40),
        ]
        
        for method, params, expected_nn in test_cases:
            merged = merge_params_with_defaults(method, params)
            if merged["g_params"] is not None:
                actual_nn = merged["g_params"]["nn"]
                assert actual_nn == expected_nn, (
                    f"{method} with {params} gave nn={actual_nn}, "
                    f"expected {expected_nn}"
                )

    def test_no_silent_parameter_loss(self):
        """Ensure parameters don't silently get lost."""
        # All these parameters should end up in the right place
        params = {
            "nn": 30,
            "weighted": 1,
            "dist_to_aff": "gaussian",
            "metric": "cosine",
            "sigma": 2.0,
            "dim": 3,
        }
        
        merged = merge_params_with_defaults("le", params)
        
        # Verify each parameter ended up in the right place
        assert merged["g_params"]["nn"] == 30
        assert merged["g_params"]["weighted"] == 1
        assert merged["g_params"]["dist_to_aff"] == "gaussian"
        assert merged["m_params"]["metric_name"] == "cosine"
        assert merged["m_params"]["sigma"] == 2.0
        assert merged["e_params"]["dim"] == 3


if __name__ == "__main__":
    # Run critical tests to verify the fix
    test = TestCriticalParameterMapping()
    test.test_nn_parameter_mapping()
    test.test_k_parameter_mapping()
    print("✅ Critical parameter mapping tests passed!")