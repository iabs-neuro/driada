"""Test DR method defaults and simplified API."""

import pytest
import numpy as np
from driada.dim_reduction import MVData, METHODS_DICT
from driada.dim_reduction.dr_base import merge_params_with_defaults


class TestDRMethodDefaults:
    """Test the DRMethod class with default parameters."""

    def test_drmethod_has_defaults(self):
        """Test that DRMethod objects have default parameters."""
        # Check a few key methods
        assert hasattr(METHODS_DICT["pca"], "default_params")
        assert hasattr(METHODS_DICT["pca"], "default_graph_params")
        assert hasattr(METHODS_DICT["pca"], "default_metric_params")

        # PCA should have dim default
        assert METHODS_DICT["pca"].default_params["dim"] == 2

        # Graph-based methods should have graph defaults
        assert METHODS_DICT["isomap"].default_graph_params is not None
        assert METHODS_DICT["isomap"].default_graph_params["nn"] == 15

        # UMAP should have min_dist default
        assert METHODS_DICT["umap"].default_params["min_dist"] == 0.1


class TestMergeParamsWithDefaults:
    """Test the parameter merging function."""

    def test_merge_with_no_user_params(self):
        """Test merging with no user parameters."""
        params = merge_params_with_defaults("pca")

        assert params["e_params"]["dim"] == 2
        assert params["e_params"]["e_method_name"] == "pca"
        assert params["e_params"]["e_method"] == METHODS_DICT["pca"]
        assert params["g_params"] is None
        assert params["m_params"] is None

    def test_merge_with_flat_user_params(self):
        """Test merging with flat user parameters."""
        params = merge_params_with_defaults("pca", {"dim": 3})
        assert params["e_params"]["dim"] == 3

        params = merge_params_with_defaults(
            "umap", {"n_neighbors": 30, "min_dist": 0.3}
        )
        assert params["g_params"]["nn"] == 30
        assert params["e_params"]["min_dist"] == 0.3

    def test_merge_with_structured_params(self):
        """Test merging with structured parameters (legacy format)."""
        user_params = {"e_params": {"dim": 5}, "g_params": {"nn": 50}}
        params = merge_params_with_defaults("isomap", user_params)

        assert params["e_params"]["dim"] == 5
        assert params["g_params"]["nn"] == 50
        # Check that other defaults are preserved
        assert params["g_params"]["g_method_name"] == "knn"

    def test_invalid_method_raises_error(self):
        """Test that invalid method name raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            merge_params_with_defaults("invalid_method")


class TestMVDataSimplifiedAPI:
    """Test MVData's simplified get_embedding API."""

    @pytest.fixture
    def sample_mvdata(self):
        """Generate sample MVData for testing."""
        np.random.seed(42)
        n_features, n_samples = 50, 100
        data = np.random.randn(n_features, n_samples)
        return MVData(data)

    def test_simplified_api_pca(self, sample_mvdata):
        """Test simplified API with PCA."""
        # Test with defaults
        emb = sample_mvdata.get_embedding(method="pca")
        assert emb.coords.shape == (2, 100)

        # Test with custom dimension
        emb = sample_mvdata.get_embedding(method="pca", dim=3)
        assert emb.coords.shape == (3, 100)

    def test_simplified_api_graph_method(self, sample_mvdata):
        """Test simplified API with graph-based method."""
        # Test Isomap
        emb = sample_mvdata.get_embedding(method="isomap", n_neighbors=10)
        assert emb.coords.shape[0] == 2  # Default dim
        assert hasattr(emb, "graph")
        assert emb.graph.nn == 10

    def test_simplified_api_tsne(self, sample_mvdata):
        """Test simplified API with t-SNE."""
        # Test with custom perplexity and reduced iterations
        emb = sample_mvdata.get_embedding(method="tsne", perplexity=15, n_iter=250)
        assert emb.coords.shape[0] == 2

    def test_simplified_api_mds(self, sample_mvdata):
        """Test simplified API with MDS."""
        # MDS requires distance matrix
        sample_mvdata.get_distmat()
        emb = sample_mvdata.get_embedding(method="mds")
        assert emb.coords.shape == (2, 100)

    def test_backward_compatibility(self, sample_mvdata):
        """Test that legacy API still works."""
        # Legacy format
        e_params = {"e_method_name": "pca", "e_method": METHODS_DICT["pca"], "dim": 3}
        emb = sample_mvdata.get_embedding(e_params)
        assert emb.coords.shape == (3, 100)

    def test_error_handling(self, sample_mvdata):
        """Test error handling in simplified API."""
        # No method or e_params provided
        with pytest.raises(ValueError, match="Either 'method' or 'e_params'"):
            sample_mvdata.get_embedding()

        # Invalid method
        with pytest.raises(ValueError, match="Unknown method"):
            sample_mvdata.get_embedding(method="invalid_method")


class TestParameterPropagation:
    """Test that parameters are correctly propagated to methods."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        return np.random.randn(20, 50)

    def test_umap_parameters(self, sample_data):
        """Test UMAP-specific parameters."""
        mvdata = MVData(sample_data)

        # Test min_dist parameter
        emb = mvdata.get_embedding(method="umap", min_dist=0.5, n_neighbors=10)
        assert emb.e_method_name == "umap"
        assert emb.min_dist == 0.5

    def test_graph_parameters(self, sample_data):
        """Test graph construction parameters."""
        mvdata = MVData(sample_data)

        # Test with custom neighbors for Isomap
        emb = mvdata.get_embedding(method="isomap", n_neighbors=20)
        assert emb.graph.nn == 20


# Additional tests for uncovered get_embedding functionality


def test_get_embedding_graph_required_no_params():
    """Test error when graph method lacks graph parameters."""
    mvdata = MVData(np.random.rand(5, 20))

    # Mock a method that requires graph but no params provided
    from unittest.mock import MagicMock

    e_params = {
        "e_method_name": "le",
        "e_method": MagicMock(requires_graph=True, requires_distmat=False),
        "dim": 2,
    }

    with pytest.raises(ValueError, match="requires proximity graph"):
        mvdata.get_embedding(e_params=e_params)


def test_get_embedding_weighted_graph_no_metric():
    """Test error when weighted graph needs metric params."""
    mvdata = MVData(np.random.rand(5, 20))

    from unittest.mock import MagicMock

    e_params = {
        "e_method_name": "le",
        "e_method": MagicMock(requires_graph=True, requires_distmat=False),
        "dim": 2,
    }
    g_params = {"weighted": True, "g_method_name": "knn"}

    with pytest.raises(ValueError, match="requires weights for proximity graph"):
        mvdata.get_embedding(e_params=e_params, g_params=g_params)


def test_get_embedding_legacy_method_lookup():
    """Test legacy e_method lookup from METHODS_DICT."""
    mvdata = MVData(np.random.rand(5, 20))

    # e_method is None, should be looked up
    e_params = {"e_method_name": "pca", "e_method": None, "dim": 2}

    emb = mvdata.get_embedding(e_params=e_params)
    assert emb.coords.shape == (2, 20)


def test_get_embedding_nn_kwargs_extraction():
    """Test NN method kwargs extraction."""
    mvdata = MVData(np.random.rand(5, 30))

    # Test with autoencoder and verify params are passed
    emb = mvdata.get_embedding(
        method="ae", dim=2, epochs=5, lr=0.001, batch_size=10, verbose=False, seed=42
    )

    # Should complete without error
    assert emb.coords.shape == (2, 30)
