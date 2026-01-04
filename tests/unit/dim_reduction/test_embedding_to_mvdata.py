"""
Test suite for Embedding.to_mvdata() conversion functionality.
"""

import numpy as np
import pytest
from driada.dim_reduction import MVData


class TestEmbeddingToMVData:
    """Test conversion of Embedding to MVData."""

    @pytest.fixture
    def sample_data(self):
        """Create sample high-dimensional data."""
        np.random.seed(42)
        # 20 features, 100 samples
        return np.random.randn(20, 100)

    @pytest.fixture
    def sample_labels(self):
        """Create sample labels."""
        return np.array([0] * 50 + [1] * 50)

    @pytest.fixture
    def mvdata(self, sample_data, sample_labels):
        """Create MVData instance."""
        return MVData(sample_data, labels=sample_labels)

    def test_to_mvdata_basic_conversion(self, mvdata):
        """Test basic conversion from Embedding to MVData."""
        # Create embedding
        embedding = mvdata.get_embedding(method="pca", dim=3)

        # Convert to MVData
        embedding_mvdata = embedding.to_mvdata()

        # Check type
        assert isinstance(embedding_mvdata, MVData)

        # Check dimensions
        assert embedding_mvdata.n_dim == 3  # PCA dim
        assert embedding_mvdata.n_points == mvdata.n_points

        # Check data matches embedding coordinates
        np.testing.assert_array_equal(embedding_mvdata.data, embedding.coords)

        # Check labels are preserved
        np.testing.assert_array_equal(embedding_mvdata.labels, mvdata.labels)

        # Check data name
        assert embedding_mvdata.data_name == "pca_embedding"

    def test_to_mvdata_validates_coords(self, mvdata):
        """Test that to_mvdata validates that coords exist."""
        # Get a valid embedding
        embedding = mvdata.get_embedding(method="pca", dim=3)

        # Manually set coords to None to test validation
        original_coords = embedding.coords
        embedding.coords = None

        # Should raise error
        with pytest.raises(ValueError, match="Embedding has not been built yet"):
            embedding.to_mvdata()

        # Restore coords for other tests
        embedding.coords = original_coords

    def test_recursive_embedding_pipeline(self, mvdata):
        """Test recursive embedding: PCA -> UMAP."""
        # First reduction: 20D -> 10D with PCA
        pca_embedding = mvdata.get_embedding(method="pca", dim=10)

        # Convert to MVData
        pca_mvdata = pca_embedding.to_mvdata()

        # Second reduction: 10D -> 2D with UMAP
        umap_embedding = pca_mvdata.get_embedding(method="umap", dim=2, n_neighbors=15)

        # Check final dimensions
        assert umap_embedding.coords.shape == (2, mvdata.n_points)

        # Labels should be preserved through the pipeline
        np.testing.assert_array_equal(umap_embedding.labels, mvdata.labels)

    def test_different_embedding_methods(self, mvdata):
        """Test to_mvdata with different embedding methods."""
        methods = [
            ("pca", {"dim": 3}),
            ("isomap", {"dim": 2, "n_neighbors": 10}),
            ("umap", {"dim": 2, "n_neighbors": 15}),
            ("tsne", {"dim": 2}),
        ]

        for method_name, params in methods:
            # Skip if method requires additional setup
            try:
                embedding = mvdata.get_embedding(method=method_name, **params)
                embedding_mvdata = embedding.to_mvdata()

                # Check basic properties
                assert embedding_mvdata.n_dim == params["dim"]
                assert embedding_mvdata.data_name == f"{method_name}_embedding"
                
                # Labels may be filtered if graph preprocessing removed nodes
                # So we just check that labels have the correct length
                assert len(embedding_mvdata.labels) == embedding_mvdata.n_points
                # And that they maintain the same unique values
                assert set(np.unique(embedding_mvdata.labels)) <= set(np.unique(mvdata.labels))
            except Exception as e:
                # Some methods might fail due to missing dependencies
                if "No module named" in str(e):
                    pytest.skip(f"Skipping {method_name} due to missing dependency")
                else:
                    raise

    def test_to_mvdata_properties(self, mvdata):
        """Test specific properties of converted MVData."""
        embedding = mvdata.get_embedding(method="pca", dim=5)
        embedding_mvdata = embedding.to_mvdata()

        # Check that rescale_rows is False (embeddings already scaled)
        assert embedding_mvdata.rescale_rows is False

        # Check that distmat is None (needs recomputation)
        assert embedding_mvdata.distmat is None

        # Check that downsampling is preserved
        assert embedding_mvdata.ds == 1

    def test_to_mvdata_with_neural_methods(self, mvdata):
        """Test to_mvdata with neural network methods if torch available."""
        try:
            import torch

            # Test with autoencoder
            ae_embedding = mvdata.get_embedding(
                method="ae", dim=3, epochs=5, verbose=False  # Quick test
            )
            ae_mvdata = ae_embedding.to_mvdata()

            assert ae_mvdata.n_dim == 3
            assert ae_mvdata.data_name == "ae_embedding"

        except ImportError:
            pytest.skip("PyTorch not available, skipping neural methods test")

    def test_mvdata_methods_on_converted(self, mvdata):
        """Test that MVData methods work on converted embedding."""
        # Create embedding and convert
        embedding = mvdata.get_embedding(method="pca", dim=5)
        embedding_mvdata = embedding.to_mvdata()

        # Test correlation matrix
        corr_mat = embedding_mvdata.corr_mat()
        assert corr_mat.shape == (5, 5)

        # Test distance matrix computation
        distmat = embedding_mvdata.get_distmat()
        assert distmat.shape == (mvdata.n_points, mvdata.n_points)

        # Test that we can create another embedding
        second_embedding = embedding_mvdata.get_embedding(method="pca", dim=2)
        assert second_embedding.coords.shape == (2, mvdata.n_points)
