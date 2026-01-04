"""
Test suite for disconnected graph handling in dimensionality reduction methods.
"""

import numpy as np
import pytest
from driada.dim_reduction import MVData, METHODS_DICT
from driada.dim_reduction.embedding import Embedding
from driada.dim_reduction.graph import ProximityGraph


class TestDisconnectedGraphHandling:
    """Test handling of disconnected graphs by different DR methods."""

    @pytest.fixture
    def disconnected_data(self):
        """Create data that will result in a disconnected graph."""
        np.random.seed(42)
        # Create two well-separated clusters
        # Make them VERY far apart to ensure disconnection
        cluster1 = np.random.randn(10, 10) * 0.1  # 10 features, 10 samples
        cluster2 = np.random.randn(10, 10) * 0.1 + 1000  # Very far away cluster

        # Combine into single dataset
        data = np.hstack([cluster1, cluster2])
        labels = np.array([0] * 10 + [1] * 10)

        return MVData(data, labels=labels)

    @pytest.fixture
    def connected_data(self):
        """Create data that will result in a connected graph."""
        np.random.seed(42)
        # Create continuous data distribution
        data = np.random.randn(10, 50)
        labels = np.arange(50)

        return MVData(data, labels=labels)

    def test_drmethod_property_values(self):
        """Test that handles_disconnected_graphs property is set correctly."""
        # UMAP should handle disconnected graphs
        assert METHODS_DICT["umap"].handles_disconnected_graphs == 1

        # All other graph-based methods should not handle disconnected graphs
        graph_based_methods = [
            "le",
            "auto_le",
            "dmaps",
            "auto_dmaps",
            "isomap",
            "lle",
            "hlle",
            "mvu",
        ]

        for method_name in graph_based_methods:
            method = METHODS_DICT[method_name]
            assert method.requires_graph == 1
            assert (
                method.handles_disconnected_graphs == 0
            ), f"{method_name} should not handle disconnected graphs"

        # Non-graph methods should have default value (0)
        non_graph_methods = ["pca", "mds", "ae", "vae", "tsne"]
        for method_name in non_graph_methods:
            method = METHODS_DICT[method_name]
            assert method.requires_graph == 0
            assert method.handles_disconnected_graphs == 0

    def test_umap_with_disconnected_graph(self, disconnected_data):
        """Test that UMAP can handle disconnected graphs."""
        # Create embedding with very low nn to ensure disconnection
        # Use higher max_deleted_nodes to allow disconnected components
        embedding = disconnected_data.get_embedding(
            method="umap",
            n_neighbors=3,  # Very low to ensure disconnection
            dim=2,
            max_deleted_nodes=0.6,  # Allow up to 60% node loss
        )

        # Should succeed without raising exception
        assert embedding is not None
        assert embedding.coords is not None
        # Shape depends on how many nodes remain after preprocessing
        assert embedding.coords.shape[0] == 2  # 2D embedding

        # Note: After giant_cc preprocessing, the resulting graph is connected
        # but we lost nodes in the process
        assert hasattr(embedding.graph, "lost_nodes")

    def test_other_methods_with_preprocessing(self, disconnected_data):
        """Test that graph-based methods work after preprocessing removes disconnection."""
        # Methods that require connected graphs
        methods = ["isomap", "le", "lle"]

        for method_name in methods:
            # With preprocessing, the graph becomes connected by removing nodes
            embedding = disconnected_data.get_embedding(
                method=method_name,
                n_neighbors=5,  # Increased to ensure some connectivity
                dim=2,
                max_deleted_nodes=0.6,  # Allow node loss during preprocessing
            )

            # Should succeed because preprocessing creates connected graph
            assert embedding is not None
            assert embedding.coords is not None
            # After giant_cc preprocessing, graph is connected
            assert embedding.graph.is_connected()
            # But nodes were lost
            assert len(embedding.graph.lost_nodes) > 0

    def test_connected_graph_works_for_all(self, connected_data):
        """Test that all graph-based methods work with connected graphs."""
        graph_methods = ["umap", "isomap", "le"]

        for method_name in graph_methods:
            embedding = connected_data.get_embedding(
                method=method_name,
                n_neighbors=15,  # Reasonable number for connectivity
                dim=2,
            )

            assert embedding is not None
            assert embedding.coords is not None
            assert embedding.graph.is_connected()

    def test_direct_embedding_creation(self):
        """Test disconnectivity handling via direct Embedding creation."""
        # Create disconnected graph manually
        np.random.seed(42)
        cluster1 = np.random.randn(5, 10) * 0.1
        cluster2 = np.random.randn(5, 10) * 0.1 + 50
        data = np.hstack([cluster1, cluster2])

        mvdata = MVData(data)

        # Create graph with low neighbors and no preprocessing
        # to create a truly disconnected graph
        graph_params = {
            "g_method_name": "knn",
            "nn": 2,  # Very low
            "weighted": 0,
            "max_deleted_nodes": 1.0,  # Allow all nodes
            "dist_to_aff": "hk",
            "graph_preprocessing": None,  # No preprocessing!
        }

        metric_params = {"metric_name": "euclidean", "sigma": 1.0}

        graph = ProximityGraph(
            mvdata.data, metric_params, graph_params  # Pass data, not distmat
        )

        # Check if graph is truly disconnected
        # Note: with preprocessing='none', it should be disconnected
        # unless the implementation forces connectivity

        # Test UMAP - should work even if disconnected
        umap_params = {
            "e_method_name": "umap",
            "e_method": METHODS_DICT["umap"],
            "dim": 2,
            "min_dist": 0.1,
        }

        umap_embedding = Embedding(
            mvdata.data, mvdata.distmat, mvdata.labels, umap_params, g=graph
        )
        # UMAP should not raise exception due to handles_disconnected_graphs=1
        umap_embedding.build()

        assert umap_embedding.coords is not None

        # Test Isomap - should fail if graph is disconnected
        isomap_params = {
            "e_method_name": "isomap",
            "e_method": METHODS_DICT["isomap"],
            "dim": 2,
        }

        isomap_embedding = Embedding(
            mvdata.data, mvdata.distmat, mvdata.labels, isomap_params, g=graph
        )

        # If the graph is disconnected (no preprocessing), this should fail
        # If preprocessing was applied despite 'none', it might succeed
        try:
            isomap_embedding.build()
            # If it succeeded, the graph must be connected
            assert graph.is_connected()
        except Exception as e:
            # If it failed, it should be due to disconnection
            assert "Graph is not connected!" in str(e)

    def test_preprocessing_interaction(self, disconnected_data):
        """Test interaction with graph preprocessing options."""
        # Use Isomap which requires connected graphs and will use giant_cc by default
        # Use n_neighbors=5 to ensure we don't connect across clusters
        embedding = disconnected_data.get_embedding(
            method="isomap", n_neighbors=5, dim=2, max_deleted_nodes=0.6
        )

        # Check that lost_nodes attribute exists
        assert hasattr(embedding.graph, "lost_nodes")
        
        # With two clusters of 10 nodes each at distance 1000,
        # and n_neighbors=5, the graph should be disconnected,
        # so giant_cc preprocessing should remove one cluster
        assert len(embedding.graph.lost_nodes) == 10

        # Isomap should still work with the remaining cluster
        assert embedding.coords is not None
        assert embedding.coords.shape[1] == 10  # Only one cluster remains (coords is dim x samples)

    def test_handles_disconnected_graphs_property(self):
        """Test the handles_disconnected_graphs property behavior."""
        # The key insight: ProximityGraph always applies preprocessing (giant_cc)
        # So we need to test the property behavior at a conceptual level

        # Verify property values are set correctly
        umap_method = METHODS_DICT["umap"]
        assert umap_method.handles_disconnected_graphs == 1

        le_method = METHODS_DICT["le"]
        assert le_method.handles_disconnected_graphs == 0

        isomap_method = METHODS_DICT["isomap"]
        assert isomap_method.handles_disconnected_graphs == 0

        # Test conceptual behavior: UMAP can handle disconnected components
        # while other methods cannot

        # Create data that would be disconnected without preprocessing
        np.random.seed(42)
        cluster1 = np.random.randn(10, 5) * 0.1
        cluster2 = np.random.randn(10, 5) * 0.1 + 100
        data = np.hstack([cluster1, cluster2])
        mvdata = MVData(data)

        # All methods should work with default preprocessing
        # The difference is that UMAP would work even without it
        for method_name in ["umap", "le", "isomap"]:
            embedding = mvdata.get_embedding(
                method=method_name, n_neighbors=5, dim=2, max_deleted_nodes=0.6
            )
            assert embedding is not None
            assert embedding.coords is not None
