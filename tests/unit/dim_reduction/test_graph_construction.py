"""Tests for graph construction methods in ProximityGraph"""

import pytest
import numpy as np
import scipy.sparse as sp
from driada.dim_reduction.graph import ProximityGraph


class TestGraphConstruction:
    """Test various graph construction methods"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for graph construction"""
        np.random.seed(42)
        # 3D data with 50 points
        return np.random.randn(3, 50)

    @pytest.fixture
    def small_data(self):
        """Create small data for testing edge cases"""
        np.random.seed(42)
        # 2D data with 10 points
        return np.random.randn(2, 10)

    def test_knn_graph_construction(self, sample_data):
        """Test standard k-nearest neighbors graph construction"""
        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "knn",
            "nn": 5,
            "weighted": False,
            "dist_to_aff": None,
            "max_deleted_nodes": 0.5,
        }

        graph = ProximityGraph(sample_data, m_params, g_params, create_nx_graph=False)

        # Check basic properties
        assert graph.adj is not None
        assert graph.bin_adj is not None
        assert graph.neigh_distmat is not None
        assert graph.adj.shape == (graph.n, graph.n)
        assert graph.adj.nnz > 0

        # Check that each node has at most nn neighbors (bidirectional)
        degrees = np.array(graph.adj.sum(axis=1)).flatten()
        assert np.all(degrees <= 2 * g_params["nn"])

    def test_knn_graph_weighted(self, sample_data):
        """Test weighted k-nearest neighbors graph with affinities"""
        m_params = {"metric_name": "euclidean", "sigma": 2.0}
        g_params = {
            "g_method_name": "knn",
            "nn": 5,
            "weighted": True,
            "dist_to_aff": "hk",  # Heat kernel affinities
            "max_deleted_nodes": 0.5,
        }

        graph = ProximityGraph(sample_data, m_params, g_params, create_nx_graph=False)

        # Check weighted graph properties
        assert graph.adj is not None
        assert graph.neigh_distmat is not None

        # Affinities should be between 0 and 1
        assert np.all(graph.adj.data > 0)
        assert np.all(graph.adj.data <= 1)

        # Check symmetry
        assert np.allclose(graph.adj.toarray(), graph.adj.toarray().T)

    def test_knn_graph_different_metrics(self, sample_data):
        """Test knn graph with different distance metrics"""
        metrics = ["manhattan", "cosine", "euclidean"]

        for metric in metrics:
            m_params = {"metric_name": metric, "sigma": 1.0}
            g_params = {
                "g_method_name": "knn",
                "nn": 5,  # Increased from 3 to ensure better connectivity
                "weighted": False,
                "dist_to_aff": None,
                "max_deleted_nodes": 0.5,
            }

            graph = ProximityGraph(
                sample_data, m_params, g_params, create_nx_graph=False
            )
            assert graph.adj is not None
            assert graph.adj.nnz > 0

    def test_auto_knn_graph_construction(self, sample_data):
        """Test auto k-nearest neighbors graph construction using sklearn"""
        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "auto_knn",
            "nn": 5,
            "weighted": False,
            "dist_to_aff": None,
            "max_deleted_nodes": 0.5,
        }

        graph = ProximityGraph(sample_data, m_params, g_params, create_nx_graph=False)

        # Check basic properties
        assert graph.adj is not None
        assert graph.adj.shape == (graph.n, graph.n)
        assert graph.adj.nnz > 0

        # Should be symmetric
        assert np.allclose(graph.adj.toarray(), graph.adj.toarray().T)

    def test_umap_graph_construction(self, small_data):
        """Test UMAP graph construction"""
        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "umap",
            "nn": 3,
            "weighted": True,  # UMAP graphs are inherently weighted
            "dist_to_aff": None,
            "max_deleted_nodes": 0.5,
        }

        graph = ProximityGraph(small_data, m_params, g_params, create_nx_graph=False)

        # Check basic properties
        assert graph.adj is not None
        assert graph.adj.shape == (graph.n, graph.n)
        assert graph.adj.nnz > 0

        # UMAP produces weighted graph
        assert not np.all(np.isin(graph.adj.data, [0, 1]))

    def test_graph_with_manhattan_metric(self, sample_data):
        """Test graph construction with Manhattan (L1) metric"""
        # Use manhattan metric which is supported
        m_params = {"metric_name": "manhattan", "sigma": 1.0}
        g_params = {
            "g_method_name": "knn",
            "nn": 5,
            "weighted": False,
            "dist_to_aff": None,
            "max_deleted_nodes": 0.5,
        }

        graph = ProximityGraph(sample_data, m_params, g_params, create_nx_graph=False)

        # Check basic properties
        assert graph.adj is not None
        assert graph.adj.shape == (graph.n, graph.n)
        assert graph.adj.nnz > 0

        # Should be using manhattan metric
        assert graph.metric == "manhattan"

    def test_graph_with_minkowski_metric(self, sample_data):
        """Test graph construction with minkowski metric and p parameter"""
        m_params = {"metric_name": "minkowski", "p": 3, "sigma": 1.0}
        g_params = {
            "g_method_name": "knn",
            "nn": 5,
            "weighted": False,
            "dist_to_aff": None,
            "max_deleted_nodes": 0.5,
        }

        graph = ProximityGraph(sample_data, m_params, g_params, create_nx_graph=False)

        # Check basic properties
        assert graph.adj is not None
        assert graph.adj.nnz > 0

    def test_graph_giant_component_extraction(self, sample_data):
        """Test that giant component extraction works properly"""
        # Create disconnected data by spreading points far apart
        disconnected_data = sample_data.copy()
        # Move half the points far away
        disconnected_data[:, 25:] += 100

        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "knn",
            "nn": 5,  # Increased from 3 to ensure better connectivity within clusters
            "weighted": False,
            "dist_to_aff": None,
            "max_deleted_nodes": 0.6,  # Allow up to 60% node loss
        }
        # Note: nn=3 was too small after fixing the graph symmetrization to use
        # minimum instead of addition, which creates a properly symmetric but
        # potentially sparser graph

        graph = ProximityGraph(
            disconnected_data, m_params, g_params, create_nx_graph=False
        )

        # Should have lost some nodes
        assert graph.n < disconnected_data.shape[1]
        # Should have reported lost nodes
        assert hasattr(graph, "lost_nodes")
        assert len(graph.lost_nodes) > 0

    def test_graph_too_many_nodes_lost_error(self, sample_data):
        """Test error when too many nodes are lost during giant component extraction"""
        # Create highly disconnected data
        disconnected_data = sample_data.copy()
        # Spread all points far apart
        for i in range(disconnected_data.shape[1]):
            disconnected_data[:, i] += i * 100

        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "knn",
            "nn": 1,  # Very small nn
            "weighted": False,
            "dist_to_aff": None,
            "max_deleted_nodes": 0.2,  # Only allow 20% node loss
        }

        with pytest.raises(Exception, match="more than.*% of nodes discarded"):
            ProximityGraph(disconnected_data, m_params, g_params, create_nx_graph=False)

    def test_checkpoint_validation(self, sample_data):
        """Test _checkpoint method validation"""
        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "knn",
            "nn": 5,
            "weighted": False,
            "dist_to_aff": None,
            "max_deleted_nodes": 0.5,
        }

        graph = ProximityGraph(sample_data, m_params, g_params, create_nx_graph=False)

        # Checkpoint should have been called during initialization
        # Verify matrices are symmetric
        assert np.allclose(graph.adj.toarray(), graph.adj.toarray().T)
        assert np.allclose(graph.bin_adj.toarray(), graph.bin_adj.toarray().T)
        assert np.allclose(
            graph.neigh_distmat.toarray(), graph.neigh_distmat.toarray().T
        )

    def test_checkpoint_with_no_adjacency_error(self, sample_data):
        """Test _checkpoint raises error when adjacency is None"""
        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "knn",
            "nn": 5,
            "weighted": False,
            "dist_to_aff": None,
            "max_deleted_nodes": 0.5,
        }

        graph = ProximityGraph(sample_data, m_params, g_params, create_nx_graph=False)

        # Manually set adj to None and try checkpoint
        graph.adj = None
        with pytest.raises(Exception, match="Adjacency matrix is not constructed"):
            graph._checkpoint()

    def test_checkpoint_with_dense_matrix_error(self, sample_data):
        """Test _checkpoint raises error for non-sparse adjacency"""
        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "knn",
            "nn": 5,
            "weighted": False,
            "dist_to_aff": None,
            "max_deleted_nodes": 0.5,
        }

        graph = ProximityGraph(sample_data, m_params, g_params, create_nx_graph=False)

        # Convert to dense matrix
        graph.adj = graph.adj.toarray()
        with pytest.raises(Exception, match="Adjacency matrix is not sparse"):
            graph._checkpoint()

    def test_distances_to_affinities_hk(self, sample_data):
        """Test heat kernel affinity transformation"""
        m_params = {"metric_name": "euclidean", "sigma": 2.0}
        g_params = {
            "g_method_name": "knn",
            "nn": 5,
            "weighted": True,
            "dist_to_aff": "hk",
            "max_deleted_nodes": 0.5,
        }

        # Create graph (distances_to_affinities is called internally for weighted graphs)
        graph = ProximityGraph(sample_data, m_params, g_params, create_nx_graph=False)

        # Check that affinities are computed correctly
        assert graph.adj is not None
        # All affinities should be positive and <= 1
        assert np.all(graph.adj.data > 0)
        assert np.all(graph.adj.data <= 1)

        # Should be symmetric
        assert np.allclose(graph.adj.toarray(), graph.adj.toarray().T)

    def test_distances_to_affinities_error_conditions(self, sample_data):
        """Test error conditions in distances_to_affinities"""
        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "knn",
            "nn": 5,
            "weighted": False,  # Not weighted
            "dist_to_aff": "hk",
            "max_deleted_nodes": 0.5,
        }

        graph = ProximityGraph(sample_data, m_params, g_params, create_nx_graph=False)

        # Test error when neigh_distmat is None
        graph.neigh_distmat = None
        with pytest.raises(
            Exception, match="distances between nearest neighbors not available"
        ):
            graph.distances_to_affinities()

        # Test error when graph is not weighted
        graph.neigh_distmat = sp.csr_matrix(graph.adj.shape)  # Restore it
        graph.weighted = False
        with pytest.raises(
            Exception, match="no need to construct affinities for binary graph"
        ):
            graph.distances_to_affinities()


class TestIntrinsicDimension:
    """Test intrinsic dimension estimation methods in ProximityGraph"""

    @pytest.fixture
    def swiss_roll_data(self):
        """Create Swiss roll data for testing dimension estimation"""
        from sklearn.datasets import make_swiss_roll

        np.random.seed(42)
        data, _ = make_swiss_roll(n_samples=500, noise=0.05)
        return data.T  # ProximityGraph expects (features, samples)

    @pytest.fixture
    def linear_subspace_data(self):
        """Create data in a 2D linear subspace of 5D ambient space"""
        np.random.seed(42)
        n_samples = 200
        # Create orthonormal basis for 2D subspace
        basis = np.random.randn(5, 2)
        basis = np.linalg.qr(basis)[0]
        # Generate 2D coefficients and embed in 5D
        coeffs = np.random.randn(n_samples, 2)
        data = coeffs @ basis.T
        # Add small noise
        data += 1e-4 * np.random.randn(*data.shape)
        return data.T  # (features, samples)

    def test_get_int_dim_geodesic_method(self, swiss_roll_data):
        """Test geodesic dimension estimation method"""
        # Import for direct comparison
        from driada.dimensionality import geodesic_dimension
        
        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "knn",
            "nn": 30,  # Need larger k for geodesic dimension to work well
            "weighted": False,  # Use unweighted graph for consistent geodesic dimension
            "max_deleted_nodes": 0.5,
            "graph_preprocessing": "giant_cc",  # Use giant connected component
        }

        graph = ProximityGraph(
            swiss_roll_data, m_params, g_params, create_nx_graph=False
        )

        # Test with ProximityGraph
        dim = graph.get_int_dim(method="geodesic")
        assert isinstance(dim, float)
        
        # Compare with direct geodesic_dimension on the transposed data
        # swiss_roll_data is (features, samples) so transpose back to (samples, features)
        dim_direct = geodesic_dimension(swiss_roll_data.T, k=30)
        
        # Debug the difference
        print(f"ProximityGraph dimension: {dim:.3f}")
        print(f"Direct geodesic dimension: {dim_direct:.3f}")
        print(f"Lost nodes: {len(graph.lost_nodes)}")
        
        # They should match closely if no nodes were lost
        if len(graph.lost_nodes) == 0:
            assert abs(dim - dim_direct) < 0.5, f"ProximityGraph {dim} vs direct {dim_direct}"
        
        assert 1.8 < dim < 2.5, f"Expected dimension ~2 for Swiss roll, got {dim}"

        # Test caching
        dim_cached = graph.get_int_dim(method="geodesic")
        assert dim == dim_cached
        assert "geodesic_full_f2" in graph.intrinsic_dimensions

    def test_get_int_dim_nn_method(self, linear_subspace_data):
        """Test nn dimension estimation method with k-NN graph"""
        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "knn",
            "nn": 10,
            "weighted": False,
            "dist_to_aff": None,
            "max_deleted_nodes": 0.5,
        }

        graph = ProximityGraph(
            linear_subspace_data, m_params, g_params, create_nx_graph=False
        )

        # Test nn method
        dim = graph.get_int_dim(method="nn")
        assert isinstance(dim, float)
        assert 1.5 < dim < 2.5, f"Expected dimension ~2 for 2D subspace, got {dim}"

        # Check that k-NN data was saved
        assert hasattr(graph, "knn_indices")
        assert hasattr(graph, "knn_distances")
        assert graph.knn_indices is not None
        assert graph.knn_distances is not None

    def test_get_int_dim_force_recompute(self, swiss_roll_data):
        """Test force_recompute flag"""
        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "knn",
            "nn": 15,
            "weighted": True,
            "dist_to_aff": "hk",
            "max_deleted_nodes": 0.5,
        }

        graph = ProximityGraph(
            swiss_roll_data, m_params, g_params, create_nx_graph=False
        )

        # First computation
        dim1 = graph.get_int_dim(method="geodesic")

        # Check it's cached
        assert "geodesic_full_f2" in graph.intrinsic_dimensions

        # Force recomputation
        dim2 = graph.get_int_dim(method="geodesic", force_recompute=True)

        # Should be the same value (same data and method)
        assert abs(dim1 - dim2) < 1e-10

    def test_get_int_dim_fast_mode(self, swiss_roll_data):
        """Test geodesic method with fast mode"""
        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "knn",
            "nn": 15,
            "weighted": True,
            "dist_to_aff": "hk",
            "max_deleted_nodes": 0.5,
        }

        graph = ProximityGraph(
            swiss_roll_data, m_params, g_params, create_nx_graph=False
        )

        # Test fast mode
        np.random.seed(42)  # For reproducibility of subsampling
        dim_fast = graph.get_int_dim(method="geodesic", mode="fast", factor=4)
        assert isinstance(dim_fast, float)
        # Weighted graphs with heat kernel can give higher estimates
        assert 1.5 < dim_fast < 3.5, f"Fast mode dimension out of range: {dim_fast}"

        # Check different cache key
        assert "geodesic_fast_f4" in graph.intrinsic_dimensions

    def test_get_int_dim_nn_method_not_available(self, swiss_roll_data):
        """Test error when nn method is used with incompatible graph type"""
        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "auto_knn",  # This method doesn't save k-NN data
            "nn": 15,
            "weighted": False,
            "dist_to_aff": None,
            "max_deleted_nodes": 0.5,
        }

        graph = ProximityGraph(
            swiss_roll_data, m_params, g_params, create_nx_graph=False
        )

        # Should raise error for nn method
        with pytest.raises(ValueError, match="nn method requires k-NN graph data"):
            graph.get_int_dim(method="nn")

    def test_get_int_dim_invalid_method(self, swiss_roll_data):
        """Test error for invalid method"""
        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "knn",
            "nn": 15,
            "weighted": True,
            "dist_to_aff": "hk",
            "max_deleted_nodes": 0.5,
        }

        graph = ProximityGraph(
            swiss_roll_data, m_params, g_params, create_nx_graph=False
        )

        with pytest.raises(ValueError, match="Unknown method"):
            graph.get_int_dim(method="invalid")

    def test_get_int_dim_multiple_methods(self, linear_subspace_data):
        """Test using multiple methods and check cache"""
        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "knn",
            "nn": 10,
            "weighted": True,
            "dist_to_aff": "hk",
            "max_deleted_nodes": 0.5,
        }

        graph = ProximityGraph(
            linear_subspace_data, m_params, g_params, create_nx_graph=False
        )

        # Compute with different methods
        dim_geo = graph.get_int_dim(method="geodesic")
        dim_nn = graph.get_int_dim(method="nn")

        # Both should give reasonable estimates for 2D data
        assert 1.5 < dim_geo < 2.5
        assert 1.5 < dim_nn < 2.5

        # Check cache contains both
        assert "geodesic_full_f2" in graph.intrinsic_dimensions
        assert "nn" in graph.intrinsic_dimensions
        assert len(graph.intrinsic_dimensions) == 2

    def test_get_int_dim_with_lost_nodes(self):
        """Test dimension estimation when some nodes are lost in giant component"""
        # Create disconnected data
        np.random.seed(42)
        cluster1 = np.random.randn(3, 25)
        cluster2 = np.random.randn(3, 25) + 100  # Far away
        data = np.hstack([cluster1, cluster2])

        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "knn",
            "nn": 5,  # Increased to ensure better connectivity
            "weighted": True,
            "dist_to_aff": "hk",
            "max_deleted_nodes": 0.6,
        }

        graph = ProximityGraph(data, m_params, g_params, create_nx_graph=False)

        # Should have lost some nodes
        assert hasattr(graph, "lost_nodes")
        assert len(graph.lost_nodes) > 0

        # Dimension estimation should still work
        dim = graph.get_int_dim(method="geodesic")
        assert isinstance(dim, float)
        assert 0 < dim < 10

        # Check k-NN arrays were properly filtered
        if hasattr(graph, "knn_indices") and graph.knn_indices is not None:
            assert len(graph.knn_indices) == graph.n


class TestEpsilonGraph:
    """Test epsilon-ball graph construction"""

    @pytest.fixture
    def clustered_data(self):
        """Create clustered data for epsilon graph testing"""
        np.random.seed(42)
        # Create 3 well-separated clusters
        cluster1 = np.random.randn(20, 3) * 0.5
        cluster2 = np.random.randn(20, 3) * 0.5 + np.array([5, 0, 0])
        cluster3 = np.random.randn(20, 3) * 0.5 + np.array([0, 5, 0])
        # Note: hstack horizontally concatenates, we want vstack for samples
        return np.vstack([cluster1, cluster2, cluster3])

    def test_eps_graph_construction(self, clustered_data):
        """Test basic epsilon-ball graph construction"""
        # Debug: check data shape
        assert clustered_data.shape == (60, 3), f"Wrong shape: {clustered_data.shape}"

        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "eps",
            "eps": 6.0,  # Larger radius to connect clusters (distance between clusters is ~5)
            "min_density": 0.001,  # Lower minimum density requirement
            "weighted": False,
            "dist_to_aff": None,
            "max_deleted_nodes": 0.5,
        }

        # ProximityGraph expects data in (n_features, n_samples) format
        graph = ProximityGraph(
            clustered_data.T, m_params, g_params, create_nx_graph=False
        )

        # Check basic properties
        assert graph.adj is not None
        assert graph.adj.shape == (graph.n, graph.n)
        assert graph.adj.nnz > 0

        # Should be symmetric
        assert np.allclose(graph.adj.toarray(), graph.adj.toarray().T)

    def test_eps_graph_weighted(self, clustered_data):
        """Test weighted epsilon-ball graph"""
        m_params = {"metric_name": "euclidean", "sigma": 2.0}
        g_params = {
            "g_method_name": "eps",
            "eps": 6.0,  # Larger radius to connect clusters
            "min_density": 0.001,
            "weighted": True,
            "dist_to_aff": "hk",  # Heat kernel affinities
            "max_deleted_nodes": 0.5,
        }

        # ProximityGraph expects data in (n_features, n_samples) format
        graph = ProximityGraph(
            clustered_data.T, m_params, g_params, create_nx_graph=False
        )

        # Check weighted properties
        assert graph.adj is not None
        assert not np.all(np.isin(graph.adj.data, [0, 1]))  # Should have weights

        # Weights should be positive
        assert np.all(graph.adj.data > 0)

    def test_eps_graph_too_sparse_error(self, clustered_data):
        """Test error when epsilon graph is too sparse"""
        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "eps",
            "eps": 0.1,  # Very small radius
            "min_density": 0.1,  # High minimum density requirement
            "weighted": False,
            "dist_to_aff": None,
            "max_deleted_nodes": 0.5,
        }

        with pytest.raises(ValueError, match="Epsilon graph too sparse"):
            ProximityGraph(clustered_data.T, m_params, g_params, create_nx_graph=False)

    def test_eps_graph_dense_warning(self, clustered_data):
        """Test warning when epsilon graph is too dense"""
        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "eps",
            "eps": 100.0,  # Very large radius
            "min_density": 0.01,
            "weighted": False,
            "dist_to_aff": None,
            "max_deleted_nodes": 0.5,
        }

        # Should construct but print warning
        graph = ProximityGraph(
            clustered_data.T, m_params, g_params, create_nx_graph=False
        )
        assert graph.adj is not None

        # Should be very dense
        nnz_ratio = graph.adj.nnz / (graph.n * (graph.n - 1))
        assert nnz_ratio > 0.5

    def test_eps_graph_different_metrics(self, clustered_data):
        """Test epsilon graph with different distance metrics"""
        metrics = ["manhattan", "chebyshev"]

        for metric in metrics:
            m_params = {"metric_name": metric, "sigma": 1.0}
            g_params = {
                "g_method_name": "eps",
                "eps": 8.0,  # Larger for manhattan/chebyshev metrics
                "min_density": 0.001,
                "weighted": False,
                "dist_to_aff": None,
                "max_deleted_nodes": 0.5,
            }

            graph = ProximityGraph(
                clustered_data.T, m_params, g_params, create_nx_graph=False
            )
            assert graph.adj is not None
            assert graph.adj.nnz > 0


class TestGraphMethods:
    """Test various graph methods"""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing methods"""
        np.random.seed(42)
        data = np.random.randn(3, 30)
        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "knn",
            "nn": 5,
            "weighted": True,
            "dist_to_aff": "hk",
            "max_deleted_nodes": 0.5,
        }
        return ProximityGraph(data, m_params, g_params, create_nx_graph=False)

    def test_scaling_method(self, sample_graph):
        """Test the scaling method"""
        # Get diagonal sums
        diagsums = sample_graph.scaling()

        # Should return list of diagonal sums
        assert isinstance(diagsums, list)
        assert len(diagsums) == sample_graph.n - 1

        # Values should be non-negative (trace of adjacency powers)
        assert all(d >= 0 for d in diagsums)


    def test_custom_metric_function(self):
        """Test graph construction with custom metric function"""
        from numba import njit

        # Define custom metric - must be numba compiled for pynndescent
        @njit
        def custom_metric(x, y):
            return np.sum(np.abs(x - y) ** 1.5) ** (1 / 1.5)

        np.random.seed(42)
        data = np.random.randn(3, 20)

        # Pass the callable metric function directly
        m_params = {"metric_name": custom_metric, "sigma": 1.0}
        g_params = {
            "g_method_name": "knn",
            "nn": 5,  # Increased for better connectivity
            "weighted": False,
            "dist_to_aff": None,
            "max_deleted_nodes": 0.5,
        }

        graph = ProximityGraph(data, m_params, g_params, create_nx_graph=False)
        assert graph.adj is not None
        assert graph.adj.nnz > 0
        assert graph.metric is custom_metric

    def test_get_int_dim_with_logger(self):
        """Test intrinsic dimension estimation with custom logger"""
        import logging

        # Create logger
        logger = logging.getLogger("test_logger")
        logger.setLevel(logging.INFO)

        np.random.seed(42)
        data = np.random.randn(3, 50)

        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "knn",
            "nn": 10,
            "weighted": True,
            "dist_to_aff": "hk",
            "max_deleted_nodes": 0.5,
        }

        graph = ProximityGraph(data, m_params, g_params, create_nx_graph=False)

        # Test with logger
        dim = graph.get_int_dim(method="geodesic", logger=logger)
        assert isinstance(dim, float)

    def test_lost_nodes_handling_with_knn_arrays(self):
        """Test that k-NN arrays are properly filtered when nodes are lost"""
        # Create disconnected data
        np.random.seed(42)
        cluster1 = np.random.randn(3, 15)
        cluster2 = np.random.randn(3, 15) + 100
        data = np.hstack([cluster1, cluster2])

        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "knn",
            "nn": 5,  # Increased for better connectivity
            "weighted": True,
            "dist_to_aff": "hk",
            "max_deleted_nodes": 0.6,
        }

        graph = ProximityGraph(data, m_params, g_params, create_nx_graph=False)

        # Should have lost nodes
        assert hasattr(graph, "lost_nodes")
        assert len(graph.lost_nodes) > 0

        # k-NN arrays should be filtered
        assert graph.knn_indices.shape[0] == graph.n
        assert graph.knn_distances.shape[0] == graph.n

    def test_knn_indices_distances_none_for_non_knn_methods(self):
        """Test that k-NN arrays are None for methods that don't compute them"""
        np.random.seed(42)
        data = np.random.randn(3, 30)

        # Test with auto_knn method
        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "auto_knn",
            "nn": 5,
            "weighted": False,
            "dist_to_aff": None,
            "max_deleted_nodes": 0.5,
        }

        graph = ProximityGraph(data, m_params, g_params, create_nx_graph=False)

        # Should not have k-NN arrays
        assert graph.knn_indices is None
        assert graph.knn_distances is None

    def test_neigh_distmat_initialization_for_eps_graph(self):
        """Test neighbor distance matrix initialization for epsilon graphs"""
        np.random.seed(42)
        data = np.random.randn(3, 30)

        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "eps",
            "eps": 2.0,
            "min_density": 0.001,
            "weighted": False,
            "dist_to_aff": None,
            "max_deleted_nodes": 0.5,
        }

        graph = ProximityGraph(data, m_params, g_params, create_nx_graph=False)

        # Should have neighbor distance matrix initialized
        assert graph.neigh_distmat is not None
        assert sp.issparse(graph.neigh_distmat)
        assert graph.neigh_distmat.shape == graph.adj.shape
