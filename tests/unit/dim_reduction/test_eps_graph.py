"""Tests for epsilon graph construction in ProximityGraph"""

import pytest
import numpy as np
from driada.dim_reduction.graph import ProximityGraph


class TestEpsGraph:
    """Test epsilon-ball graph construction"""

    def test_eps_graph_basic(self):
        """Test basic epsilon graph construction"""
        np.random.seed(42)
        data = np.random.randn(2, 50)  # 2D data, 50 points

        m_params = {"metric_name": "euclidean", "sigma": None}
        g_params = {
            "g_method_name": "eps",
            "eps": 1.0,
            "min_density": 0.01,
            "weighted": False,
            "dist_to_aff": None,
            "max_deleted_nodes": 0.5,
        }

        graph = ProximityGraph(data, m_params, g_params, create_nx_graph=False)

        # Check basic properties
        assert graph.adj is not None
        assert graph.bin_adj is not None
        assert graph.adj.shape == (graph.n, graph.n)
        assert graph.adj.nnz > 0  # Should have some edges

    def test_eps_graph_weighted(self):
        """Test weighted epsilon graph with affinities"""
        np.random.seed(42)
        data = np.random.randn(2, 30)

        m_params = {"metric_name": "euclidean", "sigma": 1.0}
        g_params = {
            "g_method_name": "eps",
            "eps": 1.5,
            "min_density": 0.01,
            "weighted": True,
            "dist_to_aff": "hk",
            "max_deleted_nodes": 0.5,
        }

        graph = ProximityGraph(data, m_params, g_params, create_nx_graph=False)

        # Check weighted graph properties
        assert graph.adj is not None
        assert graph.neigh_distmat is not None
        assert graph.adj.nnz == graph.bin_adj.nnz
        # Affinities should be between 0 and 1
        assert np.all(graph.adj.data >= 0)
        assert np.all(graph.adj.data <= 1)

    def test_eps_graph_too_sparse(self):
        """Test error when epsilon graph is too sparse"""
        np.random.seed(42)
        data = np.random.randn(2, 50)

        m_params = {"metric_name": "euclidean", "sigma": None}
        g_params = {
            "g_method_name": "eps",
            "eps": 0.01,  # Very small epsilon
            "min_density": 0.1,  # High minimum density
            "weighted": False,
            "dist_to_aff": None,
            "max_deleted_nodes": 0.5,
        }

        with pytest.raises(ValueError, match="Epsilon graph too sparse"):
            ProximityGraph(data, m_params, g_params, create_nx_graph=False)

    def test_eps_graph_different_metrics(self):
        """Test epsilon graph with different distance metrics"""
        np.random.seed(42)
        data = np.random.rand(3, 40)  # 3D data, 40 points

        for metric in ["manhattan", "cosine"]:
            m_params = {"metric_name": metric, "sigma": None}
            g_params = {
                "g_method_name": "eps",
                "eps": 1.0 if metric == "manhattan" else 0.5,
                "min_density": 0.01,
                "weighted": False,
                "dist_to_aff": None,
                "max_deleted_nodes": 0.5,
            }

            graph = ProximityGraph(data, m_params, g_params, create_nx_graph=False)
            assert graph.adj is not None
            assert graph.adj.nnz > 0

    def test_eps_graph_dense_warning(self, capsys):
        """Test warning when epsilon graph is too dense"""
        np.random.seed(42)
        # Create clustered data for dense graph
        data = np.random.randn(2, 20) * 0.1  # Small variance = dense graph

        m_params = {"metric_name": "euclidean", "sigma": None}
        g_params = {
            "g_method_name": "eps",
            "eps": 2.0,  # Large epsilon for dense graph
            "min_density": 0.01,
            "weighted": False,
            "dist_to_aff": None,
            "max_deleted_nodes": 0.9,  # High tolerance for dense graph
        }

        graph = ProximityGraph(data, m_params, g_params, create_nx_graph=False, verbose=True)
        captured = capsys.readouterr()

        # Should see density warning
        assert "WARNING: Epsilon graph is dense" in captured.out
