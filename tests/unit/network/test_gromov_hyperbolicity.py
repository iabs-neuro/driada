"""Test Gromov hyperbolicity calculation in Network class."""

import numpy as np
import scipy.sparse as sp
import networkx as nx
import pytest

from driada.network.net_base import Network


class TestGromovHyperbolicity:
    """Test suite for Gromov hyperbolicity calculation."""

    def test_tree_hyperbolicity(self):
        """Test that trees have hyperbolicity 0."""
        # Create a simple tree
        edges = [(0, 1), (1, 2), (1, 3), (3, 4)]
        G = nx.Graph()
        G.add_edges_from(edges)
        
        adj = nx.adjacency_matrix(G)
        net = Network(adj=sp.csr_matrix(adj), create_nx_graph=True)
        
        # Trees should have hyperbolicity 0
        hyp = net.calculate_gromov_hyperbolicity(num_samples=1000)
        assert abs(hyp) < 1e-10, f"Tree should have hyperbolicity 0, got {hyp}"

    def test_cycle_hyperbolicity(self):
        """Test hyperbolicity of a cycle graph."""
        # Create a cycle with n nodes
        n = 8
        G = nx.cycle_graph(n)
        
        adj = nx.adjacency_matrix(G)
        net = Network(adj=sp.csr_matrix(adj), create_nx_graph=True)
        
        # For a cycle, hyperbolicity depends on n
        # For n=8, it should be around 0.5-1.0
        hyp = net.calculate_gromov_hyperbolicity(num_samples=5000)
        assert 0 <= hyp <= 1.5, f"Cycle should have moderate hyperbolicity, got {hyp}"

    def test_complete_graph_hyperbolicity(self):
        """Test hyperbolicity of a complete graph."""
        # Create a complete graph
        n = 6
        G = nx.complete_graph(n)
        
        adj = nx.adjacency_matrix(G)
        net = Network(adj=sp.csr_matrix(adj), create_nx_graph=True)
        
        # Complete graphs have hyperbolicity 0 (all distances are 1)
        hyp = net.calculate_gromov_hyperbolicity(num_samples=5000)
        expected = 0.0
        assert abs(hyp - expected) < 0.1, f"Complete graph should have hyperbolicity ~{expected}, got {hyp}"

    def test_grid_hyperbolicity(self):
        """Test hyperbolicity of a grid graph."""
        # Create a small grid
        G = nx.grid_2d_graph(4, 4)
        
        # Convert node labels to integers
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        
        adj = nx.adjacency_matrix(G)
        net = Network(adj=sp.csr_matrix(adj), create_nx_graph=True)
        
        # Grids have moderate hyperbolicity
        hyp = net.calculate_gromov_hyperbolicity(num_samples=5000)
        assert 0 <= hyp <= 2.0, f"Grid should have moderate hyperbolicity, got {hyp}"

    def test_return_list_option(self):
        """Test return_list option."""
        # Create a simple graph
        G = nx.karate_club_graph()
        
        adj = nx.adjacency_matrix(G)
        net = Network(adj=sp.csr_matrix(adj), create_nx_graph=True)
        
        # Test return_list=True
        num_samples = 1000
        hyp_list = net.calculate_gromov_hyperbolicity(num_samples=num_samples, return_list=True)
        
        assert isinstance(hyp_list, list)
        assert len(hyp_list) == num_samples
        assert all(isinstance(h, (int, float)) for h in hyp_list)
        assert all(h >= 0 for h in hyp_list)  # Hyperbolicity is non-negative
        
        # Average should match non-list return
        hyp_avg = net.calculate_gromov_hyperbolicity(num_samples=num_samples, return_list=False)
        # Allow small difference due to random sampling
        assert abs(np.mean(hyp_list) - hyp_avg) < 0.05

    def test_no_networkx_graph_error(self):
        """Test error when NetworkX graph is not available."""
        # Create network without NetworkX graph
        A = sp.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        net = Network(adj=A, create_nx_graph=False)
        
        with pytest.raises(ValueError, match="NetworkX graph not available"):
            net.calculate_gromov_hyperbolicity()

    def test_directed_graph(self):
        """Test hyperbolicity calculation on directed graphs."""
        # Create a directed graph
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
        
        adj = nx.adjacency_matrix(G)
        net = Network(adj=sp.csr_matrix(adj), create_nx_graph=True)
        
        # Should work on directed graphs too
        hyp = net.calculate_gromov_hyperbolicity(num_samples=1000)
        assert hyp >= 0  # Basic sanity check

    def test_small_graph_samples(self):
        """Test with small graphs where we can't sample many 4-tuples."""
        # Create a very small graph (4 nodes - minimum for hyperbolicity)
        G = nx.path_graph(4)
        
        adj = nx.adjacency_matrix(G)
        net = Network(adj=sp.csr_matrix(adj), create_nx_graph=True)
        
        # With 4 nodes, there's only 1 possible 4-tuple
        hyp = net.calculate_gromov_hyperbolicity(num_samples=10)
        assert hyp >= 0  # Should still work

    def test_disconnected_graph_error(self):
        """Test error with disconnected graphs after preprocessing."""
        # Create a disconnected graph with small components
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3), (4, 5)])  # Three disconnected components
        
        adj = nx.adjacency_matrix(G)
        # Network will take giant component by default, which has only 2 nodes
        net = Network(adj=sp.csr_matrix(adj), create_nx_graph=True)
        
        # Should raise error because giant component has < 4 nodes
        with pytest.raises(ValueError, match="Graph must have at least 4 nodes"):
            net.calculate_gromov_hyperbolicity(num_samples=100)