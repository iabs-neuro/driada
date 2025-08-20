"""
Tests for network drawing functions.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from driada.network.net_base import Network
from driada.network.drawing import (
    draw_degree_distr,
    draw_spectrum,
    draw_net,
    show_mat,
    plot_lem_embedding,
)


def create_network_from_edges(edges, directed=False, preprocessing=None):
    """Helper function to create Network from edge list."""
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()
    graph.add_edges_from(edges)
    
    # For empty graphs, don't use preprocessing
    if len(edges) == 0:
        preprocessing = None
    
    return Network(graph=graph, directed=directed, preprocessing=preprocessing)


class TestDegreeDistribution:
    """Test degree distribution plotting."""
    
    def test_draw_degree_distr_undirected(self):
        """Test degree distribution for undirected network."""
        # Create simple undirected network
        edges = [(1, 2), (2, 3), (3, 1), (3, 4)]
        net = create_network_from_edges(edges, directed=False)
        
        # Should not raise error
        draw_degree_distr(net)
        plt.close()
    
    def test_draw_degree_distr_directed(self):
        """Test degree distribution for directed network."""
        # Create directed network
        edges = [(1, 2), (2, 3), (3, 1), (1, 4), (4, 2)]
        net = create_network_from_edges(edges, directed=True)
        
        # Test default (plots all degree types)
        draw_degree_distr(net)
        plt.close()
        
        # Test specific modes
        for mode in ['all', 'in', 'out']:
            draw_degree_distr(net, mode=mode)
            plt.close()
    
    def test_draw_degree_distr_cumulative(self):
        """Test cumulative degree distribution."""
        edges = [(i, i+1) for i in range(10)]  # Chain
        net = create_network_from_edges(edges, directed=False)
        
        # Test cumulative distribution
        draw_degree_distr(net, cumulative=1)
        plt.close()
        
        # Test survival function
        draw_degree_distr(net, cumulative=1, survival=1)
        plt.close()
        
        # Test CDF (not survival)
        draw_degree_distr(net, cumulative=1, survival=0)
        plt.close()
    
    def test_draw_degree_distr_loglog(self):
        """Test log-log degree distribution."""
        # Create scale-free-like network
        edges = []
        for i in range(1, 20):
            for j in range(i):
                if np.random.random() < 0.3:
                    edges.append((i, j))
        
        net = create_network_from_edges(edges, directed=False)
        
        # Test log-log plot
        draw_degree_distr(net, log_log=1)
        plt.close()
        
        # Test log-log with cumulative
        draw_degree_distr(net, log_log=1, cumulative=1)
        plt.close()
    
    def test_draw_degree_distr_single_node_network(self):
        """Test with single node network (no edges)."""
        # Create network with one node but no edges
        graph = nx.Graph()
        graph.add_node(1)
        net = Network(graph=graph, directed=False, preprocessing=None)
        
        # Should handle network with no edges gracefully
        draw_degree_distr(net)
        plt.close()


class TestSpectrum:
    """Test spectrum visualization."""
    
    def test_draw_spectrum_adjacency(self):
        """Test adjacency matrix spectrum."""
        edges = [(1, 2), (2, 3), (3, 1), (3, 4)]
        net = create_network_from_edges(edges, directed=False)
        
        # Test adjacency spectrum
        fig, ax = plt.subplots()
        draw_spectrum(net, mode="adj", ax=ax)
        plt.close(fig)
    
    def test_draw_spectrum_laplacian(self):
        """Test Laplacian matrix spectrum."""
        edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
        net = create_network_from_edges(edges, directed=False)
        
        # Test Laplacian spectrum
        fig, ax = plt.subplots()
        draw_spectrum(net, mode="lap", ax=ax)
        plt.close(fig)
        
        # Test normalized Laplacian
        fig, ax = plt.subplots()
        draw_spectrum(net, mode="nlap", ax=ax)
        plt.close(fig)
    
    def test_draw_spectrum_custom_colors(self):
        """Test spectrum with custom colors."""
        edges = [(i, j) for i in range(5) for j in range(i+1, 5)]  # Complete graph
        net = create_network_from_edges(edges, directed=False)
        
        # Test with custom colors
        fig, ax = plt.subplots()
        colors = ['red', 'blue', 'green', 'yellow', 'purple']
        draw_spectrum(net, mode="adj", ax=ax, colors=colors)
        plt.close(fig)
        
        # Test with colormap
        fig, ax = plt.subplots()
        draw_spectrum(net, mode="adj", ax=ax, cmap='viridis')
        plt.close(fig)
    
    def test_draw_spectrum_nbins(self):
        """Test spectrum with different bin numbers."""
        # Create larger network for meaningful histogram
        edges = [(i, j) for i in range(10) for j in range(i+1, 10) if np.random.random() < 0.3]
        net = create_network_from_edges(edges, directed=False)
        
        # Test with different nbins
        for nbins in [5, 10, 20]:
            fig, ax = plt.subplots()
            draw_spectrum(net, mode="adj", ax=ax, nbins=nbins)
            plt.close(fig)


class TestNetworkDrawing:
    """Test network graph drawing."""
    
    def test_draw_net_basic(self):
        """Test basic network drawing."""
        edges = [(1, 2), (2, 3), (3, 1), (1, 4)]
        net = create_network_from_edges(edges, directed=False)
        
        # Test basic drawing
        fig, ax = plt.subplots()
        draw_net(net, ax=ax)
        plt.close(fig)
    
    def test_draw_net_with_colors(self):
        """Test network drawing with node colors."""
        edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
        net = create_network_from_edges(edges, directed=False)
        
        # Test with colors
        fig, ax = plt.subplots()
        colors = [0.1, 0.3, 0.5, 0.7]  # Numeric values for colormap
        draw_net(net, colors=colors, ax=ax)
        plt.close(fig)
    
    def test_draw_net_with_sizes(self):
        """Test network drawing with node sizes."""
        edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
        net = create_network_from_edges(edges, directed=False)
        
        # Test with node sizes
        fig, ax = plt.subplots()
        nodesize = [100, 200, 300, 400]
        draw_net(net, nodesize=nodesize, ax=ax)
        plt.close(fig)
    
    def test_draw_net_directed(self):
        """Test directed network drawing."""
        edges = [(1, 2), (2, 3), (3, 1)]
        net = create_network_from_edges(edges, directed=True)
        
        # Should handle directed networks
        fig, ax = plt.subplots()
        draw_net(net, ax=ax)
        plt.close(fig)
    
    def test_draw_net_single_node(self):
        """Test drawing network with single node."""
        graph = nx.Graph()
        graph.add_node(1)
        net = Network(graph=graph, directed=False, preprocessing=None)
        
        # Should handle gracefully
        fig, ax = plt.subplots()
        draw_net(net, ax=ax)
        plt.close(fig)


class TestShowMat:
    """Test matrix visualization."""
    
    def test_show_mat_adjacency(self):
        """Test showing adjacency matrix."""
        edges = [(1, 2), (2, 3), (3, 1)]
        net = create_network_from_edges(edges, directed=False)
        
        fig, ax = plt.subplots()
        show_mat(net, mode="adj", ax=ax)
        plt.close(fig)
    
    def test_show_mat_laplacian(self):
        """Test showing Laplacian matrix."""
        edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
        net = create_network_from_edges(edges, directed=False)
        
        fig, ax = plt.subplots()
        show_mat(net, mode="lap", ax=ax)
        plt.close(fig)
    
    def test_show_mat_directed(self):
        """Test showing matrix for directed network."""
        edges = [(1, 2), (2, 3), (3, 1), (1, 3)]
        net = create_network_from_edges(edges, directed=True)
        
        fig, ax = plt.subplots()
        show_mat(net, mode="adj", ax=ax)
        plt.close(fig)
    
    def test_show_mat_binary(self):
        """Test showing matrix with binary visualization."""
        # Create a larger sparse network
        edges = [(i, i+1) for i in range(20)]  # Chain
        net = create_network_from_edges(edges, directed=False)
        
        fig, ax = plt.subplots()
        # Use bool dtype for binary visualization
        show_mat(net, dtype=bool, ax=ax)
        plt.close(fig)


class TestLEMEmbedding:
    """Test Laplacian Eigenmaps embedding visualization."""
    
    def test_plot_lem_2d(self):
        """Test 2D LEM embedding."""
        # Create a simple connected network (not complete graph to avoid eigenvector issues)
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (1, 3)]
        net = create_network_from_edges(edges, directed=False)
        
        # Test 2D embedding
        plot_lem_embedding(net, ndim=2)
        plt.close()
    
    def test_plot_lem_3d(self):
        """Test 3D LEM embedding."""
        # Create a larger network for 3D embedding (avoid complete/regular graphs)
        edges = [(i, (i+1)%10) for i in range(10)]  # Ring
        edges += [(i, (i+3)%10) for i in range(10)]  # Additional connections
        edges += [(0, 5), (2, 7), (4, 9)]  # Some cross connections
        net = create_network_from_edges(edges, directed=False)
        
        # Test 3D embedding
        plot_lem_embedding(net, ndim=3)
        plt.close()
    
    def test_plot_lem_with_colors(self):
        """Test LEM embedding with node colors."""
        # Create a deterministic network (avoid random in tests)
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), 
                 (0, 3), (1, 4), (2, 5)]
        net = create_network_from_edges(edges, directed=False)
        
        # Node colors
        colors = list(range(len(net.graph.nodes())))
        
        # Test with colors
        plot_lem_embedding(net, ndim=2, colors=colors)
        plt.close()
    
    def test_plot_lem_disconnected(self):
        """Test LEM embedding with disconnected graph."""
        # Create disconnected components
        edges = [(1, 2), (2, 3), (4, 5), (5, 6)]
        net = create_network_from_edges(edges, directed=False)
        
        # Should raise exception for disconnected graphs
        with pytest.raises(Exception, match="graph is not connected"):
            plot_lem_embedding(net, ndim=2)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_node(self):
        """Test network with single node."""
        graph = nx.Graph()
        graph.add_node(1)
        net = Network(graph=graph, preprocessing=None)
        
        # Should handle single node
        fig, ax = plt.subplots()
        draw_net(net, ax=ax)
        plt.close(fig)
        
        # Degree distribution of single node
        draw_degree_distr(net)
        plt.close()
    
    def test_self_loops(self):
        """Test network with self-loops."""
        edges = [(1, 1), (1, 2), (2, 2), (2, 3)]
        net = create_network_from_edges(edges, directed=False)
        
        # Should handle self-loops
        draw_degree_distr(net)
        plt.close()
        
        fig, ax = plt.subplots()
        draw_net(net, ax=ax)
        plt.close(fig)